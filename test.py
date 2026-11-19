#!/usr/bin/env python3
"""
Interactive PYNQ test for the FP32 systolic-array MATRIX-MULTIPLY accelerator.

Operation checked:
        Y = X . W        X, W, Y are L x L FP32 matrices, L = 4, 8 or 16
        Y[m][n] = sum over k of X[m][k] * W[k][n]

Runs on the PYNQ-Z2 together with:
    design_1.bit
    design_1.hwh
    test.py

Usage:   sudo python3 test.py            (or run in a Jupyter terminal)
         python3 test.py my_design.bit   (another bitstream)

Menu:    1 / 2 / 3  -> 4x4 / 8x8 / 16x16, then type X and W row by row,
                       or type  r  at the first prompt for random matrices
         4          -> run all three sizes with random data and print a
                       side-by-side performance / power comparison
"""
import math
import random
import struct
import sys
import time
from pathlib import Path

from pynq import Overlay

try:
    from pynq import Clocks
except ImportError:          # older PYNQ images
    Clocks = None

BITSTREAM = "design_1.bit"
IP_NAME = "cnn_accelerator_axi_0"

# Clock of the accelerator, used for latency / throughput.
#   None -> read the PL clock 0 from PYNQ, fall back to FALLBACK_CLOCK_MHZ
#   or set a number (e.g. 100.0) if the IP runs on another clock.
CLOCK_MHZ = 94.118
FALLBACK_CLOCK_MHZ = 94.118
# ----------------------------------------------------------------------
# AXI register map (matches cnn_accelerator_axi.v)
# ----------------------------------------------------------------------
REG_CTRL = 0x000      # [0] start, [1] soft reset (self-clearing pulses)
REG_SIZE = 0x004      # [1:0] 0 = 4x4, 1 = 8x8, 2 = 16x16
REG_STATUS = 0x008
REG_FEATURE = 0x00C   # X, one word per write, row-major
REG_WEIGHT = 0x010    # W, one word per write, row-major
REG_PERF = 0x014      # 7 counters, 4 bytes apart
REG_ID = 0x030
REG_OUT = 0x400       # Y[m][n] at REG_OUT + 4*(m*L + n)

ID_VALUE = 0x4D4D3031           # "MM01"

CTRL_START = 0x1
CTRL_RESET = 0x2
ST_DONE = 0x01
ST_BUSY = 0x02
ST_IN_LOADED = 0x04
ST_W_LOADED = 0x08

# order of the counters at REG_PERF
PERF_TOTAL, PERF_FEATURE, PERF_WEIGHT, PERF_COMPUTE, PERF_STORE, \
    PERF_BUSY, PERF_IDLE = range(7)
NUM_PERF = 7

SIZES = {1: 4, 2: 8, 3: 16}
SIZE_CODE = {4: 0, 8: 1, 16: 2}
PHYSICAL_N = 4                  # physical systolic array is 4x4 (16 PEs)
LOAD_TIMEOUT_S = 2.0
RUN_TIMEOUT_S = 5.0

# ----------------------------------------------------------------------
# Power model constants.  Preliminary values taken from the Vivado power
# report of the earlier design; replace them with the numbers from the
# report of THIS bitstream (Report Power -> total / static / PS7) to get
# estimates that fit the matrix design.
# ----------------------------------------------------------------------
REF_TOTAL_W = 1.705
REF_STATIC_W = 0.137
REF_PS7_DYNAMIC_W = 1.525
REF_BUSY_RATIO = 96.17          # busy ratio the reference report was made at


# ----------------------------------------------------------------------
# Number <-> register word
# ----------------------------------------------------------------------
def to_word(value):
    """FP32 bit pattern of a number."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def from_word(word):
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


def to_f32(value):
    """Round a Python float to the nearest FP32 value."""
    return from_word(to_word(value))


def parse_value(token):
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"{token!r} is not a valid number.") from exc
    if not math.isfinite(value):
        raise ValueError("NaN and infinity are not supported.")
    try:
        struct.pack("<f", value)
    except OverflowError as exc:
        raise ValueError("Value is outside the finite FP32 range.") from exc
    return to_f32(value)


def fmt(value):
    return f"{value:.7g}"


# ----------------------------------------------------------------------
# User input
# ----------------------------------------------------------------------
def ask_size():
    print("\nSelect logical size:")
    print("1. 4 x 4")
    print("2. 8 x 8")
    print("3. 16 x 16")
    print("4. Compare all three (random data)")
    while True:
        choice = input("\nEnter choice (1/2/3/4, q to quit): ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return None
        if choice in ("1", "2", "3"):
            return SIZES[int(choice)]
        if choice == "4":
            return 0
        print("Invalid choice. Select 1, 2, 3 or 4.")


def read_row(prompt, count):
    """Read exactly `count` numbers (spaces or commas). Returns None if the
    user typed r / random, 'q' returns the string 'q'."""
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("r", "random"):
            return None
        try:
            values = [parse_value(tok) for tok in raw.replace(",", " ").split()]
        except ValueError as exc:
            print(exc)
            continue
        if len(values) != count:
            print(f"Please enter exactly {count} values (got {len(values)}).")
            continue
        return values


def random_matrix(L):
    return [[to_f32(round(random.uniform(-4.0, 4.0), 3)) for _ in range(L)]
            for _ in range(L)]


def ask_matrices(L):
    """Returns (X, W). At the first X-row prompt 'r' gives random X and W.
    At any later prompt 'r' fills only that matrix's remaining rows randomly."""
    print(f"\nEnter X ({L} x {L}) row by row, {L} numbers per row.")
    print("Type  r  at the first row for random X and W.")
    X = []
    for i in range(L):
        row = read_row(f"  X row {i} ({L} numbers): ", L)
        if row is None:
            if i == 0:
                return random_matrix(L), random_matrix(L)
            print("  'r' is only allowed at the first row; enter the numbers.")
            row = read_row(f"  X row {i} ({L} numbers): ", L)
            if row is None:
                row = [0.0] * L
        X.append(row)

    print(f"\nEnter W ({L} x {L}) row by row, {L} numbers per row.")
    print("Type  r  at the first row for a random W.")
    W = []
    for k in range(L):
        row = read_row(f"  W row {k} ({L} numbers): ", L)
        if row is None:
            if k == 0:
                return X, random_matrix(L)
            print("  'r' is only allowed at the first row; enter the numbers.")
            row = read_row(f"  W row {k} ({L} numbers): ", L)
            if row is None:
                row = [0.0] * L
        W.append(row)
    return X, W


# ----------------------------------------------------------------------
# Hardware access
# ----------------------------------------------------------------------
def load_overlay():
    bit_path = Path(BITSTREAM)
    if not bit_path.is_file():
        raise FileNotFoundError(
            f"Bitstream not found: {bit_path.resolve()}. "
            "Place design_1.bit in the current directory or pass its path."
        )

    hwh_path = bit_path.with_suffix(".hwh")
    if not hwh_path.is_file():
        print(f"WARNING: matching HWH file not found: {hwh_path.resolve()}")

    print("Loading FPGA bitstream...")
    overlay = Overlay(str(bit_path))
    available = list(overlay.ip_dict.keys())
    print("Available IPs:", available)

    if IP_NAME not in overlay.ip_dict:
        raise RuntimeError(
            f"IP '{IP_NAME}' not found. Available IPs: {available}. "
            "Set IP_NAME to the exact accelerator IP name."
        )

    ip = getattr(overlay, IP_NAME)
    ident = ip.read(REG_ID) & 0xFFFFFFFF
    if ident != ID_VALUE:
        raise RuntimeError(
            f"ID register = 0x{ident:08X}, expected 0x{ID_VALUE:08X}. "
            "This bitstream is not the matrix-multiply design."
        )
    print("Accelerator connected:", IP_NAME, "(matrix-multiply design)")
    return ip


def get_clock_mhz():
    if CLOCK_MHZ:
        return float(CLOCK_MHZ)
    try:
        mhz = float(Clocks.fclk0_mhz)
        if mhz > 0:
            return mhz
    except Exception:
        pass
    return FALLBACK_CLOCK_MHZ


def wait_for_status(ip, mask, timeout_s, description, poll_sleep_s=0.0):
    start = time.monotonic()
    while True:
        status = ip.read(REG_STATUS)
        if (status & mask) == mask:
            return status
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for {description}; STATUS=0x{status:08X}"
            )
        if poll_sleep_s:
            time.sleep(poll_sleep_s)


def load_inputs(ip, L, X, W):
    """Reset the core, select the size, write X then W (row-major)."""
    if L not in SIZE_CODE:
        raise ValueError(f"Unsupported logical size: {L}")
    if len(X) != L or len(W) != L or any(len(r) != L for r in X + W):
        raise ValueError("X and W must both be L x L.")

    ip.write(REG_CTRL, CTRL_RESET)            # clears pipelines, done, flags
    time.sleep(0.01)
    ip.write(REG_SIZE, SIZE_CODE[L])          # a size change clears write ptrs
    time.sleep(0.01)

    for row in X:
        for value in row:
            ip.write(REG_FEATURE, to_word(value))
    for row in W:
        for value in row:
            ip.write(REG_WEIGHT, to_word(value))

    wait_for_status(
        ip, ST_IN_LOADED | ST_W_LOADED, LOAD_TIMEOUT_S,
        "X and W buffers to load", poll_sleep_s=0.001
    )


def run_and_read(ip, L):
    """Start one run, wait for DONE, read Y and the counters.
    Returns (Y matrix, performance counters, PS-observed time in us)."""
    t0 = time.perf_counter()
    ip.write(REG_CTRL, CTRL_START)
    wait_for_status(ip, ST_DONE, RUN_TIMEOUT_S, "accelerator DONE")
    ps_us = (time.perf_counter() - t0) * 1e6

    Y = [[from_word(ip.read(REG_OUT + 4 * (m * L + n))) for n in range(L)]
         for m in range(L)]
    perf = [ip.read(REG_PERF + 4 * i) & 0xFFFFFFFF for i in range(NUM_PERF)]
    return Y, perf, ps_us


# ----------------------------------------------------------------------
# Reference model
# ----------------------------------------------------------------------
def reference(X, W):
    """Expected Y (double precision) and the magnitude sum|x*w| per element,
    used to scale the FP32 tolerance (robust against cancellation)."""
    L = len(X)
    Y = [[0.0] * L for _ in range(L)]
    mag = [[0.0] * L for _ in range(L)]
    for m in range(L):
        for n in range(L):
            s = 0.0
            a = 0.0
            for k in range(L):
                p = X[m][k] * W[k][n]
                s += p
                a += abs(p)
            Y[m][n] = s
            mag[m][n] = a
    return Y, mag


def element_ok(actual, expected, scale):
    # FP32 hardware adds the products in its own order; allow a few ulp
    # relative to the magnitude of the terms.
    return abs(actual - expected) <= 1e-5 * max(1.0, scale)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def print_matrix(title, M):
    print(f"\n{title}")
    for row in M:
        print("  " + " ".join(f"{v:9.4g}" for v in row))


def check_outputs(X, W, Y, verbose=True):
    """Self-check. Returns (all_ok, number_wrong, max_abs_error)."""
    L = len(X)
    expected, mag = reference(X, W)
    wrong = 0
    max_err = 0.0
    lines = []
    for m in range(L):
        for n in range(L):
            a, e = Y[m][n], expected[m][n]
            ok = element_ok(a, e, mag[m][n])
            err = abs(a - e)
            max_err = max(max_err, err)
            if not ok:
                wrong += 1
            lines.append((m, n, a, e, ok))

    if verbose:
        print_matrix("Y = X . W  (hardware)", Y)
        print_matrix("Y expected (software reference)", expected)
        print("\nOUTPUT VERIFICATION")
        show_all = L <= 4
        for m, n, a, e, ok in lines:
            if show_all or not ok:
                print(f"Y[{m}][{n}]: actual={fmt(a)}, expected={fmt(e)} "
                      f"-> {'PASS' if ok else 'FAIL'}")
        print(f"{L * L - wrong} of {L * L} outputs match "
              f"(max abs error {max_err:.3g}) -> "
              f"{'PASS' if wrong == 0 else 'FAIL'}")
    return wrong == 0, wrong, max_err


def estimate_power_w(busy_ratio):
    """Preliminary model-based power estimate, not a hardware measurement.

    Total = static + PS7 dynamic + (remaining dynamic scaled by the busy
    ratio relative to the reference busy ratio).  All three logical sizes
    run on the same physical 4x4 array, so they differ only through the
    busy ratio.  Workload-specific SAIF or rail measurements are needed for
    verified power values.
    """
    other_dynamic_w = REF_TOTAL_W - REF_STATIC_W - REF_PS7_DYNAMIC_W
    if busy_ratio <= 0:
        return REF_TOTAL_W
    scaled = other_dynamic_w * min(100.0, busy_ratio) / REF_BUSY_RATIO
    return REF_STATIC_W + REF_PS7_DYNAMIC_W + scaled


def metrics(L, perf, ps_us):
    """All derived numbers for one run."""
    total = perf[PERF_TOTAL]
    compute = perf[PERF_COMPUTE]
    busy = perf[PERF_BUSY]
    clock_mhz = get_clock_mhz()

    macs = L * L * L
    latency_us = total / clock_mhz                     # cycles / MHz = us
    compute_s = compute / (clock_mhz * 1e6)
    total_s = total / (clock_mhz * 1e6)
    busy_ratio = 100.0 * busy / total if total else 0.0
    power_w = estimate_power_w(busy_ratio)
    return {
        "L": L,
        "passes": (L // PHYSICAL_N) ** 2,
        "macs": macs,
        "total": total,
        "compute": compute,
        "latency_us": latency_us,
        "mmac_s": (macs / compute_s / 1e6) if compute_s > 0 else 0.0,
        "gflops": (2.0 * macs / total_s / 1e9) if total_s > 0 else 0.0,
        "util": (100.0 * macs / (compute * PHYSICAL_N * PHYSICAL_N)
                 if compute else 0.0),
        "busy_ratio": busy_ratio,
        "power_w": power_w,
        "energy_uj": power_w * latency_us,
        "ps_us": ps_us,
        "clock_mhz": clock_mhz,
    }


def show_performance(L, perf, ps_us):
    s = metrics(L, perf, ps_us)
    print("\n" + "-" * 54)
    print(f"             {L}x{L} PERFORMANCE SUMMARY")
    print("-" * 54)
    rows = [
        ("Matrix multiply",      f"{L}x{L} . {L}x{L}"),
        ("Tile passes (4x4)",    f"{s['passes']}"),
        ("MAC operations",       f"{s['macs']:,}"),
        ("Total cycles",         f"{perf[PERF_TOTAL]:,}"),
        ("Feature load cycles",  f"{perf[PERF_FEATURE]:,}"),
        ("Weight load cycles",   f"{perf[PERF_WEIGHT]:,}"),
        ("Compute cycles",       f"{perf[PERF_COMPUTE]:,}"),
        ("Output store cycles",  f"{perf[PERF_STORE]:,}"),
        ("Busy cycles",          f"{perf[PERF_BUSY]:,}"),
        ("Idle cycles",          f"{perf[PERF_IDLE]:,}"),
        ("Clock",                f"{s['clock_mhz']:.1f} MHz"),
        ("Hardware latency",     f"{s['latency_us']:.3f} µs"),
        ("Throughput",           f"{s['mmac_s']:.2f} MMAC/s"),
        ("Performance",          f"{s['gflops']:.4f} GFLOPS"),
        ("PE utilisation",       f"{s['util']:.2f}%"),
        ("Busy-cycle ratio",     f"{s['busy_ratio']:.2f}%"),
        ("PS-observed time",     f"{ps_us:.2f} µs"),
    ]
    for label, value in rows:
        print(f"  {label:<22} {value:>18}")
    print("-" * 54)
    print(f"  {'Power consumed (estimated)':<26} {s['power_w']:>11.3f} W")
    print(f"  {'Energy per run (estimated)':<26} {s['energy_uj']:>10.3f} µJ")
    print("-" * 54)


def show_comparison(results):
    """results: list of (metrics dict, passed bool)."""
    width = 98
    print("\n" + "=" * width)
    print("        ARCHITECTURE COMPARISON  (same physical 4x4 array, "
          "logical size selected at run time)")
    print("=" * width)
    head = (f"{'Size':<7}{'Passes':>7}{'MACs':>8}{'Cycles':>9}{'Latency µs':>12}"
            f"{'MMAC/s':>9}{'GFLOPS':>9}{'Util %':>8}{'Power W*':>10}"
            f"{'Energy µJ*':>12}{'Check':>7}")
    print(head)
    print("-" * width)
    for s, ok in results:
        print(f"{str(s['L']) + 'x' + str(s['L']):<7}{s['passes']:>7}"
              f"{s['macs']:>8,}{s['total']:>9,}{s['latency_us']:>12.3f}"
              f"{s['mmac_s']:>9.2f}{s['gflops']:>9.4f}{s['util']:>8.2f}"
              f"{s['power_w']:>10.3f}{s['energy_uj']:>12.3f}"
              f"{'PASS' if ok else 'FAIL':>7}")
    print("-" * width)
    print("* power / energy are model-based estimates, not measurements.")


# ----------------------------------------------------------------------
# One test
# ----------------------------------------------------------------------
def run_one(ip, L, X, W, verbose=True):
    """Load, run, check. Returns (metrics, passed) or None on error."""
    if verbose:
        print_matrix("X", X)
        print_matrix("W", W)
    try:
        load_inputs(ip, L, X, W)
        if verbose:
            print("\nInputs submitted.")
        Y, perf, ps_us = run_and_read(ip, L)
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"\nHardware test error: {exc}")
        return None

    ok, _, _ = check_outputs(X, W, Y, verbose=verbose)
    if verbose:
        show_performance(L, perf, ps_us)
    return metrics(L, perf, ps_us), ok


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global BITSTREAM
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".bit"):
        BITSTREAM = sys.argv[1]

    try:
        ip = load_overlay()
    except Exception as exc:
        print(f"\nCould not load accelerator: {exc}")
        return

    print("=" * 54)
    print("   SYSTOLIC-ARRAY MATRIX MULTIPLY INTERACTIVE TEST")
    print("=" * 54)

    while True:
        L = ask_size()
        if L is None:
            break

        if L == 0:                                   # compare all sizes
            results = []
            for size in (4, 8, 16):
                print("\n" + "=" * 54)
                print(f"RUNNING {size}x{size} MATRIX MULTIPLY (random data)")
                print("=" * 54)
                X, W = random_matrix(size), random_matrix(size)
                res = run_one(ip, size, X, W, verbose=False)
                if res is not None:
                    s, ok = res
                    print(f"  {s['total']:,} cycles, latency "
                          f"{s['latency_us']:.3f} µs -> {'PASS' if ok else 'FAIL'}")
                    results.append(res)
            if results:
                show_comparison(results)
        else:
            X, W = ask_matrices(L)
            print("\n" + "=" * 54)
            print(f"RUNNING {L}x{L} MATRIX MULTIPLY TEST")
            print("=" * 54)
            run_one(ip, L, X, W, verbose=True)

        again = input("\nRun another test? (y/n) [y]: ").strip().lower()
        if again in ("n", "no"):
            break

    print("Done.")


if __name__ == "__main__":
    main()
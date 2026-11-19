#!/usr/bin/env python3
"""
cnn_pynq.py  -  MNIST CNN on the PYNQ board with the systolic-array
                matrix-multiply accelerator.

    28x28 digit -> conv 3x3 x8 -> ReLU -> max-pool 2x2 -> fully connected -> digit

The two matrix-heavy layers (convolution and fully connected) run on the FPGA,
16 x 16 matrix products at a time; ReLU, max-pool and the final argmax run in
software.  The mapping of the layers onto the 16 x 16 products is in
cnn_reference.py, the board access is in accel_driver.py.

Put these files in one folder on the board:
    cnn_pynq.py  cnn_reference.py  accel_driver.py
    design_1.bit  design_1.hwh
    model/mnist_cnn.npz  model/mnist_test_subset.npz      (from train_mnist.py)

Usage
    sudo python3 cnn_pynq.py                 menu: evaluate / classify one image
    sudo python3 cnn_pynq.py --n 200         evaluate 200 test images and exit
    python3 cnn_pynq.py --sim --n 64         no board: software model of the accelerator
Options: --model --subset --bitstream --ip --show --clock-mhz --slow-io
"""
import argparse
import sys
import time

import numpy as np

import cnn_reference as cr

BATCH = cr.L                       # images per fully-connected group (16)

# ----------------------------------------------------------------------
# Power model constants (same idea as the earlier test script): replace the
# reference values with the numbers of the Vivado power report of YOUR
# bitstream.  The result is an ESTIMATE, not a measurement.
# ----------------------------------------------------------------------
REF_TOTAL_W = 1.705
REF_STATIC_W = 0.137
REF_PS7_DYNAMIC_W = 1.525
REF_BUSY_RATIO = 96.17             # busy ratio of the reference report

MACS_PER_IMAGE = 676 * 9 * 8 + 1352 * 10          # useful multiply-adds: 62,192


# ======================================================================
# Helpers
# ======================================================================
def softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


def ascii_image(img):
    chars = " .:-=+*#%@"
    lines = []
    for row in img[::1]:
        lines.append("  " + "".join(chars[min(9, int(p) * 10 // 256)] * 2 for p in row))
    return "\n".join(lines)


def snap(accel):
    s = accel.stats
    return {"runs": s.runs, "cycles": s.cycles, "xw": s.x_writes, "ww": s.w_writes,
            "rd": s.reads,
            "t_write": getattr(accel, "time_write", 0.0),
            "t_run": getattr(accel, "time_run", 0.0),
            "t_read": getattr(accel, "time_read", 0.0)}


class LayerStat:
    def __init__(self):
        self.runs = self.cycles = self.words = 0
        self.wall = 0.0

    def add(self, a, b, seconds):
        self.runs += b["runs"] - a["runs"]
        self.cycles += b["cycles"] - a["cycles"]
        self.words += (b["xw"] - a["xw"]) + (b["ww"] - a["ww"]) + (b["rd"] - a["rd"])
        self.wall += seconds


def estimate_power_w(busy_ratio):
    other = REF_TOTAL_W - REF_STATIC_W - REF_PS7_DYNAMIC_W
    if busy_ratio <= 0:
        return REF_TOTAL_W
    return REF_STATIC_W + REF_PS7_DYNAMIC_W + other * min(100.0, busy_ratio) / REF_BUSY_RATIO


def make_accel(args):
    if args.sim:
        return (cr.SoftwareAccel(),
                "SIMULATED - software model of the accelerator (no board)",
                args.clock_mhz or 94.117645)
    from accel_driver import connect
    drv = connect(args.bitstream, args.ip, fast_io=not args.slow_io)
    return drv, f"board - {args.ip}", args.clock_mhz or drv.clock_mhz()


def forward_timed(accel, model, x, stat, progress=None):
    """Run the network on normalised images x (N,28,28); returns logits (N,10)
    and fills `stat` (dict of LayerStat) with runs / cycles / AXI words / wall time."""
    n = x.shape[0]
    logits = np.zeros((n, 10), dtype=np.float32)
    for i0 in range(0, n, BATCH):
        xb = x[i0:i0 + BATCH]
        s0 = snap(accel)
        t0 = time.perf_counter()
        conv = cr.conv_layer(accel, xb, model)
        s1 = snap(accel)
        t1 = time.perf_counter()
        flat = cr.flatten_channel_major(cr.relu_pool(conv))
        t2 = time.perf_counter()
        out = cr.fc_layer(accel, flat, model)
        s2 = snap(accel)
        t3 = time.perf_counter()
        logits[i0:i0 + len(xb)] = out
        stat["conv"].add(s0, s1, t1 - t0)
        stat["act"].wall += t2 - t1
        stat["fc"].add(s1, s2, t3 - t2)
        if progress:
            progress(min(i0 + BATCH, n), logits[:min(i0 + BATCH, n)])
    return logits


# ======================================================================
# Evaluation on many test images
# ======================================================================
def evaluate(accel, label, model, images, labels, tlogits, n, clock_mhz, show, sim):
    n = min(n, len(images))
    x = cr.prep(images[:n])
    stat = {"conv": LayerStat(), "act": LayerStat(), "fc": LayerStat()}
    s_begin = snap(accel)

    def progress(done, partial):
        acc = 100.0 * np.mean(partial.argmax(1) == labels[:done])
        print(f"\r  {done}/{n} images   running accuracy {acc:6.2f}%", end="", flush=True)

    print("\n" + "=" * 62)
    print(f"RUNNING THE CNN ON {n} TEST IMAGES")
    print("=" * 62)
    t0 = time.perf_counter()
    logits = forward_timed(accel, model, x, stat, progress)
    wall = time.perf_counter() - t0
    print()
    s_end = snap(accel)

    pred = logits.argmax(1)
    tpred = tlogits[:n].argmax(1)
    acc = 100.0 * np.mean(pred == labels[:n])
    tacc = 100.0 * np.mean(tpred == labels[:n])
    agree = 100.0 * np.mean(pred == tpred)
    maxdiff = float(np.max(np.abs(logits - tlogits[:n])))

    print(f"\nPREDICTIONS (first {min(show, n)})")
    for i in range(min(show, n)):
        ok = pred[i] == labels[i]
        print(f"  image {i:4d} : predicted {pred[i]}   label {labels[i]}   "
              f"PyTorch {tpred[i]}   {'PASS' if ok else 'FAIL'}")

    print("\nACCURACY")
    print(f"  accelerator            : {acc:6.2f}%  ({int(np.sum(pred == labels[:n]))}/{n})")
    print(f"  PyTorch (same images)  : {tacc:6.2f}%")
    print(f"  same prediction as PyTorch : {agree:6.2f}%")
    print(f"  max |logit difference vs PyTorch| : {maxdiff:.2e}")

    if not sim:
        ref = cr.forward_accel(cr.SoftwareAccel(), x[:BATCH], model)
        got = logits[:len(ref)]
        a = np.where(ref == 0, np.float32(0), ref).view(np.uint32)
        b = np.where(got == 0, np.float32(0), got).view(np.uint32)
        bad = int(np.sum(a != b))
        print(f"  bit-exact vs accelerator model (first {len(ref)} images) : "
              f"{'YES' if bad == 0 else f'NO ({bad} of {ref.size} logits differ)'}")

    # ---------------------------------------------------------- performance
    runs = s_end["runs"] - s_begin["runs"]
    cycles = s_end["cycles"] - s_begin["cycles"]
    words = (s_end["xw"] - s_begin["xw"]) + (s_end["ww"] - s_begin["ww"]) + \
            (s_end["rd"] - s_begin["rd"])
    hw_us = cycles / clock_mhz / n
    wall_ms = 1e3 * wall / n
    if sim:
        busy_ratio = 100.0 * (2 + cr.PASSES * (4 + 4 * cr.L + 32)) / cr.CYCLES_PER_RUN
    else:
        p = accel.read_perf()
        busy_ratio = 100.0 * p["busy"] / p["total"] if p["total"] else 0.0
    power = estimate_power_w(busy_ratio)

    print("\n" + "-" * 62)
    print(f"        PERFORMANCE SUMMARY  ({n} images)")
    print("-" * 62)
    rows = [
        ("Hardware runs (16x16 products)", f"{runs:,}"),
        ("  per image",                    f"{runs / n:.2f}"),
        ("Hardware cycles per image",      f"{cycles / n:,.0f}"),
        ("Clock",                          f"{clock_mhz:.1f} MHz"),
        ("Hardware time per image",        f"{hw_us:,.1f} us"),
        ("Total time per image (PS)",      f"{wall_ms:,.2f} ms"),
        ("Images per second",              f"{n / wall:,.1f}"),
        ("Hardware-only images per second", f"{1e6 / hw_us:,.0f}"),
        ("AXI data words per image",       f"{words / n:,.0f}"),
        ("Useful MACs per image",          f"{MACS_PER_IMAGE:,}"),
        ("Useful MACs per hardware cycle", f"{MACS_PER_IMAGE / (cycles / n):.2f}"),
        ("Busy-cycle ratio",               f"{busy_ratio:.2f}%"),
    ]
    for name, value in rows:
        print(f"  {name:<34} {value:>22}")

    print("\n  LAYER BREAKDOWN (per image)")
    print(f"  {'layer':<14}{'runs':>8}{'cycles':>11}{'hw us':>10}{'PS ms':>10}{'AXI words':>12}")
    for key, name in (("conv", "convolution"), ("act", "ReLU + pool"), ("fc", "fully conn.")):
        st = stat[key]
        if key == "act":
            print(f"  {name:<14}{'-':>8}{'-':>11}{'-':>10}{1e3 * st.wall / n:>10.2f}{'-':>12}")
        else:
            print(f"  {name:<14}{st.runs / n:>8.2f}{st.cycles / n:>11,.0f}"
                  f"{st.cycles / clock_mhz / n:>10.1f}{1e3 * st.wall / n:>10.2f}"
                  f"{st.words / n:>12,.0f}")

    if not sim:
        tw = s_end["t_write"] - s_begin["t_write"]
        tr = s_end["t_run"] - s_begin["t_run"]
        td = s_end["t_read"] - s_begin["t_read"]
        other = max(wall - tw - tr - td, 0.0)
        print("\n  WHERE THE TIME GOES (per image)")
        print(f"  {'AXI writes (X and W)':<34}{1e3 * tw / n:>12.2f} ms")
        print(f"  {'accelerator compute + polling':<34}{1e3 * tr / n:>12.2f} ms")
        print(f"  {'AXI reads (results)':<34}{1e3 * td / n:>12.2f} ms")
        print(f"  {'Python / NumPy (patches, ReLU, pool)':<34}{1e3 * other / n:>12.2f} ms")

    print("-" * 62)
    print(f"  {'Power consumed (estimated)':<34}{power:>19.3f} W")
    print(f"  {'Energy per image (estimated)':<34}{power * hw_us:>16.2f} uJ")
    print("-" * 62)
    print("  power / energy are model-based estimates, not measurements.")
    return acc


# ======================================================================
# Classify one image
# ======================================================================
def classify_one(accel, label, model, images, labels, tlogits, idx, clock_mhz):
    x = cr.prep(images[idx:idx + 1])
    stat = {"conv": LayerStat(), "act": LayerStat(), "fc": LayerStat()}
    s0 = snap(accel)
    t0 = time.perf_counter()
    logits = forward_timed(accel, model, x, stat)[0]
    wall = time.perf_counter() - t0
    s1 = snap(accel)

    prob = softmax(logits.astype(np.float64))
    pred = int(logits.argmax())
    print("\n" + "=" * 62)
    print(f"TEST IMAGE {idx}")
    print("=" * 62)
    print(ascii_image(images[idx]))
    print("\nSCORES (hardware)             probability")
    for d in np.argsort(-prob)[:5]:
        bar = "#" * int(round(40 * prob[d]))
        print(f"  digit {d}: {logits[d]:9.3f}   {100 * prob[d]:6.2f}%  {bar}")
    print(f"\n  predicted digit : {pred}")
    print(f"  label           : {labels[idx]}   -> "
          f"{'CORRECT' if pred == labels[idx] else 'WRONG'}")
    print(f"  PyTorch result  : {int(tlogits[idx].argmax())}   "
          f"(max |logit difference| {np.max(np.abs(logits - tlogits[idx])):.2e})")
    cyc = s1["cycles"] - s0["cycles"]
    print(f"  hardware runs {s1['runs'] - s0['runs']}, {cyc:,} cycles "
          f"= {cyc / clock_mhz:,.1f} us at {clock_mhz:g} MHz;  total time {1e3 * wall:.1f} ms")


# ======================================================================
# Main
# ======================================================================
def ask_int(prompt, default, lo, hi):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if lo <= v <= hi:
            return v
        print(f"Please enter a number between {lo} and {hi}.")


def main():
    ap = argparse.ArgumentParser(description="MNIST CNN on the systolic-array accelerator.")
    ap.add_argument("--model", default="model/mnist_cnn.npz")
    ap.add_argument("--subset", default="model/mnist_test_subset.npz")
    ap.add_argument("--bitstream", default="design_1.bit")
    ap.add_argument("--ip", default="cnn_accelerator_axi_0")
    ap.add_argument("--n", type=int, default=None, help="evaluate N test images and exit")
    ap.add_argument("--show", type=int, default=10, help="predictions to print")
    ap.add_argument("--clock-mhz", type=float, default=None)
    ap.add_argument("--sim", action="store_true", help="use the software model, no board")
    ap.add_argument("--slow-io", action="store_true", help="ip.read / ip.write only")
    args = ap.parse_args()

    try:
        model = cr.load_model(args.model)
        images, labels, tlogits = cr.load_subset(args.subset)
    except FileNotFoundError as exc:
        print(f"File not found: {exc.filename}\nRun train_mnist.py first and copy the "
              "model/ folder next to this script.")
        sys.exit(1)

    try:
        accel, label, clock = make_accel(args)
    except Exception as exc:
        print(f"\nCould not connect to the accelerator: {exc}")
        sys.exit(1)

    print("=" * 62)
    print("   MNIST CNN ON THE SYSTOLIC-ARRAY ACCELERATOR")
    print("=" * 62)
    print(f"accelerator : {label}")
    print(f"test images : {len(images)} available in {args.subset}")

    if args.n:
        evaluate(accel, label, model, images, labels, tlogits, args.n, clock, args.show, args.sim)
        return

    while True:
        print("\nMenu:")
        print("1. Evaluate accuracy on test images")
        print("2. Classify one test image")
        print("3. Quit")
        try:
            choice = input("\nEnter choice (1/2/3): ").strip().lower()
        except EOFError:
            break
        if choice in ("3", "q", "quit", "exit"):
            break
        if choice == "1":
            n = ask_int("How many test images", 100, 1, len(images))
            evaluate(accel, label, model, images, labels, tlogits, n, clock, args.show, args.sim)
        elif choice == "2":
            idx = ask_int("Test image number", 0, 0, len(images) - 1)
            classify_one(accel, label, model, images, labels, tlogits, idx, clock)
        else:
            print("Invalid choice. Select 1, 2 or 3.")
    print("Done.")


if __name__ == "__main__":
    main()
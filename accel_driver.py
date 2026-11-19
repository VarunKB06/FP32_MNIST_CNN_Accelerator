#!/usr/bin/env python3
"""
accel_driver.py  -  PYNQ driver for the FP32 systolic-array matrix-multiply
                    accelerator (axi.v + design.v)

The driver offers the same call as the software model (cnn_reference.SoftwareAccel):

        Y = drv.matmul16(X, W, w_key=None, rows=16)

    X, W   : 16 x 16 float32 matrices      Y = X . W  (16 x 16 float32)
    w_key  : label of W. When it equals the label of the previous call the
             weights are still loaded in the hardware and are NOT written again
             (the convolution layer uses one W for all its runs)
    rows   : number of result rows that are read back (the others are zeros)

What one call does on the board
    1. write W (256 words, only if it changed)      register WEIGHT
    2. write X (256 words)                          register FEATURE
    3. start, poll DONE                             registers CTRL / STATUS
    4. read rows*16 result words                    window OUT (0x400 ..)
    5. read the cycle counter of the run            register PERF_TOTAL

AXI register map (from cnn_accelerator_axi.v), word index = byte offset / 4
    0x000 CTRL    [0] start  [1] soft reset         0x00C FEATURE  X, row-major
    0x004 SIZE    0/1/2 = 4x4 / 8x8 / 16x16         0x010 WEIGHT   W, row-major
    0x008 STATUS  [0] done [1] busy [2] X loaded    0x014 .. 0x02C  7 perf counters
                  [3] W loaded [4] output valid     0x030 ID = 0x4D4D3031
    0x400 .. 0x7FC  OUT  Y[m][n] at 0x400 + 4*(m*16 + n)

Fast I/O: the driver uses the memory-mapped numpy view of the IP
(ip.mmio.array) when it is available; otherwise ip.write / ip.read.

Self-test on the board (compares every result bit for bit with the software
model, which reproduces the hardware arithmetic exactly):
        sudo python3 accel_driver.py
        sudo python3 accel_driver.py --bitstream design_1.bit --tests 10
"""
import argparse
import sys
import time

import numpy as np

from cnn_reference import RunStats, SoftwareAccel, L, CYCLES_PER_RUN

BITSTREAM = "design_1.bit"
IP_NAME = "cnn_accelerator_axi_0"
# Accelerator clock of the implemented design (FCLK_CLK0): 94.117645 MHz, period 10.625 ns
# (design_1.hwh and the Vivado timing report). The PYNQ clock API reported 58.8 MHz during the first
# CNN board runs, which did not match the implemented design; the fixed value is used instead.
DESIGN_CLOCK_MHZ = 94.117645

# word indices of the registers
IDX_CTRL, IDX_SIZE, IDX_STATUS, IDX_FEATURE, IDX_WEIGHT = 0, 1, 2, 3, 4
IDX_PERF = 5                      # 7 counters: total, feature, weight, compute, store, busy, idle
IDX_ID = 12
IDX_OUT = 0x400 // 4              # 256

CTRL_START = 0x1
CTRL_RESET = 0x2
ST_DONE = 0x01
ST_IN_LOADED = 0x04
ST_W_LOADED = 0x08
SIZE_16 = 2
ID_VALUE = 0x4D4D3031             # "MM01"

PERF_NAMES = ("total", "feature_load", "weight_load", "compute",
              "output_store", "busy", "idle")

_NOKEY = object()


def _to_words(M):
    a = np.ascontiguousarray(M, dtype=np.float32)
    if a.shape != (L, L):
        raise ValueError("matmul16 needs 16 x 16 matrices")
    if not np.all(np.isfinite(a)):
        raise ValueError("NaN and infinity are not supported by the accelerator")
    return a.reshape(-1).view(np.uint32).tolist()


class AccelDriver:
    """Driver object; `ip` is the PYNQ IP object of cnn_accelerator_axi."""

    def __init__(self, ip, fast_io=True, run_timeout_s=5.0):
        self.ip = ip
        self.run_timeout_s = run_timeout_s
        self.stats = RunStats()
        self.time_write = 0.0          # seconds spent writing X / W
        self.time_run = 0.0            # seconds from start until DONE was seen
        self.time_read = 0.0           # seconds spent reading results
        self._w_key = _NOKEY
        self.arr = None
        if fast_io:
            mm = getattr(ip, "mmio", None)
            self.arr = getattr(mm, "array", None)
        self._check_id()
        self.reset()

    # ------------------------------------------------------------ low level
    def _rd(self, idx):
        if self.arr is not None:
            return int(self.arr[idx]) & 0xFFFFFFFF
        return self.ip.read(idx * 4) & 0xFFFFFFFF

    def _wr(self, idx, value):
        if self.arr is not None:
            self.arr[idx] = value
        else:
            self.ip.write(idx * 4, value)

    def _stream(self, idx, words):
        """Write many words to the same (FIFO-like) register."""
        if self.arr is not None:
            arr = self.arr
            for v in words:
                arr[idx] = v
        else:
            write = self.ip.write
            off = idx * 4
            for v in words:
                write(off, v)

    def _read_block(self, idx, n):
        if self.arr is not None:
            return np.array(self.arr[idx:idx + n], dtype=np.uint32)
        return np.array([self.ip.read((idx + i) * 4) & 0xFFFFFFFF for i in range(n)],
                        dtype=np.uint32)

    def _wait_status(self, mask, timeout_s, what):
        t0 = time.monotonic()
        while True:
            st = self._rd(IDX_STATUS)
            if (st & mask) == mask:
                return st
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError(f"timed out waiting for {what}; STATUS=0x{st:08X}")

    def _check_id(self):
        ident = self._rd(IDX_ID)
        if ident != ID_VALUE:
            raise RuntimeError(
                f"ID register = 0x{ident:08X}, expected 0x{ID_VALUE:08X}: this is not "
                "the matrix-multiply bitstream (or the wrong IP name / address).")
        if self.arr is not None:                       # fast path must agree with ip.read
            try:
                if self.ip.read(IDX_ID * 4) & 0xFFFFFFFF != ident:
                    self.arr = None
            except Exception:
                pass

    # ------------------------------------------------------------ control
    def reset(self):
        """Soft reset, select the 16 x 16 size.  X and W must be written again."""
        self._wr(IDX_CTRL, CTRL_RESET)
        time.sleep(0.01)
        self._wr(IDX_SIZE, SIZE_16)
        time.sleep(0.01)
        self._w_key = _NOKEY

    def read_perf(self):
        """The seven cycle counters of the last run."""
        return {name: self._rd(IDX_PERF + i) for i, name in enumerate(PERF_NAMES)}

    # ------------------------------------------------------------ the matrix product
    def matmul16(self, X, W, w_key=None, rows=L):
        xw = _to_words(X)
        need_w = (w_key is None) or (w_key != self._w_key)
        ww = _to_words(W) if need_w else None
        rows = max(1, min(L, int(rows)))

        t0 = time.perf_counter()
        if need_w:
            self._stream(IDX_WEIGHT, ww)
        self._stream(IDX_FEATURE, xw)
        self._wait_status(ST_IN_LOADED | ST_W_LOADED, 2.0, "X and W to be loaded")
        t1 = time.perf_counter()

        self._wr(IDX_CTRL, CTRL_START)
        try:
            self._wait_status(ST_DONE, self.run_timeout_s, "DONE")
        except RuntimeError:
            self.reset()
            raise
        t2 = time.perf_counter()

        data = self._read_block(IDX_OUT, rows * L)
        total = self._rd(IDX_PERF)
        t3 = time.perf_counter()

        Y = np.zeros((L, L), dtype=np.float32)
        Y[:rows] = data.view(np.float32).reshape(rows, L)

        if need_w:
            self._w_key = w_key if w_key is not None else _NOKEY
        s = self.stats
        s.runs += 1
        s.cycles += total
        s.x_writes += L * L
        s.w_writes += (L * L) if need_w else 0
        s.reads += rows * L
        self.time_write += t1 - t0
        self.time_run += t2 - t1
        self.time_read += t3 - t2
        return Y

    # ------------------------------------------------------------ info
    def clock_mhz(self):
        """Accelerator clock in MHz: ACCEL_CLOCK_MHZ environment variable if set, else the design clock."""
        import os
        env = os.environ.get("ACCEL_CLOCK_MHZ")
        return float(env) if env else DESIGN_CLOCK_MHZ


def connect(bitstream=BITSTREAM, ip_name=IP_NAME, fast_io=True):
    """Load the bitstream and return an AccelDriver."""
    from pathlib import Path
    from pynq import Overlay

    path = Path(bitstream)
    if not path.is_file():
        raise FileNotFoundError(f"Bitstream not found: {path.resolve()}")
    print("Loading FPGA bitstream...")
    overlay = Overlay(str(path))
    if ip_name not in overlay.ip_dict:
        raise RuntimeError(f"IP '{ip_name}' not found. Available IPs: "
                           f"{list(overlay.ip_dict.keys())}")
    drv = AccelDriver(getattr(overlay, ip_name), fast_io=fast_io)
    print("Accelerator connected:", ip_name,
          "(memory-mapped fast I/O)" if drv.arr is not None else "(ip.read / ip.write)")
    drv.overlay = overlay                         # keep the overlay alive
    return drv


# ======================================================================
# Self-test on the board
# ======================================================================
def _same(a, b):
    """Bit-exact comparison; +0 and -0 are the same number for the hardware."""
    a = np.where(a == 0, np.float32(0), a).astype(np.float32)
    b = np.where(b == 0, np.float32(0), b).astype(np.float32)
    return np.array_equal(a.view(np.uint32), b.view(np.uint32))


def self_test(drv, tests=5):
    rng = np.random.default_rng(1)
    sw = SoftwareAccel()
    ok_all = True
    rows_out = []

    def check(name, hw, ref):
        nonlocal ok_all
        ok = _same(hw, ref)
        ok_all &= ok
        rows_out.append((name, ok))
        print(f"  {name:<46} {'PASS' if ok else 'FAIL'}")

    print("\nBIT-EXACT TESTS  (board result vs software model of the accelerator)")
    X = rng.normal(0, 2, (L, L)).astype(np.float32)
    check("identity weights, Y = X",
          drv.matmul16(X, np.eye(L, dtype=np.float32)), X)

    for t in range(tests):
        X = rng.normal(0, 2, (L, L)).astype(np.float32)
        W = rng.normal(0, 2, (L, L)).astype(np.float32)
        X[rng.random((L, L)) < 0.1] = 0
        check(f"random matrices {t + 1}", drv.matmul16(X, W), sw.matmul16(X, W))

    # weights stay loaded between calls with the same key
    W = rng.normal(0, 2, (L, L)).astype(np.float32)
    w0 = drv.stats.w_writes
    for t in range(3):
        X = rng.normal(0, 2, (L, L)).astype(np.float32)
        check(f"same W kept in hardware, run {t + 1}",
              drv.matmul16(X, W, w_key="keep"), sw.matmul16(X, W))
    kept = (drv.stats.w_writes - w0) == L * L
    ok_all &= kept
    print(f"  {'W written once for three runs':<46} {'PASS' if kept else 'FAIL'}")

    # partial read-back
    X = rng.normal(0, 2, (L, L)).astype(np.float32)
    W = rng.normal(0, 2, (L, L)).astype(np.float32)
    got = drv.matmul16(X, W, rows=4)
    ref = sw.matmul16(X, W)
    check("only 4 result rows read back", got[:4], ref[:4])

    perf = drv.read_perf()
    cyc_ok = perf["total"] == CYCLES_PER_RUN
    ok_all &= cyc_ok
    print(f"  {'cycles per run = ' + str(CYCLES_PER_RUN):<46} "
          f"{'PASS' if cyc_ok else 'FAIL (hardware reports ' + str(perf['total']) + ')'}")

    s = drv.stats
    words = s.x_writes + s.w_writes + s.reads
    clock = drv.clock_mhz()
    print("\nPERFORMANCE OF THE INTERFACE")
    print(f"  matrix products          : {s.runs}")
    print(f"  clock                    : {clock:.1f} MHz")
    print(f"  hardware time per run    : {CYCLES_PER_RUN / clock:.2f} us "
          f"({CYCLES_PER_RUN} cycles)")
    print(f"  measured per run (PS)    : write {1e6 * drv.time_write / s.runs:8.1f} us,"
          f" compute+poll {1e6 * drv.time_run / s.runs:6.1f} us,"
          f" read {1e6 * drv.time_read / s.runs:7.1f} us")
    print(f"  AXI data words moved     : {words:,}  "
          f"({1e6 * (drv.time_write + drv.time_read) / max(words, 1):.2f} us per word)")
    print("\nRESULT: " + ("PASS - the board reproduces the model bit for bit"
                          if ok_all else "FAIL - see the lines above"))
    return ok_all


def main():
    ap = argparse.ArgumentParser(description="Self-test of the accelerator on the PYNQ board.")
    ap.add_argument("--bitstream", default=BITSTREAM)
    ap.add_argument("--ip", default=IP_NAME)
    ap.add_argument("--tests", type=int, default=5)
    ap.add_argument("--slow-io", action="store_true", help="use ip.read / ip.write only")
    args = ap.parse_args()

    print("=" * 62)
    print("   ACCELERATOR DRIVER SELF-TEST")
    print("=" * 62)
    try:
        drv = connect(args.bitstream, args.ip, fast_io=not args.slow_io)
    except Exception as exc:
        print(f"\nCould not connect to the accelerator: {exc}")
        sys.exit(1)
    sys.exit(0 if self_test(drv, args.tests) else 2)


if __name__ == "__main__":
    main()
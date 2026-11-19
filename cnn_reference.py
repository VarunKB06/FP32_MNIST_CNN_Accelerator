#!/usr/bin/env python3
"""
cnn_reference.py  -  NumPy reference of the MNIST CNN, the mapping of its
                     layers onto the accelerator, and a SOFTWARE MODEL of
                     the accelerator.  Runs on the PC and on the board.

The accelerator does one operation:   Y = X . W      (16 x 16 FP32 matrices)

    convolution : X = 16 image patches (10 values each: 9 pixels + a 1.0
                      bias slot, zero padded to 16 columns)
                  W = hw_conv_w (16 x 16: 9 taps + bias row, 8 filters)
                  -> 676 patches per image, 16 patches per run
    fully conn. : X = 16 images x 16 features of the flattened feature maps
                  W = 16 x 16 block of fc_w.T (10 of the 16 columns used)
                  -> 1352 features = 85 blocks of 16, the partial results
                     and the bias are added in software
    ReLU, max-pool and argmax are done in software.

SoftwareAccel reproduces the hardware arithmetic exactly (same order of
additions, FP32 rounding after every multiply and add, denormals flushed),
so the whole pipeline can be tested on the PC before the board is used.
The board driver (accel_driver.py) has the same  matmul16(X, W, w_key)
interface, so the mapping code below is shared.

Files used (written by train_mnist.py)
    model/mnist_cnn.npz           weights
    model/mnist_test_subset.npz   test images, labels, PyTorch logits

Usage
    python3 cnn_reference.py                          # 200 test images
    python3 cnn_reference.py --n 1000 --clock-mhz 100
    python3 cnn_reference.py --selftest               # random weights, no files
"""
import argparse
import math
import sys
import time

import numpy as np

MEAN = 0.1307
STD = 0.3081
FP32_TINY = float(np.finfo(np.float32).tiny)

L = 16                                   # hardware matrix size used by the mapping
PASSES = (L // 4) ** 2                   # 16 passes of the 4x4 array
CYCLES_PER_RUN = 11 + PASSES * (4 * L + 37)          # 1627, see systolic_matmul_final.v
FEATURES = 8 * 13 * 13                   # 1352 inputs of the fully-connected layer


# ======================================================================
# Loading
# ======================================================================
def load_model(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_subset(path):
    data = np.load(path)
    return data["images"], data["labels"], data["torch_logits"]


# ======================================================================
# Plain NumPy layers (the reference)
# ======================================================================
def prep(images_u8):
    """uint8 images (B,28,28) -> normalised float32."""
    x = images_u8.astype(np.float32) / np.float32(255.0)
    return ((x - np.float32(MEAN)) / np.float32(STD)).astype(np.float32)


def im2col_3x3(x):
    """(B,28,28) -> (B*676, 10): row-major 3x3 pixels + bias 1.0.

    Uses basic slicing/stacking instead of sliding_window_view so this also
    runs with the older NumPy 1.13 shipped with PYNQ 2.5.
    """
    if x.ndim != 3 or x.shape[1] < 3 or x.shape[2] < 3:
        raise ValueError("Expected input with shape (B,H,W), H and W >= 3")
    b, h, w = x.shape
    patches = []
    for kh in range(3):
        for kw in range(3):
            patches.append(x[:, kh:kh + h - 2, kw:kw + w - 2])
    cols = np.stack(patches, axis=-1).reshape(b, (h - 2) * (w - 2), 9)
    ones = np.ones((b, (h - 2) * (w - 2), 1), dtype=np.float32)
    return np.concatenate([cols, ones], axis=2).reshape(b * (h - 2) * (w - 2), 10)


def relu_pool(conv):
    """conv (B,26,26,8) -> ReLU -> 2x2 max-pool -> (B,13,13,8)."""
    b = conv.shape[0]
    act = np.maximum(conv, np.float32(0.0))
    return act.reshape(b, 13, 2, 13, 2, 8).max(axis=(2, 4))


def flatten_channel_major(pool):
    """(B,13,13,8) -> (B,1352), index = channel*169 + row*13 + col."""
    return pool.transpose(0, 3, 1, 2).reshape(pool.shape[0], FEATURES)


def numpy_forward(x, w):
    """Reference forward pass with ordinary matrix products. -> logits (B,10)."""
    b = x.shape[0]
    conv = (im2col_3x3(x) @ w["hw_conv_w"][:10, :8]).reshape(b, 26, 26, 8)
    flat = flatten_channel_major(relu_pool(conv))
    return flat @ w["fc_w"].T + w["fc_b"]


# ======================================================================
# Software model of the accelerator
# ======================================================================
def _flush(a):
    """Hardware flushes FP32 denormals to zero."""
    a = np.asarray(a, dtype=np.float32)
    return np.where(np.abs(a) < FP32_TINY, np.float32(0.0), a).astype(np.float32)


class RunStats:
    """Counts the hardware work done by a sequence of matmul16 calls."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.runs = 0
        self.cycles = 0
        self.x_writes = 0          # AXI word writes of X
        self.w_writes = 0          # AXI word writes of W (skipped when unchanged)
        self.reads = 0             # AXI word reads of Y

    @property
    def axi_accesses(self):
        return self.x_writes + self.w_writes + self.reads


class SoftwareAccel:
    """Software twin of the accelerator, same interface as the board driver.

    matmul16(X, W, w_key, rows)  ->  Y = X . W   (all 16 x 16 float32)
    rows  : number of result rows the caller needs (the board driver reads
            only those rows back; the others are returned as zeros there)
    w_key : optional label of W; when it equals the label of the previous
            call the weights are assumed to be still loaded in the hardware
            (counted as "no W writes").
    Arithmetic order of the hardware, per output element:
        for each group of 4 k values (c_tile = 0..3):
            partial = ((x0*w0 + x1*w1) + x2*w2) + x3*w3     (FP32 each step)
        Y = ((partial0 + partial1) + partial2) + partial3
    """

    def __init__(self):
        self.stats = RunStats()
        self._last_key = object()

    def matmul16(self, X, W, w_key=None, rows=L):
        X = _flush(X)
        W = _flush(W)
        if X.shape != (L, L) or W.shape != (L, L):
            raise ValueError("matmul16 needs 16 x 16 matrices")

        prod = _flush(X[:, None, :] * W.T[None, :, :])         # prod[m, n, k] = X[m,k]*W[k,n]
        total = None
        for kt in range(L // 4):
            part = prod[:, :, 4 * kt]
            for c in range(1, 4):
                part = _flush(part + prod[:, :, 4 * kt + c])
            total = part if total is None else _flush(total + part)

        s = self.stats
        s.runs += 1
        s.cycles += CYCLES_PER_RUN
        s.x_writes += L * L
        s.reads += rows * L            # only the needed rows are read back
        if w_key is None or w_key != self._last_key:
            s.w_writes += L * L
        self._last_key = w_key
        return total


# ======================================================================
# Mapping of the layers onto 16 x 16 matrix products
# ======================================================================
def conv_layer(accel, x, w):
    """Convolution of a batch of normalised images.  x (B,28,28) -> (B,26,26,8).
    All patches of the batch are packed 16 per run; W stays loaded."""
    b = x.shape[0]
    patches = im2col_3x3(x)                         # (B*676, 10)
    n = patches.shape[0]
    wc = w["hw_conv_w"]                             # (16, 16)
    out = np.zeros((n, 8), dtype=np.float32)
    for r0 in range(0, n, L):
        rows = min(L, n - r0)
        blk = np.zeros((L, L), dtype=np.float32)
        blk[:rows, :10] = patches[r0:r0 + rows]
        y = accel.matmul16(blk, wc, w_key="conv", rows=rows)
        out[r0:r0 + rows] = y[:rows, :8]
    return out.reshape(b, 26, 26, 8)


def fc_layer(accel, flat, w):
    """Fully-connected layer.  flat (B,1352) -> logits (B,10).
    16 images per group, K split in 85 blocks of 16, summed in software."""
    b = flat.shape[0]
    wt = w["fc_w"].T                                # (1352, 10)
    logits = np.zeros((b, 10), dtype=np.float32)
    for i0 in range(0, b, L):
        rows = min(L, b - i0)
        acc = np.zeros((L, L), dtype=np.float32)
        for j0 in range(0, FEATURES, L):
            kk = min(L, FEATURES - j0)
            xb = np.zeros((L, L), dtype=np.float32)
            xb[:rows, :kk] = flat[i0:i0 + rows, j0:j0 + kk]
            wb = np.zeros((L, L), dtype=np.float32)
            wb[:kk, :10] = wt[j0:j0 + kk, :]
            acc = acc + accel.matmul16(xb, wb, w_key=("fc", j0), rows=rows)
        logits[i0:i0 + rows] = acc[:rows, :10] + w["fc_b"]
    return logits


def forward_accel(accel, x, w):
    """Whole network with every matrix product done by `accel`."""
    conv = conv_layer(accel, x, w)
    flat = flatten_channel_major(relu_pool(conv))
    return fc_layer(accel, flat, w)


# ======================================================================
# Report helpers
# ======================================================================
def run_counts(batch_images, batch=L):
    """(conv runs, fc runs) needed for `batch_images` images in groups of `batch`."""
    conv_runs = math.ceil(676 * batch_images / L)
    fc_runs = math.ceil(batch_images / batch) * math.ceil(FEATURES / L)
    return conv_runs, fc_runs


def evaluate(model, images, labels, torch_logits, clock_mhz, batch=L):
    n = len(images)
    x = prep(images)

    t0 = time.perf_counter()
    ref = np.concatenate([numpy_forward(x[i:i + batch], model)
                          for i in range(0, n, batch)])
    t_ref = time.perf_counter() - t0

    accel = SoftwareAccel()
    t0 = time.perf_counter()
    hw = np.concatenate([forward_accel(accel, x[i:i + batch], model)
                         for i in range(0, n, batch)])
    t_hw = time.perf_counter() - t0
    s = accel.stats

    print("=" * 66)
    print("   MNIST CNN  -  NumPy reference and accelerator model")
    print("=" * 66)
    print(f"images evaluated : {n}")

    def line(name, logits):
        acc = 100.0 * np.mean(logits.argmax(1) == labels)
        d = float(np.max(np.abs(logits - torch_logits)))
        ag = 100.0 * np.mean(logits.argmax(1) == torch_logits.argmax(1))
        print(f"  {name:<20} accuracy {acc:6.2f}%   max|logit - PyTorch| {d:.2e}"
              f"   same prediction as PyTorch {ag:6.2f}%")
        return d, ag

    print("\nACCURACY")
    print(f"  {'PyTorch':<20} accuracy "
          f"{100.0 * np.mean(torch_logits.argmax(1) == labels):6.2f}%")
    d_ref, ag_ref = line("NumPy reference", ref)
    d_hw, ag_hw = line("accelerator model", hw)
    dd = float(np.max(np.abs(hw - ref)))
    print(f"  accelerator model vs NumPy reference: max|diff| {dd:.2e}")

    conv_runs, fc_runs = run_counts(n, batch)
    cyc = s.cycles
    print("\nHARDWARE WORK  (what the board will do for these images)")
    print(f"  convolution runs      : {conv_runs:6d}   ({conv_runs / n:.2f} per image)")
    print(f"  fully-connected runs  : {fc_runs:6d}   ({fc_runs / n:.2f} per image)")
    print(f"  total runs            : {s.runs:6d}   ({s.runs / n:.2f} per image)")
    print(f"  cycles per run        : {CYCLES_PER_RUN:6d}   (16 x 16 matrix multiply)")
    print(f"  hardware cycles       : {cyc:,}   ({cyc / n:,.0f} per image)")
    print(f"  hardware time @ {clock_mhz:g} MHz : {cyc / clock_mhz / 1e3:.2f} ms total, "
          f"{cyc / clock_mhz / n / 1e3:.3f} ms per image")
    print(f"  AXI word accesses     : {s.axi_accesses:,}   "
          f"({s.axi_accesses / n:,.0f} per image;  X writes {s.x_writes:,}, "
          f"W writes {s.w_writes:,}, Y reads {s.reads:,})")
    print(f"  PC run time (NumPy)   : reference {t_ref:.2f} s, model {t_hw:.2f} s")

    ok = ag_hw == 100.0 and d_hw < 1e-3
    print("\nRESULT: " + ("PASS - the accelerator mapping reproduces the PyTorch "
                           "predictions" if ok else
                           "FAIL - check the weight export / mapping"))
    return ok


# ======================================================================
# Self-test (random weights, no files)
# ======================================================================
def _random_model(rng):
    conv_w = rng.normal(0, 0.3, (8, 3, 3)).astype(np.float32)
    conv_b = rng.normal(0, 0.1, 8).astype(np.float32)
    hw = np.zeros((16, 16), dtype=np.float32)
    hw[0:9, 0:8] = conv_w.reshape(8, 9).T
    hw[9, 0:8] = conv_b
    return {"conv_w": conv_w, "conv_b": conv_b, "hw_conv_w": hw,
            "fc_w": rng.normal(0, 0.05, (10, FEATURES)).astype(np.float32),
            "fc_b": rng.normal(0, 0.1, 10).astype(np.float32)}


def selftest():
    rng = np.random.default_rng(3)
    ok = True

    # 1. matmul16 against a float64 matrix product
    accel = SoftwareAccel()
    X = rng.normal(0, 1.5, (16, 16)).astype(np.float32)
    W = rng.normal(0, 1.5, (16, 16)).astype(np.float32)
    Y = accel.matmul16(X, W)
    ref = X.astype(np.float64) @ W.astype(np.float64)
    scale = np.abs(X.astype(np.float64)) @ np.abs(W.astype(np.float64))
    err = float(np.max(np.abs(Y - ref) / np.maximum(scale, 1e-30)))
    good = err < 1e-5
    ok &= good
    print(f"selftest 1: matmul16 vs float64 product, max relative error "
          f"{err:.1e} -> {'PASS' if good else 'FAIL'}")

    # 2. identity weights reproduce X exactly
    eye = np.eye(16, dtype=np.float32)
    good = np.array_equal(SoftwareAccel().matmul16(X, eye), X)
    ok &= good
    print(f"selftest 2: X . I == X (bit exact) -> {'PASS' if good else 'FAIL'}")

    # 3. whole network through the accelerator model vs the NumPy reference
    model = _random_model(rng)
    imgs = rng.integers(0, 256, (20, 28, 28), dtype=np.uint8)
    x = prep(imgs)
    accel = SoftwareAccel()
    got = np.concatenate([forward_accel(accel, x[i:i + 16], model) for i in (0, 16)])
    ref = np.concatenate([numpy_forward(x[i:i + 16], model) for i in (0, 16)])
    d = float(np.max(np.abs(got - ref)))
    good = d < 1e-3 and np.array_equal(got.argmax(1), ref.argmax(1))
    ok &= good
    print(f"selftest 3: full network via accelerator model vs NumPy, max|diff| "
          f"{d:.1e} -> {'PASS' if good else 'FAIL'}")

    # 4. run counts
    conv_runs, fc_runs = run_counts(20)
    good = accel.stats.runs == conv_runs + fc_runs
    ok &= good
    print(f"selftest 4: hardware runs {accel.stats.runs} == {conv_runs} conv + "
          f"{fc_runs} fc -> {'PASS' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="MNIST CNN reference and accelerator model.")
    ap.add_argument("--model", default="model/mnist_cnn.npz")
    ap.add_argument("--subset", default="model/mnist_test_subset.npz")
    ap.add_argument("--n", type=int, default=200, help="number of test images")
    ap.add_argument("--clock-mhz", type=float, default=94.117645)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 2)

    model = load_model(args.model)
    images, labels, torch_logits = load_subset(args.subset)
    n = min(args.n, len(images))
    ok = evaluate(model, images[:n], labels[:n], torch_logits[:n], args.clock_mhz)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
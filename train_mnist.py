#!/usr/bin/env python3
"""
train_mnist.py  -  train the small CNN on MNIST (runs on the PC, PyTorch)
                   and export the weights in the layout used by the
                   systolic-array matrix-multiply accelerator.

Network (inference path that will run on the FPGA board)
    input   28 x 28 grayscale digit, normalised with (x/255 - 0.1307)/0.3081
    conv    8 filters, 3 x 3, stride 1, no padding, with bias   -> 26 x 26 x 8
    ReLU
    pool    max-pool 2 x 2                                      -> 13 x 13 x 8
    flatten channel-major  (index = channel*169 + row*13 + col) -> 1352
    fc      1352 -> 10, with bias
    argmax  predicted digit

How the layers become matrix products  Y = X . W  (done by the accelerator)
    conv :  X = image patches, one patch per row = 9 pixels (row-major 3x3)
                followed by a constant 1.0  (the "bias slot")      -> 10 columns
            W = hw_conv_w : rows 0..8 = filter taps, row 9 = bias, columns = filters
            Y = convolution output (bias already included)
    fc   :  X = flattened feature maps (one image per row)
            W = fc_w.T,  the bias is added in software

Files written to --out-dir (default ./model)
    mnist_cnn.npz          conv_w (8,3,3)  conv_b (8)  fc_w (10,1352)  fc_b (10)
                           hw_conv_w (16,16)  mean  std
    mnist_test_subset.npz  images (N,28,28) uint8, labels (N), torch_logits (N,10)

Usage
    python3 train_mnist.py                      # 3 epochs, downloads MNIST
    python3 train_mnist.py --epochs 5 --out-dir model
    python3 train_mnist.py --selftest           # no torch, no data needed:
                                                # checks the export layout only
Requirements for training:  pip install torch torchvision numpy
"""
import argparse
import os
import sys
import time

import numpy as np

MEAN = 0.1307
STD = 0.3081
FP32_TINY = float(np.finfo(np.float32).tiny)        # 1.1754944e-38 (smallest normal)


# ======================================================================
# NumPy side : export layout and a plain forward pass  (no torch needed)
# ======================================================================
def prep(images_u8):
    """uint8 images (B,28,28) -> normalised float32."""
    x = images_u8.astype(np.float32) / np.float32(255.0)
    return ((x - np.float32(MEAN)) / np.float32(STD)).astype(np.float32)


def flush_denormals(a):
    """The accelerator flushes FP32 denormals to zero; do the same to the
    exported weights so software and hardware see identical numbers."""
    a = np.array(a, dtype=np.float32, copy=True)
    a[np.abs(a) < FP32_TINY] = 0.0
    return a


def build_weights(conv_w, conv_b, fc_w, fc_b):
    """Weight dictionary in the layout of the accelerator pipeline.

    conv_w : (8, 3, 3)   [filter, kh, kw]
    conv_b : (8,)
    fc_w   : (10, 1352)  [class, channel*169 + row*13 + col]
    fc_b   : (10,)
    hw_conv_w : (16, 16) hardware weight matrix of the convolution
                rows 0..8 : taps (kh*3 + kw)   row 9 : bias   columns 0..7 : filters
    """
    conv_w = flush_denormals(conv_w).reshape(8, 3, 3)
    conv_b = flush_denormals(conv_b).reshape(8)
    fc_w = flush_denormals(fc_w).reshape(10, 1352)
    fc_b = flush_denormals(fc_b).reshape(10)
    for name, arr in (("conv_w", conv_w), ("conv_b", conv_b),
                      ("fc_w", fc_w), ("fc_b", fc_b)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf")

    hw = np.zeros((16, 16), dtype=np.float32)
    hw[0:9, 0:8] = conv_w.reshape(8, 9).T          # tap k of filter n
    hw[9, 0:8] = conv_b                            # bias slot
    return {
        "conv_w": conv_w, "conv_b": conv_b, "fc_w": fc_w, "fc_b": fc_b,
        "hw_conv_w": hw,
        "mean": np.float32(MEAN), "std": np.float32(STD),
    }


def im2col_3x3(x):
    """x : (B,28,28) float32 -> (B*676, 10) float32.
    Each row = 9 pixels of one 3x3 window (row-major) followed by 1.0."""
    b = x.shape[0]
    win = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))   # (B,26,26,3,3)
    cols = win.reshape(b, 26, 26, 9)
    ones = np.ones((b, 26, 26, 1), dtype=np.float32)
    return np.concatenate([cols, ones], axis=3).reshape(b * 676, 10)


def numpy_forward(x, w):
    """Forward pass written as matrix products (the formulation used for the
    hardware).  x : (B,28,28) normalised float32 -> logits (B,10)."""
    b = x.shape[0]
    patches = im2col_3x3(x)                                   # (B*676, 10)
    conv = (patches @ w["hw_conv_w"][:10, :8]).reshape(b, 26, 26, 8)
    act = np.maximum(conv, np.float32(0.0))
    pool = act.reshape(b, 13, 2, 13, 2, 8).max(axis=(2, 4))   # (B,13,13,8)
    flat = pool.transpose(0, 3, 1, 2).reshape(b, 8 * 13 * 13) # channel-major
    return flat @ w["fc_w"].T + w["fc_b"]


def naive_forward(x, w):
    """Independent direct implementation with plain loops (self-test only)."""
    b = x.shape[0]
    out = np.zeros((b, 10), dtype=np.float64)
    for n in range(b):
        fmap = np.zeros((8, 26, 26))
        for f in range(8):
            for i in range(26):
                for j in range(26):
                    s = float(w["conv_b"][f])
                    for kh in range(3):
                        for kw in range(3):
                            s += float(w["conv_w"][f, kh, kw]) * float(x[n, i + kh, j + kw])
                    fmap[f, i, j] = max(s, 0.0)
        pooled = np.zeros((8, 13, 13))
        for f in range(8):
            for i in range(13):
                for j in range(13):
                    pooled[f, i, j] = fmap[f, 2*i:2*i+2, 2*j:2*j+2].max()
        flat = pooled.reshape(-1)                              # channel-major
        out[n] = w["fc_w"].astype(np.float64) @ flat + w["fc_b"].astype(np.float64)
    return out


def save_weights(path, w):
    np.savez(path, **w)


def selftest():
    """Check the export layout without torch or MNIST."""
    rng = np.random.default_rng(0)
    w = build_weights(rng.normal(0, 0.3, (8, 3, 3)), rng.normal(0, 0.1, 8),
                      rng.normal(0, 0.05, (10, 1352)), rng.normal(0, 0.1, 10))
    imgs = rng.integers(0, 256, (2, 28, 28), dtype=np.uint8)
    x = prep(imgs)
    a = numpy_forward(x, w).astype(np.float64)
    b = naive_forward(x, w)
    err = float(np.max(np.abs(a - b)))
    ok = err < 1e-3 and np.array_equal(a.argmax(1), b.argmax(1))
    print(f"selftest: matrix formulation vs direct convolution, max |diff| = {err:.2e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    print(f"          hw_conv_w shape {w['hw_conv_w'].shape}, patches shape "
          f"{im2col_3x3(x).shape}")
    return ok


# ======================================================================
# PyTorch side : training
# ======================================================================
def train(args):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import datasets, transforms
    except ImportError:
        print("PyTorch / torchvision not found.  Install them with\n"
              "    pip install torch torchvision\n"
              "(or run  python3 train_mnist.py --selftest  to check the layout only).")
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 8, kernel_size=3)          # 28x28 -> 26x26
            self.fc = nn.Linear(8 * 13 * 13, 10)                # after 2x2 pool: 13x13

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv(x)), 2)
            return self.fc(x.flatten(1))                        # channel-major flatten

    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((MEAN,), (STD,))])
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def evaluate():
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                correct += (model(xb.to(device)).argmax(1).cpu() == yb).sum().item()
        return 100.0 * correct / len(test_set)

    print("=" * 62)
    print("   MNIST CNN TRAINING   (conv 3x3 x8 -> ReLU -> pool -> fc 10)")
    print("=" * 62)
    print(f"device {device}, {len(train_set)} training / {len(test_set)} test images, "
          f"{args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, seen, hit = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            seen += len(yb)
            hit += (out.argmax(1) == yb).sum().item()
        print(f"  epoch {epoch}/{args.epochs}   loss {loss_sum / seen:.4f}   "
              f"train acc {100.0 * hit / seen:5.2f}%   test acc {evaluate():5.2f}%   "
              f"({time.time() - t0:.0f} s)")

    test_acc = evaluate()
    print(f"\nfinal test accuracy: {test_acc:.2f}%  ({len(test_set)} images)")

    # ---------------------------------------------------------------- export
    model = model.cpu().eval()
    w = build_weights(model.conv.weight.detach().numpy()[:, 0],
                      model.conv.bias.detach().numpy(),
                      model.fc.weight.detach().numpy(),
                      model.fc.bias.detach().numpy())

    os.makedirs(args.out_dir, exist_ok=True)
    wpath = os.path.join(args.out_dir, "mnist_cnn.npz")
    save_weights(wpath, w)

    n = args.subset
    images = test_set.data[:n].numpy().astype(np.uint8)             # raw pixels
    labels = test_set.targets[:n].numpy().astype(np.int64)
    with torch.no_grad():
        xt = torch.from_numpy(prep(images)).unsqueeze(1)
        torch_logits = model(xt).numpy().astype(np.float32)
    spath = os.path.join(args.out_dir, "mnist_test_subset.npz")
    np.savez(spath, images=images, labels=labels, torch_logits=torch_logits)

    # ------------------------------------------------- export self-check
    np_logits = numpy_forward(prep(images), w)
    diff = float(np.max(np.abs(np_logits - torch_logits)))
    agree = float(np.mean(np_logits.argmax(1) == torch_logits.argmax(1))) * 100.0
    acc_np = float(np.mean(np_logits.argmax(1) == labels)) * 100.0
    print("\nEXPORT CHECK  (matrix formulation in NumPy vs PyTorch, "
          f"{n} test images)")
    print(f"  max |logit difference|  : {diff:.2e}")
    print(f"  prediction agreement    : {agree:.2f}%")
    print(f"  NumPy accuracy on subset: {acc_np:.2f}%")
    ok = diff < 1e-3 and agree == 100.0
    print(f"  -> {'PASS' if ok else 'FAIL: the export layout does not match PyTorch'}")

    print("\nfiles written:")
    print(f"  {wpath}")
    print(f"  {spath}")
    print("next: cnn_reference.py  (NumPy reference + software model of the accelerator)")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description="Train the MNIST CNN and export weights.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./model")
    ap.add_argument("--subset", type=int, default=1000,
                    help="number of test images saved for the board test")
    ap.add_argument("--cpu", action="store_true", help="force CPU training")
    ap.add_argument("--selftest", action="store_true",
                    help="check the export layout with random weights (no torch)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 2)
    sys.exit(train(args))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Generate vectors/golden_16x16.hex for tb/tb_cnn_accelerator_axi.v.

8 sets x (X, W, Y) x 256 FP32 words (hex, one word per line, row-major).  Y comes from the bit-accurate software
twin of the accelerator (cnn_reference.SoftwareAccel), which reproduces the hardware arithmetic order and rounding.
Sets 0-3 random in [-8, 8); 4 wide dynamic range; 5 sparse; 6 first convolution block and 7 first fully-connected
block of the real network on real MNIST images.   Usage: python3 tools/gen_golden_vectors.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cnn_reference as cr

rng = np.random.RandomState(20261116)
acc = cr.SoftwareAccel()
sets = []

def add(X, W):
    X = np.asarray(X, dtype=np.float32); W = np.asarray(W, dtype=np.float32)
    sets.append((X, W, np.asarray(acc.matmul16(X, W), dtype=np.float32)))

for _ in range(4):
    add(rng.uniform(-8, 8, (16, 16)), rng.uniform(-8, 8, (16, 16)))
def wide():
    return (rng.choice([-1, 1], (16, 16)) * np.exp2(rng.randint(-10, 11, (16, 16))) * rng.uniform(1, 2, (16, 16)))
add(wide(), wide())
def sparse():
    return rng.uniform(-8, 8, (16, 16)) * (rng.rand(16, 16) < 0.3)
add(sparse(), sparse())

class Rec(cr.SoftwareAccel):
    calls = []
    def matmul16(self, X, W, w_key=None, rows=cr.L):
        Rec.calls.append((X.copy(), W.copy()))
        return super().matmul16(X, W, w_key, rows)

model = cr.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "mnist_cnn.npz"))
images, labels, tl = cr.load_subset(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "mnist_test_subset.npz"))
cr.forward_accel(Rec(), cr.prep(images[:16]), model)
conv_call = Rec.calls[0]
fc_call = [c for c in Rec.calls if c[1][:, 10:].sum() == 0 and c[1][:, :10].any()][0]
add(*conv_call); add(*fc_call)
assert len(sets) == 8

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vectors", "golden_16x16.hex")
with open(out, "w") as f:
    for X, W, Y in sets:
        for M in (X, W, Y):
            for w in np.ascontiguousarray(M, dtype=np.float32).reshape(-1).view(np.uint32):
                f.write("%08x\n" % int(w))
print("wrote", out, len(sets) * 768, "words")

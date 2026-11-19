# Test-image results

The table below records the first ten predictions printed by the 200-image PYNQ-Z2 run. Image indices are zero-based and refer to the bundled `model/mnist_test_subset.npz` file.

| Image index | FPGA prediction | Ground-truth label | PyTorch prediction | Result |
|---:|---:|---:|---:|---|
| 0 | 7 | 7 | 7 | PASS |
| 1 | 2 | 2 | 2 | PASS |
| 2 | 1 | 1 | 1 | PASS |
| 3 | 0 | 0 | 0 | PASS |
| 4 | 4 | 4 | 4 | PASS |
| 5 | 1 | 1 | 1 | PASS |
| 6 | 4 | 4 | 4 | PASS |
| 7 | 9 | 9 | 9 | PASS |
| 8 | 6 | 5 | 6 | FAIL (same as PyTorch) |
| 9 | 9 | 9 | 9 | PASS |

The application reports 100% prediction agreement with PyTorch for all 200 images. The one failure in this displayed first-ten table is a shared model classification error, not a disagreement between the FPGA and PyTorch.

See `first_10_mnist_samples.png` for a visual reference of the ten input images. It is a visualization of the included subset, not a separate source of model predictions.

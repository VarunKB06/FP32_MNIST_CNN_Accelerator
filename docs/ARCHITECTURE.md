# Architecture and interface notes

## Hardware datapath

`design.v` contains the FP32 arithmetic units, processing element, systolic array, controller, address generation, vector memories, output buffer, performance monitor, and `cnn_accelerator_top`.

The top-level parameter defaults to `N=4` and `DATA_WIDTH=32`. The logical matrix size is selectable as 4×4, 8×8, or 16×16. The physical array remains 4×4; larger logical products are decomposed into tiles. The CNN application uses logical 16×16 matrix products.

## AXI4-Lite wrapper

`axi.v` exposes `cnn_accelerator_axi` with a 32-bit AXI4-Lite data path and an 11-bit address bus. The wrapper independently captures AXI write address and write data before committing a register write. The driver uses 32-bit words; FP32 operands are transferred as IEEE-754 single-precision bit patterns.

| Byte offset | Register | Purpose |
|---:|---|---|
| `0x000` | CTRL | Start and soft reset control |
| `0x004` | SIZE | Logical matrix size selector |
| `0x008` | STATUS | Done/busy/load/output status |
| `0x00C` | FEATURE | Feature/input word stream |
| `0x010` | WEIGHT | Weight word stream |
| `0x014`–`0x02C` | PERF | Seven performance counters |
| `0x030` | ID | Hardware identifier `0x4D4D3031` |
| `0x400`–`0x7FC` | OUT | Result words |

Refer to the RTL and `accel_driver.py` for the exact bit definitions and access sequence. The register interface is word-oriented; the driver writes full words.

## Software/PS responsibilities

`cnn_pynq.py` loads the model and test subset, normalizes images, prepares convolution patches, dispatches matrix operations through `accel_driver.py`, performs ReLU and pooling, flattens feature maps, performs the fully connected layer through FPGA matrix products, and compares predictions with stored PyTorch reference logits.

`cnn_reference.py` contains the numerical reference and software model. Its `im2col_3x3` implementation uses basic NumPy slicing/stacking to remain compatible with the older NumPy version on PYNQ 2.5 while preserving row-major 3×3 patch ordering.

## Clock
The accelerator runs on FCLK_CLK0 = **94.117645 MHz** (period 10.625 ns): `design_1.hwh` and the Vivado timing report (`reports/vivado/`) agree. The first CNN board runs printed 58.8 MHz from the PYNQ clock API; that value did not match the implemented design and the time-based figures were recomputed at 94.1176 MHz (cycle counts were unaffected).

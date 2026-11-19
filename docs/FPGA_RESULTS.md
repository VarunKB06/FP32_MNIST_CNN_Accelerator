# FPGA results and methodology

Board: PYNQ-Z2 (Zynq-7020). Accelerator clock: **94.1176 MHz** (FCLK_CLK0, period 10.625 ns; `design_1.hwh` and the Vivado
timing report). Flow: ARM Python (`cnn_pynq.py`) forms the patches, the accelerator performs the 4x4-tiled FP32 matrix
products over AXI4-Lite, Python applies ReLU / max-pool and the final bias.

## Clock note (resolved)
The first CNN board runs printed 58.8 MHz, taken from the PYNQ clock API. The implemented design runs at 94.1176 MHz, so the
time-based figures were recomputed from the measured cycle counts. Measured values (cycle counts, wall-clock times, accuracy,
counts of runs/words) are unchanged. `accel_driver.py` now uses the design clock (`DESIGN_CLOCK_MHZ`, override with the
`ACCEL_CLOCK_MHZ` environment variable). The raw 58.8 MHz figures (1.3214 ms/image, 757 images/s) are superseded.

## 200-image board run
| Metric | Result | Note |
|---|---:|---|
| Images evaluated | 200 | subset, not the full test set |
| FPGA classifications correct | 197 / 200 (98.50%) | |
| PyTorch accuracy, same subset | 98.50% | |
| FPGA / PyTorch prediction agreement | 200 / 200 (100%) | same predictions, not "all correct" |
| Max absolute logit difference vs PyTorch | 1.14e-05 | numerical comparison, not bit-exact vs PyTorch |
| Hardware cycles per image | 77,730 | measured |
| Hardware latency per image | 825.9 us | derived: 77,730 cycles at 94.1176 MHz |
| Hardware-only throughput | about 1,211 images/s | derived; not application throughput |
| End-to-end time on the PS | 57.15 ms/image (17.5 images/s) | measured wall time |
| Useful MACs per image / per hardware cycle | 62,192 / about 0.80 | |
| Busy-cycle ratio | 98.46% | |
| AXI data words per image | 25,837 | |
| Hardware runs (total / per image) | 9,555 / 47.77 | |

Layer breakdown per image: convolution 42.25 runs, 68,741 cycles (730.4 us), 46.16 ms PS time, 21,649 AXI words; ReLU + pooling
about 0.83 ms PS time; fully connected 5.53 runs, 8,989 cycles (95.5 us), 10.11 ms PS time, 4,189 AXI words.

Approximate PS time per image: AXI writes of X/W 22.19 ms; accelerator compute and polling 2.60 ms; AXI reads 5.90 ms;
Python/NumPy patch generation, ReLU and pooling 26.46 ms. The ARM side, not the accelerator, dominates application time.

## 10-image board run
9 / 10 correct (90%), PyTorch 9 / 10, prediction agreement 100%, max logit difference 7.63e-06, 82,652 cycles/image
(878.2 us at 94.1176 MHz, about 1,139 images/s hardware-only), end-to-end about 16.1 images/s. The 10- and 200-image runs differ
in cycles per image (different batch composition); both are kept, not merged.

## Matrix-level board test (`test.py`, clock 94.118 MHz)
Verbatim summaries: `reports/matrix_test_console_excerpt.txt`.

| Logical size | Tile passes (4x4) | MACs | Cycles | Latency (us) | Result check | PE utilisation | Busy-cycle ratio |
|---|---:|---:|---:|---:|---|---:|---:|
| 4x4 | 1 | 64 | 64 | 0.680 | 16/16 match, max abs error 0 | 8.33% | 84.38% |
| 8x8 | 4 | 512 | 287 | 3.049 | 64/64 match, max abs error 2.13e-06 | 12.50% | 95.47% |
| 16x16 | 16 | 4,096 | 1,627 | 17.287 | 256/256 match, max abs error 5.49e-06 | 16.67% | 98.46% |

These cycle counts are also checked by the RTL testbench (`tb/tb_cnn_accelerator_axi.v`, test T06).

## Vivado implementation (xc7z020clg400-1, Vivado v2026.1, routed)
Source files: `reports/vivado/` (`utilization_report.txt`, `timing_1.rpx`, `timing_summary_extract.txt`,
`implementation_summary.csv`).

| Resource | Used | Available | Util. |
|---|---:|---:|---:|
| Slice LUTs | 10,366 | 53,200 | 19.48% |
| of which LUT as logic / as memory | 8,587 / 1,779 | | |
| Slice registers (FF) | 8,184 | 106,400 | 7.69% |
| Slices | 3,328 | 13,300 | 25.02% |
| Block RAM | 0 | 140 | 0% |
| DSP48E1 | 32 | 220 | 14.55% |
| BUFG | 1 | 32 | 3.13% |

Timing (decoded from `timing_1.rpx`, clock `clk_fpga_0` 10.625 ns): setup WNS +0.179 ns, hold WHS +0.024 ns, pulse-width slack
+4.0625 ns, 0 failing endpoints out of 28,549 (setup/hold) and 9,996 (pulse width). All slacks are positive, so the constraints
are met at 94.1 MHz with little setup margin (about 0.18 ns).

## Power and energy (estimates, not measurements)
- Vivado post-implementation power estimate (`docs/images/vivado_power_summary.png`): total on-chip **1.735 W** = dynamic 1.598 W
  (PS7 1.526 W; PL clocks 0.040, signals 0.017, logic 0.014, DSP < 0.001) + device static 0.138 W; confidence level Medium, junction
  temperature 45.0 C. The programmable-logic part of the dynamic power is about 0.07 W; the ARM processing system dominates.
- The application's own model prints about 1.706 W. With 825.9 us/image that gives about 1,409 uJ/image (about 1,433 uJ/image
  with the Vivado figure). Both are estimates; no board power measurement exists.

## Interpretation rules
- 1,211 images/s is hardware-only; the measured application throughput is 17.5 images/s.
- 98.50% is the accuracy on a 200-image subset, not the full 10,000-image MNIST test set.
- 100% prediction agreement does not mean all labels were classified correctly (3 of 200 were wrong, in both FPGA and PyTorch).
- The accelerator is a physical 4x4 array (16 PEs); larger matrices are tiled.

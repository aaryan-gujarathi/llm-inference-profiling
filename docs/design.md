# Phase 2: Accelerator Design Notes

## Why weight-stationary

Phase 1 showed batch-1 decode is limited by weight traffic, not compute (I ≈ 4 FLOPs/byte vs. a ridge of ≈ 203). The dataflow that minimizes weight movement is the one to build: each weight is loaded into a PE once, then reused for every activation that streams past it.

## Tile

- **Array:** 8 × 8 processing elements
- **Target layer:** `gate_proj`, $K = 2048$, $N = 5632$
- **Tiling:** the $K \times N$ weight matrix is processed as $\lceil 2048/8 \rceil \times \lceil 5632/8 \rceil = 256 \times 704$ tiles
- **PE datapath:** INT8 × INT8 → INT16 product, INT32 accumulate (see [`rtl/pe.sv`](../rtl/pe.sv))

Accumulator headroom: worst-case column sum over $K$ is
$$
2048 \times 128 \times 128 = 2^{25} \ll 2^{31}
$$
so INT32 never overflows for this layer. Requantize to INT8 at the array edge.

## Latency estimate

Estimated tile latency: **~230 ns**.

<!-- EXAMPLE DERIVATION: replace with your Phase 2 numbers/assumptions. -->

**Assumptions**

| Parameter | Value | Why |
| --- | --- | --- |
| Array size $N$ | 8 | 8×8 tile |
| Clock $f_{clk}$ | 100 MHz ($T_{clk} = 10$ ns) | Conservative target for an FPGA / open-PDK ASIC |
| Batch $B$ | 1 | Matches the profiled decode workload |
| PE latency | 1 cycle (registered MAC, see `rtl/pe.sv`) | |

**Cycle count for one tile**

1. **Weight load:** weights shift down each column one row per cycle, so filling all $N$ rows takes
$$
C_{load} = N = 8 \text{ cycles}
$$
2. **Skewed fill and drain:** row $i$'s activation enters $i$ cycles late so it meets the right partial sum. The last activation (row $N-1$) enters at cycle $N-1$ and reaches column $N-1$ after another $N-1$ cycles. Its partial sum then leaves the array one cycle later:
$$
C_{compute} = (N-1) + (N-1) + 1 = 2N - 1 = 15 \text{ cycles}
$$
3. **Total:**
$$
C_{tile} = C_{load} + C_{compute} = 8 + 15 = 23 \text{ cycles}
$$
$$
t_{tile} = C_{tile} \cdot T_{clk} = 23 \times 10\,\text{ns} = \mathbf{230\,ns}
$$

**What this says about utilization**

The tile does $N^2 = 64$ useful MACs but occupies $64 \times 23$ PE-cycles:
$$
U = \frac{N^2}{N^2 \cdot C_{tile}} = \frac{1}{23} \approx 4.3\%
$$
At $B = 1$ each weight is used once, so the array sits mostly idle during load, fill and drain. This is the same memory-bound result the roofline gave, now seen from the hardware side. Utilization only improves when the load cost is amortized over more activations: for $B$ activation vectors per tile, $C_{tile} = N + (2N - 1) + (B - 1)$.

**Whole-layer estimate (no overlap)**

`gate_proj` needs $256 \times 704 = 180{,}224$ tiles:
$$
t_{layer} \approx 180{,}224 \times 230\,\text{ns} \approx 41\,\text{ms}
$$
That's a naive upper bound. Double-buffering the weights (loading the next tile during the current tile's compute) hides $C_{load}$ and brings the bound down to $\approx 180{,}224 \times 15 \times 10\,\text{ns} \approx 27\,\text{ms}$. Even then, streaming 5.77 MB of 4-bit weights needs memory bandwidth to keep up, so a single 8×8 tile won't be competitive without many tiles in parallel.

## Open questions for Phase 3

- Weight load path: column shift (current PE supports it via `w_in`/`w_out`) vs. broadcast
- Double-buffered weights to hide reload between tiles
- Edge requantization: rounding mode and saturation

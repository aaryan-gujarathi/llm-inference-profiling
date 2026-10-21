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

<!-- TODO(Aaryan): paste the derivation from your Phase 2 notes here:
     clock assumption, fill/drain cycles (2N-1 skew), compute cycles, and
     how they add up to 230 ns. Reviewers will check this. -->

## Open questions for Phase 3

- Weight load path: column shift (current PE supports it via `w_in`/`w_out`) vs. broadcast
- Double-buffered weights to hide reload between tiles
- Edge requantization: rounding mode and saturation

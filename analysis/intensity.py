"""Arithmetic-intensity model for the linear layers of a decoder-only LLM.

A linear layer computes Y[B, N] = X[B, K] @ W[K, N]. Per forward pass:

    FLOPs  = 2 * B * K * N                       (one multiply + one add per MAC)
    Bytes  = K*N*w_bytes + B*K*a_bytes + B*N*a_bytes

    I = FLOPs / Bytes   [FLOPs/byte]

At B = 1 the weight term dominates, so I ~= 2 / w_bytes regardless of K and N.
For 4-bit weights (w_bytes = 0.5) that gives I ~= 4 FLOPs/byte.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Hardware:
    """Peak compute and memory bandwidth of an accelerator.

    Attributes:
        name: Display name.
        peak_flops: Peak throughput in FLOP/s for the relevant datatype.
        mem_bw: Peak DRAM bandwidth in bytes/s.
    """

    name: str
    peak_flops: float
    mem_bw: float

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOPs/byte) where the roofline turns flat.

        Complexity: O(1) time, O(1) space.
        """
        return self.peak_flops / self.mem_bw


# NVIDIA T4: 65 TFLOPS FP16 (Tensor Cores), 320 GB/s GDDR6.
T4 = Hardware(name="NVIDIA T4", peak_flops=65e12, mem_bw=320e9)


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LinearLayer:
    """Shape of one linear projection, Y = X @ W with W of shape (K, N)."""

    name: str
    k: int  # in_features
    n: int  # out_features


# TinyLlama-1.1B: hidden 2048, intermediate 5632, 32 query heads,
# 4 KV heads (GQA, head_dim 64) -> k_proj / v_proj are 2048 -> 256.
TINYLLAMA_LAYERS: tuple[LinearLayer, ...] = (
    LinearLayer("q_proj", 2048, 2048),
    LinearLayer("k_proj", 2048, 256),
    LinearLayer("v_proj", 2048, 256),
    LinearLayer("o_proj", 2048, 2048),
    LinearLayer("gate_proj", 2048, 5632),
    LinearLayer("up_proj", 2048, 5632),
    LinearLayer("down_proj", 5632, 2048),
)


def _validate(batch: int, w_bytes: float, a_bytes: float) -> None:
    """Raise ValueError on non-physical inputs."""
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if w_bytes <= 0 or a_bytes <= 0:
        raise ValueError("w_bytes and a_bytes must be positive")


def layer_flops(layer: LinearLayer, batch: int = 1) -> int:
    """FLOPs for one forward pass of a linear layer.

    Args:
        layer: Layer shape.
        batch: Number of tokens processed together (B).

    Returns:
        2 * B * K * N.

    Complexity: O(1) time, O(1) space.
    """
    _validate(batch, 1.0, 1.0)
    return 2 * batch * layer.k * layer.n


def layer_bytes(
    layer: LinearLayer, batch: int = 1, w_bytes: float = 0.5, a_bytes: float = 2.0
) -> float:
    """DRAM bytes moved for one forward pass, assuming no cross-call reuse.

    Args:
        layer: Layer shape.
        batch: Number of tokens (B).
        w_bytes: Bytes per weight (0.5 for 4-bit, 2 for FP16).
        a_bytes: Bytes per activation element (2 for FP16).

    Returns:
        Weight + input-activation + output-activation bytes.

    Note:
        Ignores bitsandbytes quantization metadata (absmax per 64-weight
        block, ~0.06 B/weight), so it slightly overestimates intensity.

    Complexity: O(1) time, O(1) space.
    """
    _validate(batch, w_bytes, a_bytes)
    weights = layer.k * layer.n * w_bytes
    acts = batch * (layer.k + layer.n) * a_bytes
    return weights + acts


def arithmetic_intensity(
    layer: LinearLayer, batch: int = 1, w_bytes: float = 0.5, a_bytes: float = 2.0
) -> float:
    """FLOPs per DRAM byte for one forward pass.

    Complexity: O(1) time, O(1) space.
    """
    return layer_flops(layer, batch) / layer_bytes(layer, batch, w_bytes, a_bytes)


def attainable_flops(hw: Hardware, intensity: float) -> float:
    """Roofline: min(peak compute, bandwidth * intensity).

    Complexity: O(1) time, O(1) space.
    """
    if intensity <= 0:
        raise ValueError("intensity must be positive")
    return min(hw.peak_flops, hw.mem_bw * intensity)


def is_memory_bound(hw: Hardware, intensity: float) -> bool:
    """True when the layer sits left of the ridge point.

    Complexity: O(1) time, O(1) space.
    """
    return intensity < hw.ridge_point


def summarize(
    layers: tuple[LinearLayer, ...] = TINYLLAMA_LAYERS,
    hw: Hardware = T4,
    batch: int = 1,
    w_bytes: float = 0.5,
) -> list[dict]:
    """Per-layer intensity table.

    Complexity: O(L) time and space for L layers.
    """
    rows = []
    for layer in layers:
        ai = arithmetic_intensity(layer, batch, w_bytes)
        rows.append(
            {
                "layer": layer.name,
                "K": layer.k,
                "N": layer.n,
                "intensity": ai,
                "memory_bound": is_memory_bound(hw, ai),
            }
        )
    return rows


if __name__ == "__main__":
    print(f"{T4.name} ridge point: {T4.ridge_point:.1f} FLOPs/byte\n")
    print(f"{'layer':<10}{'K':>6}{'N':>6}{'I (FLOP/B)':>12}  bound")
    for r in summarize():
        bound = "memory" if r["memory_bound"] else "compute"
        print(f"{r['layer']:<10}{r['K']:>6}{r['N']:>6}{r['intensity']:>12.2f}  {bound}")

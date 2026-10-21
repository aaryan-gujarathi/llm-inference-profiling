"""Render a T4 roofline with TinyLlama linear layers at several batch sizes.

Usage:
    python -m analysis.roofline --out docs/figures/roofline_t4.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.intensity import (  # noqa: E402
    T4,
    TINYLLAMA_LAYERS,
    Hardware,
    arithmetic_intensity,
    attainable_flops,
)

BATCHES = (1, 16, 256)


def roofline_curve(hw: Hardware, lo: float = 0.5, hi: float = 2000.0, n: int = 400):
    """Sample the roofline on a log-spaced intensity grid.

    Complexity: O(n) time and space.
    """
    x = np.logspace(np.log10(lo), np.log10(hi), n)
    y = np.minimum(hw.peak_flops, hw.mem_bw * x)
    return x, y


def plot(hw: Hardware, out: Path) -> None:
    """Draw the roofline and the gate_proj / q_proj points per batch size.

    Complexity: O(L * |BATCHES|) time.
    """
    x, y = roofline_curve(hw)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.loglog(x, y / 1e12, color="#333", lw=2, label=f"{hw.name} roofline")
    ax.axvline(hw.ridge_point, color="#999", ls="--", lw=1)
    ax.text(hw.ridge_point * 1.08, 0.25, f"ridge ≈ {hw.ridge_point:.0f}", color="#555")

    markers = {"gate_proj": "o", "q_proj": "s"}
    colors = {1: "#d1495b", 16: "#edae49", 256: "#00798c"}
    for layer in TINYLLAMA_LAYERS:
        if layer.name not in markers:
            continue
        for b in BATCHES:
            ai = arithmetic_intensity(layer, batch=b)
            ax.scatter(
                ai,
                attainable_flops(hw, ai) / 1e12,
                marker=markers[layer.name],
                color=colors[b],
                s=50,
                zorder=3,
                label=f"{layer.name}, B={b}",
            )

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Attainable TFLOP/s")
    ax.set_title("TinyLlama-1.1B (4-bit weights) on T4")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/roofline_t4.png"))
    args = parser.parse_args()
    plot(T4, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

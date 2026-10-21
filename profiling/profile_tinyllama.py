"""Profile 4-bit TinyLlama-1.1B decode on a CUDA GPU (tested target: Colab T4).

Usage (GPU runtime required):
    pip install -r requirements-gpu.txt
    python -m profiling.profile_tinyllama --tokens 32 --out results/

Outputs:
    results/kernel_times.csv   top CUDA kernels by device time
    results/layer_shapes.csv   (K, N) of every Linear4bit layer

Gotchas this script already handles:
    * `load_in_4bit=` as a from_pretrained kwarg was removed; use BitsAndBytesConfig.
    * `cuda_time_total` was renamed `device_time_total` in recent PyTorch.
    * The profiler's record_shapes misses bitsandbytes 4-bit kernels, so shapes
      are read from the Linear4bit modules directly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model(model_id: str = MODEL_ID):
    """Load tokenizer and 4-bit NF4 model onto the GPU.

    Raises:
        RuntimeError: if no CUDA device is available.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (bitsandbytes 4-bit is GPU-only).")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, device_map="cuda"
    )
    model.eval()
    return tok, model


def linear_shapes(model) -> list[dict]:
    """Return (name, K, N) for every Linear4bit / Linear module.

    Complexity: O(M) time and space for M modules.
    """
    rows = []
    for name, mod in model.named_modules():
        if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
            rows.append(
                {"module": name, "K": mod.in_features, "N": mod.out_features,
                 "type": type(mod).__name__}
            )
    return rows


def profile_decode(tok, model, prompt: str, new_tokens: int, top: int = 25) -> list[dict]:
    """Run batch-1 generation under torch.profiler; return top kernels.

    A warm-up pass runs first so CUDA context / autotuning stays out of the trace.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    if new_tokens < 1:
        raise ValueError("new_tokens must be >= 1")

    inputs = tok(prompt, return_tensors="pt").to("cuda")
    gen = dict(max_new_tokens=new_tokens, do_sample=False)

    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=4, do_sample=False)  # warm-up
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model.generate(**inputs, **gen)
            torch.cuda.synchronize()

    events = sorted(
        prof.key_averages(), key=lambda e: e.device_time_total, reverse=True
    )[:top]
    return [
        {"kernel": e.key, "calls": e.count,
         "device_time_us": round(e.device_time_total, 1)}
        for e in events
    ]


def write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to CSV, creating parent dirs."""
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--prompt", default="Explain why the sky is blue.")
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args()

    tok, model = load_model()
    write_csv(linear_shapes(model), a.out / "layer_shapes.csv")
    write_csv(profile_decode(tok, model, a.prompt, a.tokens), a.out / "kernel_times.csv")
    print(f"wrote results to {a.out}/")


if __name__ == "__main__":
    main()

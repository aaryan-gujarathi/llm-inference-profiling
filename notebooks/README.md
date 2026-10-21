`tinyllama_profiling.ipynb` is the original Phase 1 Colab notebook (T4 runtime). It shows:

- the profiler run over `model.generate()` with `record_shapes=True`
- why per-shape GEMM timing came back empty for bitsandbytes 4-bit kernels
- the fallback that reads `(in_features, out_features)` straight from every `Linear4bit` layer

[`profiling/profile_tinyllama.py`](../profiling/profile_tinyllama.py) is the cleaned-up, scriptable version of this notebook.

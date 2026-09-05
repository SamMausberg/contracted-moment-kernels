# Actual model activation diagnostics

`manifest.json` pins Qwen2.5-0.5B weights and records token/array/source hashes.
`scripts/model_traces.py` generates three original probe texts, captures the
actual post-RoPE attention arguments at the final prefix position, and records
all 24 layers and 14 query heads. Prefix lengths are 1,024 and 4,096; seven query
heads share each KV head. The stored query includes exact 1/8 scaling.

`diagnostics.json` records numerical box-screen diagnostics at rank 8 for every
head and rank 64 in layers 0, 12, 23. Coupled screening on these traces is unmeasured.
Large NPZ arrays and model weights are regenerated locally. The inputs are
controlled diagnostic texts, not model accuracy data or a production sample.

Reproduce with `OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/model_traces.py all`.
The source tree must remain fixed throughout the run. Each saved result records
its execution revision and source hashes; formatting changes after a run do not
retroactively change its provenance. Figures use the saved diagnostic rows.

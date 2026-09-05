# Recorded validation

**These are authoring-environment CPU results. No Lean build, CUDA build, GPU benchmark, or remote CI run occurred.**

| Check | Recorded result |
|---|---|
| pytest | 10 test groups passed, including 500 randomized numerical cases and 30 rational attention cases. |
| Outward conversion | 1,000 randomized exact-rational endpoint cases plus edge cases passed. |
| Host numerical core | 48 coordinates, maximum scaled envelope difference 2.82991e-16 versus NumPy. |
| Imported interval checker | 160 output rows; 74 accepted by both the rational and conservative host checkers; no false host certificates. |
| Lean | Not executed: `lake` is unavailable; attempted check exited 127. |
| CUDA | No compiler/GPU run; draft source only. |

## Synthetic ablation

`N=8192, d=16, h=8, B=16, Q=24`. One BLAS thread. Blocks and coordinate structure are supplied by the synthetic generator, not discovered from model traces.

| Case, rank 4 | Initial full-output numerical passes | Mean original-token fraction scanned by adaptive refinement |
|---|---:|---:|
| tight | 24/24 | 0.0000 |
| mixed_one_broad_block | 0/24 | 0.0625 |
| broad_negative_control | 0/24 | 1.0000 |

Summary scalar arrays: 72,832 bytes at full rank; 17,536 bytes at rank 4 (4.1533x smaller). This excludes original K/V, fallback indices, scratch, and allocations.

**The recorded Python screening paths are slower than dense NumPy in every configuration.** The exact-fraction candidate rounding is intentionally conservative and expensive. Fewer read tokens and smaller summaries do not establish wall-clock speedup. Construction and full fallback are included in the machine-readable measurements.

The mathematical sharp-tail coefficient is smaller than the older bound by the factors below. These are coefficient ratios, not speedups or acceptance rates.

| Radius | Old coefficient / new coefficient |
|---:|---:|
| 0.1 | 1.077681 |
| 1 | 2.075514 |
| 2 | 4.123836 |
| 4 | 14.000148 |
| 8 | 86.523373 |

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest -q
OPENBLAS_NUM_THREADS=1 python scripts/experiments.py
python scripts/make_fixture.py /tmp/cmk-fixture.bin
python scripts/make_imported_fixture.py /tmp/cmk-imported.txt
cmake -S kernels -B build/host -DCMAKE_BUILD_TYPE=Release
cmake --build build/host -j
build/host/cmk_host_check /tmp/cmk-fixture.bin
build/host/cmk_imported_check /tmp/cmk-imported.txt
```

[Experiment data](experiments.json), [host parity](host-check.json), [imported checker](imported-check.json), [pytest log](pytest.log), [verification scope](verification.json).

A future passing Lean CI run verifies only the explicit theorem statements in the source, not the unformalized analytic and implementation bridges. The recorded JSON is a historical authoring record, not an automatic statement about later commits.

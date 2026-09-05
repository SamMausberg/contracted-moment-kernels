# Recorded validation

The September 5, 2026 results were produced on the repository's GH200. The
paper distinguishes the proved real-arithmetic contract, exact-rational
implementation checks, and numerical GPU screens.

| Evidence | Recorded outcome | Artifact |
| --- | --- | --- |
| Lean build and complete axiom audit | 64 theorems; only `propext`, `Classical.choice`, `Quot.sound` | [Build](lean-build.log), [audit](lean-axioms.log), [source hashes](lean-verification.json) |
| GitHub checks | CPU, host C++, formatting, documentation, and Lean pass | [CI record](ci.json) |
| Python tests | 46 passed after formatting | [Log](pytest.log) |
| Ruff / clang-format | Formatting and lint checks pass | [Log](lint.log) |
| Rational coupling | 61/149 initial accepts versus 57/149 boxes; 0 observed false certificates | [Data](certification/coupling.json), [interpretation](certification/validation.md) |
| Imported CUDA checker | 74/160 accepted, matching rational decisions; 13 controls rejected | [Result](gh200/final_imported_check.json) |
| Numerical CUDA paths | CPU comparison, three evaluator variants, correction, and four ABI rejection controls pass | [Result](gh200/final_numerical_check.json) |
| CUDA memory checking | No errors on final numerical and imported fixtures | [Numerical](gh200/final_memcheck_numerical.log), [imported](gh200/final_memcheck_imported.log) |
| GH200 performance | 66 configurations; complete screened path slower than dense in all cases | [Raw samples](gh200/benchmark.json) |
| Same-machine reproduction | 66 cases pass again; unchanged coverage; complete path still slower | [Second run](gh200/reproduction.json), [source and input provenance](gh200/provenance.json) |
| Model diagnostics | 2,016 rank-eight and 252 full-rank head instances; zero numerical passes | [Data](model_traces/diagnostics.json), [capture manifest](model_traces/manifest.json) |

The optimized resident screening stage is faster in one favorable long-context
single-query case. That is a different measurement from the complete path,
which includes its host decision and fallback. Setup, uploads, allocation,
graph capture, scratch, failed controls, and all raw timing samples are retained.
The source data reused across ranks/configurations are not independent trials.

The scalar-only GPU sweep and initial absolute-tolerance parity failures remain
in [the GPU artifacts](../docs/GH200.md). The historical authoring environment's
failed Lean attempt remains in `lean-attempt.log`; its original status is archived
in [authoring-verification.json](historical/authoring-verification.json). Both predate the successful
GH200 build. A built real-attention theorem does not specify bitwise equality to
FlashAttention or verify the deployed Python/CUDA arithmetic.

See [reproduction commands](../docs/REPRODUCING.md) and the
[12-figure index](../paper/figures/README.md). CPU synthetic refinement is in
[experiments.json](experiments.json); it reports actual CPU costs and selected
correction tokens. The counter excludes full-source validation/fingerprint reads
and is not total memory traffic or GPU performance. The current evidence index
and artifact hashes are in [verification.json](verification.json).

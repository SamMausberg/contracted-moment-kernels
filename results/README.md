# Recorded validation

The September 5, 2026 results were produced on the repository's GH200. The
native [paper](../paper/PAPER.tex) distinguishes the proved real-arithmetic
contract, exact-rational implementation checks, and numerical GPU screens.

The [built PDF](../paper/PAPER.pdf) uses official ICML 2026 preprint style;
[latex-source.zip](../paper/latex-source.zip) contains its editable sources.
The [Markdown companion](../paper/PAPER.md) preserves the expanded exposition.
The manuscript contains two native TikZ drawings and ten imported vector plots.

| Evidence | Recorded outcome | Artifact |
| --- | --- | --- |
| Lean build and complete axiom audit | 64 theorems; only `propext`, `Classical.choice`, `Quot.sound` | [Build](lean-build.log), [audit](lean-axioms.log), [source hashes](lean-verification.json) |
| GitHub checks | CPU, host C++, formatting, documentation, and Lean pass | [CI record](ci.json) |
| Manuscript GitHub build | Fresh Ubuntu build, figure captions, embedded fonts, no overfull boxes | [Paper CI record](paper-ci.json) |
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

`make paper` builds the native source with latexmk, pdfLaTeX, and BibTeX,
then runs the Poppler-based publication checker. Its
[layout report](paper-layout.json) records reference, caption, font, page, and
text-bound checks plus source and imported-figure hashes. The built PDF has
seven main-text pages, nine appendix pages, and one references page; all twelve
figure captions occur once, all 28 fonts are embedded, and the final LaTeX log
has no overfull boxes. The [visual review](paper-visual-review.json) covers all
17 rendered pages, including equation tags, legend placement, labels, and float
order. Plots use the manuscript's physical width, with legends below data panels.
The [figure layout report](figure-layout.json) records legend/data intersections,
canvas bounds, and export hashes; these checks also run in CI. Publication
checks have a separate scope from the scientific evidence above.
The build uses committed plots and
does not rerun experiments. See the [paper build guide](../paper/README.md).
The [standalone source build](paper-source-build.json) also compiles the extracted
archive independently and confirms that its PDF text matches the repository PDF.

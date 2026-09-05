# Reproducing the research

The recorded machine is an aarch64 GH200 with CUDA 12.8 and a CUDA-enabled
PyTorch 2.7.0 supplied by its image. `requirements-gh200.txt` pins the installed
Python research tools. The local virtual environment inherits the image's
PyTorch and CUDA libraries; the setup script checks that they actually work.

## Environment and development tools

```sh
sudo apt-get update
sudo apt-get install -y python3-venv clang-format latexmk texlive-latex-extra \
  texlive-science texlive-fonts-recommended poppler-utils
bash scripts/setup_gh200.sh
source .venv/bin/activate
make format
make lint
make test
```

Ruff 0.16.6 formats and lints Python. clang-format 14 formats C++/CUDA using the
checked-in style. `make format` applies changes; `make lint` checks without
editing. Formatting does not require new unit tests. Substantive numerical,
identity, and kernel changes require the relevant correctness checks.

Lean installation and the complete build/axiom audit are described in
[FORMALIZATION.md](FORMALIZATION.md). CMake and CUDA build/sanitizer commands
are in [GH200.md](GH200.md). The source tree must remain fixed while recording
an experiment; each artifact's scope and revision identify what was measured.

## Experiments

Run these from the repository root after activating `.venv`:

```sh
OPENBLAS_NUM_THREADS=1 python scripts/certification_experiments.py
OPENBLAS_NUM_THREADS=1 python scripts/experiments.py
OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python scripts/model_traces.py all
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_gpu.py \
  --trace results/model_traces/gpu_prose_4096_layer0.npz \
  --trace results/model_traces/gpu_prose_4096_layer12.npz \
  --trace results/model_traces/gpu_prose_4096_layer23.npz
make figures
make paper
python scripts/check_docs.py
```

The trace script downloads the pinned public Qwen2.5-0.5B weights and generates
three deterministic original probe texts. It records final-prefix attention
arguments, token and array hashes, exact GQA mapping, scaling, and every layer's
diagnostics. Capturing at the actual attention call avoids reconstructing RoPE
or projecting queries again with a different GEMM shape. A supplied effective
mask must expose every captured key to the final query. Large model files and
NPZ arrays are regenerated locally; the manifests and diagnostic rows are
committed. Use an otherwise idle GPU for dedicated timing.

The benchmark saves raw samples and all fallback/setup costs. It is a
microbenchmark with resident inputs, explicit masks, and declared dtype/graph
contracts. Model inference capture is an activation diagnostic, not a model
accuracy or serving throughput evaluation. CPU diagnostic times are never used
as GPU latency evidence.

## Paper and figures

The canonical manuscript is [paper/PAPER.tex](../paper/PAPER.tex), using the
official ICML 2026 two-column style in `preprint` mode. Edit its native prose
and proofs in `paper/sections/`, the two TikZ drawings in `paper/diagrams/`,
and references in `paper/references.bib`. The
[Markdown companion](../paper/PAPER.md) retains the expanded mathematical
exposition and research references. Style provenance is retained in
[paper/vendor/README.md](../paper/vendor/README.md).

From the repository root, `make paper` runs `scripts/build_paper.sh`: `latexmk`
coordinates pdfLaTeX and BibTeX, writes auxiliary files to `build/paper/`, copies
the finished [PDF](../paper/PAPER.pdf), runs the publication checker, and creates
[paper/latex-source.zip](../paper/latex-source.zip). It uses the committed vector
plots without rerunning experiments or regenerating figures. The standalone
archive includes the native source, style, bibliography, drawings, and figure
PDFs; extract it and run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error PAPER.tex
```

`make figures` separately reads the recorded JSON and writes vector SVG/PDF
plots plus PNG previews. The manuscript imports ten vector plots and compiles
two drawings directly from TikZ. The figure manifest records plot data hashes
and labels analytic examples; the companion's geometric SVG exports remain
available. The [figure index](../paper/figures/README.md) describes each figure.

The build runs
`python3 scripts/check_paper.py --output results/paper-layout.json`.
This checker uses Poppler's `pdftotext`, `pdfinfo`, and `pdffonts`, together with
the final LaTeX log and recorder file, to check references, captions, embedded
fonts, page dimensions, and extracted text bounds. Its report records source
and imported-figure hashes. Visual review is still needed for graphical overlap
and reading order; these checks do not validate scientific claims or proofs.

The research GitHub workflow runs Ruff, clang-format, Python tests, documentation/source
checks, host C++ fixtures, and the Lean build/audit. GPU evidence comes from this
GH200's saved runs; the hosted CPU workflow does not claim GPU validation.
The separate manuscript workflow installs LaTeX, builds and checks the paper,
and uploads the PDF, source archive, layout report, and final LaTeX log.

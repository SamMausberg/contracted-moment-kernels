# Reproducing the research

The recorded machine is an aarch64 GH200 with CUDA 12.8 and a CUDA-enabled
PyTorch 2.7.0 supplied by its image. `requirements-gh200.txt` pins the installed
Python research tools. The local virtual environment inherits the image's
PyTorch and CUDA libraries; the setup script checks that they actually work.

## Environment and development tools

```sh
sudo apt-get update
sudo apt-get install -y python3-venv clang-format pandoc texlive-xetex texlive-latex-extra fonts-dejavu-core
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

`make figures` reads committed JSON and writes vector SVG/PDF plus PNG previews.
The figure manifest records data hashes and labels analytic examples. `make
paper` additionally renders `paper/PAPER.pdf` and a local HTML version with
MathJax. The Markdown retains SVG links for GitHub; the PDF renderer uses vector
PDF siblings. The [figure index](../paper/figures/README.md) states the question
each plot answers.

The GitHub workflow runs Ruff, clang-format, Python tests, documentation/source
checks, host C++ fixtures, and the Lean build/audit. GPU evidence comes from this
GH200's saved runs; the hosted CPU workflow does not claim GPU validation.

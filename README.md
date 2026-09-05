# Contracted moment kernels

When can moment summaries certify an attention output? This repository combines
real-arithmetic proofs, exact-rational checking, and measured GH200 experiments.

[Paper PDF](paper/PAPER.pdf) · [LaTeX source](paper/PAPER.tex) ·
[Source archive](paper/latex-source.zip) · [Mathematical companion](paper/PAPER.md) ·
[12 figures](paper/figures/README.md) · [Proof scope](docs/FORMALIZATION.md) ·
[GPU execution](docs/GH200.md) · [Results](results/README.md)

**Validated:** 64 Lean theorems with a complete axiom audit, 46 Python tests,
and executed CUDA checks. Value coupling tightens some certificates. Actual
model traces expose large bound inflation; the complete measured GPU pipeline
remains slower than fused dense attention. The paper explains why.

```sh
bash scripts/setup_gh200.sh
source .venv/bin/activate
make test
make lint
bash scripts/check_lean.sh
```

Use `make format` for Ruff and clang-format. `make paper` builds the native
LaTeX PDF in official ICML 2026 preprint style, checks it, and packages the
source archive; `make figures` regenerates plots separately. See
[reproduction](docs/REPRODUCING.md) for dependencies and experiment commands.
[Apache-2.0](LICENSE).

## References

[Multipole Attention](https://arxiv.org/abs/2506.13059) ·
[COBS](https://arxiv.org/abs/2607.09052) ·
[Adaptive predicates](https://www.cs.cmu.edu/~quake/robust.html) ·
[Complete bibliography](paper/PAPER.md#references) ·
[Primary-literature comparison](docs/NOVELTY.md)

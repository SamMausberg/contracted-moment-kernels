# Contracted moment kernels

When can moment summaries certify an attention output? This repository combines
real-arithmetic proofs, exact-rational checking, and measured GH200 experiments.

[Paper](paper/PAPER.md) · [PDF](paper/PAPER.pdf) ·
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

Use `make format` for Ruff and clang-format, and `make paper` to regenerate the
figures and PDF. See [reproduction](docs/REPRODUCING.md) for dependencies and
experiment commands. [Apache-2.0](LICENSE).

## References

[Multipole Attention](https://arxiv.org/abs/2506.13059) ·
[COBS](https://arxiv.org/abs/2607.09052) ·
[Adaptive predicates](https://www.cs.cmu.edu/~quake/robust.html) ·
[Complete bibliography](paper/PAPER.md#references) ·
[Primary-literature comparison](docs/NOVELTY.md)

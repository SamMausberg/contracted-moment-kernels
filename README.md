# Contracted moment kernels

Boundary certificates for positive normalized reductions, with projected signed
moments and selective block refinement.

[Paper](paper/PAPER.md) · [Lean scope](docs/FORMALIZATION.md) ·
[GH200 notes](docs/GH200.md) · [Results](results/README.md) ·
[Novelty audit](docs/NOVELTY.md)

Research prototype. CPU tests pass. Lean source is **uncompiled**; CUDA is
**untested**. Analytic and implementation proof bridges remain. No general
speedup or uniqueness claim.

```sh
python -m pip install -e '.[test]'
python -m pytest -q
bash scripts/check_lean.sh
```

[Apache-2.0](LICENSE). [Publication helper](scripts/publish.sh) creates a private
GitHub repository when run with an authenticated GitHub CLI.

## References

[Multipole Attention](https://arxiv.org/abs/2506.13059) ·
[TaylorShift](https://arxiv.org/abs/2403.02920) ·
[Symmetry-aware Taylor attention](https://arxiv.org/abs/2602.00294) ·
[COBS](https://arxiv.org/abs/2607.09052) ·
[SPLA](https://arxiv.org/abs/2601.22379) ·
[WitCert](https://arxiv.org/abs/2607.28699) ·
[Attention-memory contracts](https://arxiv.org/abs/2608.05863) ·
[Locks](https://arxiv.org/html/2607.24555) ·
[Adaptive predicates](https://www.cs.cmu.edu/~quake/robust.html) ·
[Complete bibliography](paper/PAPER.md#references)

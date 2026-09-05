# Novelty audit

**Status: originality and priority are not established.** This is a scoped
primary-literature review conducted September 4, 2026, not proof that no prior
construction is equivalent. A different name is not a novelty argument. No
patentability or freedom-to-operate conclusion is offered.

## Closest overlaps

| Source | Established overlap with this proposal | Proposed narrower distinction, not a priority claim |
|---|---|---|
| [Multipole Attention](https://arxiv.org/abs/2506.13059), Hooper et al., 2025 | Clustered key summaries and selected exact attention plus approximate remainder. | Strict observation-cell residuals and value-aware signed-moment witnesses rather than an accuracy-only acceptance policy. |
| [TaylorShift](https://arxiv.org/abs/2403.02920), Nauen et al., 2024 | Polynomial reformulation, moment-like contractions, changed attention scaling. | Certify the original exponential reduction conditionally rather than identify a polynomial replacement as the same mathematical operator. |
| [Symmetry-Aware Taylor Attention](https://arxiv.org/abs/2602.00294), Heinsen and Kozachkov, 2026 | Constant-cost-per-token polynomial features, precision-dependent representations. | Blockwise signed value moments, explicit rank-discard inflation, and cell tests. |
| [COBS](https://arxiv.org/abs/2607.09052), Tian et al., 2026 | Compressed second-order block statistics for attention-mass selection. | Enclosures of the centered value numerator as well as mass; direct certification of the output observation. |
| [SPLA](https://arxiv.org/abs/2601.22379), Wang et al., 2026 | Second-order block selection and a compact linear representation of the unselected tail. | Strict residual certificates with explicit fallback rather than an unqualified exactness claim for the compressed tail. |
| [WitCert](https://arxiv.org/abs/2607.28699), Wei et al., 2026 | Runtime KV quantization witnesses, gating, formal artifacts. | Omission of moment-represented computation, with residual sign tests at numerical output boundaries. |
| [Runtime Observability for Heterogeneous Attention Memory](https://arxiv.org/abs/2608.05863), Wei et al., 2026 | Typed contracts, explicit certification scope, Lean artifacts, local fallback. | The particular algebra and summaries, not the general idea of local certificates or proof-linked runtimes. |
| [Locks](https://arxiv.org/html/2607.24555), Hwang, 2026 | Compact page-local low-rank summaries and exact selected-page work; explicit comparison to moment-based selectors. | A value-aware output certificate instead of only mass-based page selection. |
| [Fast Multipole Attention](https://arxiv.org/abs/2310.11960), Kang et al., 2023 | Hierarchical attention and coarser representations of distant interactions. | Consumer-boundary acceptance of a specified real reduction. |
| [Shewchuk](https://www.cs.cmu.edu/~quake/robust.html), 1997 | Adaptive exact predicates with cheap filters and expensive refinement. | Applying this established discipline to a specific attention-summary construction. |
| [Berz and Hoffstätter](https://doi.org/10.1023/A:1009958918582), 1998 | Taylor polynomials with bounded interval remainders. | Particular contracted signed witnesses; not invention of Taylor-model calculus. |

Some entries are recent preprints. Their existence and described methods are
relevant to novelty, but their empirical performance claims were not reproduced
here. Our tests compare no implementation against these papers. The algebraic
box extremizers in our paper are elementary and should not be presented as a
new general theorem of interval analysis.

## Defensible description

"An experimental combination of boundary-residual certificates, projected
signed-moment attention envelopes, and selective refinement, with draft Lean
formalization and CPU validation artifacts."

The exact combined construction might be independently useful; this review
does not establish that it is unprecedented. Before making a publication
priority claim, compare full derivations and implementations, including
value-aware sparse attention, interval softmax certification, Taylor-model
compilers, and adaptive-precision numerical predicates.

## Claims this release does not make

It does not introduce clustering, polynomial attention, numerical certificates,
local fallback, low-rank summaries, or Lean-verified inference as general ideas.
It does not establish a new unconditional sublinear algorithm for arbitrary
dense attention, a universal dense-GEMM improvement, or measured GH200 speedup.
It does not establish a completed Lean verification or a verified CUDA program.

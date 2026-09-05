# Formalization status

The repository builds with Lean 4.24.0 and mathlib `v4.24.0` on this GH200's
aarch64 host. All 64 local theorem declarations have an axiom-audit entry.
The successful build and complete audit are recorded in
[`results/lean-build.log`](../results/lean-build.log) and
[`results/lean-axioms.log`](../results/lean-axioms.log). The only foundational
dependencies admitted by the audit are `propext`, `Classical.choice`, and
`Quot.sound`. [`results/lean-verification.json`](../results/lean-verification.json)
records the checked source hashes and dependencies of each theorem.

The formalization now proves the analytic chain from concrete real-valued block
witnesses to a strict observation interval for the finite real-attention
quotient. This claim concerns the Lean statements and their hypotheses. The
Python and CUDA implementations have separate verification gaps below.

## Theorem scope

| File | Theorems | Coverage |
| --- | ---: | --- |
| `CMK/Analytic.lean` | 14 | Actual exponential-series tail, sharp quadratic remainder, nonnegative and optimal coefficient, strict improvement over the Lagrange coefficient, and discarded-score exponential bounds. |
| `CMK/Quadratic.lean` | 2 | Symmetric row-sum quadratic bound and coordinate-radius contraction. |
| `CMK/FiniteMoments.lean` | 9 | Exact finite-mean centering, projected-score centering, symmetric finite tensors, linear and quadratic contractions, and omitted signed-tensor bound. |
| `CMK/Attention.lean` | 8 | Summed remainder bounds, centered mass floor, projected enclosures, discarded-coordinate inflation, and integrated full-score block enclosure. |
| `CMK/CertifiedAttention.lean` | 4 | Common-offset scaling, positive block mass, original per-token numerator identity, and full real-attention observation certificate. |
| `CMK/Envelopes.lean` | 13 | Residual interval bounds, endpoint attainment, refinement, and necessary-and-sufficient strict observation tests over the box. |
| `CMK/Observation.lean` | 5 | Constant consumer, abstract monotone rounding, strict argmax, interval intersection, and equal state-transition composition. |
| `CMK/Moments.lean` | 6 | Expansion algebra, scalar remainder lifting, and generic smooth-gate residual bounds. |
| `CMK/Projection.lean` | 3 | Positive-weight perturbations and centered value-range/mass coupling. |

`CMK/Audit.lean` gives every exact theorem name. The central proof chain is:

1. `exp_tail_hasSum` and `exp_coefficient_hasSum` connect the actual `Real.exp`
   series to the quadratic coefficient. `exp_remainder_sharp`,
   `exp_remainder_endpoint`, and `exp_coefficient_optimal` prove its sharpness.
   `exp_coefficient_lt_lagrange` proves the strict coefficient improvement for
   positive radius.
2. `finite_mean_centering`, `projected_score_centering`,
   `second_moment_contraction`, and `signed_moment_contraction` establish the
   finite-sum identities. `quadratic_rowsum_bound` derives the error from an
   explicit symmetric tensor and its absolute row sums.
3. `projected_mass_enclosure` and `projected_center_enclosure` use these analytic
   and tensor results. `full_score_enclosure` restores the discarded score
   coordinates and proves the four full-score block endpoints.
4. `moment_block_enclosure` includes a common exponential offset.
   `moment_block_numerator` identifies the centered representation with the
   original per-token weighted sum. `full_attention_observation` applies the
   derived block bounds to the actual finite real-attention quotient.

`MomentBlock.Witness` contains concrete premises: nonempty tokens, nonnegative
radii, key and value centering, score/value radius bounds, retained-tensor
symmetry, and an absolute row-sum bound on the actual omitted signed tensor.
It does not assume a Taylor remainder or a final mass/numerator enclosure.
`coordinate_radius_bound` proves the score-radius contraction from coordinate
radii. `finite_mean_centering` proves that exact finite means supply centering.

The Lean finite tensors use sums. For a block with cardinality `n`, the paper's
mean tensors correspond to `H_sum = n H_mean`, `D_sum = n D_mean`, and
`eta_sum = n eta_mean`. Accordingly, `massApprox` is `n + sum(t²)/2` and the
common multiplier in `MomentBlock` is `exp(offset)`. These are the unnormalized
finite-sum forms of the manuscript's block formulas. The retained tensor may
be any symmetric tensor; retaining its diagonal is a special case.

`observation_cell_iff` proves the box certificate converse when the sum of
lower mass bounds is positive. If either strict test fails, an admissible box
vertex witnesses failure of the corresponding universal cell claim. This
characterizes the information in that independent interval box; it does not
assert that every such vertex can arise from the original token data.
`centered_value_mass_coupling` proves an additional constraint that can exclude
box vertices. The rational coupled-support optimizer is not extracted from
this theorem.

## Remaining implementation bridges

The exact-rational executable is not extracted from Lean. Its interval
exponential algorithm, rational conversion, rounding-cell construction,
polytope support calculation, and runtime validation have not been connected
to the formal real-number semantics. An executable acceptance claim therefore
still depends on those implementation steps. The numerical summary path does
not produce outward intervals and cannot discharge a directed checker's
sound-input premises.

The concrete IEEE BF16 model, including signed zero, infinities, NaNs, and
exact midpoint ties, is not formalized. `monotone_rounding` is an abstract real
function theorem; `observation_cell` and `full_attention_observation` use strict
open intervals. Neither specifies the rounded accumulation order or
exponential approximation of a particular GPU attention kernel.

The C++/CUDA memory accesses, instruction rounding, overflow behavior, compiler
transformations, input identity, and state mutations have not been formally
verified. The generic state-transition theorem assumes pointwise equal
transitions; it does not establish that any executor or fallback satisfies
that premise. The SwiGLU-specific analytic curvature witness also remains
outside the formalization.

A proof of exact-real attention alone does not establish bitwise equality to a
particular GPU reference. GPU timings and the literature comparison are
empirical evidence, not consequences of the algebraic theorems.

## Reproduce the build and audit

Install the pinned toolchain on a fresh aarch64 or x86-64 host:

```sh
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o /tmp/elan-init.sh
sh /tmp/elan-init.sh -y --default-toolchain leanprover/lean4:v4.24.0
export PATH="$HOME/.elan/bin:$PATH"
bash scripts/check_lean.sh
```

The committed Lake manifest pins mathlib to
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7` and records its dependency revisions.
The script downloads mathlib's build cache, builds all imported modules, and
runs `#print axioms` for every local theorem. It checks that the source and
audit declaration sets match exactly, then rejects every axiom outside the
three foundational dependencies listed above. Missing, duplicate, malformed,
or unexpected audit results also fail the check.

`python3 scripts/check_source.py` checks source/audit consistency and Markdown
source conventions only. It does not run Lean or establish proof validity.

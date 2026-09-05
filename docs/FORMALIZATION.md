# Formalization status

**No Lean build has run for this release. No theorem in this repository is
claimed to have been machine-checked.** Lean/Lake were absent, and compiler
retrieval was unavailable. The build scripts and CI describe checks to execute;
they are not evidence that those checks already passed.

## Scope

The project pins Lean 4.24.0 and the mathlib `v4.24.0` tag. There are 25 named
Lean theorem declarations with proof scripts, no intentional proof holes, and
an axiom-audit entry for every declaration. Source inspection found no `sorry`,
`admit`, new `axiom`, `unsafe`, or `native_decide` declaration/use in our Lean
files. Absence of those strings is not evidence of successful elaboration.

| File | Source statements | Coverage |
|---|---|---|
| `CMK/Envelopes.lean` | 12 | Multiplication intervals, residual identity/enclosures, strict observation cell, endpoint extremizers, monotone refinement. |
| `CMK/Observation.lean` | 5 | Constant consumer, abstract monotone rounding, strict argmax, interval intersection, composition of equal state transitions. |
| `CMK/Moments.lean` | 6 | Exponential algebraic decomposition, centered expansions, conditional scalar remainder lifting, generic smooth-gate residual identity/bound. |
| `CMK/Projection.lean` | 2 | Conditional multiplicative positive-weight bounds and centered-value perturbation. |

Every listed statement has a proof-script body, but may still require source
repairs after the first actual Lean run. `CMK/Audit.lean` lists the exact names.
The paper points to these names, rather than presenting a generic certification
badge over unrelated theorems.

## Missing proof bridges

The sharp exponential tail inequality is proved in the manuscript by power
series, not formalized from `Real.exp`. Centering, coordinate-radius bounds,
Jensen's mass lower bound, the row-sum quadratic witness, and the combination
that produces Theorem 7 still need a complete matrix/finite-sum instantiation.
The gate-specific analytic curvature bound for SwiGLU is also absent.

The exact-rational executable is not extracted from Lean. Its interval
exponential algorithm, data conversion, and rounding implementation have not
been connected to the real-number theorem. The concrete IEEE BF16 model,
including signed zero, infinities, NaNs, and exact midpoint tie handling, is not
formalized. The supplied strict-cell theorem deliberately avoids ties.

The C++/CUDA memory accesses, instruction rounding, overflow behavior, compiler
transformations, and state identity have not been formally verified. Directed
rounding in the final checker is conditional on sound imported inputs; it does
not establish soundness of a numerical summary builder. Complexity arguments
are ordinary arithmetic-model arguments, not Lean cost-semantics theorems.

## Reproduce the intended check

```sh
bash scripts/check_lean.sh
```

The script builds all imported modules and runs `#print axioms` for all 25
statements. It rejects `sorryAx` and native proof-evaluation shortcuts. Standard
Lean/mathlib foundations, such as propositional extensionality, classical
choice, and quotient soundness, are not advertised as eliminated.

A successful future build proves exactly the statements in the source, under
their explicit hypotheses. In particular, proving a conditional lifting lemma
with a remainder bound supplied as a premise does not prove the missing
remainder bound. Do not change a status label to "fully verified" merely because
`lake build` succeeds.

## Release gate for a verified inference claim

A verified real-attention result requires the full analytic chain, an executable
witness checker connected to its formal semantics, and exact input identity.
A claim of bitwise equality to a particular GPU reference additionally requires
that reference's numerical contract. Formal real attention alone does not
specify its rounded accumulation order or exponential approximation.

All state, including cache mutations and fallback behavior, must be included
before claiming equal network execution. Empirical GPU benchmarks and empirical
novelty reviews cannot be proved by an algebraic Lean theorem.

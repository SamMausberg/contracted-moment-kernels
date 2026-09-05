# Research questions and evidence

The study asks when a block summary can establish a consumer's decision without
scanning every token, and what prevents this from being useful. A successful
experiment must explain its result through the uncertainty terms and measured
execution costs.

| Question | Evidence | Interpretation |
|---|---|---|
| Which information makes the boundary decision possible? | Box extrema, value-range coupling, exact rational oracle, strict-improvement example. | Separate a better representation from a faster implementation. |
| Why does a screen fail? | Score radii, discarded-coordinate inflation, quadratic and Taylor terms, distance to the observation boundary. | Attribute rejection to an identifiable mathematical term. |
| Does refining one block help? | Fixed-cell residual margins after each intersection, original-token fraction, broad-block controls. | Measure localization of uncertainty and its limits. |
| Where does compression stop helping? | Rank and block-size sweeps, summary bytes, retained original K/V. | Explain the tradeoff between storage and bound width. |
| Does GPU execution repay its overhead? | Dense fused and binary64 baselines, shared-scalar ablation, setup, transfers, screening, fallback, raw timing samples. | Determine the measured reuse and coverage needed to break even. |
| Does the synthetic mechanism survive real activations? | Post-RoPE traces from pinned public model weights across layers and heads. | Report scope and failures before inferring model-wide usefulness. |
| What has actually been proved? | Lean build, complete axiom audit, theorem-to-paper mapping. | Distinguish proved real arithmetic from executable interval and GPU contracts. |

All figures are generated from committed measurements or explicitly labeled
analytic examples. The trained-model inputs are deterministic, original probe
texts; they are diagnostic inputs rather than an accuracy benchmark. Model
weights and large activation arrays are downloaded or regenerated locally.

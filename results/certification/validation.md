# Rational certificate audit and coupling ablation

The exact-rational implementation and its independent rational exponential oracle
certify rounding of exact-real softmax attention on supported rational inputs.
They do not establish bitwise equality to a GPU exp/accumulation kernel. The
NumPy implementation remains a numerical research path without outward rounding.
The new polygon optimizer is Python source validated by independent geometry and
high-precision tests; these tests are not a universal proof of its implementation.

## Actual arithmetic defect corrected

`Interval(1, 1) / Interval(3, 3)` previously evaluated the reciprocal endpoints as
Python `1 / int`, obtaining binary64 `1/3`, which is less than exact rational
`1/3`. Its upper endpoint was inward. Interval construction now converts every
endpoint to `Fraction`; scalar multiplication also converts its scalar before
arithmetic. `test_integer_interval_reciprocal_stays_exact` reproduces the trigger
and requires both endpoints to equal exact `Fraction(1, 3)`.

## Input and refinement contracts

Rational dot products and the independent direct oracle now reject inconsistent
query/key/value dimensions instead of silently truncating with `zip`. Both
summary implementations reject fractional or Boolean ranks and noninteger group
indices rather than coercing them. Groups must partition exactly the visible
source. Unsupported exponential domains and invalid/nonpositive envelopes are
rejected.

Rational summaries are frozen dataclasses whose array contents are immutable
tuples. Numerical summaries defensively copy arrays and mark them read-only.
The exact ordered visible key/value source has a SHA-256 identity. Rational
summaries from different sources cannot be combined, and refinement requires the
original source, query, summary and shift. Numerical refinement requires the same
source and summary object, query and shift. Tests cover changed rows both inside
and outside the refined block, truncated sources, changed queries and shifts,
duplicated partitions and mismatched summary dimensions.

These checks detect stale or malformed ordinary inputs. They are not an
authenticated import boundary and do not prove moment identities for arbitrary
hand-constructed summaries. Sound use requires `summarize` on the intended visible
source; a shape check alone is not a moment certificate. The CPU interfaces have
no separate epoch argument: their identity is the exact supplied ordered data.

## Coupled bound and exact witnesses

For one block/channel let `Z` be positive mass, `M` its centered numerator, and
`a=min(v−nu)`, `b=max(v−nu)`. Positive weights imply `a Z <= M <= b Z`. The new
optional `coupled=True` residual computes the minimum and maximum of
`M + (nu−boundary) Z` over the existing rectangle intersected with these two
inequalities. Candidate mass values are the rectangle's two mass endpoints and
the intersections of its horizontal central bounds with each nonzero slope
`a,b`. Feasible endpoints at those mass values contain every polygon vertex.
The resulting support interval is contained in the original box support interval.
Intersecting refinement boxes preserves this containment monotonically.

For `Z in [1,4]`, `M in [1/2,3/2]`, `a=−1/2`, `b=1/2`, and coefficient `−1/4`,
box-only optimization gives `[−1/2,5/4]`, range-only optimization gives `[−3,1]`,
and their scalar interval intersection gives `[−1/2,1]`. Joint polygon
optimization gives the strictly tighter `[−1/2,3/4]`. This is a feasible-metadata
geometry witness, not a generated attention instance.

The generated attention witness has `q=1`, keys `−2,2`, values
`1−1/2560,1+1/2560`, one rank-zero block. The box screen abstains; coupling
certifies BF16 value 1. Both values already lie in its BF16 cell, so a global
value-hull screen also explains that example. Adding a singleton key `−16` with
value `3` makes the global hull fail, while block mass/range coupling still
certifies 1. The direct rational oracle independently verifies both certificates.

## Reproducible results and limitations

Run `.venv/bin/python scripts/certification_experiments.py` to regenerate
`coupling.json`. Seed 962605 covers 53 cases and 149 initial coordinates, including
four ranks, four key spreads, narrow/broad value channels, exact BF16 midpoint
channels, constant no-effect cases and named witnesses. Every sequential block
refinement is retained. The file stores source inputs, oracle intervals and
coverage, cell boundaries, both residual intervals, widths, acceptance, and
separate CPU setup, evaluation, screening, refinement and actually executed full
rational fallback costs. Stored decimal endpoints are display approximations;
certificate-to-oracle comparisons in the script use exact fractions.

Initial box acceptance is 57/149; coupled acceptance is 61/149. Residual width is
strictly improved for 30/149 initial coordinates and unchanged for 119/149.
The global value hull accepts 50/149; only one of the four added coupling
acceptances goes beyond that baseline, in the prescribed suppressed-outlier
witness. These are synthetic examples, not a claim of improved model workloads.
No false certificate or independent coverage failure occurred in the saved run.
Broad-value failures and exact-midpoint abstentions remain in the data. The
exported directed GPU format remains box-only and does not implement coupling.

The first experiment attempt used zero-tolerance comparison of an approximate
mpmath result with an exact zero-width interval for a constant channel and
failed on numerical cancellation. The recorded run uses 110 decimal digits and
an explicit scaled `1e-95` comparison tolerance, saved per coordinate. Exact
rational certificate decisions are unchanged and use no tolerance.

The test log is `pytest.txt`: 46 tests passed. Substantive checks include 500
independent exact half-plane LP comparisons, 40 randomized 100-digit direct
attention checks across all ranks and refinement, 40 numerical-versus-exact
geometry comparisons, arithmetic regression and stale-input rejection. Both
existing numerical and rational exported fixture generators also completed with
the compatible box API. These CPU checks do not supply GPU performance evidence.

# SPDX-License-Identifier: Apache-2.0
"""Independent geometry and exact-real enclosure checks for coupled residuals."""

import random
from fractions import Fraction as F
from itertools import combinations

import mpmath as mp
import numpy as np
import pytest

from cmk import rational as r
from cmk import reference as m


def polygon_oracle(z, c, lower, upper, coefficient):
    # Independent generic two-variable half-plane intersection. Each triple
    # means A*Z + B*M <= C; enumerate all pairs, without the specialized solver.
    constraints = [
        (-1, 0, -z.lo),
        (1, 0, z.hi),
        (0, -1, -c.lo),
        (0, 1, c.hi),
        (lower, -1, 0),
        (-upper, 1, 0),
    ]
    objectives = []
    for (a, b, t), (x, y, s) in combinations(constraints, 2):
        det = a * y - b * x
        if not det:
            continue
        zz, mm = F(t * y - b * s, det), F(a * s - t * x, det)
        if all(A * zz + B * mm <= C for A, B, C in constraints):
            objectives.append(mm + coefficient * zz)
    if not objectives:
        raise ValueError("Empty independent polygon")
    return r.Interval(min(objectives), max(objectives))


def test_coupled_support_matches_independent_halfplane_oracle():
    rng = random.Random(581004)
    checked = 0
    for _ in range(500):
        z = r.Interval(F(rng.randrange(0, 5), 3), F(rng.randrange(5, 12), 3))
        c = r.Interval(F(rng.randrange(-10, 0), 4), F(rng.randrange(0, 10), 4))
        lower, upper = F(rng.randrange(-5, 1), 3), F(rng.randrange(0, 6), 3)
        coefficient = F(rng.randrange(-12, 13), 4)
        coupled = r.coupled_block_residual(z, c, lower, upper, coefficient)
        assert coupled == polygon_oracle(z, c, lower, upper, coefficient)
        box = c + z.scale(coefficient)
        assert box.lo <= coupled.lo <= coupled.hi <= box.hi
        checked += 1
    assert checked == 500
    assert r.coupled_block_residual(
        r.Interval(1, 4), r.Interval(-2, 2), F(-1, 2), F(1, 2), 1
    ) == r.Interval(F(1, 2), 6)
    # Sharing Z is stronger than intersecting two separately optimized scalar
    # residual intervals: box is [-1/2,5/4], range-only is [-3,1].
    assert r.coupled_block_residual(
        r.Interval(1, 4), r.Interval(F(1, 2), F(3, 2)), F(-1, 2), F(1, 2), F(-1, 4)
    ) == r.Interval(F(-1, 2), F(3, 4))
    with pytest.raises(ValueError, match="Inconsistent"):
        r.coupled_block_residual(r.Interval(1, 2), r.Interval(5, 6), -1, 1, 0)


def test_coupled_residual_dense_high_precision_and_refinement():
    rng = random.Random(500731)

    def high(x):
        return mp.mpf(x.numerator) / x.denominator

    with mp.workdps(100):
        for trial in range(40):
            K = [[F(rng.randrange(-16, 17), 16) for _ in range(3)] for _ in range(8)]
            V = [[F(rng.randrange(-16, 17), 16) for _ in range(2)] for _ in range(8)]
            q = [F(rng.randrange(-4, 5), 8) for _ in range(3)]
            s = r.summarize(
                K, V, [range(4), range(4, 8)], rank=trial % 4, keep_diagonal=trial % 2 == 0
            )
            e, shift = r.evaluate(q, s, bits=72)
            weights = [
                mp.exp(sum(high(x) * high(y) for x, y in zip(q, k)) - high(shift)) for k in K
            ]
            for j in range(2):
                boundary = F(rng.randrange(-10, 11), 16)
                actual = sum(w * (high(v[j]) - high(boundary)) for w, v in zip(weights, V))
                box = r.residual(e, boundary, j)
                coupled = r.residual(e, boundary, j, coupled=True)
                assert high(coupled.lo) <= actual <= high(coupled.hi)
                assert box.lo <= coupled.lo <= coupled.hi <= box.hi
                fine = [r.refine(q, K, V, block, old, shift, bits=112) for block, old in zip(s, e)]
                after = r.residual(fine, boundary, j, coupled=True)
                assert coupled.lo <= after.lo <= after.hi <= coupled.hi
                assert high(after.lo) <= actual <= high(after.hi)


def test_strict_acceptance_witness_and_constant_no_effect():
    K = [[F(-2)], [F(2)]]
    V = [[F(1) - F(1, 2560)], [F(1) + F(1, 2560)]]
    q = [F(1)]
    e, _ = r.evaluate(q, r.summarize(K, V, [[0, 1]], rank=0))
    assert r.screen(e) == [None]
    assert r.screen(e, coupled=True) == [F(1)]
    oracle = r.direct_oracle(q, K, V)[0]
    assert r.rounded_interval(oracle) == 1
    # A negligible outlier rules out a global convex-hull explanation while
    # block mass bounds retain its negligible contribution.
    outlier_k, outlier_v = K + [[F(-16)]], V + [[F(3)]]
    outlier, _ = r.evaluate(q, r.summarize(outlier_k, outlier_v, [[0, 1], [2]], rank=0))
    assert r.screen(outlier) == [None]
    assert r.screen(outlier, coupled=True) == [F(1)]
    assert r.rounded_interval(r.direct_oracle(q, outlier_k, outlier_v)[0]) == 1
    assert max(v[0] for v in outlier_v) > r.bf16_cell(F(1))[1]
    V = [[F(257, 256)], [F(257, 256)]]
    e, _ = r.evaluate(q, r.summarize(K, V, [[0, 1]], rank=0))
    assert r.screen(e) == r.screen(e, coupled=True) == [None]
    for boundary in (F(1), F(257, 256), F(2)):
        assert r.residual(e, boundary, 0) == r.residual(e, boundary, 0, coupled=True)


def test_numerical_coupling_matches_rational_geometry():
    rng = np.random.default_rng(71290)
    for _ in range(40):
        k = rng.normal(size=(12, 3))
        v = rng.normal(size=(12, 2))
        q = rng.normal(size=3) / 3
        s = m.summarize(k, v, [range(6), range(6, 12)], rank=1)
        e = m.evaluate(q, s)
        boundary = rng.normal(size=2)
        low, high = e.residual(boundary, coupled=True)
        exact_low, exact_high = [], []
        for j in range(2):
            sums = r.Interval.point(0)
            for b in range(2):
                sums += polygon_oracle(
                    r.Interval(e.zlo[b], e.zhi[b]),
                    r.Interval(e.mlo[b, j], e.mhi[b, j]),
                    F(e.value_lower[b, j]),
                    F(e.value_upper[b, j]),
                    F(e.nu[b, j] - boundary[j]),
                )
            exact_low.append(float(sums.lo))
            exact_high.append(float(sums.hi))
        np.testing.assert_allclose(low, exact_low, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(high, exact_high, rtol=2e-14, atol=2e-14)

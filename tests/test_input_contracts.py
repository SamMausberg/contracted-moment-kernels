# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import numpy as np
import pytest

from cmk import rational as r
from cmk import reference as m
from cmk.export import export_rows


def test_integer_interval_reciprocal_stays_exact():
    # Binary64 1/3 is below exact 1/3: keeping integer endpoints used to make
    # the upper reciprocal endpoint inward, an actual enclosure defect.
    out = r.Interval(1, 1) / r.Interval(3, 3)
    assert out.lo == out.hi == F(1, 3)
    assert isinstance(out.lo, F) and isinstance(out.hi, F)
    assert r.Interval(F(1, 3), F(2, 3)).scale(0.1) == r.Interval(F(0.1) / 3, 2 * F(0.1) / 3)


@pytest.mark.parametrize("oracle", [r.direct_oracle, m.dense_attention])
@pytest.mark.parametrize(
    "q,k,v",
    [
        ([1, 2], [[1]], [[1]]),
        ([1], [[1], [2]], [[1]]),
        ([1], [[1]], [[1], [2]]),
        ([1], [[1], [1, 2]], [[1], [2]]),
        ([1], [[1], [2]], [[1], [1, 2]]),
        ([], [], []),
        ([1], [[math.nan]], [[1]]),
        ([1], [[1]], [[math.inf]]),
    ],
)
def test_oracles_reject_invalid_shapes_and_domains(oracle, q, k, v):
    with pytest.raises((ValueError, OverflowError)):
        oracle(q, k, v)


@pytest.mark.parametrize("implementation", [r, m])
@pytest.mark.parametrize(
    "rank,groups",
    [
        (0.5, [[0, 1]]),
        (True, [[0, 1]]),
        (0, [[0.0, 1.0]]),
        (0, [[0], [0]]),
        (0, [[0, 2]]),
        (0, [[-1, 0]]),
    ],
)
def test_summary_rejects_lossy_indices_and_rank(implementation, rank, groups):
    with pytest.raises(ValueError):
        implementation.summarize([[0], [1]], [[1], [2]], groups, rank=rank)


def test_rational_summary_immutable_and_source_query_shift_bound():
    K, V, q = [[F(0)], [F(1)]], [[F(1)], [F(2)]], [F(1)]
    s = r.summarize(K, V, [[0], [1]])
    e, shift = r.evaluate(q, s)
    with pytest.raises(FrozenInstanceError):
        s[0].rank = 0
    with pytest.raises(TypeError):
        s[0].mu[0] = 99
    for args in (
        ([F(2)], K, V, shift),
        (q, K, [[F(3)], [F(2)]], shift),
        (q, K, [[F(1)], [F(3)]], shift),
        (q, K, V, shift + 1),
        (q, K[:1], V[:1], shift),
    ):
        qq, kk, vv, ss = args
        with pytest.raises(ValueError):
            r.refine(qq, kk, vv, s[0], e[0], ss)
    other = r.summarize(K, [[F(3)], [F(2)]], [[0], [1]])
    with pytest.raises(ValueError, match="different visible sources"):
        r.evaluate(q, [s[0], other[1]])
    with pytest.raises(ValueError, match="partition"):
        r.evaluate(q, [s[0], s[0]])
    with pytest.raises(ValueError):
        replace(s[0], kr=(-1,))
    with pytest.raises(ValueError):
        replace(s[0], cross=((F(0), F(1)),))


def test_numerical_source_and_summary_identity_checked_before_refinement():
    K, V, q = np.array([[0.0], [1.0]]), np.array([[1.0], [2.0]]), np.array([1.0])
    s = m.summarize(K, V, [[0], [1]])
    e = m.evaluate(q, s)
    with pytest.raises(ValueError):
        s.mu[0, 0] = 20
    with pytest.raises(ValueError):
        m.refine_block(q * 2, s, e, 0, K, V)
    with pytest.raises(ValueError):
        m.refine_block(q, s, e, 0, K, V + 1)
    with pytest.raises(ValueError):
        m.refine_block(q, replace(s), e, 0, K, V)
    e.shift += 1
    with pytest.raises(ValueError):
        m.refine_block(q, s, e, 0, K, V)
    with pytest.raises(ValueError):
        m.adaptive(q, s, K, V + 1)
    with pytest.raises(ValueError):
        replace(s, count=np.array([1.0, 0.0]))


def test_empty_bad_envelopes_and_unsupported_exponential_domain():
    for fn in (r.screen, export_rows):
        with pytest.raises(ValueError):
            fn([])
    with pytest.raises(ValueError):
        r.Envelope(r.Interval(-1, 1), (r.Interval(0, 1),), (F(1),))
    with pytest.raises(ValueError):
        r.exp_interval(F(33))
    with pytest.raises(ValueError):
        r.exp_interval(F(0), bits=32.5)
    with pytest.raises(ValueError):
        r.dot([F(1)], [F(1), F(2)])

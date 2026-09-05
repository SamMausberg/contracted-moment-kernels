# SPDX-License-Identifier: Apache-2.0
import math
from itertools import product

import numpy as np
import pytest

from cmk import reference as m


def test_randomized_envelopes():
    rng = np.random.default_rng(9042601)
    checks = 0
    for t in range(500):
        n = int(rng.integers(4, 50))
        d = int(rng.integers(1, 9))
        h = int(rng.integers(1, 6))
        B = min(n, 4)
        r = int(rng.integers(0, d + 1))
        groups = np.array_split(np.arange(n), B)
        k = rng.normal(size=(n, d)) * 10 ** rng.uniform(-3, 0)
        v = rng.normal(size=(n, h))
        if t % 19 == 0:
            v[:] = rng.normal(size=h)
        q = rng.normal(size=d) / np.sqrt(d)
        s = m.summarize(k, v, groups, r, keep_diagonal=(t % 2 == 0))
        e = m.evaluate(q, s)
        y = m.dense_attention(q, k, v)
        for b, ids in enumerate(groups):
            p = np.exp(k[ids] @ q - e.shift)
            z = p.sum()
            c = p @ (v[ids] - s.nu[b])
            slack = 1e-10 * (1 + np.max(np.abs(c)) + z)
            assert e.zlo[b] - slack <= z <= e.zhi[b] + slack
            assert np.all(e.mlo[b] - slack <= c) and np.all(c <= e.mhi[b] + slack)
        low, high = e.residual(y)
        assert np.all(low <= 1e-9) and np.all(high >= -1e-9)
        if np.max(np.abs(e.candidate())) < 2**100:
            lo, hi, b = m.bf16_cells(e.candidate())
            gate = e.contains_cell(lo, hi)
            assert np.all((y[gate] > lo[gate]) & (y[gate] < hi[gate]))
        checks += 1
    assert checks == 500


def test_extremal_residuals_and_refinement():
    rng = np.random.default_rng(15926)
    for _ in range(100):
        B = 3
        zl = rng.uniform(0.2, 2, B)
        zu = zl + rng.uniform(0.1, 2, B)
        ml = rng.normal(size=(B, 1))
        mu = ml + rng.uniform(0.1, 1, (B, 1))
        nu = rng.normal(size=(B, 1))
        e = m.Envelopes(zl, zu, ml, mu, nu, (zl + zu) / 2, (ml + mu) / 2, 0.0)
        a = rng.normal(size=1)
        low, high = e.residual(a)
        vals = []
        for signs in product((0, 1), repeat=2 * B):
            z = np.where(signs[:B], zu, zl)
            c = np.where(np.array(signs[B:])[:, None], mu, ml)
            vals.append(float(((nu - a) * z[:, None] + c).sum()))
        assert math.isclose(min(vals), low[0], abs_tol=1e-12)
        assert math.isclose(max(vals), high[0], abs_tol=1e-12)
        e2 = m.Envelopes(
            (2 * zl + zu) / 3,
            (zl + 2 * zu) / 3,
            (2 * ml + mu) / 3,
            (ml + 2 * mu) / 3,
            nu,
            (zl + zu) / 2,
            (ml + mu) / 2,
            0.0,
        )
        lo2, hi2 = e2.residual(a)
        assert low[0] <= lo2[0] + 1e-12 and hi2[0] <= high[0] + 1e-12


def test_box_test_stronger_than_symmetric_bound_example():
    e = m.Envelopes(
        np.array([1.0, 1.0]),
        np.array([2.0, 2.0]),
        np.zeros((2, 1)),
        np.zeros((2, 1)),
        np.array([[0.0], [2.0]]),
        np.array([1.5, 1.5]),
        np.zeros((2, 1)),
        0.0,
    )
    assert e.contains_cell(np.array([0.6]), np.array([1.4])).all()
    # The coarse triangle bound at candidate 1 is radius .5: it cannot prove this cell.
    assert 1 - 0.5 < 0.6 and 1 + 0.5 > 1.4


def test_tail_improvement():
    for rho in np.geomspace(1e-12, 30, 300):
        sharp = m.tail_coefficient(float(rho))
        old = m.tail_coefficient(float(rho), False)
        assert 0 <= sharp <= old * (1 + 1e-14)
        for t in (-rho, -rho / 3, rho / 3, rho):
            rem = abs(math.expm1(t) - t - t * t / 2)
            assert rem <= sharp * t * t + 1e-12 * (1 + math.exp(rho))


def test_bad_inputs_and_abstention():
    k = np.ones((4, 2))
    v = np.ones((4, 1))
    with pytest.raises(ValueError):
        m.summarize(k, v, [[0, 1], [1, 2, 3]])
    with pytest.raises(ValueError):
        m.summarize(k, v, [list(range(4))], 3)
    with pytest.raises(ValueError):
        m.summarize(k * np.nan, v, [list(range(4))])
    e = m.Envelopes(
        np.array([1.0]),
        np.array([1.0]),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.array([[257 / 256]]),
        np.array([1.0]),
        np.zeros((1, 1)),
        0.0,
    )
    lo, hi, _ = m.bf16_cells(e.candidate())
    assert not e.contains_cell(lo, hi).any()  # Exact midpoint, strict test refuses.

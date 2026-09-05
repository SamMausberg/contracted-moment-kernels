# SPDX-License-Identifier: Apache-2.0
"""Numerical research path, NOT a certified floating-point implementation.

Input queries already include attention scaling. Summaries cover exactly the
visible keys, after positional transformations. Every approximate result and
screen in this module is heuristic with respect to machine roundoff.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def tail_coefficient(rho: float, sharp: bool = True) -> float:
    """(exp(rho)-1-rho-rho**2/2)/rho**2, evaluated stably near zero.

    The exact-real formula is a rigorous majorant; this floating evaluation is
    not outward rounded. Overflow is an explicit abstention condition.
    """
    if not math.isfinite(rho) or rho < 0:
        return math.inf
    if rho == 0:
        return 0.0
    if rho > 650:
        return math.inf
    if not sharp:
        return math.exp(rho) * rho / 6
    if rho < 0.5:
        term, total = rho / 6, rho / 6
        for n in range(4, 36):
            term *= rho / n
            total += term
        return total
    return (math.expm1(rho) - rho - rho * rho / 2) / (rho * rho)


@dataclass
class Summary:
    groups: tuple[np.ndarray, ...]
    mu: np.ndarray
    nu: np.ndarray
    cov: np.ndarray
    cross: np.ndarray
    diagonal: np.ndarray
    eta: np.ndarray
    key_radius: np.ndarray
    value_radius: np.ndarray
    count: np.ndarray
    rank: int

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in
                   ("mu", "nu", "cov", "cross", "diagonal", "eta",
                    "key_radius", "value_radius", "count"))


def summarize(keys, values, groups, rank=None, keep_diagonal=True) -> Summary:
    k, v = np.asarray(keys, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if k.ndim != 2 or v.ndim != 2 or not len(k) or len(k) != len(v):
        raise ValueError("Expected nonempty K[N,d], V[N,h]")
    if not k.shape[1] or not v.shape[1] or not np.isfinite(k).all() or not np.isfinite(v).all():
        raise ValueError("Finite, positive dimensions required")
    n, d = k.shape
    h = v.shape[1]
    r = d if rank is None else int(rank)
    if not 0 <= r <= d:
        raise ValueError("rank must be in [0,d]")
    groups = tuple(np.asarray(g, dtype=np.int64) for g in groups)
    if not groups or any(g.ndim != 1 or not len(g) for g in groups):
        raise ValueError("Empty/invalid group")
    if not np.array_equal(np.sort(np.concatenate(groups)), np.arange(n)):
        raise ValueError("Groups must partition all and only the visible keys")
    B = len(groups)
    mu, nu = np.empty((B, d)), np.empty((B, h))
    cov, cross = np.empty((B, r, r)), np.empty((B, r, h))
    diagonal, eta = np.zeros((B, r, h)), np.empty((B, h))
    kr, vr = np.empty((B, d)), np.empty((B, h))
    for b, ids in enumerate(groups):
        mu[b], nu[b] = k[ids].mean(0), v[ids].mean(0)
        dk, dv = k[ids] - mu[b], v[ids] - nu[b]
        z = dk[:, :r]
        cov[b], cross[b] = z.T @ z / len(ids), z.T @ dv / len(ids)
        H = np.einsum('ni,nj,nk->kij', z, z, dv, optimize=True) / len(ids)
        if r:
            if keep_diagonal:
                diagonal[b] = np.diagonal(H, axis1=1, axis2=2).T
                H[:, np.arange(r), np.arange(r)] = 0
            # Symmetric matrix spectral norm <= max absolute row sum.
            eta[b] = np.max(np.sum(np.abs(H), axis=2), axis=1)
        else:
            eta[b] = 0
        kr[b], vr[b] = np.abs(dk).max(0), np.abs(dv).max(0)
    return Summary(groups, mu, nu, cov, cross, diagonal, eta, kr, vr,
                   np.array([len(g) for g in groups], dtype=np.float64), r)


@dataclass
class Envelopes:
    zlo: np.ndarray
    zhi: np.ndarray
    mlo: np.ndarray
    mhi: np.ndarray
    nu: np.ndarray
    zhat: np.ndarray
    mhat: np.ndarray
    shift: float

    def candidate(self) -> np.ndarray:
        return ((self.nu * self.zhat[:, None] + self.mhat).sum(0)
                / self.zhat.sum())

    def residual(self, boundary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = self.nu - np.asarray(boundary)
        low = (self.mlo + np.minimum(a * self.zlo[:, None], a * self.zhi[:, None])).sum(0)
        high = (self.mhi + np.maximum(a * self.zlo[:, None], a * self.zhi[:, None])).sum(0)
        return low, high

    def contains_cell(self, lower, upper) -> np.ndarray:
        """Strict endpoint screen. Numerical only; ties are rejected."""
        lo, _ = self.residual(lower)
        _, hi = self.residual(upper)
        finite = np.isfinite(self.zlo).all() and np.isfinite(self.zhi).all()
        return (lo > 0) & (hi < 0) & finite & (self.zlo.sum() > 0)


def evaluate(q, s: Summary, sharp=True) -> Envelopes:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (s.mu.shape[1],) or not np.isfinite(q).all():
        raise ValueError("Invalid scaled query")
    r = s.rank
    u = q[:r]
    logits = s.mu @ q
    shift = float(np.max(logits + np.log(s.count)))
    w = s.count * np.exp(logits - shift)
    variance = np.maximum(0, np.einsum('i,bij,j->b', u, s.cov, u))
    rho = s.key_radius[:, :r] @ np.abs(u)
    eps = s.key_radius[:, r:] @ np.abs(q[r:])
    if np.any(rho > 650) or np.any(eps > 650):
        raise OverflowError("Summary bound overflow: use the declared reference fallback")
    tau = np.array([tail_coefficient(float(x), sharp) for x in rho]) * variance
    A = 1 + variance / 2
    zhat = w * A
    projected_lo, projected_hi = w * np.maximum(1, A - tau), w * (A + tau)
    mhat = w[:, None] * (np.einsum('i,bih->bh', u, s.cross)
                           + np.einsum('i,bih->bh', u*u, s.diagonal) / 2)
    beta = w[:, None] * (s.eta * float(u@u) / 2 + tau[:, None] * s.value_radius)
    # Bound discarded-coordinate scores rather than silently dropping them.
    inflate = np.exp(eps)
    beta += np.expm1(eps)[:, None] * projected_hi[:, None] * s.value_radius
    return Envelopes(projected_lo / inflate, projected_hi * inflate,
                     mhat-beta, mhat+beta, s.nu, zhat, mhat, shift)


def dense_attention(q, keys, values) -> np.ndarray:
    score = np.asarray(keys) @ np.asarray(q)
    p = np.exp(score - np.max(score))
    return p @ np.asarray(values) / p.sum()


def bf16_cells(x):
    """Exact cell endpoints after a float64 candidate is chosen.

    Uses the rational BF16 conversion to avoid a float32 double-rounding step.
    Distinguishes numerical values, not signed-zero bit patterns.
    """
    from fractions import Fraction
    from .rational import bf16_round, bf16_cell
    x = np.asarray(x, dtype=np.float64)
    lower, upper, rounded = [], [], []
    for t in x:
        b = bf16_round(Fraction(float(t)))
        lo, hi = bf16_cell(b)
        lower.append(float(lo)); upper.append(float(hi)); rounded.append(float(b))
    return np.array(lower), np.array(upper), np.array(rounded)


def refine_block(q, s, e: Envelopes, b, keys, values):
    """Dense numerical block scan. Not an exact arithmetic certificate."""
    ids = s.groups[b]
    p = np.exp(np.asarray(keys)[ids] @ q - e.shift)
    z = p.sum()
    m = p @ (np.asarray(values)[ids] - s.nu[b])
    # Numerical path deliberately does not advertise proof-carrying refinement.
    e.zlo[b] = e.zhi[b] = e.zhat[b] = z
    e.mlo[b] = e.mhi[b] = e.mhat[b] = m


def adaptive(q, s, keys, values, max_blocks=None):
    """Numerical scheduling experiment; never call it verified inference."""
    e = evaluate(q, s)
    B = len(s.groups)
    limit = B if max_blocks is None else min(B, max(0, int(max_blocks)))
    remaining = set(range(B))
    scans = []
    for step in range(limit + 1):
        candidate = e.candidate()
        lower, upper, rounded = bf16_cells(candidate)
        accepted = e.contains_cell(lower, upper)
        if accepted.all() or step == limit:
            return dict(candidate=candidate, rounded=rounded, accepted=accepted,
                        scanned_blocks=scans, scanned_tokens=sum(len(s.groups[b]) for b in scans))
        width = e.mhi - e.mlo + np.abs(e.nu-candidate) * (e.zhi-e.zlo)[:, None]
        priority = np.max(width[:, ~accepted], axis=1)
        b = max(remaining, key=lambda j: float(priority[j]))
        refine_block(q, s, e, b, keys, values)
        remaining.remove(b); scans.append(b)
    raise AssertionError("unreachable")

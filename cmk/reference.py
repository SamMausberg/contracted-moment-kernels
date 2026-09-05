# SPDX-License-Identifier: Apache-2.0
"""Numerical research path, NOT a certified floating-point implementation.

Input queries already include attention scaling. Summaries cover exactly the
visible keys, after positional transformations. Every approximate result and
screen in this module is heuristic with respect to machine roundoff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .rational import _integer


def _inputs(keys, values, q=None):
    k, v = np.asarray(keys, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if k.ndim != 2 or v.ndim != 2 or not len(k) or len(k) != len(v):
        raise ValueError("Expected nonempty K[N,d], V[N,h]")
    if not k.shape[1] or not v.shape[1] or not np.isfinite(k).all() or not np.isfinite(v).all():
        raise ValueError("Finite, positive dimensions required")
    query = None if q is None else np.asarray(q, dtype=np.float64)
    if query is not None and (query.shape != (k.shape[1],) or not np.isfinite(query).all()):
        raise ValueError("Invalid scaled query")
    return k, v, query


def _fingerprint(k, v):
    digest = sha256()
    for a in (k, v):
        digest.update(str(a.shape).encode())
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


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


@dataclass(frozen=True)
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
    value_lower: np.ndarray
    value_upper: np.ndarray
    source_identity: str

    def __post_init__(self):
        # Defensive copies keep subsequent caller input edits from changing
        # existing summaries. This is not an authenticated import format.
        for name in (
            "mu",
            "nu",
            "cov",
            "cross",
            "diagonal",
            "eta",
            "key_radius",
            "value_radius",
            "count",
            "value_lower",
            "value_upper",
        ):
            a = np.array(getattr(self, name), dtype=np.float64, copy=True)
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        groups = tuple(np.array(g, copy=True) for g in self.groups)
        for g in groups:
            g.setflags(write=False)
        object.__setattr__(self, "groups", groups)
        _validate_summary(self)

    @property
    def nbytes(self) -> int:
        return sum(
            getattr(self, name).nbytes
            for name in (
                "mu",
                "nu",
                "cov",
                "cross",
                "diagonal",
                "eta",
                "key_radius",
                "value_radius",
                "count",
                "value_lower",
                "value_upper",
            )
        )


def _validate_summary(s):
    if s.mu.ndim != 2 or s.nu.ndim != 2 or not s.mu.size or not s.nu.size:
        raise ValueError("Invalid summary center dimensions")
    B, d = s.mu.shape
    h, r = s.nu.shape[1], _integer(s.rank, "rank")
    if not 0 <= r <= d or len(s.groups) != B:
        raise ValueError("Invalid summary rank or groups")
    shapes = {
        "nu": (B, h),
        "cov": (B, r, r),
        "cross": (B, r, h),
        "diagonal": (B, r, h),
        "eta": (B, h),
        "key_radius": (B, d),
        "value_radius": (B, h),
        "count": (B,),
        "value_lower": (B, h),
        "value_upper": (B, h),
    }
    for name, shape in shapes.items():
        a = getattr(s, name)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError("Invalid summary array: " + name)
    if not np.isfinite(s.mu).all() or any(
        np.any(getattr(s, n) < 0) for n in ("eta", "key_radius", "value_radius")
    ):
        raise ValueError("Nonfinite center or negative summary bound")
    if any(g.ndim != 1 or not len(g) or not np.issubdtype(g.dtype, np.integer) for g in s.groups):
        raise ValueError("Invalid summary group")
    flat = np.concatenate(s.groups)
    if not np.array_equal(np.sort(flat), np.arange(len(flat))) or not np.array_equal(
        s.count, [len(g) for g in s.groups]
    ):
        raise ValueError("Summary groups must partition visible rows with matching counts")
    if (
        np.any(s.value_lower > s.value_upper)
        or np.any(s.value_lower < -s.value_radius)
        or np.any(s.value_upper > s.value_radius)
    ):
        raise ValueError("Invalid centered value range")


def summarize(keys, values, groups, rank=None, keep_diagonal=True) -> Summary:
    k, v, _ = _inputs(keys, values)
    n, d = k.shape
    h = v.shape[1]
    r = d if rank is None else _integer(rank, "rank")
    if not 0 <= r <= d:
        raise ValueError("rank must be in [0,d]")
    groups = tuple(np.asarray(g) for g in groups)
    if not groups or any(g.ndim != 1 or not len(g) for g in groups):
        raise ValueError("Empty/invalid group")
    if any(not np.issubdtype(g.dtype, np.integer) for g in groups):
        raise ValueError("Group indices must be integers")
    if not np.array_equal(np.sort(np.concatenate(groups)), np.arange(n)):
        raise ValueError("Groups must partition all and only the visible keys")
    B = len(groups)
    mu, nu = np.empty((B, d)), np.empty((B, h))
    cov, cross = np.empty((B, r, r)), np.empty((B, r, h))
    diagonal, eta = np.zeros((B, r, h)), np.empty((B, h))
    kr, vr = np.empty((B, d)), np.empty((B, h))
    lower, upper = np.empty((B, h)), np.empty((B, h))
    for b, ids in enumerate(groups):
        mu[b], nu[b] = k[ids].mean(0), v[ids].mean(0)
        dk, dv = k[ids] - mu[b], v[ids] - nu[b]
        z = dk[:, :r]
        cov[b], cross[b] = z.T @ z / len(ids), z.T @ dv / len(ids)
        H = np.einsum("ni,nj,nk->kij", z, z, dv, optimize=True) / len(ids)
        if r:
            if keep_diagonal:
                diagonal[b] = np.diagonal(H, axis1=1, axis2=2).T
                H[:, np.arange(r), np.arange(r)] = 0
            # Symmetric matrix spectral norm <= max absolute row sum.
            eta[b] = np.max(np.sum(np.abs(H), axis=2), axis=1)
        else:
            eta[b] = 0
        kr[b], vr[b] = np.abs(dk).max(0), np.abs(dv).max(0)
        lower[b], upper[b] = dv.min(0), dv.max(0)
    return Summary(
        groups,
        mu,
        nu,
        cov,
        cross,
        diagonal,
        eta,
        kr,
        vr,
        np.array([len(g) for g in groups], dtype=np.float64),
        r,
        lower,
        upper,
        _fingerprint(k, v),
    )


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
    value_lower: np.ndarray | None = None
    value_upper: np.ndarray | None = None
    query: np.ndarray | None = None
    source_identity: str | None = None
    summary: Summary | None = None

    def candidate(self) -> np.ndarray:
        return (self.nu * self.zhat[:, None] + self.mhat).sum(0) / self.zhat.sum()

    def residual(self, boundary: np.ndarray, *, coupled=False) -> tuple[np.ndarray, np.ndarray]:
        a = self.nu - np.asarray(boundary)
        low = self.mlo + np.minimum(a * self.zlo[:, None], a * self.zhi[:, None])
        high = self.mhi + np.maximum(a * self.zlo[:, None], a * self.zhi[:, None])
        if coupled:
            if self.value_lower is None or self.value_upper is None:
                raise ValueError("Coupled residual requires source value extrema")
            for b in range(len(self.zlo)):
                for j in range(self.nu.shape[1]):
                    lower, upper = self.value_lower[b, j], self.value_upper[b, j]
                    zs = [self.zlo[b], self.zhi[b]]
                    for slope in (lower, upper):
                        if slope:
                            zs.extend((self.mlo[b, j] / slope, self.mhi[b, j] / slope))
                    vals = []
                    for z in zs:
                        if self.zlo[b] <= z <= self.zhi[b]:
                            lo = max(self.mlo[b, j], lower * z)
                            hi = min(self.mhi[b, j], upper * z)
                            if lo <= hi:
                                vals.extend((lo + a[b, j] * z, hi + a[b, j] * z))
                    # Rounding may make a numerical polygon appear empty.
                    # Abstain from tightening in that case. Still heuristic.
                    if vals:
                        low[b, j] = max(low[b, j], min(vals))
                        high[b, j] = min(high[b, j], max(vals))
        low, high = low.sum(0), high.sum(0)
        return low, high

    def contains_cell(self, lower, upper, *, coupled=False) -> np.ndarray:
        """Strict endpoint screen. Numerical only; ties are rejected."""
        lo, _ = self.residual(lower, coupled=coupled)
        _, hi = self.residual(upper, coupled=coupled)
        finite = np.isfinite(self.zlo).all() and np.isfinite(self.zhi).all()
        return (lo > 0) & (hi < 0) & finite & (self.zlo.sum() > 0)


def evaluate(q, s: Summary, sharp=True) -> Envelopes:
    _validate_summary(s)
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (s.mu.shape[1],) or not np.isfinite(q).all():
        raise ValueError("Invalid scaled query")
    r = s.rank
    u = q[:r]
    logits = s.mu @ q
    shift = float(np.max(logits + np.log(s.count)))
    w = s.count * np.exp(logits - shift)
    variance = np.maximum(0, np.einsum("i,bij,j->b", u, s.cov, u))
    rho = s.key_radius[:, :r] @ np.abs(u)
    eps = s.key_radius[:, r:] @ np.abs(q[r:])
    if np.any(rho > 650) or np.any(eps > 650):
        raise OverflowError("Summary bound overflow: use the declared reference fallback")
    tau = np.array([tail_coefficient(float(x), sharp) for x in rho]) * variance
    A = 1 + variance / 2
    zhat = w * A
    projected_lo, projected_hi = w * np.maximum(1, A - tau), w * (A + tau)
    mhat = w[:, None] * (
        np.einsum("i,bih->bh", u, s.cross) + np.einsum("i,bih->bh", u * u, s.diagonal) / 2
    )
    beta = w[:, None] * (s.eta * float(u @ u) / 2 + tau[:, None] * s.value_radius)
    # Bound discarded-coordinate scores rather than silently dropping them.
    inflate = np.exp(eps)
    beta += np.expm1(eps)[:, None] * projected_hi[:, None] * s.value_radius
    out = Envelopes(
        projected_lo / inflate,
        projected_hi * inflate,
        mhat - beta,
        mhat + beta,
        s.nu,
        zhat,
        mhat,
        shift,
        s.value_lower,
        s.value_upper,
        q.copy(),
        s.source_identity,
        s,
    )
    out.query.setflags(write=False)
    if (
        any(
            not np.isfinite(getattr(out, name)).all()
            for name in ("zlo", "zhi", "mlo", "mhi", "zhat", "mhat")
        )
        or out.zlo.sum() <= 0
    ):
        raise OverflowError("Nonfinite or degenerate numerical envelope: use reference fallback")
    return out


def dense_attention(q, keys, values) -> np.ndarray:
    keys, values, q = _inputs(keys, values, q)
    score = keys @ q
    if not np.isfinite(score).all():
        raise OverflowError("Nonfinite numerical attention scores")
    p = np.exp(score - np.max(score))
    out = p @ values / p.sum()
    if not np.isfinite(out).all():
        raise OverflowError("Nonfinite numerical attention output")
    return out


def bf16_cells(x):
    """Exact cell endpoints after a float64 candidate is chosen.

    Uses the rational BF16 conversion to avoid a float32 double-rounding step.
    Distinguishes numerical values, not signed-zero bit patterns.
    """
    from fractions import Fraction

    from .rational import bf16_cell, bf16_round

    x = np.asarray(x, dtype=np.float64)
    lower, upper, rounded = [], [], []
    for t in x:
        b = bf16_round(Fraction(float(t)))
        lo, hi = bf16_cell(b)
        lower.append(float(lo))
        upper.append(float(hi))
        rounded.append(float(b))
    return np.array(lower), np.array(upper), np.array(rounded)


def refine_block(q, s, e: Envelopes, b, keys, values):
    """Dense numerical block scan. Not an exact arithmetic certificate."""
    keys, values, q = _inputs(keys, values, q)
    b = _integer(b, "Block index")
    if not 0 <= b < len(s.groups):
        raise ValueError("Invalid block index")
    shift = float(np.max(s.mu @ q + np.log(s.count)))
    if (
        _fingerprint(keys, values) != s.source_identity
        or e.summary is not s
        or e.source_identity != s.source_identity
        or not np.array_equal(q, e.query)
        or e.shift != shift
    ):
        raise ValueError("Refinement source, summary or query is stale")
    ids = s.groups[b]
    p = np.exp(np.asarray(keys)[ids] @ q - e.shift)
    z = p.sum()
    m = p @ (np.asarray(values)[ids] - s.nu[b])
    # Numerical path deliberately does not advertise proof-carrying refinement.
    e.zlo[b] = e.zhi[b] = e.zhat[b] = z
    e.mlo[b] = e.mhi[b] = e.mhat[b] = m


def adaptive(q, s, keys, values, max_blocks=None, *, coupled=False):
    """Numerical scheduling experiment; never call it verified inference."""
    keys, values, q = _inputs(keys, values, q)
    if _fingerprint(keys, values) != s.source_identity:
        raise ValueError("Adaptive source differs from summarized visible data")
    e = evaluate(q, s)
    B = len(s.groups)
    limit = B if max_blocks is None else min(B, max(0, _integer(max_blocks, "max_blocks")))
    remaining = set(range(B))
    scans = []
    for step in range(limit + 1):
        candidate = e.candidate()
        lower, upper, rounded = bf16_cells(candidate)
        accepted = e.contains_cell(lower, upper, coupled=coupled)
        if accepted.all() or step == limit:
            return dict(
                candidate=candidate,
                rounded=rounded,
                accepted=accepted,
                scanned_blocks=scans,
                scanned_tokens=sum(len(s.groups[b]) for b in scans),
            )
        width = e.mhi - e.mlo + np.abs(e.nu - candidate) * (e.zhi - e.zlo)[:, None]
        priority = np.max(width[:, ~accepted], axis=1)
        b = max(remaining, key=lambda j: float(priority[j]))
        refine_block(q, s, e, b, keys, values)
        remaining.remove(b)
        scans.append(b)
    raise AssertionError("unreachable")

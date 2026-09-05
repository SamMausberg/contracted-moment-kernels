"""Small, slow rational-interval attention certificate and independent oracle.

All algebra uses fractions.Fraction. Exponential enclosures use an explicitly
bounded positive Taylor series, reciprocal, and repeated squaring. No float
arithmetic is used in the certificate or BF16 rounding decision. This certifies
rounding of REAL softmax attention on rational inputs, NOT the result of any
particular GPU accumulation/exp implementation. Not a machine-checked proof.
"""
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F


def dot(a: list[F], b: list[F]) -> F:
    return sum((x*y for x, y in zip(a, b)), F(0))


def pow2(p: int) -> F:
    return F(2**p) if p >= 0 else F(1, 2**(-p))


@dataclass(frozen=True)
class Interval:
    lo: F
    hi: F

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("Reversed interval")

    @classmethod
    def point(cls, x: F) -> 'Interval':
        return cls(F(x), F(x))

    def __add__(self, other: 'Interval') -> 'Interval':
        return Interval(self.lo+other.lo, self.hi+other.hi)

    def scale(self, x: F) -> 'Interval':
        return Interval(min(self.lo*x, self.hi*x), max(self.lo*x, self.hi*x))

    def __mul__(self, other: 'Interval') -> 'Interval':
        vals = (self.lo*other.lo, self.lo*other.hi,
                self.hi*other.lo, self.hi*other.hi)
        return Interval(min(vals), max(vals))

    def __truediv__(self, other: 'Interval') -> 'Interval':
        if other.lo <= 0:
            raise ValueError("Only positive denominator intervals supported")
        return self * Interval(1/other.hi, 1/other.lo)


def exp_interval(x: F, bits: int = 64) -> Interval:
    """Sound interval for exp(x). bits sets base-series remainder target.

    The actual final interval width is allowed to grow during squaring. The
    resulting interval remains sound; bits is NOT a promised final error bound.
    Range is limited to avoid pathological resource use in this slow prototype.
    """
    x = F(x)
    if bits < 16 or abs(x) > 32:
        raise ValueError("Use bits >= 16 and |x| <= 32 in this reference")
    if x == 0:
        return Interval.point(F(1))
    if x < 0:
        p = exp_interval(-x, bits)
        return Interval(1/p.hi, 1/p.lo)
    squarings = 0
    while x > F(1, 2):
        x /= 2
        squarings += 1
    total, term, degree = F(1), F(1), 0
    target = pow2(-bits)
    while True:
        next_term = term*x/(degree+1)
        # Remaining term ratios are <= x/(degree+2).
        tail = next_term/(1-x/(degree+2))
        if tail <= target:
            out = Interval(total, total+tail)
            break
        total += next_term
        term = next_term
        degree += 1
    for _ in range(squarings):
        out = Interval(out.lo*out.lo, out.hi*out.hi)
    return out


def bf16_round(x: F) -> F:
    """Exact nearest-even BF16 rounding in the supported finite range.

    Numerically treats +0 and -0 as the same. Raises before overflow.
    """
    x = F(x)
    if abs(x) > pow2(127):
        raise ValueError("Out-of-range BF16 test value")
    if x == 0:
        return F(0)
    sign, a = (1 if x > 0 else -1), abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if a < pow2(e):
        e -= 1
    step = pow2(max(e-7, -133))
    z = a/step
    whole, remainder = divmod(z.numerator, z.denominator)
    twice = 2*remainder
    if twice > z.denominator or (twice == z.denominator and whole % 2):
        whole += 1
    return sign*whole*step


def rounded_interval(interval: Interval) -> F | None:
    a, b = bf16_round(interval.lo), bf16_round(interval.hi)
    return a if a == b else None



def bf16_cell(b: F) -> tuple[F, F]:
    """Open cell around a finite BF16 numerical value; excludes all ties."""
    b = F(b)
    if bf16_round(b) != b or abs(b) > pow2(120):
        raise ValueError("Expected supported finite BF16 value")
    if b == 0:
        return -pow2(-134), pow2(-134)
    if b < 0:
        lo, hi = bf16_cell(-b)
        return -hi, -lo
    e = b.numerator.bit_length() - b.denominator.bit_length()
    if b < pow2(e):
        e -= 1
    step = pow2(max(e-7, -133))
    previous_step = step/2 if b == pow2(e) and e > -126 else step
    return b-previous_step/2, b+step/2


@dataclass
class Summary:
    ids: list[int]
    mu: list[F]
    nu: list[F]
    cov: list[list[F]]
    cross: list[list[F]]
    diagonal: list[list[F]]
    eta: list[F]
    kr: list[F]
    vr: list[F]
    rank: int


def summarize(keys, values, groups, rank=None, keep_diagonal=True):
    if not keys or not values or not keys[0] or not values[0]:
        raise ValueError("Nonempty arrays required")
    n, d, h = len(keys), len(keys[0]), len(values[0])
    if len(values) != n or any(len(k)!=d for k in keys) or any(len(v)!=h for v in values):
        raise ValueError("Invalid dimensions")
    if not groups or any(not g for g in groups) or sorted(i for g in groups for i in g) != list(range(n)):
        raise ValueError("Groups must be an exact visible-key partition")
    r = d if rank is None else int(rank)
    if not 0 <= r <= d:
        raise ValueError("Invalid rank")
    K = [[F(x) for x in row] for row in keys]
    V = [[F(x) for x in row] for row in values]
    out = []
    for ids in groups:
        nb = len(ids)
        mu = [sum((K[i][a] for i in ids),F(0))/nb for a in range(d)]
        nu = [sum((V[i][j] for i in ids),F(0))/nb for j in range(h)]
        dk = [[K[i][a]-mu[a] for a in range(d)] for i in ids]
        dv = [[V[i][j]-nu[j] for j in range(h)] for i in ids]
        cov = [[sum((z[a]*z[b] for z in dk),F(0))/nb for b in range(r)] for a in range(r)]
        cross = [[sum((z[a]*v[j] for z,v in zip(dk,dv)),F(0))/nb for j in range(h)] for a in range(r)]
        diagonal = [[F(0) for _ in range(h)] for _ in range(r)]
        eta = []
        for j in range(h):
            H = [[sum((z[a]*z[b]*v[j] for z,v in zip(dk,dv)),F(0))/nb
                  for b in range(r)] for a in range(r)]
            if keep_diagonal:
                for a in range(r):
                    diagonal[a][j], H[a][a] = H[a][a], F(0)
            eta.append(max((sum((abs(x) for x in row),F(0)) for row in H),default=F(0)))
        kr = [max(abs(z[a]) for z in dk) for a in range(d)]
        vr = [max(abs(v[j]) for v in dv) for j in range(h)]
        out.append(Summary(list(ids),mu,nu,cov,cross,diagonal,eta,kr,vr,r))
    return out


@dataclass
class Envelope:
    mass: Interval
    central: list[Interval]
    center: list[F]


def evaluate(q, summaries, bits=80, sharp=True):
    q = [F(x) for x in q]
    if not summaries or len(q) != len(summaries[0].mu):
        raise ValueError("Invalid query")
    shift = max(dot(q,s.mu) for s in summaries)
    envelopes = []
    for s in summaries:
        r = s.rank
        u = q[:r]
        w = exp_interval(dot(q,s.mu)-shift,bits).scale(F(len(s.ids)))
        variance = sum((u[a]*u[b]*s.cov[a][b] for a in range(r) for b in range(r)),F(0))
        rho, eps = dot([abs(x) for x in u],s.kr[:r]), dot([abs(x) for x in q[r:]],s.kr[r:])
        assert variance >= 0
        if rho == 0:
            tau = F(0)
        elif sharp:
            tau = (exp_interval(rho,bits).hi-1-rho-rho*rho/2)*variance/(rho*rho)
        else:
            tau = exp_interval(rho,bits).hi*rho*variance/6
        A = 1+variance/2
        projected = w*Interval(max(F(1),A-tau),A+tau)
        ex = exp_interval(eps,bits).hi
        mass = Interval(projected.lo/ex, projected.hi*ex)
        central = []
        for j in range(len(s.nu)):
            c = sum((u[a]*s.cross[a][j]+u[a]*u[a]*s.diagonal[a][j]/2 for a in range(r)),F(0))
            beta = s.eta[j]*dot(u,u)/2 + tau*s.vr[j]
            m = w*Interval(c-beta,c+beta)
            discarded = (ex-1)*projected.hi*s.vr[j]
            central.append(Interval(m.lo-discarded,m.hi+discarded))
        envelopes.append(Envelope(mass,central,s.nu))
    return envelopes, shift


def residual(envelopes, boundary, j):
    total = Interval.point(F(0))
    for b in envelopes:
        total = total + b.central[j] + b.mass.scale(b.center[j]-boundary)
    return total


def candidate(envelopes, j):
    den, num = F(0),F(0)
    for b in envelopes:
        z=(b.mass.lo+b.mass.hi)/2
        m=(b.central[j].lo+b.central[j].hi)/2
        den += z; num += b.center[j]*z+m
    return num/den


def screen(envelopes):
    if sum((b.mass.lo for b in envelopes),F(0)) <= 0:
        raise ValueError("Nonpositive denominator lower bound")
    out = []
    for j in range(len(envelopes[0].center)):
        b = bf16_round(candidate(envelopes,j))
        lo,hi = bf16_cell(b)
        good = residual(envelopes,lo,j).lo > 0 and residual(envelopes,hi,j).hi < 0
        out.append(b if good else None)
    return out


def intersect(a: Interval,b: Interval):
    return Interval(max(a.lo,b.lo),min(a.hi,b.hi))


def refine(q, keys, values, summary, old, shift, bits=112):
    """Intersect with a direct rational exponential-interval block scan."""
    mass=Interval.point(F(0)); central=[Interval.point(F(0)) for _ in summary.nu]
    for i in summary.ids:
        e=exp_interval(dot(q,keys[i])-shift,bits)
        mass=mass+e
        for j in range(len(central)):
            central[j]=central[j]+e.scale(F(values[i][j])-summary.nu[j])
    return Envelope(intersect(old.mass,mass),
                    [intersect(a,b) for a,b in zip(old.central,central)],old.center)


def direct_oracle(q,keys,values,bits=128):
    """Independent direct normalized sum; no summaries or moment formulas."""
    shift=max(dot(q,k) for k in keys)
    den=Interval.point(F(0));num=[Interval.point(F(0)) for _ in values[0]]
    for k,v in zip(keys,values):
        e=exp_interval(dot(q,k)-shift,bits)
        den=den+e
        for j,x in enumerate(v):num[j]=num[j]+e.scale(F(x))
    return [Interval.point(F(values[0][j])) if all(v[j]==values[0][j] for v in values)
            else n/den for j,n in enumerate(num)]

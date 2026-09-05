# SPDX-License-Identifier: Apache-2.0
"""Outward binary64 export of exact rational block envelopes.

Each conversion is checked against Fraction.from_float. This is separate from
(the uncertified) numerical moment implementation. It is not extracted from Lean.
"""

import math
from fractions import Fraction as F

from .rational import _validate_envelopes, bf16_cell, bf16_round, candidate


def outward(x, upper: bool) -> float:
    x = F(x)
    try:
        f = float(x)
    except OverflowError as exc:
        raise ValueError("Value outside finite binary64 export domain") from exc
    if not math.isfinite(f):
        raise ValueError("Value outside finite binary64 export domain")
    if (upper and F(f) < x) or (not upper and F(f) > x):
        f = math.nextafter(f, math.inf if upper else -math.inf)
    if not math.isfinite(f):
        raise ValueError("Outward endpoint is not finite")
    assert (F(f) >= x) if upper else (F(f) <= x)
    return f


def export_rows(envelopes):
    """Return box rows [zlo,zhi,mlo,mhi,centerlo,centerhi] and exact cell bounds.

    Value-range coupling is deliberately not part of the GPU wire format.
    A coupled Python acceptance is not a prediction of this box check.
    """
    _validate_envelopes(envelopes)
    rows = []
    for j in range(len(envelopes[0].center)):
        b = bf16_round(candidate(envelopes, j))
        lo, hi = bf16_cell(b)
        # BF16 finite cell boundaries are exactly representable in binary64.
        a, z = float(lo), float(hi)
        if F(a) != lo or F(z) != hi:
            raise ValueError("Cell boundary not exactly representable")
        boxes = [
            [
                outward(e.mass.lo, False),
                outward(e.mass.hi, True),
                outward(e.central[j].lo, False),
                outward(e.central[j].hi, True),
                outward(e.center[j], False),
                outward(e.center[j], True),
            ]
            for e in envelopes
        ]
        rows.append((boxes, a, z))
    return rows

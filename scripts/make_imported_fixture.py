#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import random
import sys
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cmk import rational as r
from cmk.export import export_rows

rng = random.Random(70914)
rows = []
for trial in range(80):
    n, d, h = 12, 3, 2
    scale = 10000 if trial < 40 else 20
    k = [[F(rng.randrange(-16, 17), scale) for _ in range(d)] for _ in range(n)]
    v = [[F(rng.randrange(110, 145), 128) for _ in range(h)] for _ in range(n)]
    if trial in (10, 50):
        v = [[F(257, 256)] * h for _ in range(n)]
    q = [F(rng.randrange(-8, 9), 8) for _ in range(d)]
    s = r.summarize(k, v, [list(range(6)), list(range(6, 12))], rank=trial % 4)
    e, _ = r.evaluate(q, s)
    for j, (boxes, lo, hi) in enumerate(export_rows(e)):
        expected = int(r.residual(e, F(lo), j).lo > 0 and r.residual(e, F(hi), j).hi < 0)
        # Independent attention oracle must be strictly inside every certified cell.
        if expected:
            y = r.direct_oracle(q, k, v)[j]
            assert F(lo) < y.lo <= y.hi < F(hi)
        rows.append((lo, hi, expected, boxes))
path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cmk-imported.txt")
with path.open("w") as f:
    f.write(f"{len(rows)} 2\n")
    for lo, hi, expected, boxes in rows:
        f.write(f"{lo:.17g} {hi:.17g} {expected}\n")
        for b in boxes:
            f.write(" ".join(f"{x:.17g}" for x in b) + "\n")
print(path)

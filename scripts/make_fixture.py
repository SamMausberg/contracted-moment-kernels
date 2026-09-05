#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export a small numerical CPU/CUDA parity fixture, NOT a sound-box certificate."""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from cmk import reference as m


def main(path):
    rng = np.random.default_rng(204901)
    n, B, d, r, h, Q = 128, 4, 8, 3, 4, 12
    groups = np.array_split(np.arange(n), B)
    centers = rng.normal(size=(B, d))
    k = np.empty((n, d))
    v = np.empty((n, h))
    for b, ids in enumerate(groups):
        k[ids] = centers[b] + rng.normal(size=(len(ids), d)) * np.array(
            [0.008] * r + [0.00001] * (d - r)
        )
        v[ids] = 1 + rng.normal(size=(len(ids), h)) * 0.2
    q = rng.normal(size=(Q, d)) / np.sqrt(d)
    s = m.summarize(k, v, groups, r)
    boxes = []
    shifts = []
    ys = []
    gates = []
    for x in q:
        e = m.evaluate(x, s)
        shifts.append(e.shift)
        ys.append(m.dense_attention(x, k, v))
        lo, hi, _ = m.bf16_cells(e.candidate())
        gates.append(e.contains_cell(lo, hi).astype(np.int32))
        boxes.append(
            np.stack(
                [
                    np.repeat(e.zlo[:, None], h, axis=1),
                    np.repeat(e.zhi[:, None], h, axis=1),
                    e.mlo,
                    e.mhi,
                    np.repeat(e.zhat[:, None], h, axis=1),
                    e.mhat,
                ],
                axis=-1,
            )
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<8s6i", b"CMK0001\n", n, B, d, r, h, Q))
        f.write(np.array([0] + [int(g[-1]) + 1 for g in groups], dtype="<i4").tobytes())
        for a in [
            k,
            v,
            s.count,
            s.mu,
            s.nu,
            s.cov,
            s.cross,
            s.diagonal,
            s.eta,
            s.key_radius,
            s.value_radius,
            q,
            shifts,
            boxes,
            ys,
        ]:
            f.write(np.array(a, dtype="<f8").tobytes())
        f.write(np.array(gates, dtype="<i4").tobytes())
    print(path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cmk-fixture.bin")

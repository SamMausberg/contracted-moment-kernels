#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact-rational coupling ablation with an independent 100-digit direct sum.

CPU setup, screening, refinement and fallback costs are reported separately.
No GPU timing or bitwise GPU equivalence is inferred from these measurements.
"""
from fractions import Fraction as F
import json
from pathlib import Path
import platform
import random
import sys
import time

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cmk import rational as r

SEED = 962605


def decimal(x, digits=40):
    return mp.nstr(mp.mpf(x.numerator)/x.denominator,digits)


def run_case(name,K,V,q,groups,rank,spread,trial):
    start = time.perf_counter()
    summaries = r.summarize(K,V,groups,rank=rank)
    setup_ms = 1000*(time.perf_counter()-start)
    start = time.perf_counter()
    env,shift = r.evaluate(q,summaries,bits=80)
    evaluation_ms = 1000*(time.perf_counter()-start)
    start = time.perf_counter()
    oracle = r.direct_oracle(q,K,V,bits=144)
    oracle_ms = 1000*(time.perf_counter()-start)
    weights = [mp.exp(sum(mp.mpf(a.numerator)/a.denominator * (mp.mpf(b.numerator)/b.denominator)
                         for a,b in zip(q,k))) for k in K]
    dense = [sum(w*(mp.mpf(v[j].numerator)/v[j].denominator) for w,v in zip(weights,V))/sum(weights)
             for j in range(len(V[0]))]
    rows = []
    elapsed_refinement = 0.0
    for stage in range(len(groups)+1):
        start = time.perf_counter(); box_gate = r.screen(env)
        box_ms = 1000*(time.perf_counter()-start)
        start = time.perf_counter(); coupled_gate = r.screen(env,coupled=True)
        coupled_ms = 1000*(time.perf_counter()-start)
        coordinates = []
        for j,actual in enumerate(dense):
            b = r.bf16_round(r.candidate(env,j)); lo,hi = r.bf16_cell(b)
            box_l,box_u = r.residual(env,lo,j),r.residual(env,hi,j)
            cp_l,cp_u = r.residual(env,lo,j,coupled=True),r.residual(env,hi,j,coupled=True)
            assert box_l.lo <= cp_l.lo <= cp_l.hi <= box_l.hi
            assert box_u.lo <= cp_u.lo <= cp_u.hi <= box_u.hi
            tolerance = mp.mpf("1e-95")*(1+abs(actual))
            oracle_covered = mp.mpf(decimal(oracle[j].lo,100))-tolerance <= actual <= mp.mpf(decimal(oracle[j].hi,100))+tolerance
            # Decimal re-rendering is not used to certify a gate. The actual
            # gate comparison below uses exact rational oracle endpoints.
            for c in (box_gate[j],coupled_gate[j]):
                if c is not None:
                    cell_l,cell_u = r.bf16_cell(c)
                    assert cell_l < oracle[j].lo <= oracle[j].hi < cell_u
            assert box_gate[j] is None or coupled_gate[j] == box_gate[j]
            actual_res_lo = sum(weights)*(actual-mp.mpf(lo.numerator)/lo.denominator)/mp.exp(mp.mpf(shift.numerator)/shift.denominator)
            actual_res_hi = sum(weights)*(actual-mp.mpf(hi.numerator)/hi.denominator)/mp.exp(mp.mpf(shift.numerator)/shift.denominator)
            residual_tolerance = mp.mpf("1e-95")*(1+sum(weights)/mp.exp(mp.mpf(shift.numerator)/shift.denominator))
            contained = (mp.mpf(decimal(cp_l.lo,100))-residual_tolerance <= actual_res_lo <= mp.mpf(decimal(cp_l.hi,100))+residual_tolerance and
                         mp.mpf(decimal(cp_u.lo,100))-residual_tolerance <= actual_res_hi <= mp.mpf(decimal(cp_u.hi,100))+residual_tolerance)
            assert contained and oracle_covered
            coordinates.append(dict(coordinate=j,candidate=str(b),cell=[str(lo),str(hi)],
                oracle_interval=[decimal(oracle[j].lo,60),decimal(oracle[j].hi,60)],
                oracle_width=decimal(oracle[j].hi-oracle[j].lo),dense_100_digits=mp.nstr(actual,100),
                high_precision_covered=contained,oracle_covers_high_precision=oracle_covered,
                high_precision_absolute_tolerance=mp.nstr(tolerance,10),residual_absolute_tolerance=mp.nstr(residual_tolerance,10),
                box_accepted=box_gate[j] is not None,coupled_accepted=coupled_gate[j] is not None,
                global_value_hull_accepts=lo < min(v[j] for v in V) <= max(v[j] for v in V) < hi,
                box_residual_lower=[decimal(box_l.lo),decimal(box_l.hi)],
                coupled_residual_lower=[decimal(cp_l.lo),decimal(cp_l.hi)],
                box_residual_upper=[decimal(box_u.lo),decimal(box_u.hi)],
                coupled_residual_upper=[decimal(cp_u.lo),decimal(cp_u.hi)],
                width_box=float(box_l.hi-box_l.lo+box_u.hi-box_u.lo),
                width_coupled=float(cp_l.hi-cp_l.lo+cp_u.hi-cp_u.lo)))
        rows.append(dict(refined_blocks=stage,refined_tokens=sum(len(g) for g in groups[:stage]),
                         cumulative_refinement_ms=elapsed_refinement,box_screen_ms=box_ms,
                         coupled_screen_ms=coupled_ms,coordinates=coordinates))
        if stage < len(groups):
            start = time.perf_counter()
            env[stage] = r.refine(q,K,V,summaries[stage],env[stage],shift,bits=112)
            elapsed_refinement += 1000*(time.perf_counter()-start)
    # Direct rational fallback is actually executed for initial gate failures;
    # this separate timing retains the expensive fallback and excludes oracle
    # verification work from the pipeline accounting.
    pipelines = {}
    for mode in ("box","coupled"):
        initial = rows[0]
        accepted = all(c[mode+"_accepted"] for c in initial["coordinates"])
        fallback_ms = 0.0
        fallback_roundings = []
        if not accepted:
            start = time.perf_counter()
            fallback = r.direct_oracle(q,K,V,bits=144)
            fallback_roundings = [None if r.rounded_interval(x) is None else str(r.rounded_interval(x)) for x in fallback]
            fallback_ms = 1000*(time.perf_counter()-start)
        pipelines[mode] = dict(initial_full_acceptance=accepted,full_fallback_executed=not accepted,
            fallback_ms=fallback_ms,fallback_roundings=fallback_roundings,
            setup_evaluation_screen_fallback_ms=setup_ms+evaluation_ms+initial[mode+"_screen_ms"]+fallback_ms)
    return dict(case=name,trial=trial,rank=rank,spread=str(spread),n=len(K),d=len(q),h=len(V[0]),
                inputs=dict(keys=[[str(x) for x in row] for row in K],values=[[str(x) for x in row] for row in V],
                            query=[str(x) for x in q],groups=groups),
                setup_ms=setup_ms,evaluation_ms=evaluation_ms,oracle_verification_ms=oracle_ms,
                pipelines=pipelines,stages=rows)


def suite():
    rng = random.Random(SEED)
    data = []
    with mp.workdps(110):
        for spread in (F(1,64),F(1,4),F(1),F(2)):
            for rank in range(4):
                for trial in range(3):
                    K = [[spread*F(rng.randrange(-16,17),16) for _ in range(3)] for _ in range(8)]
                    q = [F(rng.randrange(-4,5),8) for _ in range(3)]
                    if not any(q):q[0] = F(1)
                    V = [[F(1)+F(rng.randrange(-8,9),16384),
                          F(1)+F(rng.randrange(-8,9),16),
                          F(257,256)] for _ in range(8)]
                    data.append(run_case("narrow_broad_and_exact_midpoint",K,V,q,[list(range(4)),list(range(4,8))],rank,spread,trial))
        for name,V in (("strict_improvement_witness",[[F(1)-F(1,2560)],[F(1)+F(1,2560)]]),
                       ("constant_no_effect",[[F(1)],[F(1)]]),
                       ("exact_midpoint_refusal",[[F(257,256)],[F(257,256)]]),
                       ("broad_negative_control",[[F(-1)],[F(3)]])):
            data.append(run_case(name,[[F(-2)],[F(2)]],V,[F(1)],[[0,1]],0,F(2),0))
        data.append(run_case("suppressed_outlier_witness",[[F(-2)],[F(2)],[F(-16)]],
                             [[F(1)-F(1,2560)],[F(1)+F(1,2560)],[F(3)]],
                             [F(1)],[[0,1],[2]],0,F(2),0))
    coordinates = [c for row in data for c in row["stages"][0]["coordinates"]]
    summary = dict(cases=len(data),initial_coordinates=len(coordinates),
        box_accepted=sum(c["box_accepted"] for c in coordinates),
        coupled_accepted=sum(c["coupled_accepted"] for c in coordinates),
        global_value_hull_accepted=sum(c["global_value_hull_accepts"] for c in coordinates),
        strict_width_improvements=sum(c["width_coupled"] < c["width_box"] for c in coordinates),
        no_width_change=sum(c["width_coupled"] == c["width_box"] for c in coordinates),
        gains_beyond_global_value_hull=sum(c["coupled_accepted"] and not c["box_accepted"] and not c["global_value_hull_accepts"] for c in coordinates),
        false_certifications=0,failed_high_precision_coverage=0,
        notes="Coverage asserted for every saved case/stage. This does not prove all inputs. Midpoint refusals and broad failures retained.")
    geometry = dict(mass=["1","4"],central=["1/2","3/2"],centered_values=["-1/2","1/2"],
        coefficient="-1/4",box_residual=["-1/2","5/4"],range_only_residual=["-3","1"],
        independent_interval_intersection=["-1/2","1"],coupled_polygon_residual=["-1/2","3/4"],
        scope="Exact feasible-metadata geometry witness, not a generated attention instance")
    return dict(schema_version=1,seed=SEED,status="synthetic CPU exact-rational research; not GPU performance or bitwise GPU equivalence",
        arithmetic="Fractions for certificates/oracle intervals; mpmath 110 decimal digits with recorded 1e-95 scaled comparison tolerance for independent dense check; stored interval decimals are display approximations",
        environment=dict(python=sys.version,platform=platform.platform(),mpmath=mp.__version__),summary=summary,
        polygon_witness=geometry,cases=data)


if __name__ == "__main__":
    result = suite()
    output = ROOT/"results"/"certification"/"coupling.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["summary"],indent=2))
    print(output)

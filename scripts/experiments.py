#!/usr/bin/env python3
"""Reproducible CPU ablations. All fast gates are numerical, not verified."""
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cmk.reference import summarize, evaluate, adaptive, bf16_cells, dense_attention, tail_coefficient
from cmk import rational as rat
from fractions import Fraction as F


def timing(fn, repeats=5):
    fn()
    times=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); times.append(time.perf_counter()-t)
    return 1000*statistics.median(times)


def suite():
    rng=np.random.default_rng(100406)
    n,d,h,B,Q=8192,16,8,16,24
    groups=np.array_split(np.arange(n),B)
    qset=rng.normal(size=(Q,d))/np.sqrt(d)
    results=[]
    for name,active,discarded,mixed in [
        ('tight',0.003,0.000001,False),
        ('mixed_one_broad_block',0.003,0.000001,True),
        ('broad_negative_control',0.3,0.3,False),
    ]:
        centers=rng.normal(size=(B,d))*0.3
        k=np.concatenate([centers[b]+rng.normal(size=(len(g),d))*
                          np.r_[np.full(4,active),np.full(d-4,discarded)]
                          for b,g in enumerate(groups)])
        if mixed:
            k[groups[3]] += rng.normal(size=(len(groups[3]),d))*0.3
        v=rng.normal(size=(n,h))*0.3+1.0
        dense=np.array([dense_attention(q,k,v) for q in qset])
        dense_ms=timing(lambda:[dense_attention(q,k,v) for q in qset])/Q
        for r,diag,sharp in [(d,False,False),(d,True,True),(4,True,True)]:
            start=time.perf_counter(); s=summarize(k,v,groups,rank=r,keep_diagonal=diag)
            setup_ms=1000*(time.perf_counter()-start)
            passes=0; coords=0; ratios=[]
            for qi,q in enumerate(qset):
                e=evaluate(q,s,sharp=sharp)
                low,up,out=bf16_cells(e.candidate())
                gate=e.contains_cell(low,up)
                assert np.all(~gate|((dense[qi]>low)&(dense[qi]<up)))
                passes+=int(gate.all());coords+=int(gate.sum())
            eval_ms=timing(lambda:[evaluate(q,s,sharp=sharp) for q in qset])/Q
            # Same actual screening function with a complete numerical fallback.
            def screened():
                for q in qset:
                    e=evaluate(q,s,sharp=sharp)
                    low,up,_=bf16_cells(e.candidate())
                    if not e.contains_cell(low,up).all():
                        dense_attention(q,k,v)
            screen_ms=timing(screened)/Q
            row=dict(case=name,rank=r,keep_diagonal=diag,sharp_tail=sharp,
                     full_gate_passes=passes,coordinate_gate_passes=coords,
                     queries=Q,coordinates=Q*h,summary_bytes=s.nbytes,
                     kv_bytes=k.nbytes+v.nbytes,setup_ms=setup_ms,
                     dense_ms_per_query=dense_ms,evaluation_ms_per_query=eval_ms,
                     screen_and_full_fallback_ms_per_query=screen_ms,
                     query_speed_ratio=dense_ms/screen_ms,
                     batch_speed_ratio_including_setup=(Q*dense_ms)/(setup_ms+Q*screen_ms))
            if diag and sharp:
                scanned=[]; accepted=0
                start=time.perf_counter()
                for qi,q in enumerate(qset):
                    a=adaptive(q,s,k,v)
                    low,up,_=bf16_cells(a['candidate'])
                    assert np.all(~a['accepted']|((dense[qi]>low)&(dense[qi]<up)))
                    scanned.append(a['scanned_tokens']);accepted+=int(a['accepted'].all())
                row.update(adaptive_full_gate_passes=accepted,
                           adaptive_scanned_tokens=scanned,
                           adaptive_mean_scanned_fraction=float(np.mean(scanned)/n),
                           adaptive_ms_per_query=1000*(time.perf_counter()-start)/Q)
            results.append(row)
    # An exact rational arithmetic certificate example, independent of CPU timing.
    k=[[F(i,1000),F((-1)**i,100000)] for i in range(12)]
    v=[[F(1)+F(i-6,100), F(3,2)+F((i*7)%11-5,100)] for i in range(12)]
    groups=[list(range(0,6)),list(range(6,12))]
    s=rat.summarize(k,v,groups,rank=1)
    counts={'coordinates':0,'initial_certified':0,'after_refinement_certified':0,'false_certifications':0}
    for i in range(12):
        q=[F(i-6,10),F(1,10)]
        env,shift=rat.evaluate(q,s)
        initial=rat.screen(env)
        oracle=rat.direct_oracle(q,k,v)
        for j,c in enumerate(initial):
            counts['coordinates']+=1
            if c is not None:
                counts['initial_certified']+=1
                lo,hi=rat.bf16_cell(c)
                assert lo<oracle[j].lo<=oracle[j].hi<hi
        for b in range(2):
            env[b]=rat.refine(q,k,v,s[b],env[b],shift)
        after=rat.screen(env)
        counts['after_refinement_certified']+=sum(c is not None for c in after)
    return dict(status='synthetic_CPU_only; fast gates are not roundoff certified',
                seed=100406,environment=dict(python=sys.version,numpy=np.__version__,
                platform=platform.platform(),blas_threads=os.environ.get('OPENBLAS_NUM_THREADS')),
                dimensions=dict(N=n,d=d,h=h,B=B,Q=Q),ablations=results,rational_example=counts,
                tail_old_over_new={str(x):tail_coefficient(x,False)/tail_coefficient(x,True)
                                   for x in [0.1,1,2,4,8]})

if __name__=='__main__':
    result=suite()
    path=ROOT/'results'/'experiments.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(path)

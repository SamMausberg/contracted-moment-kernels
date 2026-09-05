# SPDX-License-Identifier: Apache-2.0
from fractions import Fraction as F
import random
from cmk import rational as r


def test_bf16_boundaries():
    for b in [F(0),F(1),F(-1),F(2),F(-2),F(3,2),r.pow2(-126),r.pow2(-133)]:
        lo,hi=r.bf16_cell(b)
        assert lo<b<hi
        assert r.bf16_round((lo+b)/2)==b
        assert r.bf16_round((hi+b)/2)==b
    assert r.bf16_round(F(257,256))==1
    assert r.bf16_round(F(259,256))==F(130,128)


def test_rational_envelopes_against_independent_oracle():
    rng=random.Random(926041)
    for trial in range(30):
        n,d,h=8,3,2;rank=trial%4
        K=[[F(rng.randrange(-32,33),256) for _ in range(d)] for _ in range(n)]
        V=[[F(rng.randrange(64,192),128) for _ in range(h)] for _ in range(n)]
        q=[F(rng.randrange(-4,5),8) for _ in range(d)]
        groups=[list(range(4)),list(range(4,8))]
        ss=r.summarize(K,V,groups,rank)
        env,shift=r.evaluate(q,ss,bits=72)
        oracle=r.direct_oracle(q,K,V,bits=112)
        for b,s in enumerate(ss):
            fine=r.refine(q,K,V,s,env[b],shift,bits=112)
            assert env[b].mass.lo<=fine.mass.lo<=fine.mass.hi<=env[b].mass.hi
            for a,c in zip(env[b].central,fine.central):
                assert a.lo<=c.lo<=c.hi<=a.hi
        for j,b in enumerate(r.screen(env)):
            if b is not None:
                lo,hi=r.bf16_cell(b)
                assert lo<oracle[j].lo<=oracle[j].hi<hi
        # Monotone intersection refinement, including the discarded coordinates.
        for b,s in enumerate(ss):
            env[b]=r.refine(q,K,V,s,env[b],shift,bits=112)
        for j,b in enumerate(r.screen(env)):
            if b is not None:
                assert r.rounded_interval(oracle[j])==b


def test_midpoint_refusal():
    K=[[F(-1,8)],[F(1,8)]];V=[[F(257,256)],[F(257,256)]];q=[F(1)]
    s=r.summarize(K,V,[[0,1]])
    env,_=r.evaluate(q,s)
    assert r.screen(env)==[None]
    assert r.direct_oracle(q,K,V)[0]==r.Interval.point(F(257,256))


def test_exact_constant_channel_and_rank_zero():
    K=[[F(0),F(0)],[F(1,8),F(-1,8)]];V=[[F(1)],[F(1)]]
    s=r.summarize(K,V,[[0,1]],rank=0)
    env,_=r.evaluate([F(1),F(1)],s)
    assert r.screen(env)==[F(1)]

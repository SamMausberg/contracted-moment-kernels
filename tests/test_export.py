# SPDX-License-Identifier: Apache-2.0
from fractions import Fraction as F
import random
import pytest
from cmk.export import outward

def test_outward_export():
 rng=random.Random(118141)
 for _ in range(1000):
  x=F(rng.randrange(-1000000,1000001),rng.randrange(1,100000))*F(2)**rng.randrange(-1000,950)
  lo,hi=outward(x,False),outward(x,True)
  assert F(lo)<=x<=F(hi)
 for x in [F(0),F(1),F(-1),F(1,2**1100),-F(1,2**1100)]:
  assert F(outward(x,False))<=x<=F(outward(x,True))
 with pytest.raises(ValueError):outward(F(2)**1100,True)

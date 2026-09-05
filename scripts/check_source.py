#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static source consistency only. This does NOT run or validate Lean."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
files=list((ROOT/'lean'/'CMK').glob('*.lean'))
names=[]
for p in files:
    s=p.read_text()
    assert not re.search(r'\b(sorry|admit|axiom|unsafe|native_decide)\b',s),p
    names+=re.findall(r'^theorem\s+(\w+)',s,re.M)
audit=(ROOT/'lean'/'CMK'/'Audit.lean').read_text()
audited=re.findall(r'^#print axioms CMK\.(\w+)',audit,re.M)
assert len(names)==25 and len(set(names))==25
assert set(names)==set(audited) and len(audited)==25
for p in [ROOT/'README.md',ROOT/'paper'/'PAPER.md',*list((ROOT/'docs').glob('*.md'))]:
    assert '\u2014' not in p.read_text(),f'Unexpected em dash in {p}'
print('Static source checks passed: 25 theorem scripts and 25 audit entries. Lean NOT executed.')

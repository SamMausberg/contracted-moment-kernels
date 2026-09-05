#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../lean"
if ! command -v lake >/dev/null && [[ -x "$HOME/.elan/bin/lake" ]]; then
  export PATH="$HOME/.elan/bin:$PATH"
fi
command -v lake >/dev/null || { echo 'Lean/Lake is required; no verification performed.' >&2; exit 127; }
mkdir -p ../results
python3 ../scripts/check_source.py --lean-only
if [[ ! -f lake-manifest.json ]]; then
  lake update
fi
lake exe cache get
{
  date -u '+Build date: %Y-%m-%dT%H:%M:%SZ'
  lean --version
  lake build
} 2>&1 | tee ../results/lean-build.log
lake env lean CMK/Audit.lean | tee ../results/lean-axioms.log
python3 ../scripts/check_source.py --lean-only --audit-log ../results/lean-axioms.log --write-manifest ../results/lean-verification.json

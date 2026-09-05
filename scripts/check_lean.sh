#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../lean"
command -v lake >/dev/null || { echo 'Lean/Lake is required; no verification performed.' >&2; exit 127; }
# Reject proof escape hatches in our source. Comments are deliberately included.
if grep -RnwE --include='*.lean' '(sorry|admit|axiom|unsafe|native_decide)' CMK CMK.lean; then
  echo 'Rejected proof source escape hatch.' >&2; exit 1
fi
lake update
lake exe cache get
lake build
lake env lean CMK/Audit.lean | tee ../results/lean-axioms.log
# Standard foundational axioms such as propext/choice/quotient soundness are not holes.
if grep -E 'sorryAx|ofReduceBool|Lean\.ofReduceBool' ../results/lean-axioms.log; then
  echo 'Rejected non-kernel proof dependency.' >&2; exit 1
fi

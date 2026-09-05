#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Explicitly run by the repository owner; not executed during authoring.
set -euo pipefail
cd "$(dirname "$0")/.."
visibility=private
if [[ ${1:-} == --public ]]; then visibility=public; shift; fi
if (($#)); then echo 'Usage: bash scripts/publish.sh [--public]' >&2; exit 2; fi
command -v gh >/dev/null || { echo 'GitHub CLI is required.' >&2; exit 127; }
command -v git >/dev/null || { echo 'Git is required.' >&2; exit 127; }
gh auth status >/dev/null
owner=$(gh api user --jq .login)
if [[ $owner != SamMausberg ]]; then
  echo "Authenticated as $owner, not the intended owner SamMausberg; refusing to publish." >&2
  exit 1
fi
name=contracted-moment-kernels
# Never initialize inside a parent repository or attach this to an unrelated repo.
if [[ ! -d .git ]]; then
  if git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo 'An enclosing repository exists; move this directory outside it before publishing.' >&2
    exit 1
  fi
  git init -b main
  git add .
  git -c user.name='Sam Mausberg' \
      -c user.email='210202706+SamMausberg@users.noreply.github.com' \
      commit -m 'Initial research prototype; Lean and CUDA explicitly unverified'
fi
[[ $(git rev-parse --show-toplevel) == "$PWD" ]] || { echo 'Repository root mismatch.' >&2; exit 1; }
if git remote get-url origin >/dev/null 2>&1; then
  echo 'An origin remote already exists. Inspect it rather than overwriting it.' >&2; exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
  echo 'Commit or remove local changes before publishing; no edits will be silently omitted.' >&2; exit 1
fi
# gh repo create fails rather than overwrites if this name is already in use.
gh repo create "$owner/$name" "--$visibility" --source=. --remote=origin --push \
  --description 'Boundary certificates and moment summaries. Research prototype; verification incomplete.'

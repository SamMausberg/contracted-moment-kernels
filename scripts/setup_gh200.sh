#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-gh200.txt -e '.[test,research,traces]'
.venv/bin/python - <<'PY'
import platform
import numpy
import torch

if not torch.cuda.is_available():
    raise SystemExit('A CUDA-enabled PyTorch installation is required for GPU experiments.')
print({
    'platform': platform.platform(), 'numpy': numpy.__version__,
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0),
    'capability': torch.cuda.get_device_capability(0),
})
PY
printf '%s\n' 'Activate with: source .venv/bin/activate' 'Lean setup and audit: bash scripts/check_lean.sh'

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/.."
command -v latexmk >/dev/null
command -v pdflatex >/dev/null
command -v bibtex >/dev/null
mkdir -p build/paper
if ! (
  cd paper
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
    -outdir=../build/paper PAPER.tex > ../build/paper/latexmk.stdout 2>&1
); then
  tail -80 build/paper/latexmk.stdout >&2
  exit 1
fi
cp build/paper/PAPER.pdf paper/PAPER.pdf
python3 scripts/check_paper.py --output results/paper-layout.json
python3 - <<'PY'
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

root = Path('paper')
paths = [root / 'README.md']
for pattern in ('*.tex', '*.bib', '*.bst', '*.sty', 'sections/*.tex',
                'diagrams/*.tex', 'vendor/*'):
    paths.extend(root.glob(pattern))
manifest = json.loads(Path('results/paper-layout.json').read_text())
paths.extend(Path(path) for path in manifest['vector_figure_inputs'])
with ZipFile(root / 'latex-source.zip', 'w', compression=ZIP_DEFLATED) as archive:
    for path in sorted(set(paths)):
        info = ZipInfo(str(path.relative_to(root)), date_time=(2026, 9, 5, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, path.read_bytes())
print('Built paper/PAPER.pdf and paper/latex-source.zip from native LaTeX.')
PY

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/.."
command -v pandoc >/dev/null
command -v xelatex >/dev/null
mkdir -p build/paper
pandoc paper/PAPER.md --from=markdown+tex_math_dollars-implicit_figures --standalone \
  --resource-path=paper --lua-filter=scripts/svg_to_pdf.lua \
  --pdf-engine=xelatex -V documentclass=article -V fontsize=10pt \
  -V geometry:margin=0.8in -V colorlinks=true -V urlcolor=blue \
  -V mainfont='DejaVu Serif' -V sansfont='DejaVu Sans' \
  -V monofont='DejaVu Sans Mono' --highlight-style=tango \
  -o paper/PAPER.pdf
pandoc paper/PAPER.md --from=markdown+tex_math_dollars --standalone --mathjax \
  --resource-path=paper --metadata title='When moment summaries can certify attention' \
  -o paper/PAPER.html
printf '%s\n' 'Built paper/PAPER.pdf and paper/PAPER.html'

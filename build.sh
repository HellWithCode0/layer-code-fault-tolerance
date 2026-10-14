#!/usr/bin/env bash
# Build manuscript.pdf and supplement.pdf, then the linked and combined copies.
set -euo pipefail
cd "$(dirname "$0")"
python3 build_bibliography.py
for name in supplement supplement supplement manuscript manuscript manuscript; do
  pdflatex -interaction=nonstopmode -halt-on-error "$name.tex" > "build_$name.log" || { echo "PDF build failed: $name"; exit 1; }
done
python3 finalize_pdf_links.py

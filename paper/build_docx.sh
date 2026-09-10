#!/usr/bin/env bash
# Convert paper.md → paper.docx using the Springer LNCS reference template.
#
# Requires: pandoc 3.x (pre-installed at ~/.local/bin/pandoc)
# Reference doc: ../guidelines.docm (LNCS Springer template, copied as .docx for pandoc)
# Bibliography: references.bib (BibTeX), Springer LNCS CSL.

set -euo pipefail

cd "$(dirname "$0")"

PANDOC="${PANDOC:-$HOME/.local/bin/pandoc}"
if [[ ! -x "$PANDOC" ]]; then
  PANDOC="$(command -v pandoc || true)"
fi
if [[ -z "$PANDOC" ]]; then
  echo "pandoc not found. Install via apt or download from https://github.com/jgm/pandoc/releases."
  exit 1
fi

CSL_URL="https://www.zotero.org/styles/springer-lecture-notes-in-computer-science"
CSL_FILE="lncs.csl"
if [[ ! -f "$CSL_FILE" ]]; then
  echo "Downloading LNCS CSL..."
  curl -sSLo "$CSL_FILE" "$CSL_URL"
fi

REFDOC="../guidelines.docm"
REFDOC_TMP=""
# Pandoc needs the reference doc to be a real .docx (not .docm with macros).
# Copy guidelines.docm to a temp .docx for the duration of this build.
if [[ -f "$REFDOC" ]]; then
  REFDOC_TMP="$(mktemp --suffix=.docx)"
  cp "$REFDOC" "$REFDOC_TMP"
  REFDOC="$REFDOC_TMP"
fi

"$PANDOC" paper.md \
  -o paper.docx \
  --reference-doc="$REFDOC" \
  --bibliography=references.bib \
  --citeproc \
  --csl="$CSL_FILE" \
  --resource-path=.:figures \
  --metadata link-citations=true

if [[ -n "$REFDOC_TMP" ]]; then
  rm -f "$REFDOC_TMP"
fi

echo "Wrote paper.docx ($(stat -c%s paper.docx) bytes)"

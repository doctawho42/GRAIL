#!/usr/bin/env bash
# Build both documents in the order the cross-references need: the SI first, so its .aux exists
# when the manuscript's \externaldocument reads it. A pointer into the SI that cannot resolve
# prints ?? and is caught by the undefined-reference count, which is the property a hand-typed
# table number does not have.
set -uo pipefail
cd "$(dirname "$0")/../paper2"
for pass in 1 2; do
  pdflatex -interaction=nonstopmode si          >/dev/null 2>&1
done
bibtex si >/dev/null 2>&1
pdflatex -interaction=nonstopmode si            >/dev/null 2>&1
pdflatex -interaction=nonstopmode si            >/dev/null 2>&1
for pass in 1 2; do
  pdflatex -interaction=nonstopmode grail_jcim  >/dev/null 2>&1
done
bibtex grail_jcim >/dev/null 2>&1
pdflatex -interaction=nonstopmode grail_jcim    >/dev/null 2>&1
pdflatex -interaction=nonstopmode grail_jcim    >/dev/null 2>&1

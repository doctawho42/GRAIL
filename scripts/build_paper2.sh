#!/usr/bin/env bash
# Build both documents in the order the cross-references need: the SI first, so its .aux exists
# when the manuscript's \externaldocument reads it. A pointer into the SI that cannot resolve
# prints ?? and is caught by the undefined-reference count, which is the property a hand-typed
# table number does not have.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/paper2"
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
# The references run both ways now, so each document has to see the other's finished numbering.
# One more round of each resolves the pointers this document makes into the other.
pdflatex -interaction=nonstopmode si            >/dev/null 2>&1
pdflatex -interaction=nonstopmode si            >/dev/null 2>&1
pdflatex -interaction=nonstopmode grail_jcim    >/dev/null 2>&1
pdflatex -interaction=nonstopmode grail_jcim    >/dev/null 2>&1

# The plain-text renderings are tracked so a reader can grep either document without a viewer.
# They are produced here rather than by hand. A tracked copy that nothing regenerates is a copy
# that goes stale silently, and both of these already had: they held prose the manuscript no
# longer contains, and no check could see it because no check knew they existed.
pdftotext grail_jcim.pdf ms.txt
pdftotext si.pdf si.txt

# Record what these logs describe, by content rather than by modification time: a generator that
# rewrites a file with identical bytes must not read as an edit.
python "${ROOT}/scripts/check_paper2_build.py" --stamp

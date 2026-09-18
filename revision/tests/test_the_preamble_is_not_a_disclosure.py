"""A package list and an author block are not prose, and must not become a concession key.

Run: python -m pytest revision/tests/test_the_preamble_is_not_a_disclosure.py -q

scripts/disclosure_inventory._plain strips LaTeX markup so a sentence survives a reflow, and
_sentences then splits what is left. Neither knows where the document starts. Everything above
\\begin{document} -- \\usepackage lines, the author list, affiliations, e-mail addresses, the title
-- comes through as text with the macro names removed, and since none of it ends in a full stop
followed by a capital, it does not split. It arrives as ONE key.

Measured on paper2/grail_jcim.tex, the wrapper that carries the preamble and the abstract: 4
pseudo-sentences, of which the first is 92 words and 688 characters of package names, author names
and affiliations run together with the opening of the abstract. It is in results/disclosure_-
inventory.json as a recorded disclosure.

The harm is not noise. A recorded key is what --check compares against, so that key makes the author
block load-bearing: correcting an affiliation, adding an author, or adding a package would move it
and the gate would report a disclosure that survives in neither the manuscript nor the Supporting
Information -- a retraction alarm raised by an edit that touched no claim. This repository has the
matching failure already written down, where a renamed heading faked a retraction.

So: nothing above \\begin{document} may reach the inventory, and the abstract's real sentences must
still reach it, because the abstract is prose and carries claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import disclosure_inventory as di  # noqa: E402

WRAPPER = ROOT / "paper2" / "grail_jcim.tex"

PREAMBLE_ONLY = r"""
\documentclass[journal=jcisd8,manuscript=article]{achemso}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\author{Ada Lovelace}
\affiliation{Analytical Engine Group, Somewhere}
\email{ada@example.org}
\title{A title that is not a disclosure}
\begin{document}
\begin{abstract}
The measurement is reported over the population it covers. A second sentence follows it.
\end{abstract}
\end{document}
"""


def test_nothing_above_begin_document_becomes_a_sentence():
    """Planted, so this holds whatever the real wrapper happens to contain today."""
    sents = di._sentences(di._plain(PREAMBLE_ONLY))
    joined = " ".join(sents).lower()
    for leak in ("usepackage", "achemso", "booktabs", "ada lovelace", "analytical engine",
                 "ada@example.org", "documentclass"):
        assert leak not in joined, (
            f"{leak!r} reached the inventory from above \\begin{{document}}. A package list and an "
            f"author block are not prose, and a recorded key built from them turns an affiliation "
            f"edit into a reported retraction.\nGot: {sents}")


def test_the_abstract_still_arrives_whole():
    """The cut must take the preamble and nothing else: the abstract carries claims."""
    sents = di._sentences(di._plain(PREAMBLE_ONLY))
    joined = " ".join(sents)
    assert "The measurement is reported over the population it covers" in joined, sents
    assert "A second sentence follows it" in joined, sents
    assert len(sents) == 2, f"the abstract's two sentences should split into two keys: {sents}"


def test_the_real_wrapper_carries_no_preamble_key():
    """The file the measurement came from, so a regression here is caught on the live document."""
    if not WRAPPER.exists():
        return
    sents = di._sentences(di._plain(WRAPPER.read_text()))
    offenders = [s for s in sents
                 if any(w in s.lower() for w in ("usepackage", "achemso", "documentclass",
                                                 "affiliation", "@"))]
    assert not offenders, (
        f"{WRAPPER.name} still yields {len(offenders)} key(s) built from its preamble, the first "
        f"{len(offenders[0].split())} words long:\n  {offenders[0][:200]}")
    assert sents, f"{WRAPPER.name} yielded no sentences at all, so the cut took the document too"

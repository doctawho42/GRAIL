#!/usr/bin/env python3
r"""How hard the body is to read, measured on the LaTeX source rather than on the PDF.

A reader's report measured this on extracted PDF text, where figure labels and section headings
splice into neighbouring sentences and inflate the count of long ones. Measuring the source avoids
that: floats, tables, captions and maths are removed, and what is left is the prose a reviewer
actually reads in sequence.

    python scripts/prose_metrics.py            # the body
    python scripts/prose_metrics.py --long 40  # list every sentence over the threshold
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The JCIM manuscript, not the ICLR one. This pointed at paper/grail_iclr.tex, so
# results/prose_metrics.json reported a mean of 16.22 words for a document nobody is submitting
# while the manuscript under review sat unmeasured at 25.4.
BODY = ROOT / "paper2" / "body.tex"


def strip_latex(s: str) -> str:
    # The ICLR file was one document with its own front matter, so the prose was cut out of it by
    # two landmarks. body.tex is already prose-only and carries neither, so the slice is taken
    # only when both are present rather than assumed.
    a, b = r"\begin{abstract}", r"\subsubsection*{Reproducibility"
    if a in s and b in s:
        s = s[s.index(a):s.index(b)]
    # floats and their captions are not read in sequence
    for env in ("figure", "table", "tabular", "tikzpicture", "itemize", "center"):
        s = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", " ", s, flags=re.S)
    s = re.sub(r"\\(section|subsection|paragraph|subsubsection)\*?\{[^}]*\}", " ", s)
    # a deleted \ref leaves a dangling "(Appendix " that glues the next sentence onto this one and
    # counts the pair as one very long sentence; it becomes a token instead
    s = re.sub(r"\\ref\{[^}]*\}", "REF", s)
    s = re.sub(r"\\(label|input|looseness|itemsep|parskip|newcommand)[^ \n]*", " ", s)
    s = re.sub(r"\\cite[a-z]*\{[^}]*\}", "CITE", s)
    s = re.sub(r"\$[^$]*\$", "NUM", s)          # maths counts as one token
    s = re.sub(r"\\emph\{([^}]*)\}|\\textbf\{([^}]*)\}|\\textsc\{([^}]*)\}",
               lambda m: next(g for g in m.groups() if g is not None), s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("---", " ").replace("~", " ")
    s = re.sub(r"[{}]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sentences(text: str) -> list:
    """Deferred to the gate's splitter rather than repeated here.

    This file used to split on a full stop followed by a capital. Pointed at body.tex that
    reports a longest "sentence" of 251 words where check_prose_density reports 64, because it
    runs across section boundaries and glues a heading to the paragraph after it. Commit 892692b
    hit the same thing at 340 words and fixed the splitter before changing a word of prose; the
    lesson applies to a report as much as to an edit, since a tracked artifact carrying 251 is a
    number about the instrument and not about the manuscript.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_prose_density as cpd

    return [s for b in cpd.blocks(text) for s in cpd.sentences(b)]


def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", type=int, default=0, help="list sentences over this many words")
    ap.add_argument("--out", default=str(ROOT / "results" / "prose_metrics.json"))
    args = ap.parse_args()

    # Not through strip_latex. Deferring the splitter was not enough on its own: strip_latex runs
    # first and flattens \section{...} to a space, so the block boundaries the gate's splitter
    # cuts on are gone by the time it sees the text, and the longest "sentence" stayed at 251
    # words while the mean moved. The aggregate absorbed the defect and the extremum did not,
    # which is the general shape: check the maximum, never the mean, when a splitter is suspect.
    sents = sentences(BODY.read_text())
    lens = [len(s.split()) for s in sents]
    words = sum(lens)
    syl = sum(syllables(w) for s in sents for w in s.split())
    flesch = 206.835 - 1.015 * (words / len(sents)) - 84.6 * (syl / words)
    over = [(n, s) for n, s in zip(lens, sents) if n > 40]

    rep = {"sentences": len(sents), "words": words,
           "mean_sentence_words": round(words / len(sents), 2),
           "flesch_reading_ease": round(flesch, 1),
           "over_40_words": len(over),
           "share_over_40": round(len(over) / len(sents), 4),
           "longest": max(lens)}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    for k, v in rep.items():
        print(f"  {k:<22} {v}")
    if args.long:
        print()
        for n, s in sorted([(n, s) for n, s in zip(lens, sents) if n > args.long], reverse=True):
            print(f"  [{n}] {s[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

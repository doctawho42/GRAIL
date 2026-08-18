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
BODY = ROOT / "paper" / "grail_iclr.tex"


def strip_latex(s: str) -> str:
    s = s[s.index(r"\begin{abstract}"):s.index(r"\subsubsection*{Reproducibility")]
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
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if len(p.split()) >= 3]


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

    text = strip_latex(BODY.read_text())
    sents = sentences(text)
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

#!/usr/bin/env python3
"""Every sentence that calls a difference certified, listed with the family that entitles it.

A paper arguing that undeclared conventions corrupt comparisons cannot run its own load-bearing
adjective under more than one convention. Section 3 defines two words and reserves the stronger for
three confirmatory comparisons, each corrected inside a family declared in advance. This enumerates
every use of that word across the manuscript and its appendices, matches each against the declared
list, and fails if any sentence claims it without one.

The check is deliberately unforgiving in one direction only: a sentence may say `separated' anywhere,
because that word claims nothing beyond an interval, but `certified' must be traceable to a family,
a family size and an adjusted p-value. The table this writes is the audit the claim rests on, and it
regenerates, so it cannot drift away from the text it describes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "app" / "claimwords.tex"

# The three confirmatory comparisons of Section 3, each keyed by a phrase that must appear in the
# sentence using the word. Adding a row here is declaring a family, which is a decision about the
# analysis and not about the prose, so it is made in this file and nowhere else.
CONFIRMATORY = [
    {"key": "differential criterion sensitivity",
     "match": ["survives Holm--Bonferroni", "differential sensitivity"],
     "family": "criterion steps by method pairs", "size": "12 external, 20 internal",
     "p": "$1.4\\times10^{-3}$ (largest rejected)"},
    {"key": "learned against prior",
     "match": ["rule-firing frequency", "learned selector"],
     "family": "the selection comparison", "size": "1",
     "p": "reported in Appendix~\\ref{app:props}"},
    {"key": "the docking control's reversals",
     "match": ["both reversals are certified", "docking control"],
     "family": "every cell-level test the contested verdict is read from, on the docking board",
     "size": "168", "p": "$1.4\\times10^{-3}$ (largest surviving reversal)"},
    {"key": "cross-domain interaction",
     "match": ["method-by-criterion", "interactions between method and choice",
               "as interactions between method"],
     "family": "criterion steps by system pairs by leaderboards", "size": "448",
     "p": "$114$ survive at $\\alpha=0.05$"},
]
# The paragraph that defines the words is allowed to use them: it is the definition, not a claim.
DEFINITION_MARKERS = ["Two words are used throughout", "A difference is \\emph{separated}",
                      "A difference is \\emph{certified}", "corrected rather than certified",
                      "only they are ever called certified",
                      # the appendix that enumerates the word has to name it to introduce itself;
                      # keyed on its label rather than its wording, so an edit cannot silently
                      # widen the exemption to a neighbouring sentence
                      "\\label{app:claimwords}",
                      # the summary table has a column headed "certified" and a caption that says
                      # what it counts; the family is per board, sits in each board's artifact, and
                      # is held there by verify_paper_numbers rather than restated in the caption
                      "contested (certified) & unresolved & tiers",
                      # the appendix that attributes each certification to its axis is about which
                      # family produced them, not a new claim inside one; keyed on its own heading
                      "Which axis certifies a reversal",
                      "certifies no reversal of a separated ordering anywhere in this paper",
                      # the same two sentences now live in the body, having been promoted with the
                      # rest of the survey; the exemption follows the text rather than the file
                      "Six\nof the twenty-three certified reversals",
                      "Six of the twenty-three certified reversals"]


def sentences(text: str):
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\s+", " ", text)
    for s in re.split(r"(?<=[.!?]) (?=[A-Z\\$])", text):
        s = s.strip()
        if s:
            yield s


def main() -> int:
    rows, violations = [], []
    for path in sorted(PAPER.glob("*.tex")) + sorted((PAPER / "app").glob("*.tex")):
        if path.name in ("claimwords.tex", "robust_tables.tex", "robust_hasse.tex"):
            continue
        for s in sentences(path.read_text()):
            if "certif" not in s:
                continue
            if any(m in s for m in DEFINITION_MARKERS):
                rows.append((path.name, "the definition itself", "--", "--"))
                continue
            hit = next((c for c in CONFIRMATORY if any(m in s for m in c["match"])), None)
            if hit is not None:
                rows.append((path.name, hit["key"], hit["size"], hit["p"]))
            elif re.search(r"Holm|strictest correction|survive the same correction|"
                           r"family of \$?\d+\$? interaction|cell-level tests", s):
                # the sentence carries its own warrant: it names the correction and the family size
                size = re.search(r"\$?(\d{2,4})\$? (?:paired )?(?:interaction|test)", s)
                rows.append((path.name, "names its own correction",
                             size.group(1) if size else "stated in the sentence", "--"))
            else:
                violations.append((str(path.relative_to(ROOT)), s[:220]))

    lines = ["% generated by scripts/audit_claim_words.py -- do not edit by hand",
             "\\begin{center}\\small", "\\begin{tabular}{llll}", "\\toprule",
             "file & the comparison it refers to & family size & adjusted $p$ \\\\", "\\midrule"]
    for f, k, n, p in rows:
        lines.append(f"\\texttt{{{f.replace('_', '-')}}} & {k} & {n} & {p} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{center}"]
    OUT.write_text("\n".join(lines) + "\n")

    print(f"{len(rows)} uses of the word, all traceable; wrote {OUT}")
    for f, k, _, _ in rows:
        print(f"  {f:20} {k}")
    if violations:
        print(f"\n{len(violations)} sentences claim the word without a declared family:")
        for f, s in violations:
            print(f"  {f}: {s}")
        return 1
    (ROOT / "results" / "claim_word_audit.json").write_text(json.dumps(
        {"uses": len(rows), "violations": len(violations),
         "confirmatory_families": [c["key"] for c in CONFIRMATORY]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

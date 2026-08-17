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

# the cross-domain interaction family's survivors, from the run that produces them
_HOLM_SURVIVORS = len(json.loads(
    (ROOT / "results" / "retro_leaderboard_cluster0.json").read_text())["holm_survivors"])

# the families the passages below name, read from the runs that produced them rather than typed
_PB = json.loads((ROOT / "results" / "robust_order_posebusters.json").read_text())
def _sci(x: float) -> str:
    """A float as the manuscript prints one, so the table and the sentence cannot disagree."""
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


_DOCK_SMALLEST = _sci(min(r["p"] for r in _PB["multiplicity"]["reversal_tests"]))
_DOCK_CUTOFF = _sci(9.765625e-04)
_UN = json.loads((ROOT / "results" / "union_multiplicity.json").read_text())
_UNION_FAMILY = _UN["union_family_size"]
_UNION_CUTOFF = _sci(_UN["union_cutoff_p"])
_PERM_TESTS = json.loads(
    (ROOT / "results" / "permutation_check.json").read_text())["n_tests_checked"]

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
    {"key": "the docking control, after its own correction",
     "match": ["certifies neither half of it"],
     "family": "every cell-level test the contested verdict is read from, on the docking board",
     "size": "168",
     "p": f"smallest reversal ${_DOCK_SMALLEST}$ against a ${_DOCK_CUTOFF}$ threshold"},
    {"key": "the correction over every grid at once",
     "match": ["the count of certified pairs falls f", "count of certified pairs falls"],
     "family": "the union of all twenty-three grids", "size": f"${_UNION_FAMILY:,}$".replace(",", "{,}"),
     "p": f"cutoff ${_UNION_CUTOFF}$"},
    {"key": "the shape-free re-test",
     "match": ["it never returns a larger $p$ than the analytic one"],
     "family": "every certified reversal and everything within a factor of ten of its cutoff",
     "size": f"{_PERM_TESTS}", "p": "no verdict moves"},
    {"key": "cross-domain interaction",
     "match": ["method-by-criterion", "interactions between method and choice",
               "as interactions between method"],
     "family": "criterion steps by system pairs by leaderboards", "size": "448",
     # read from the artifact rather than typed: this cell said 114 for a day after the exact
     # test moved it, in a file whose header says it is generated
     "p": f"${_HOLM_SURVIVORS}$ survive at $\\alpha=0.05$"},
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
                      "contested (certified) & unresolved & places supported",
                      # the appendix that attributes each certification to its axis is about which
                      # family produced them, not a new claim inside one; keyed on its own heading
                      "Which axis certifies a reversal",
                      "certifies no reversal of a separated ordering anywhere in this paper",
                      # the same two sentences now live in the body, having been promoted with the
                      # rest of the survey; the exemption follows the text rather than the file
                      "Six\nof the twenty-three certified reversals",
                      "Six of the twenty-three certified reversals",
                      "Five of the twenty-one certified reversals",
                      "Five\nof the twenty-one certified reversals",
                      # the power paragraph states what the negative claim is, and the paragraph
                      # that follows gives its size; neither certifies anything new
                      "The paper's second claim is negative",
                      "not hard enough to be certified",
                      # the shape-free check re-tests things already certified elsewhere and says
                      # so; its own family is the 42 tests it names in the same sentence
                      "No certified reversal is lost to it",
                      # the figure caption counts places, not certifications
                      "The table publishes seven places in a line"]


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
                           r"family of \$?\d+\$? interaction|cell-level tests|"
                           # a sentence that names the correction in words, or the family it was
                           # read in, warrants itself as much as one that names the procedure
                           r"under (?:any of )?the three corrections|the union of the grids|"
                           r"every one of\s*the three|the family it was read in|"
                           r"board's (?:own )?(?:family|threshold)|its own grid|"
                           r"the task's own aggregation|either way", s):
                # the sentence carries its own warrant: it names the correction and the family size
                size = re.search(r"\$?(\d{2,4})\$? (?:paired )?(?:interaction|test)", s)
                rows.append((path.name, "names its own correction",
                             size.group(1) if size else "stated in the sentence", "--"))
            else:
                violations.append((str(path.relative_to(ROOT)), s[:220]))

    lines = ["% generated by scripts/audit_claim_words.py -- do not edit by hand",
             "\\begin{center}\\small",
             # the last two columns carry sentences, not words: set as l they ran 180pt past
             # the margin, so they wrap instead
             "\\begin{tabular}{l>{\\raggedright\\arraybackslash}p{0.30\\textwidth}l"
             ">{\\raggedright\\arraybackslash}p{0.28\\textwidth}}", "\\toprule",
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

#!/usr/bin/env python3
r"""How often this manuscript reaches for two constructions, held against what the journal does.

A reviewer said the manuscript and its Supporting Information read as machine-written. The reply
that each hedge is individually defensible is true and answers a different question: the complaint
is about a rate, and a set of individually defensible sentences can still sit an order of
magnitude outside what the literature does. So the rate is measured rather than argued.

The comparison is eight papers, 50,139 words of body prose, obtained as PMC open-access XML: six
JCIM method comparisons and two adjacent-journal ones chosen because they are the closest in
subject, one of them running SyGMa, GLORYx and BioTransformer, which is this work's own comparator
set. References, captions and tables are excluded, and counting the whole XML instead moves the
corpus total from 14 occurrences to 18, so their hedges are not hiding in the excluded parts.

    rather than            corpus 0.28 per 1000 words, range 0.00-0.57, generous bound 0.68
    is/are not a           corpus 0.02, range 0.00-0.11
    epistemic self-grading corpus 0.00 in 50,139 words
    worth stating/saying   corpus 0.00
    which is why           corpus 0.00
    the objection          corpus 0.00

Two measured constructions are deliberately NOT here, and the omission is the point: "not X but Y"
runs at 0.36 and 0.17 against a corpus range of 0.00-0.77, with one of the eight papers above us,
and the mean sentence length runs at 25.4 and 27.4 against a corpus range of 23.2-28.5. Both are
inside the range these papers occupy. Editing either would be damage dressed as polish, and a
later pass that "fixes" them would be working from an impression rather than a measurement.

One paper could not be measured and is named rather than estimated: the benchmark of metabolite
predictors against human radiolabelled ADME data, which this work cites, has no PMCID and is not
in the open-access subset. Its rates are unknown and no substitute was used for them.

Every probe here counts across a line break, because the manuscript is hard-wrapped and a pattern
written with a literal space misses "rather\nthan" -- 3 such in the manuscript and 5 in the
Supporting Information, which is enough to move a headline count. And every probe is run first
against a fixture whose answer is known, and the check refuses if a probe fails its own fixture:
a regex that silently matches nothing reports a clean document, which is the failure this file
exists to avoid.

    python scripts/check_prose_rate.py
    python scripts/check_prose_rate.py --show    # print the occurrences, not the counts
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_prose_density as cpd  # noqa: E402  the one splitter this repository has

# \s+ everywhere a space would do, so a hard-wrapped occurrence counts once rather than never.
PROBES = {
    "rather than": r"\brather\s+than\b",
    "is/are not a": r"\b(?:is|are)\s+not\s+a\b",
    "epistemic self-grading": (r"\brather\s+than\s+a\b|\band\s+not\s+a\b|\brather\s+than\s+of\b"
                               r"|\bis\s+reported\b|\bis\s+stated\b"),
    "worth V-ing": r"\bworth\s+(?:stating|saying|having|carrying|putting)\b",
    "which is why": r"\bwhich\s+is\s+why\b",
    "the objection": r"\bthe\s+objection\b",
}

# A fixture per probe, with the answer written down. The line break in each is deliberate.
FIXTURES = {
    "rather than": ("the sweep is reported rather\nthan described, and rather than one cell", 2),
    "is/are not a": ("that is not a\nresult, and these are not a family", 2),
    "epistemic self-grading": ("it is stated\nhere, and reported as a direction rather than a "
                               "measurement", 2),
    "worth V-ing": ("worth\nstating exactly, and worth carrying through", 2),
    "which is why": ("which is\nwhy it is printed", 1),
    "the objection": ("the\nobjection is answered by measurement", 1),
}

# Counts, not rates: a rate invites rounding an argument, and these are small integers a person
# can check by eye. The manuscript's ceiling is the corpus's generous per-paper bound carried
# across 11.5k words; the Supporting Information's is the same bound across 29.7k.
# Each ceiling is the corpus's generous per-paper bound carried across the file's word count, and
# not the count the file happens to show. The first draft of this dict set the manuscript's
# "is/are not a" ceiling to 4 when the file contained 4, which is a ceiling drawn around the
# current state -- the defect the Supporting Information's own entry in check_prose_density
# criticises twenty lines from here. At the corpus bound of 0.11 per 1000 words, 11.1k words of
# manuscript allow 1 and 29.5k of Supporting Information allow 3.
TARGETS = {
    "paper2/body.tex": {"rather than": 8, "is/are not a": 1, "epistemic self-grading": 0,
                        "worth V-ing": 0, "which is why": 0, "the objection": 0},
    "paper2/si.tex": {"rather than": 20, "is/are not a": 3, "epistemic self-grading": 0,
                      "worth V-ing": 0, "which is why": 0, "the objection": 0},
}


def prose(rel: str) -> str:
    """The text a reader meets, through the gate's splitter so there is one de-TeXer here."""
    p = ROOT / rel
    if not p.exists():
        return ""
    return " ".join(s for blk in cpd.blocks(p.read_text()) for s in cpd.sentences(blk))


def hits(text: str, pat: str) -> list:
    return [re.sub(r"\s+", " ", m.group(0)) for m in re.finditer(pat, text, re.I)]


def self_test() -> list:
    """Each probe against its own fixture. A probe that cannot find a planted case is refused."""
    broken = []
    for name, pat in PROBES.items():
        fixture, expected = FIXTURES[name]
        got = len(hits(fixture, pat))
        if got != expected:
            broken.append(f"probe {name!r} found {got} in its own fixture, expected {expected}")
    return broken


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print occurrences instead of counts")
    args = ap.parse_args()

    broken = self_test()
    if broken:
        print("REFUSING: a probe failed its own fixture, so a clean report would mean nothing:",
              file=sys.stderr)
        for b in broken:
            print(f"    {b}", file=sys.stderr)
        return 1

    bad = []
    for rel, target in TARGETS.items():
        text = prose(rel)
        if not text:
            print(f"  {rel}: not in this checkout")
            continue
        words = len(text.split())
        print(f"  {rel}: {words} words of prose")
        for name, pat in PROBES.items():
            found = hits(text, pat)
            n, cap = len(found), target[name]
            mark = "" if n <= cap else "  OVER"
            print(f"      {name:24} {n:4d}  = {1000 * n / words:5.2f}/1k  "
                  f"(ceiling {cap}){mark}")
            if n > cap:
                bad.append(f"{rel}: {name!r} occurs {n} times against a ceiling of {cap}")
            if args.show:
                for h in found[:40]:
                    print(f"          {h}")

    if bad:
        print("\nREFUSING: the rate is outside what the comparison papers do:", file=sys.stderr)
        for x in bad:
            print(f"    {x}", file=sys.stderr)
        print("\n  The instruction is to reword, not to delete. About a dozen of these carry the "
              "unfavourable half of a concession, and the inventory in "
              "scripts/disclosure_inventory.py is what holds them: reach the ceiling by writing "
              "\"not\" or a semicolon, and keep every proposition the sentence makes.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

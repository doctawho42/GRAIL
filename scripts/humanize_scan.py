#!/usr/bin/env python3
r"""Scan the manuscript for the machine-writing patterns catalogued by WikiProject AI Cleanup.

Only the patterns that apply to a technical paper are checked. The catalogue's advice to add opinions
and first-person voice is for essays; for a methods section the plain impersonal register is the
correct human one, and injecting personality would be the defect rather than the cure.

Each family prints its hits with context so a person decides. Nothing is rewritten here.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "ai vocabulary": [
        r"\bdelve", r"\btapestry", r"\btestament\b", r"\bunderscor", r"\bshowcas",
        r"\bcrucial\b", r"\bpivotal\b", r"\bvibrant\b", r"\bintricat", r"\bmeticulous",
        r"\bseamless", r"\bleverag", r"\bgarner", r"\binterplay\b", r"\bever-\w+",
        r"\bit is important to note\b", r"\bactually\b", r"\badditionally\b",
    ],
    "aphorism formula": [
        r"\bis the \w+ of the \w+\b", r"\bthe language of\b", r"\bthe currency of\b",
        r"\bthe architecture of\b", r"\bbecomes a trap\b", r"\bnot a \w+ but a \w+\b",
        r"\bis not a \w+, it is\b",
    ],
    "authority trope": [
        r"\bthe real question\b", r"\bat its core\b", r"\bin reality\b",
        r"\bwhat really matters\b", r"\bfundamentally\b", r"\bthe deeper \w+\b",
        r"\bthe heart of the matter\b",
    ],
    "signposting": [
        r"\blet us (?:dive|explore|turn|break)\b", r"\bwe now turn to\b",
        r"\bhere is what\b", r"\bwithout further ado\b", r"\bin what follows, we will\b",
    ],
    "negative parallelism": [
        r"\bnot only\b[^.]{0,60}\bbut also\b", r"\bit is not (?:just|merely|only)\b[^.]{0,40}, it is\b",
        r"\bno guessing\b", r"\bno wasted\b",
    ],
    "copula avoidance": [
        r"\bserves as a\b", r"\bstands as a\b", r"\bboasts\b", r"\brepresents a\b(?! \w+ order)",
    ],
    "filler": [
        r"\bin order to\b", r"\bdue to the fact that\b", r"\bat this point in time\b",
        r"\bin the event that\b", r"\bhas the ability to\b", r"\bit should be noted\b",
    ],
    "hedge stack": [
        r"\b(?:could|might|may) potentially\b", r"\bpossibly\b[^.]{0,30}\b(?:might|may)\b",
        r"\bsomewhat\b", r"\brelatively\b(?! \w+ prime)", r"\bfairly\b",
    ],
    "curly quote": [r"[‘’“”]"],
    "predicate hyphen": [
        r"\bis (?:high|low|data|cross|well|long|real)-\w+\b",
        r"\bare (?:high|low|data|cross|well|long|real)-\w+\b",
    ],
    "false range": [r"\bfrom \w+ to \w+, from \w+ to\b"],
}

# One planted case per family, and the check refuses if a family cannot find its own. The question
# each fixture answers is "on what input does this family go red?" -- and a check for which that
# question has no answer is a tautology even when its code is correct. This file had exactly that:
# "negative parallelism 0" over a tree into which fourteen "and not" constructions had just been
# inserted. The count was right, the patterns ran, and the zero certified nothing.
# The assertion is "at least one", deliberately: the fixture tests whether the family is alive, and
# an exact count would test the length of the pattern list instead.
FIXTURES = {
    "ai vocabulary": "We delve into the tapestry of results.",
    "aphorism formula": "It is not a bug but a feature.",
    "authority trope": "At its core, the real question is simple.",
    "signposting": "We now turn to the results.",
    "negative parallelism": "It is not only fast but also cheap.",
    "copula avoidance": "The method serves as a baseline.",
    "filler": "In order to proceed, it should be noted that this holds.",
    "hedge stack": "It could potentially work.",
    "curly quote": "He said “yes”.",
    "predicate hyphen": "The report is high-quality.",
    "false range": "It ranges from dawn to dusk, from north to south.",
}

# What these families are KNOWN not to cover, written down because an undocumented blind spot is
# read as an absence. Negative parallelism here matches "not only ... but also", the "it is not
# just X, it is Y" shape, and two phrases lifted verbatim from the WikiProject examples -- so it
# finds the sentences the catalogue used to illustrate the pattern and not the pattern. It does not
# see a bare "and not a X", nor "not only X but Y" without "also". Those are counted by
# scripts/check_prose_rate.py, where "and not" is a probe and the corpus gives it zero occurrences
# in 49,637 words; do not read a zero here as their absence.
BLIND_SPOTS = {
    "negative parallelism": ("bare 'and not X' and 'not only X but Y' without 'also'; see "
                             "scripts/check_prose_rate.py, which measures them against the corpus"),
}


def self_test() -> list:
    broken = []
    for fam, pats in FAMILIES.items():
        fixture = FIXTURES.get(fam)
        if fixture is None:
            broken.append(f"family {fam!r} has no fixture, so its zero would mean nothing")
            continue
        if not any(re.search(p, fixture, re.IGNORECASE) for p in pats):
            broken.append(f"family {fam!r} found nothing in its own planted case")
    return broken


# Both manuscripts, because this scan was written for the first one and never read the second.
# It swept paper/ only, so every count it reported was about the ICLR submission while the
# document under review was paper2/ -- a detector that cannot go red on the file it is quoted
# about. The JCIM manuscript is listed first because it is the one being submitted.
TREES = ("paper2", "paper")


def texts() -> dict:
    out = {}
    for tree in TREES:
        for p in sorted((ROOT / tree).rglob("*.tex")):
            if "iclr2026_conference" in p.name or "iclr2027_conference" in p.name:
                continue
            out[str(p.relative_to(ROOT))] = p.read_text(errors="ignore")
    return out


def main() -> int:
    # Wired in at the top, because a fixture that is never run is the same defect one step
    # earlier: a check added and not called certifies exactly as much as a check that cannot fail.
    broken = self_test()
    if broken:
        print("REFUSING: a family cannot find its own planted case, so a zero from it would "
              "certify nothing:")
        for b in broken:
            print(f"    {b}")
        return 1

    corpus = texts()
    found = {}
    for fam, pats in FAMILIES.items():
        hits = []
        for name, body in corpus.items():
            for pat in pats:
                for m in re.finditer(pat, body, re.IGNORECASE):
                    hits.append({"file": name, "line": body[:m.start()].count("\n") + 1,
                                 "match": m.group(0)[:50],
                                 "context": re.sub(r"\s+", " ",
                                                   body[max(0, m.start() - 65):m.end() + 65]).strip()})
        found[fam] = hits

    # Counted PER FILE and never as a tree total. A tree total here was read as an indictment of
    # the submitted manuscript: "em dashes 102" across TREES, with all of them in the earlier
    # paper/ and paper2/ holding exactly zero in every file. The arithmetic was right and the
    # number was wrong, because its scope was the tree and its reader assumed the file. So the
    # scope is now printed beside every count, and the paper2/ share is printed separately.
    dashes = {name: b.count("---") for name, b in corpus.items()}
    rule3 = []
    for name, body in corpus.items():
        for m in re.finditer(r"\b(\w+), (\w+),? and (\w+)\b", body):
            rule3.append({"file": name, "line": body[:m.start()].count("\n") + 1,
                          "match": m.group(0)[:60]})

    print(f"{len(corpus)} files across {', '.join(TREES)}\n")
    for fam, hits in found.items():
        in2 = sum(1 for h in hits if h["file"].startswith("paper2/"))
        blind = BLIND_SPOTS.get(fam)
        print(f"  {fam:<22} {len(hits):4d} over the trees, {in2} in paper2/")
        if blind:
            print(f"      [does not cover: {blind}]")
        for h in hits[:6]:
            print(f"      {h['file']}:{h['line']}  [{h['match']}]")
            print(f"        ...{h['context'][:120]}")
    r3 = Counter(h["file"] for h in rule3)
    print("\n  typography, per file (a tree total here is what misled us once):")
    for name in sorted(corpus):
        if dashes[name] or r3[name]:
            print(f"      {name:36} em dashes {dashes[name]:4d}   three-item lists {r3[name]:4d}")
    print(f"      {'TREE TOTAL, scope stated':36} em dashes {sum(dashes.values()):4d}   "
          f"three-item lists {len(rule3):4d}")
    # "em_dashes" stays an INTEGER. Splitting the count per file changed this field from 102 to a
    # dict without anyone asking, and a reader of the artifact would have got an object where a
    # number had been: a field's type is part of its contract, and the per-file breakdown belongs
    # in a new field rather than inside the old name.
    Path(ROOT / "results" / "humanize_scan.json").write_text(json.dumps(
        {"counts": {k: len(v) for k, v in found.items()},
         "counts_in_paper2": {k: sum(1 for h in v if h["file"].startswith("paper2/"))
                              for k, v in found.items()},
         "em_dashes": sum(dashes.values()), "em_dashes_per_file": dashes,
         "three_item_lists": len(rule3),
         "three_item_lists_per_file": dict(Counter(h["file"] for h in rule3)),
         "scope": {"trees": list(TREES), "files": len(corpus),
                   "note": "counts span every tree listed; the paper2 share is the separate field"},
         "blind_spots": BLIND_SPOTS, "hits": found}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

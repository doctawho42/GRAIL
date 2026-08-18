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


def texts() -> dict:
    out = {}
    for p in sorted((ROOT / "paper").rglob("*.tex")):
        if "iclr2026_conference" in p.name:
            continue
        out[str(p.relative_to(ROOT))] = p.read_text(errors="ignore")
    return out


def main() -> int:
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

    # em dashes are counted apart: in LaTeX "---" is ordinary typography, so the number matters
    # more than the presence
    dashes = sum(b.count("---") for b in corpus.values())
    rule3 = []
    for name, body in corpus.items():
        for m in re.finditer(r"\b(\w+), (\w+),? and (\w+)\b", body):
            rule3.append({"file": name, "line": body[:m.start()].count("\n") + 1,
                          "match": m.group(0)[:60]})

    print(f"{len(corpus)} files\n")
    for fam, hits in found.items():
        print(f"  {fam:<22} {len(hits)}")
        for h in hits[:6]:
            print(f"      {h['file']}:{h['line']}  [{h['match']}]")
            print(f"        ...{h['context'][:120]}")
    print(f"\n  em dashes (---)        {dashes}")
    print(f"  three-item lists       {len(rule3)}")
    Path(ROOT / "results" / "humanize_scan.json").write_text(json.dumps(
        {"counts": {k: len(v) for k, v in found.items()},
         "em_dashes": dashes, "three_item_lists": len(rule3), "hits": found}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

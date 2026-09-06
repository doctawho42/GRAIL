"""Every sentence in which this work concedes something, listed so a shortening cannot drop one.

A cut is made for length and is judged on length, and what it costs is invisible at the moment it
is made: the sentences that go are the ones that read as slack, and a concession reads as slack.
This project has done it once already -- a shortening pass took out two verdicts, and nobody could
see it in the diff because the diff was large and the verdicts were one clause each.

So the concessions are enumerated before a cut and compared after it. A disclosure is a sentence in
which the paper says the work does not reach, does not establish, cannot separate, was not
measured, holds only under a condition, or is weaker than it looks. The extractor is a keyword
sweep and is therefore noisy in the harmless direction: a sentence that is not really a concession
costs a moment to confirm, while one that is a concession and vanishes is exactly what this exists
to catch.

Both documents are read together, and that is the point. Moving a concession from the manuscript to
the Supporting Information is a legitimate cut and this must not object to it; deleting one is not,
and this must. So a disclosure counts as present if it survives ANYWHERE, and the check reports
only what survives in neither.

    python scripts/disclosure_inventory.py            # record the inventory
    python scripts/disclosure_inventory.py --check    # refuse if a recorded one is now in neither
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import stamp  # noqa: E402

SOURCES = ("paper2/body.tex", "paper2/si.tex", "paper2/grail_jcim.tex")

# What a concession sounds like. Each is a claim the work makes ABOUT ITS OWN REACH, so the list is
# about limits and not about the chemistry: "does not" catches a limit on the work and also a fact
# about a molecule, and the noise that produces is cheaper than a missed concession.
MARKERS = (
    r"\bdoes not (?:follow|establish|reach|resolve|separate|say|show|support|transfer|cover)\b",
    r"\bcannot\b", r"\bis not (?:a|an|the|evidence|established|measured|checkable|comparable)\b",
    r"\bnot (?:measured|established|reported|checkable|available|redistributed|sought|corrected)\b",
    r"\bno (?:figure|number|claim|correction|model|evidence|producer|licence text)\b",
    r"\bwe do not\b", r"\bnothing (?:here|in this|that)\b",
    r"\bweaker\b", r"\bpessimistic\b", r"\bunresolved\b", r"\bopen question\b",
    r"\bis a bound\b", r"\bonly (?:under|at|when|holds|says|establishes)\b",
    r"\blimitation\b", r"\bcaveat\b", r"\bstated as (?:a|the) (?:bound|limit)\b",
    r"\bnot the (?:same|whole|only)\b", r"\brather than (?:a|an|the) (?:fact|result|verdict)\b",
    r"\bcould not be\b", r"\bwas not\b", r"\bdid not\b",
)
PAT = re.compile("|".join(MARKERS), re.I)


def _plain(tex: str) -> str:
    """LaTeX with its markup taken off, so a sentence is comparable across an edit that reflows."""
    t = re.sub(r"(?<!\\)%.*", "", tex)
    t = re.sub(r"\\begin\{(equation|align|figure|table|tabular)\*?\}.*?\\end\{\1\*?\}", " ", t,
               flags=re.S)
    t = re.sub(r"\\(cite[a-z]*|ref|label|input|includegraphics)\*?(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\num[A-Za-z]+\{?\}?", " N ", t)          # a macro's value is not the sentence
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}$\\~^_&]", " ", t)
    return re.sub(r"\s+", " ", t)


def _sentences(text: str) -> list:
    # Split on a full stop that ends a word and is followed by a capital or a quote. Abbreviations
    # inside a sentence ("Fig.", "et al.") therefore do not split it.
    parts = re.split(r"(?<=[a-z0-9\)\]])\.\s+(?=[A-Z\"'\u201c])", text)
    return [p.strip() for p in parts if p.strip()]


def _norm(s: str) -> str:
    """A form stable under reflowing, renumbering and macro changes."""
    s = re.sub(r"\bN\b", "", s)
    s = re.sub(r"[^a-z ]", " ", s.lower())
    return " ".join(s.split())


def collect() -> dict:
    found = {}
    for rel in SOURCES:
        p = ROOT / rel
        if not p.exists():
            continue
        for s in _sentences(_plain(p.read_text())):
            if len(s) < 40 or not PAT.search(s):
                continue
            k = _norm(s)
            if len(k.split()) < 6:
                continue
            found.setdefault(k, {"text": s[:300], "in": []})
            if rel not in found[k]["in"]:
                found[k]["in"].append(rel)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the recorded inventory and refuse on a disclosure that "
                         "now survives in neither document")
    ap.add_argument("--accept", metavar="REASON",
                    help="re-baseline, recording each sentence that went and why. Use only after "
                         "reading every one the check named: the extractor cannot tell a rewrite "
                         "from a deletion, so a rewritten sentence looks exactly like a dropped "
                         "one and only a reader can say which it was")
    args = ap.parse_args()

    out = ROOT / "results" / "disclosure_inventory.json"
    now = collect()

    if args.check:
        if not out.exists():
            print(f"REFUSING: {out.relative_to(ROOT)} has not been recorded, so there is nothing "
                  f"to compare a cut against. Run this without --check before shortening.",
                  file=sys.stderr)
            return 1
        before = json.loads(out.read_text())["disclosures"]
        gone = [k for k in before if k not in now]
        print(f"  {len(before)} disclosures recorded, {len(now)} present now, {len(gone)} gone")
        if gone:
            print(f"\nREFUSING: {len(gone)} disclosure(s) survive in neither the manuscript nor "
                  f"the Supporting Information:", file=sys.stderr)
            for k in gone[:25]:
                print(f"    was in {', '.join(before[k]['in'])}: {before[k]['text'][:150]}",
                      file=sys.stderr)
            print("\n  Moving one to the Supporting Information is a cut and passes here. "
                  "Deleting one is a retraction and has to be a decision, not a side effect of a "
                  "word count.", file=sys.stderr)
            return 1
        added = [k for k in now if k not in before]
        if added:
            print(f"  {len(added)} new disclosure(s) since the inventory was recorded")
        return 0

    accepted = []
    if args.accept:
        prior = json.loads(out.read_text()) if out.exists() else {}
        accepted = list(prior.get("reviewed_and_accepted") or [])
        before = prior.get("disclosures") or {}
        for k in before:
            if k not in now:
                accepted.append({"text": before[k]["text"], "was_in": before[k]["in"],
                                 "reason": args.accept})
        print(f"  {len([a for a in accepted if a['reason'] == args.accept])} sentence(s) accepted "
              f"as reworded rather than retracted, with the reason recorded")

    out.write_text(json.dumps({
        "provenance": stamp(__file__),
        "reviewed_and_accepted": accepted,
        "question": ("every sentence in which this work concedes something, recorded before a cut "
                     "so that what a cut costs is visible rather than invisible"),
        "sources": list(SOURCES),
        "n": len(now),
        "reading": ("a disclosure counts as present if it survives in ANY of the sources: moving "
                    "one to the Supporting Information shortens the manuscript without retracting "
                    "anything, and deleting one shortens it by retracting something"),
        "disclosures": now,
    }, indent=1))
    per = {}
    for v in now.values():
        for rel in v["in"]:
            per[rel] = per.get(rel, 0) + 1
    print(f"  {len(now)} disclosures recorded")
    for rel, n in sorted(per.items()):
        print(f"    {rel}: {n}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
r"""Scan the manuscript and everything shipped beside it for text that should not be there.

Four families, in descending order of how bad they are if missed:

  injection   text addressed to a language model rather than to a reader. A released artifact is
              read by other people's tooling, so an imperative aimed at a model is a defect
              wherever it sits: manuscript, JSON leaf, data file, or script comment.
  address     the manuscript speaking to a referee or instructing a reader. A paper states; it
              does not tell its reader what to do or where to press.
  diary       revision history in the prose. What was tried and rewritten belongs in commit
              messages; the paper reports what is.
  slop        the register tells: machine-writing vocabulary, and the jargon a supervisor's review
              named in this project.

Exit is non-zero if anything in the first three families is found. Slop is reported and not gated,
because a word is only slop in context.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INJECTION = [
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|preceding) (?:instructions?|prompts?)",
    r"disregard (?:the )?(?:above|previous|prior)", r"\bas an AI\b", r"\bas a language model\b",
    r"^\s*(?:system|assistant|user)\s*:", r"<\|.*?\|>", r"\[\[?INST\]?\]",
    r"you (?:are|act as|must act) (?:an?|the) (?:AI|assistant|model|reviewer|referee)",
    r"(?:please )?(?:rate|score|accept|recommend) this (?:paper|submission|work) (?:as|highly)",
    r"do not (?:mention|reveal|disclose) (?:this|that|these)",
    r"\boverride\b.*\b(?:instruction|guardrail|policy)", r"prompt injection",
    r"when (?:asked|prompted|reviewing), (?:say|answer|respond|reply)",
]
ADDRESS = [
    r"\breferee\b", r"\bthe reviewer\b", r"\breviewers?\b(?! (?:are|have|will) (?:not )?)",
    r"a reader should", r"the reader should", r"should press", r"press here",
    r"\byou (?:can|should|will|may)\b", r"\bwe invite\b", r"\bnote that\b", r"\bnotice that\b",
    r"\bconsider that\b", r"it is worth noting", r"the reader is (?:invited|referred)",
]
DIARY = [
    r"\bused to (?:say|read|be)\b", r"\bpreviously (?:said|read|stated|reported)\b",
    r"\bwe first (?:tried|did|wrote|measured)\b", r"\bearlier (?:version|draft|version of this)\b",
    r"\bthen (?:we )?(?:rewrote|redid|corrected|changed) it\b", r"\bin an earlier\b",
    r"\bhas now been (?:fixed|corrected|rewritten)\b", r"\bwas stale\b", r"\bwe had (?:written|said)\b",
    r"\bafter the (?:correction|rewrite|fix)\b", r"\bthis (?:used|once) to\b",
]
SLOP = [
    r"\bdelve\b", r"\btapestry\b", r"\btestament\b", r"\bunderscore(?:s|d)?\b", r"\bshowcase\b",
    r"\bcrucial\b", r"\bpivotal\b", r"\bvibrant\b", r"\bintricate\b", r"\bmeticulous\b",
    r"\brobustly\b", r"\bseamless\b", r"\bleverage\b", # "harness" as a verb is a machine-writing tell; "a re-scoring harness" is the ordinary
    # technical noun in this field and is not one
    r"\bharness(?:es|ed|ing)\s+(?:the|a|an|its|our)\b",
    r"\bit is important to note\b", r"\bin today's\b", r"\bever-(?:growing|evolving)\b",
    r"\bnot only .{0,40} but also\b", r"\bplays? a (?:key|vital|significant) role\b",
    # the register words a supervisor's review named on a companion manuscript
    r"\bsymptom\b", r"\banchor(?:s|ed|ing)? the\b", r"\bhomeopathically\b",
    r"\bwe do not claim\b", r"\bwe do not over-read\b", r"\bover-read\b",
]


def texts() -> dict:
    out = {}
    for p in sorted((ROOT / "paper").rglob("*.tex")):
        if "iclr2026_conference" in p.name:
            continue
        out[str(p.relative_to(ROOT))] = p.read_text(errors="ignore")
    return out


def shipped_artifacts() -> dict:
    """Committed JSON under results/, plus the audit material, read as text."""
    out = {}
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "results", "audit", "docs"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    for rel in tracked:
        p = ROOT / rel
        if p.suffix.lower() in {".json", ".md", ".csv", ".txt", ".py"} and p.exists():
            out[rel] = p.read_text(errors="ignore")
    return out


def scan(corpus: dict, patterns: list, label: str) -> list:
    hits = []
    for name, body in corpus.items():
        for pat in patterns:
            for m in re.finditer(pat, body, re.IGNORECASE | re.MULTILINE):
                line = body[:m.start()].count("\n") + 1
                ctx = re.sub(r"\s+", " ", body[max(0, m.start() - 70):m.end() + 70]).strip()
                hits.append({"family": label, "file": name, "line": line,
                             "match": m.group(0)[:60], "context": ctx})
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "polish_audit.json"))
    ap.add_argument("--all", action="store_true", help="scan shipped artifacts too, not only the paper")
    args = ap.parse_args()

    corpus = texts()
    if args.all:
        corpus.update(shipped_artifacts())

    found = {}
    for label, pats in (("injection", INJECTION), ("address", ADDRESS),
                        ("diary", DIARY), ("slop", SLOP)):
        found[label] = scan(corpus, pats, label)

    Path(args.out).write_text(json.dumps(
        {"files_scanned": len(corpus), "counts": {k: len(v) for k, v in found.items()},
         "hits": found}, indent=1))

    for label in ("injection", "address", "diary", "slop"):
        hs = found[label]
        print(f"\n=== {label}: {len(hs)}")
        for h in hs[:40]:
            print(f"  {h['file']}:{h['line']}  [{h['match']}]")
            print(f"      ...{h['context'][:130]}")
        if len(hs) > 40:
            print(f"  ... and {len(hs) - 40} more")

    gated = sum(len(found[k]) for k in ("injection", "address", "diary"))
    print(f"\n{len(corpus)} files scanned; {gated} findings in the gated families")
    return 1 if gated else 0


if __name__ == "__main__":
    raise SystemExit(main())

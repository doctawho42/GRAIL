"""Every number in the markdown draft, checked against the artifact that would produce it.

The draft was written under a rule its author held strictly: no number that is not in a report.
That rule protects against invention and not against staleness, and four of its figures moved
when the comparison was corrected for the parent-drop convention and for three comparators the
defining artifact carries and the earlier reports did not mention.

This lists every numeric literal in the draft, marks those that match a value in
results/paper2_numbers.json, and names the rest so a human decides. It does not edit the draft:
a number can be legitimately absent from the artifact set (a citation year, a dimension of a
network, a count from the companion manuscript) and only a reader can say which.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "paper2/webserver_draft.md"


def variants(v):
    """The strings an artifact value could legitimately appear as in prose."""
    out = set()
    if isinstance(v, bool):
        return out
    if isinstance(v, float):
        for d in (2, 3, 4):
            out.add(f"{v:.{d}f}")
            out.add(f"{v:.{d}f}".rstrip("0").rstrip("."))
        out.add(f"{v * 100:.1f}")
        out.add(f"{v * 100:.0f}")
        out.add(str(v))
    else:
        out.add(str(v))
        out.add(f"{v:,}")
    return {x for x in out if x}


def main() -> int:
    nums = json.loads((ROOT / "results/paper2_numbers.json").read_text())["numbers"]
    index = {}
    for k, v in nums.items():
        for s in variants(v):
            index.setdefault(s, []).append(k)

    text = DRAFT.read_text()
    body = text.split("# Placeholders to fill")[0]
    body = re.sub(r"⟨[^⟩]*⟩", " ", body)
    body = re.sub(r"\[[^\]]*\d{4}[^\]]*\]", " ", body)      # citations
    body = re.sub(r"`[^`]*`", " ", body)                     # code spans

    found = re.findall(r"(?<![A-Za-z0-9_.])(\d[\d,]*(?:\.\d+)?)(?![\d])", body)
    seen, matched, unmatched = set(), [], []
    for tok in found:
        if tok in seen:
            continue
        seen.add(tok)
        plain = tok.replace(",", "")
        keys = index.get(tok) or index.get(plain)
        (matched if keys else unmatched).append((tok, keys))

    print(f"{len(seen)} distinct numeric literals in the draft body")
    print(f"  {len(matched)} match a value in results/paper2_numbers.json")
    print(f"  {len(unmatched)} do not\n")
    print("  not matched (each needs a human decision):")
    for tok, _ in unmatched:
        ctx = ""
        m = re.search(r"[^\n.]{0,60}(?<![\d.])" + re.escape(tok) + r"(?![\d])[^\n.]{0,40}", body)
        if m:
            ctx = " ".join(m.group(0).split())
        print(f"    {tok:<12} {ctx[:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

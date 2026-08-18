#!/usr/bin/env python3
"""The one list of boards the survey covers.

Five scripts enumerated the boards independently -- two by a SOURCES constant, three inline -- and
adding a twenty-fourth updated two of them. The totals then disagreed by script: the decomposition
counted 24 boards while the screen still reported "across 23 leaderboards and 1922 pairs". A count
that depends on which file computed it is the defect this paper is about, met inside its own
analysis, so the enumeration lives here and the scripts import it.

Adding a board means editing SOURCES or COLLECTIONS once. `check_agreement` is what makes that
enough: it fails if any caller's own idea of the board count differs from this one.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (artifact, key within it) -- a key of None means the file is one board
SOURCES = [
    ("robust_order.json", "cluster0"),
    ("robust_order.json", "cluster1"),
    ("robust_order_retro_extrapolation.json", "extrapolation50k"),
    ("robust_order_metabolite.json", None),
    ("robust_order_posebusters.json", None),
    ("robust_order_wmt24_en-de.json", None),
    ("robust_order_wmt24_ja-zh.json", None),
]
# files holding several boards under "boards"
COLLECTIONS = ("robust_order_wmt24_esa.json", "robust_order_wmt23.json")

# how each board is named where a label is wanted; a board absent from here is named from its own
# system count, which is what the table rows do
LABELS = {
    ("robust_order.json", "cluster0"): "retrosynthesis, seven systems",
    ("robust_order.json", "cluster1"): "retrosynthesis, three systems",
    ("robust_order_retro_extrapolation.json", "extrapolation50k"):
        "retrosynthesis, five systems",
    ("robust_order_metabolite.json", None): "metabolites, three methods",
    ("robust_order_posebusters.json", None): "docking, seven programs",
    ("robust_order_wmt24_en-de.json", None): "translation, nineteen systems",
    ("robust_order_wmt24_ja-zh.json", None): "translation, fifteen systems",
}


def load_boards(with_labels: bool = False):
    """Every board the survey covers, in a fixed order."""
    out = []
    for fn, key in SOURCES:
        p = ROOT / "results" / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        b = d["leaderboards"][key] if key else d
        out.append((LABELS.get((fn, key), fn), b) if with_labels else b)
    for fn in COLLECTIONS:
        p = ROOT / "results" / fn
        if not p.exists():
            continue
        for name, b in json.loads(p.read_text())["boards"].items():
            out.append((f"{fn.replace('robust_order_', '').replace('.json', '')}:{name}", b)
                       if with_labels else b)
    return out


def totals() -> dict:
    bs = load_boards()
    return {"boards": len(bs),
            "places": sum(b["n_systems"] for b in bs),
            "pairs": sum(b["n_pairs"] for b in bs),
            "dominating": sum(b["n_dominating"] for b in bs),
            "contested": sum(b["n_contested"] for b in bs),
            "certified": sum(b["n_contested_after_correction"] for b in bs),
            "unresolved": sum(b["n_unresolved"] for b in bs),
            "tiers": sum(b["tiers_distinguished"] for b in bs)}


def check_agreement(n_boards: int, n_pairs: int, who: str) -> None:
    """A caller that counted for itself must get the same answer as this module."""
    t = totals()
    if n_boards != t["boards"] or n_pairs != t["pairs"]:
        raise SystemExit(
            f"{who} covers {n_boards} boards and {n_pairs} pairs; scripts/_boards.py has "
            f"{t['boards']} and {t['pairs']}. One of them is missing a board, and a total that "
            f"depends on which script computed it is exactly what this paper is about.")


if __name__ == "__main__":
    for k, v in totals().items():
        print(f"  {k:<12} {v}")

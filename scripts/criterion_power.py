#!/usr/bin/env python3
r"""A null needs a size: how far the matching rule would have to move a pair to reverse it.

The paper's second claim is negative --- on none of the twenty-three boards does the matching
criterion certify a reversal of an ordering the published cell had separated --- and a negative
claim with no power behind it is indistinguishable from a measurement that could not have seen
anything. The identity gives the power statement for free, because it says exactly what a reversal
requires: the criterion must erase the whole margin and then some. So the question "was the axis
capable of reversing this pair?" has an arithmetic answer, one signed share per pair per criterion:

    d      the margin in the published cell
    d'     the same margin under the other criterion
    share  1 - d'/d, the part of the published margin the criterion erases

The sign is the point and the first version of this got it wrong. A direction-blind
|d' - d| / max(|d|,|d'|) tends to 1 both when a margin is annihilated and when it is inflated, so
it scored the seven-system board's *widest* movement --- a margin that went from $0.0146$ to
$0.0286$, further from a reversal --- as the one that came closest to reversing. The signed share
is negative there, is $1$ exactly when the margin is annihilated, and exceeds $1$ exactly when the
ordering flips.

Nothing here is resampled and nothing re-scored: every quantity is read out of the per-cell system
means each board artifact already carries, at the board's own second-axis setting, so the criterion
is the only thing varying. The ratio is reported per board as its maximum and its median over the
pairs the published cell separates, which is the population the claim is about --- a pair the
benchmark never resolved has no established ordering to reverse, though it can certainly be
exchanged, and that exchange is the paper's first claim rather than this one.

A ratio near $1$ on a board with no certified reversal would mean the axis very nearly did it and
the test was the only thing in the way. A ratio an order of magnitude below $1$ means something
else: on that board the matching rule could not have reversed the pair whatever the test said.
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

SOURCES = [("robust_order.json", "cluster0"), ("robust_order.json", "cluster1"),
           ("robust_order_metabolite.json", None), ("robust_order_posebusters.json", None),
           ("robust_order_wmt24_en-de.json", None), ("robust_order_wmt24_ja-zh.json", None)]
COLLECTIONS = ("robust_order_wmt24_esa.json", "robust_order_wmt23.json")

# the boards on which the criterion axis is a rule for when a prediction counts as the reference,
# which is the axis the paper's negative claim is about; elsewhere the criterion axis is a
# weighting or a reading of an annotation and the claim is not made
MATCHING_AXIS = {"robust_order:cluster0", "robust_order:cluster1", "robust_order_metabolite",
                 "robust_order_posebusters"}


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def boards() -> list:
    out = []
    for name, key in SOURCES:
        p = ROOT / "results" / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out.append((f"{name[:-5]}{':' + key if key else ''}",
                    d["leaderboards"][key] if key else d))
    for name in COLLECTIONS:
        p = ROOT / "results" / name
        if not p.exists():
            continue
        for lp, b in json.loads(p.read_text())["boards"].items():
            out.append((f"{name[:-5]}:{lp}", b))
    return out


def analyse(b: dict) -> dict:
    acc = b["system_accuracy_by_cell"]
    pub = ast.literal_eval(b["published_cell"])
    cells = [ast.literal_eval(c) for c in next(iter(acc.values()))]
    # the criterion axis at the board's own setting of the other axis, so only the criterion moves
    siblings = [c for c in cells if c[1] == pub[1] and c[0] != pub[0]]

    rows = []
    for name, v in b["pairs"].items():
        if not v["per_cell"][str(pub)]["separated"]:
            continue
        hi, lo = name.split(" over ")
        d0 = acc[hi][str(pub)] - acc[lo][str(pub)]
        if d0 == 0:
            continue
        for c in siblings:
            d1 = acc[hi][str(c)] - acc[lo][str(c)]
            rows.append({"pair": name, "criterion": c[0], "margin": d0, "margin_there": d1,
                         "share_erased": 1.0 - d1 / d0,
                         "widens": abs(d1) > abs(d0),
                         "flips": (d0 > 0) != (d1 > 0)})
    r = np.array([x["share_erased"] for x in rows]) if rows else np.zeros(0)
    worst = max(rows, key=lambda x: x["share_erased"]) if rows else None
    separated = sum(1 for v in b["pairs"].values()
                    if v["per_cell"][b["published_cell"]]["separated"])
    return {"n_separated_pairs": separated,
            "n_comparisons": len(rows),
            "criteria_compared": [c[0] for c in siblings],
            "max_ratio": round(float(r.max()), 4) if len(r) else None,
            "median_ratio": round(float(np.median(r)), 4) if len(r) else None,
            "n_widening": int(sum(x["widens"] for x in rows)),
            "n_flipping": int(sum(x["flips"] for x in rows)),
            "closest_pair": ({"pair": worst["pair"], "criterion": worst["criterion"],
                              "margin": round(worst["margin"], 4),
                              "margin_there": round(worst["margin_there"], 4),
                              "ratio": round(worst["share_erased"], 4)} if worst else None)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "criterion_power.json"))
    args = ap.parse_args()

    per = {lab: analyse(b) for lab, b in boards()}
    matching = {k: v for k, v in per.items() if k in MATCHING_AXIS and v["max_ratio"] is not None}

    ratios = [v["max_ratio"] for v in matching.values()]
    # a board on which the axis does flip a separated pair is not evidence of no power, and
    # averaging it in with the boards where nothing flips would hide both facts
    quiet = {k: v for k, v in matching.items() if v["n_flipping"] == 0}
    q_ratios = [v["max_ratio"] for v in quiet.values()]
    rep = {"config": {**_code_version(),
                      "note": "read out of the per-cell system means the board artifacts carry; "
                              "the other axis is held at the board's published setting",
                      "matching_axis_boards": sorted(matching)},
           "matching_axis": {
               "n_boards": len(matching),
               "n_boards_where_nothing_flips": len(quiet),
               "largest_ratio_where_nothing_flips": round(max(q_ratios), 4) if q_ratios else None,
               "closest_such_board": max(quiet, key=lambda k: quiet[k]["max_ratio"])
               if q_ratios else None,
               "fold_short_on_that_board": round(1.0 / max(q_ratios), 2)
               if q_ratios and max(q_ratios) else None,
               "median_of_the_board_maxima": round(float(np.median(ratios)), 4) if ratios else None,
               "median_ratio_where_nothing_flips": round(
                   float(np.median([v["median_ratio"] for v in quiet.values()])), 4)
               if quiet else None,
               "boards_with_a_flip": {k: v["n_flipping"] for k, v in matching.items()
                                      if v["n_flipping"]}},
           "per_board": per}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    m = rep["matching_axis"]
    print(f"  matching-rule axis on {m['n_boards']} boards, over pairs the published cell separates")
    print(f"  on the {m['n_boards_where_nothing_flips']} where nothing flips, the largest movement "
          f"reaches {m['largest_ratio_where_nothing_flips']} of what a reversal needs "
          f"({m['closest_such_board']}), i.e. {m['fold_short_on_that_board']}x short")
    print(f"  their median comparison is {m['median_ratio_where_nothing_flips']}; "
          f"boards where the axis does flip a separated pair: {m['boards_with_a_flip']}")
    for lab, v in per.items():
        if lab in MATCHING_AXIS:
            print(f"     {lab:32s} max {v['max_ratio']}, median {v['median_ratio']}, "
                  f"{v['n_flipping']} flips of {v['n_comparisons']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

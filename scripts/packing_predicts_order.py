#!/usr/bin/env python3
"""Does the inequality predict the measurement, on the same leaderboards?

Two instruments in this paper describe the same thing from opposite ends. The robust order measures
which published pairwise orderings survive every cell of a grid, and to compute it you must score
every system in every cell. The packing condition says a pair can only be reversed when the choice
moves the two systems apart by more than the gap between them, and to compute it you need the
published cell and the size of the movement -- not the whole grid.

If the second is a theory of the first rather than a restatement, then running the screen on the
published cell alone should recover the pairs the full grid removes. That is testable here without
any new scoring, because the robust-order artifact already records the accuracy of every system in
every cell:

    gap(a,b)        their margin in the published cell
    move(a,b,c)     how much cell c moves them apart, |(a_c - a_0) - (b_c - b_0)|
    flagged         some cell has move > gap
    failed          the pair does not dominate: some cell reverses it

`flagged` is necessary for `failed` by arithmetic, so the interesting quantity is not whether it
catches them -- it must -- but how much it over-flags, and whether the over-flagging is small enough
that a maintainer given only a published table and one re-scored system learns something usable.
"""
from __future__ import annotations

import argparse

import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def analyse_board(r: dict) -> dict:
    acc = r["system_accuracy_by_cell"]
    systems = r["published_order"]
    published = r["published_cell"]
    cells = [c for c in next(iter(acc.values())) if c != published]

    rows = []
    for a, b in itertools.combinations(systems, 2):
        # The movement is compared against the larger of the two margins it connects, not against
        # the published one alone. Against one margin the condition is necessary and not
        # sufficient, and which margin that is would here be an accident of which cell a paper
        # happened to print; against the larger it holds exactly when the cell reverses the pair.
        gap = abs(acc[a][published] - acc[b][published])
        worst, worst_cell = 0.0, None
        for c in cells:
            here = acc[a][c] - acc[b][c]
            move = abs(here - (acc[a][published] - acc[b][published]))
            if move > max(gap, abs(here)) and move > worst:
                worst, worst_cell = move, c
        hi, lo = (a, b) if systems.index(a) < systems.index(b) else (b, a)
        name = f"{hi} over {lo}"
        v = r["pairs"][name]
        # Dominance requires a strictly positive margin in every cell, so an exact tie in one cell
        # ends it without any cell ordering the pair the other way. The inequality detects sign
        # changes and cannot detect a tie -- the movement then equals the gap rather than exceeding
        # it -- so the set it is checked against is the strictly reversed pairs, and tie-only
        # failures are counted apart rather than scored as misses of a thing that did not happen.
        reversed_strictly = any(c["margin"] < 0 for c in v["per_cell"].values())
        rows.append({"pair": name, "gap": round(gap, 4), "largest_move": round(worst, 4),
                     "cell_of_that_move": worst_cell, "flagged": worst_cell is not None,
                     "reversed_by_some_cell": reversed_strictly,
                     "failed_to_dominate": not v["dominates"],
                     "failed_on_a_tie_alone": (not v["dominates"]) and not reversed_strictly,
                     "contested": v["contested"]})

    tp = sum(x["flagged"] and x["reversed_by_some_cell"] for x in rows)
    fp = sum(x["flagged"] and not x["reversed_by_some_cell"] for x in rows)
    fn = sum(not x["flagged"] and x["reversed_by_some_cell"] for x in rows)
    tn = len(rows) - tp - fp - fn
    return {"n_pairs": len(rows), "flagged": tp + fp, "failed": tp + fn,
            "failed_on_a_tie_alone": sum(x["failed_on_a_tie_alone"] for x in rows),
            "flagged_and_failed": tp, "flagged_but_survived": fp,
            "failed_but_not_flagged": fn, "neither": tn,
            "sensitivity": round(tp / max(tp + fn, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "specificity": round(tn / max(tn + fp, 1), 4),
            "pairs": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "packing_predicts_order.json"))
    args = ap.parse_args()

    boards = {}
    ro = ROOT / "results/robust_order.json"
    if ro.exists():
        for name, r in json.loads(ro.read_text())["leaderboards"].items():
            if "system_accuracy_by_cell" in r:
                boards[f"retrosynthesis, {r['n_systems']} systems"] = r
    rm = ROOT / "results/robust_order_metabolite.json"
    if rm.exists():
        r = json.loads(rm.read_text())
        if "system_accuracy_by_cell" in r:
            boards[f"metabolites, {r['n_systems']} methods"] = r
    for lp in ("en-de", "ja-zh"):
        rw = ROOT / f"results/robust_order_wmt24_{lp}.json"
        if rw.exists():
            r = json.loads(rw.read_text())
            if "system_accuracy_by_cell" in r:
                boards[f"translation, WMT24 {lp}"] = r

    re_ = ROOT / "results/robust_order_wmt24_esa.json"
    if re_.exists():
        for lp, r in json.loads(re_.read_text())["boards"].items():
            if "system_accuracy_by_cell" in r:
                boards[f"translation, WMT24 {lp} (esa)"] = r

    rp = ROOT / "results/robust_order_posebusters.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        if "system_accuracy_by_cell" in r:
            boards[f"docking, {r['n_systems']} programs"] = r
    if not boards:
        raise SystemExit("no robust-order artifact carrying per-cell accuracies")

    rep = {"config": {**_code_version(),
                      "note": "the screen uses only the published cell and the movement to each "
                              "other cell; the outcome comes from the full grid"},
           "per_leaderboard": {}}
    tot = {"n_pairs": 0, "flagged": 0, "failed": 0, "flagged_and_failed": 0,
           "flagged_but_survived": 0, "failed_but_not_flagged": 0, "failed_on_a_tie_alone": 0}
    for name, r in boards.items():
        v = analyse_board(r)
        rep["per_leaderboard"][name] = v
        for k in tot:
            tot[k] += v[k]
        print(f"  {name:32} {v['flagged']:3} flagged of {v['n_pairs']:3}; "
              f"{v['flagged_and_failed']} of the {v['failed']} that failed are among them, "
              f"{v['failed_but_not_flagged']} are not; precision {v['precision']}", flush=True)

    tot["sensitivity"] = round(tot["flagged_and_failed"] / max(tot["failed"], 1), 4)
    tot["precision"] = round(tot["flagged_and_failed"] / max(tot["flagged"], 1), 4)
    rep["totals"] = tot
    print(f"\n  across {len(boards)} leaderboards and {tot['n_pairs']} pairs: the screen flags "
          f"{tot['flagged']}, {tot['failed']} fail in the full grid, sensitivity "
          f"{tot['sensitivity']}, precision {tot['precision']}")
    if tot["failed_but_not_flagged"]:
        print(f"  {tot['failed_but_not_flagged']} pairs failed without being flagged, which the "
              f"arithmetic forbids; one of the two quantities is computed wrongly")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

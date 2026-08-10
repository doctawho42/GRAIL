#!/usr/bin/env python3
"""Predicting how fragile a leaderboard is from the leaderboard alone.

Everything else in this paper needs released prediction files. Most published tables do not come
with any, and the question a reader of such a table has is the same: how much of this ordering would
an undeclared choice disturb? That is answerable from the table, because a pair reorders when the
choice moves the two systems apart by more than the margin between them, and the margins are printed.

Take the movement a choice produces as a single number for a domain -- this paper measures it, and
it is stable across the leaderboards here -- and the prediction for any table is the share of pairs
whose margin is smaller than it. This script makes that prediction on tables where the predictions
ARE available, so it can be checked against what actually happens:

    predicted     share of pairs with margin below the movement, from the published column alone
    observed      share of pairs that actually change order somewhere in the declared grid

If the two agree, a reader can run the instrument on a table in a paper they cannot reproduce, which
is most of them.
"""
from __future__ import annotations

import argparse, itertools, json, pathlib, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "fragility_from_table.json"))
    args = ap.parse_args()

    ro = json.loads((ROOT / "results/robust_order.json").read_text())["leaderboards"]
    rep = {"config": {**_code_version(),
                      "note": "the movement is estimated on each leaderboard's own grid, then the "
                              "prediction uses only the published column"},
           "leaderboards": {}}

    for name, L in ro.items():
        pairs = L["pairs"]
        # the published column is the cell the table is quoted in; margins there are all a reader has
        published_cell = L["published_cell"]
        margins, moved, observed = [], [], 0
        for k, v in pairs.items():
            cells = v["per_cell"]
            base = cells.get(published_cell)
            if base is None:
                continue
            margins.append(abs(base["margin"]))
            # the movement this pair actually sees: how far the margin travels across the grid
            span = max(abs(c["margin"] - base["margin"]) for c in cells.values())
            moved.append(span)
            observed += (not v["dominates"])
        margins, moved = np.array(margins), np.array(moved)
        movement = float(np.median(moved))
        # A single movement number treats a pair as fragile whenever the typical movement exceeds
        # its margin, which counts pairs the movement happens to miss. Using the movement's own
        # distribution instead gives an expected count: for each margin, the share of movements in
        # the domain that exceed it, summed. A reader of a table has the margins and needs only the
        # distribution, which is a domain constant rather than a per-pair quantity.
        predicted = int((margins < movement).sum())
        expected = float(sum((moved > m).mean() for m in margins))
        rep["leaderboards"][name] = {
            "n_pairs": len(margins), "published_cell": published_cell,
            "median_margin": round(float(np.median(margins)), 4),
            "median_movement": round(movement, 4),
            "predicted_fragile": predicted, "expected_fragile": round(expected, 2),
            "observed_fragile": observed,
            "predicted_share": round(predicted / max(len(margins), 1), 4),
            "observed_share": round(observed / max(len(margins), 1), 4)}
        print(f"  {name:10} pairs {len(margins):3}  median margin {np.median(margins):.4f}  "
              f"median movement {movement:.4f}")
        print(f"      threshold rule {predicted:3}   expected count {expected:5.1f}   "
              f"observed {observed:3}")

    tot_e = sum(v["expected_fragile"] for v in rep["leaderboards"].values())
    tot_p = sum(v["predicted_fragile"] for v in rep["leaderboards"].values())
    tot_o = sum(v["observed_fragile"] for v in rep["leaderboards"].values())
    rep["totals"] = {"threshold_rule": tot_p, "expected_count": round(tot_e, 2),
                     "observed": tot_o, "error_threshold": abs(tot_p - tot_o),
                     "error_expected": round(abs(tot_e - tot_o), 2)}
    print(f"\n  across leaderboards: threshold rule {tot_p}, expected count {tot_e:.1f}, "
          f"observed {tot_o}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
r"""Do the docking board's reversals survive a correction over the family they were found in?

A pair is called \emph{contested} when some cell of the grid orders it against the published table
with an interval excluding zero. On this board that verdict is read out of $21$ pairs by $8$ cells,
so the family is $168$ cell-level tests and an uncorrected interval is the wrong object to hang the
paper's control on: with that many looks, two reversals is close to what noise alone would show.

This runs the correction the paper reserves the word \emph{certified} for. Each cell-level test gets
a two-sided bootstrap $p$ from the same paired resampling the intervals come from --- twice the
smaller tail mass of the resampled margin, floored at $1/B$ since a bootstrap cannot resolve past
its own resolution --- and Holm is applied across all $168$ at $0.05$.

The answer decides what the paper may say. If no reversal survives, the control recovers nothing and
the sentence claiming it does has to be withdrawn; the grid would still be reporting that the order
is unstable, but not that any particular exchange is real.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from robust_order import N_BOOT, SEED  # noqa: E402

ALPHA = 0.05


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


def _board():
    spec = importlib.util.spec_from_file_location(
        "posebusters_board", ROOT / "scripts" / "posebusters_board.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["posebusters_board"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/external/posebusters_paper_results.csv"))
    ap.add_argument("--board", default=str(ROOT / "results/robust_order_posebusters.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "docking_multiplicity.json"))
    args = ap.parse_args()

    pbb = _board()
    hits, systems, items, cells = pbb.build_hits(pd.read_csv(args.csv))
    board = json.loads(Path(args.board).read_text())
    rank = {s: i for i, s in enumerate(board["published_order"])}

    # the same resampling the intervals came from: one index matrix, reused for every test, so the
    # per-cell margins are paired across cells as well as across systems
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(items), (N_BOOT, len(items)))

    rows = []
    for a, b in itertools.combinations(systems, 2):
        hi, lo = (a, b) if rank[a] < rank[b] else (b, a)
        for c in cells:
            d = hits[(hi, c)] - hits[(lo, c)]
            bt = d[idx].mean(axis=1)
            p = 2 * min(float((bt >= 0).mean()), float((bt <= 0).mean()))
            rows.append({"pair": f"{hi} over {lo}", "cell": str(c),
                         "margin": round(float(d.mean()), 4),
                         "p": max(p, 1.0 / N_BOOT),
                         "against_the_published_order": bool(d.mean() < 0)})

    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m = len(rows)
    survivors = []
    for step, i in enumerate(order):
        if rows[i]["p"] <= ALPHA / (m - step):
            rows[i]["holm"] = "survives"
            survivors.append(rows[i])
        else:
            for j in order[step:]:
                rows[j]["holm"] = "does not survive"
            break

    reversals = [r for r in survivors if r["against_the_published_order"]]
    pairs = sorted({r["pair"] for r in reversals})
    contested = sorted(n for n, v in board["pairs"].items() if v["contested"])

    # robust_order.analyse now folds the same correction into every board, so this script is a
    # second implementation of one number. That is the point: it is written from the hits matrix
    # rather than from analyse's internals, so agreement is evidence and disagreement is a bug in
    # one of the two. Drift is what makes duplicated statistics dangerous, and this refuses to drift.
    theirs = board.get("multiplicity", {})
    agrees = (theirs.get("family_size") == m
              and theirs.get("n_reversals_surviving") == len(reversals)
              and sorted(board.get("contested_after_correction", [])) == pairs)

    rep = {"config": {**_code_version(), "n_boot": N_BOOT, "seed": SEED, "alpha": ALPHA,
                      "family": "every cell-level test the contested verdict is read from",
                      "family_size": m, "n_pairs": len(list(itertools.combinations(systems, 2))),
                      "n_cells": len(cells),
                      "p": "two-sided bootstrap, twice the smaller tail mass, floored at 1/B"},
           "family_size": m, "n_surviving_tests": len(survivors),
           "n_surviving_reversals": len(reversals),
           "pairs_still_contested": pairs,
           "pairs_contested_before_correction": contested,
           "every_contested_pair_survives": sorted(pairs) == contested,
           "agrees_with_the_board_artifact": agrees, "board_says": theirs,
           "surviving_reversals": reversals, "tests": rows}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"  family of {m} cell-level tests ({len(contested)} pairs contested before correction)")
    print(f"  Holm at {ALPHA}: {len(survivors)} tests survive, {len(reversals)} of them reversals")
    print(f"  still contested: {', '.join(pairs) if pairs else 'none'}")
    for r in reversals:
        print(f"     {r['pair']:26s} {r['cell']:40s} {r['margin']:+.4f}  p={r['p']:.4f}")
    print(f"  independent of robust_order.analyse, which computes the same thing: "
          f"{'agrees' if agrees else 'DISAGREES -- one of the two is wrong'}")
    if not agrees:
        return 1
    if not rep["every_contested_pair_survives"]:
        print("\n  NOT every contested pair survives; the paper's control sentence overstates it")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

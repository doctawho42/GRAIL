#!/usr/bin/env python3
r"""How much of the leaderboard survives the choices nobody declares?

The paper reports reversals one at a time, which answers "does this happen" and not "how much of the
ordering is real". This gives the aggregate, and it is written to bound the paper's own claims rather
than to extend them: the striking number here is not the one to quote.

The lattice is every combination of a matching criterion and an aggregate, on the one population
where all five methods exist. Both axes are choices a published paper makes without argument -- five
criteria are in use in this literature and all four aggregates are reported in it -- so each cell is
a leaderboard someone could have published from the same frozen predictions.

Two counts are reported for every slice and they differ by a factor of four. A pair CHANGES SIGN if
the ordering is not the same in every cell, which is cheap and says little; a pair REVERSES CERTIFIED
if some cell puts A above B with disjoint intervals and some other cell puts B above A the same way.
Reporting only the first would commit exactly the fallacy \S\ref{sec:rankflip} rejects, so the first
is reported as a description and the second as the finding.

The gate is that the certified count must reproduce the two pairs the paper already names, from a
different direction: this script never reads the paper's reversal analysis and rebuilds the count
from the criterion grid alone.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "results" / "set_metrics_by_criterion.json"
AGGREGATES = ("recall", "f1", "precision", "jaccard")
# paper/app/xdomain.tex: "Those two pairs are the certified reversals"
COMMITTED_PAIRS = {("BioTransformer", "SyGMa"), ("GRAIL", "SyGMa")}


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


def build_cells(pop, modes, aggregates, methods):
    cells = []
    for mode in modes:
        for agg in aggregates:
            row = {}
            for method in methods:
                v = pop["by_mode"].get(mode, {}).get(method)
                if v and agg in v:
                    row[method] = v[agg]
            if len(row) == len(methods):
                cells.append(((mode, agg), row))
    return cells


def summarise(cells, methods) -> dict:
    orderings = {tuple(sorted(methods, key=lambda m: -row[m]["point"])) for _, row in cells}
    sign, certified, pairs = 0, 0, []
    for a, b in itertools.combinations(methods, 2):
        if len({row[a]["point"] > row[b]["point"] for _, row in cells}) > 1:
            sign += 1
        up = [k for k, r in cells if r[a]["ci95"][0] > r[b]["ci95"][1]]
        down = [k for k, r in cells if r[b]["ci95"][0] > r[a]["ci95"][1]]
        if up and down:
            certified += 1
            pairs.append({"pair": [a, b], "a_above_b_at": list(up[0]), "b_above_a_at": list(down[0]),
                          "cells_each_way": [len(up), len(down)]})
    n_pairs = len(methods) * (len(methods) - 1) // 2
    return {"cells": len(cells), "distinct_orderings": len(orderings), "pairs": n_pairs,
            "pairs_changing_sign": sign, "pairs_reversing_certified": certified,
            "certified": pairs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "ordering_stability.json"))
    args = ap.parse_args()

    grid = json.loads(GRID.read_text())
    modes = grid["modes"]
    pop = grid["populations"]["shared150"]
    methods = sorted(pop["methods"])
    print(f"population: shared subset, n={pop['n_substrates']}, methods {methods}")
    print(f"lattice: {len(modes)} matching criteria x {len(AGGREGATES)} aggregates\n")

    slices = {
        "criterion_and_aggregate": (modes, AGGREGATES),
        "criterion_alone_at_recall": (modes, ("recall",)),
        "criterion_alone_at_f1": (modes, ("f1",)),
        "aggregate_alone_at_inchikey_tautomer": (("inchikey_tautomer",), AGGREGATES),
    }
    out = {}
    print(f"  {'slice':40} {'cells':>5} {'orders':>7} {'sign':>7} {'certified':>10}")
    for name, (ms, ags) in slices.items():
        s = summarise(build_cells(pop, ms, ags, methods), methods)
        out[name] = s
        print(f"  {name:40} {s['cells']:>5} {s['distinct_orderings']:>7} "
              f"{s['pairs_changing_sign']:>4}/{s['pairs']:<2} {s['pairs_reversing_certified']:>7}/{s['pairs']:<2}")

    got = {tuple(sorted(p["pair"])) for p in out["criterion_and_aggregate"]["certified"]}
    want = {tuple(sorted(p)) for p in COMMITTED_PAIRS}
    print(f"\ngate: certified pairs {sorted(got)} against the committed {sorted(want)}")
    if got != want:
        raise SystemExit("this lattice does not reproduce the paper's certified reversals")

    rep = {"config": {**_code_version(), "population": "shared150",
                      "n_substrates": pop["n_substrates"], "methods": methods,
                      "source": "results/set_metrics_by_criterion.json",
                      "gate": "certified pairs must reproduce the two the paper names"},
           "slices": out}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

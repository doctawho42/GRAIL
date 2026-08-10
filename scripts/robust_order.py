#!/usr/bin/env python3
"""What part of a published ranking survives every choice the paper leaves undeclared?

A leaderboard reports a total order. It is computed in one cell of a grid of evaluation choices --
one matching criterion, one output budget -- and the paper reports the order without reporting the
cell, because the cell is not thought of as a parameter. This script asks what part of that order is
a property of the systems rather than of the cell.

    Fix the item set and the predictions. For every pair of systems (a, b) and every cell c of the
    declared grid, the margin is a paired quantity over items. Say a DOMINATES b when the margin is
    positive in every cell, and say the domination is CERTIFIED when a paired bootstrap interval
    excludes zero in every cell.

Domination is the intersection of the per-cell total orders, so it is a strict partial order: it is
transitive because each per-cell order is, and it is the finest relation no cell can contradict.
Certified domination is a sub-relation of it and is reported separately, because an interval
condition is not transitive and this file does not pretend otherwise.

The number the instrument produces is the share of the published pairwise claims that survive:

    robustness = |dominating pairs| / |pairs|,     certified robustness = |certified| / |pairs|

A pair that does not dominate is one whose published ordering some declared choice reverses. A pair
that dominates without being certified is one the benchmark cannot resolve at its own sample size,
which is a different failure and is counted separately rather than folded in.

The point is not that these leaderboards are wrong. It is that the part of them that is safe to
quote is computable from files that are already public, and is smaller than what is quoted.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retro_leaderboard import KS, MODES, build_keys, set_key

N_BOOT, SEED = 10000, 0


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


def analyse(hits: dict, systems: list[str], cells: list, published_cell) -> dict:
    """hits[(system, cell)] -> 0/1 vector over items, aligned across systems."""
    n = len(next(iter(hits.values())))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, (N_BOOT, n))

    published = sorted(systems, key=lambda s: -float(hits[(s, published_cell)].mean()))
    rank = {s: i for i, s in enumerate(published)}

    pairs = {}
    for a, b in itertools.combinations(systems, 2):
        # orient the pair the way the published cell orders it, so "survives" means what a reader
        # of that table would have concluded is still true in every other cell
        hi_, lo_ = (a, b) if rank[a] < rank[b] else (b, a)
        per_cell = {}
        for c in cells:
            d = hits[(hi_, c)] - hits[(lo_, c)]
            bt = d[idx].mean(axis=1)
            l, h = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            per_cell[str(c)] = {"margin": round(float(d.mean()), 4),
                                "ci95": [round(l, 4), round(h, 4)],
                                "positive": bool(d.mean() > 0),
                                "certified": bool(l > 0)}
        dominates = all(v["positive"] for v in per_cell.values())
        certified = all(v["certified"] for v in per_cell.values())
        flips = [k for k, v in per_cell.items() if not v["positive"]]
        pairs[f"{hi_} over {lo_}"] = {
            "dominates": dominates, "certified": certified,
            "cells_that_reverse_it": flips,
            "cells_it_is_not_resolved_in": [k for k, v in per_cell.items()
                                            if v["positive"] and not v["certified"]],
            "per_cell": per_cell}

    n_pairs = len(pairs)
    n_dom = sum(v["dominates"] for v in pairs.values())
    n_cert = sum(v["certified"] for v in pairs.values())

    # The share of surviving pairs is the honest number and not the readable one. The readable one
    # is how many places the leaderboard still distinguishes: the longest chain in the dominance
    # order, which is the number of tiers a reader may quote. A total order of n places whose
    # longest robust chain is t is a table asserting n ranks and supporting t.
    edges: dict = {}
    for k, v in pairs.items():
        hi_, lo_ = k.split(" over ")
        if v["dominates"]:
            edges.setdefault(hi_, set()).add(lo_)
    memo: dict = {}

    def chain(n):
        if n not in memo:
            memo[n] = 1 + max((chain(c) for c in edges.get(n, ())), default=0)
        return memo[n]

    tiers = max(chain(n) for n in systems)
    return {"published_order": published, "published_cell": str(published_cell),
            "n_systems": len(systems), "n_cells": len(cells), "n_pairs": n_pairs,
            "n_dominating": n_dom, "n_certified": n_cert,
            "tiers_distinguished": tiers,
            "robustness": round(n_dom / max(n_pairs, 1), 4),
            "certified_robustness": round(n_cert / max(n_pairs, 1), 4),
            "reversed_by_some_cell": n_pairs - n_dom,
            "unresolved_though_never_reversed": n_dom - n_cert,
            "pairs": pairs}


def retro_leaderboard(cluster: str, directory: Path, max_rank: int, workers: int) -> dict:
    ingest = json.loads((ROOT / "results/evalretro_ingest.json").read_text())
    meta = ingest["clusters"][cluster]
    systems, rows = meta["systems"], list(csv.DictReader(open(ROOT / meta["test_csv"])))
    preds = {n: json.loads((directory / "normalised" / f"{n}.json").read_text()) for n in systems}

    comps = {c for r in rows for c in r["REACTANT"].split(".") if c}
    for n in systems:
        for p in preds[n]:
            for s in p["preds"][:max_rank]:
                comps.update(c for c in s.split(".") if c)
    cache_path = directory / f"keys_{cluster}_r{max_rank}.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [c for c in sorted(comps) if c not in cached]
    if todo:
        cached.update(build_keys(todo, workers))
        cache_path.write_text(json.dumps(cached))

    cells = [(m, k) for m in MODES for k in KS]
    hits = {(n, c): np.zeros(len(rows)) for n in systems for c in cells}
    for j, row in enumerate(rows):
        for mode in MODES:
            truth = set_key(row["REACTANT"], mode, cached)
            if truth is None:
                continue
            for n in systems:
                seq = preds[n][j]["preds"][:max_rank]
                first = None
                for r, s in enumerate(seq, 1):
                    if set_key(s, mode, cached) == truth:
                        first = r
                        break
                if first is None:
                    continue
                for k in KS:
                    if first <= k:
                        hits[(n, (mode, k))][j] = 1.0
    # the cell these tables are published in: strict matching at the budget the field leads with
    return analyse(hits, systems, cells, ("canonical", 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "grail_metabolism" / "data" / "evalretro"))
    ap.add_argument("--clusters", default="cluster0,cluster1")
    ap.add_argument("--max-rank", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "robust_order.json"))
    args = ap.parse_args()

    rep = {"config": {**_code_version(), "n_boot": N_BOOT, "seed": SEED,
                      "grid": {"criteria": list(MODES), "budgets": list(KS)},
                      "note": "domination is the intersection of the per-cell orders; certified "
                              "domination additionally requires every per-cell interval to exclude "
                              "zero"},
           "leaderboards": {}}
    for cluster in args.clusters.split(","):
        r = retro_leaderboard(cluster, Path(args.dir), args.max_rank, args.workers)
        rep["leaderboards"][cluster] = r
        print(f"\n{cluster}: {r['n_systems']} systems, {r['n_cells']} cells, {r['n_pairs']} pairs")
        print(f"  published order at {r['published_cell']}: {' > '.join(r['published_order'])}")
        print(f"  survive every cell:            {r['n_dominating']}/{r['n_pairs']} "
              f"= {r['robustness']}")
        print(f"  and certified in every cell:   {r['n_certified']}/{r['n_pairs']} "
              f"= {r['certified_robustness']}")
        print(f"  tiers it still distinguishes:  {r['tiers_distinguished']} "
              f"of {r['n_systems']} published places")
        print(f"  reversed by some cell:         {r['reversed_by_some_cell']}")
        print(f"  never reversed but unresolved: {r['unresolved_though_never_reversed']}")

    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

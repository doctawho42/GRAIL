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

Sweeping a declared grid of defensible analytic choices and reporting what survives all of them is
multiverse analysis (Steegen et al., 2016) and specification-curve analysis (Simonsohn et al., 2020);
in machine learning the neighbouring practice is accounting for the variance a benchmark's own
sources induce (Bouthillier et al., 2021). What is added here is specific and small: the object
swept is a total ORDER rather than an estimate, the intersection of per-cell orders is therefore
a partial order rather than a distribution, its share and its longest chain are two scalars, and
both are computable from files the field has already released.

Since domination is an intersection over cells, the share falls as cells are added and is a
property of the pair (leaderboard, grid) rather than of the leaderboard. That is measured here
rather than conceded: each declared axis is reported alone and against the product, so a reader can
see the whole curve instead of one point on it.

The point is not that these leaderboards are wrong. It is that the part of them that is safe to
quote is computable from files that are already public, and is smaller than what is quoted.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
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
ALPHA = 0.05


def _paired_p(d) -> float:
    """Two-sided p for a paired mean, from the normal approximation to its sampling distribution.

    Not from the bootstrap draws. A resampled tail cannot resolve past $1/B$, so a bootstrap p is
    floored there, and Holm over a family of $m$ tests compares against $\\alpha/m$: once
    $m > B\\alpha$ -- which is $500$ families here at $B=10^4$ -- *no* test can survive the
    correction, and a board with many pairs would report a null that is a property of the resample
    count and not of the data. The floor was in place when this was written and did exactly that on
    a $1368$-test family. The intervals stay bootstrap; only the $p$ used for multiplicity is
    analytic, and the two agree on every family small enough for the floor not to bind.
    """
    n = len(d)
    se = float(np.std(d, ddof=1)) / np.sqrt(n)
    if se == 0.0:
        return 1.0
    z = abs(float(np.mean(d))) / se
    return float(math.erfc(z / np.sqrt(2.0)))


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


def _tiers(systems: list[str], edges: dict) -> int:
    """Longest chain in a strict partial order: how many places the order still distinguishes."""
    memo: dict = {}

    def chain(n):
        if n not in memo:
            memo[n] = 1 + max((chain(c) for c in edges.get(n, ())), default=0)
        return memo[n]

    return max(chain(n) for n in systems)


def analyse(hits: dict, systems: list[str], cells: list, published_cell,
            sub_grids: dict | None = None) -> dict:
    """hits[(system, cell)] -> 0/1 vector over items, aligned across systems.

    A pair is classified three ways rather than two. It *dominates* when the published ordering has
    a positive margin in every cell; it is *contested* when some cell puts the two systems the other
    way round with an interval excluding zero; and it is *unresolved* when neither holds, which is
    the benchmark failing to decide rather than a choice deciding against it. Reporting only the
    first would let sign noise on a tied pair count as evidence that a choice reversed something.
    """
    n = len(next(iter(hits.values())))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, (N_BOOT, n))

    published = sorted(systems, key=lambda s: -float(hits[(s, published_cell)].mean()))
    rank = {s: i for i, s in enumerate(published)}

    pairs, boot = {}, {}
    for a, b in itertools.combinations(systems, 2):
        # orient the pair the way the published cell orders it, so "survives" means what a reader
        # of that table would have concluded is still true in every other cell
        hi_, lo_ = (a, b) if rank[a] < rank[b] else (b, a)
        name = f"{hi_} over {lo_}"
        per_cell = {}
        for c in cells:
            d = hits[(hi_, c)] - hits[(lo_, c)]
            bt = d[idx].mean(axis=1)
            boot[(name, str(c))] = bt
            l, h = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            per_cell[str(c)] = {"margin": round(float(d.mean()), 4),
                                "ci95": [round(l, 4), round(h, 4)],
                                "positive": bool(d.mean() > 0),
                                "separated": bool(l > 0),
                                "separated_the_other_way": bool(h < 0)}
        dominates = all(v["positive"] for v in per_cell.values())
        separated = all(v["separated"] for v in per_cell.values())
        contested = any(v["separated_the_other_way"] for v in per_cell.values())
        flips = [k for k, v in per_cell.items() if not v["positive"]]
        pairs[name] = {
            "dominates": dominates, "separated_in_every_cell": separated,
            "contested": contested,
            "resolved_in_the_published_cell": per_cell[str(published_cell)]["separated"],
            "cells_that_reverse_it": flips,
            "cells_that_reverse_it_with_an_interval": [k for k, v in per_cell.items()
                                                       if v["separated_the_other_way"]],
            "cells_it_is_not_resolved_in": [k for k, v in per_cell.items()
                                            if v["positive"] and not v["separated"]],
            "per_cell": per_cell}

    # A contested verdict is read out of every pair by every cell, so it is read out of a family
    # whose size is the grid. On a seven-system board that is $168$ looks, and two reversals from
    # that many is close to what noise alone would give: an uncorrected interval is the wrong object
    # to hang the claim on. Holm is applied across the whole family and reported alongside, never
    # instead of, the uncorrected verdict -- correcting one board and not another, or redefining the
    # verdict so a count moves, would make the four rows incomparable.
    tests = [(name, str(c), v["per_cell"][str(c)]["margin"],
              _paired_p(hits[(name.split(" over ")[0], c)] - hits[(name.split(" over ")[1], c)]))
             for name, v in pairs.items() for c in cells]
    order = sorted(range(len(tests)), key=lambda i: tests[i][3])
    m, kept = len(tests), set()
    for step, i in enumerate(order):
        if tests[i][3] > ALPHA / (m - step):
            break
        kept.add(i)
    surviving = [tests[i] for i in sorted(kept)]
    corrected = sorted({nm for nm, _, margin, _ in surviving if margin < 0})
    for name, v in pairs.items():
        v["contested_after_correction"] = name in corrected

    n_pairs = len(pairs)
    n_dom = sum(v["dominates"] for v in pairs.values())
    n_sep = sum(v["separated_in_every_cell"] for v in pairs.values())
    n_con = sum(v["contested"] for v in pairs.values())
    n_con_corr = len(corrected)

    # The share of surviving pairs is the honest number and not the readable one. The readable one
    # is how many places the leaderboard still distinguishes: the longest chain in the dominance
    # order, which is the number of tiers a reader may quote. A total order of n places whose
    # longest robust chain is t is a table asserting n ranks and supporting t.
    edges: dict = {}
    for k, v in pairs.items():
        hi_, lo_ = k.split(" over ")
        if v["dominates"]:
            edges.setdefault(hi_, set()).add(lo_)
    tiers = _tiers(systems, edges)

    # The dual statistic: how many different total orders the same systems take across the grid.
    # Tiers say what is invariant, this says how much varies, and neither is recoverable from the
    # other.
    per_cell_orders = {str(c): tuple(sorted(systems, key=lambda s: -float(hits[(s, c)].mean())))
                       for c in cells}
    distinct = len(set(per_cell_orders.values()))

    # An interval on the share itself. Resampling items and recomputing the whole classification
    # says how much of the share is the sample rather than the systems; it says nothing about the
    # grid, which is what the sub-grid curve below is for.
    stack = np.stack([np.stack([boot[(p, str(c))] for c in cells]) for p in pairs])  # pair,cell,B
    dom_bt = (stack > 0).all(axis=1)                                                 # pair,B
    share_bt = dom_bt.mean(axis=0)
    share_ci = [round(float(np.quantile(share_bt, .025)), 4),
                round(float(np.quantile(share_bt, .975)), 4)]

    # The tier count is the headline in the unit a leaderboard is quoted in, so it gets the same
    # treatment as the share: rebuild the dominance relation on every resample and take the longest
    # chain again. A count with no interval invites the reader to treat an integer as exact when it
    # is a statistic of this sample.
    names = list(pairs)
    tier_bt = np.empty(N_BOOT, dtype=int)
    for b in range(N_BOOT):
        e: dict = {}
        for i, nm in enumerate(names):
            if dom_bt[i, b]:
                hi_, lo_ = nm.split(" over ")
                e.setdefault(hi_, set()).add(lo_)
        tier_bt[b] = _tiers(systems, e)
    tier_ci = [int(np.quantile(tier_bt, .025)), int(np.quantile(tier_bt, .975))]

    # Domination is an intersection over cells, so the share can only fall as cells are added. That
    # makes it a property of (leaderboard, grid) and not of the leaderboard, which is worth
    # measuring rather than conceding: each declared axis is reported alone and against the product.
    curve = {}
    for label, sub in (sub_grids or {}).items():
        keys = [str(c) for c in sub]
        d_ = sum(all(pairs[p]["per_cell"][k]["positive"] for k in keys) for p in pairs)
        s_ = sum(all(pairs[p]["per_cell"][k]["separated"] for k in keys) for p in pairs)
        e_: dict = {}
        for p in pairs:
            if all(pairs[p]["per_cell"][k]["positive"] for k in keys):
                hi_, lo_ = p.split(" over ")
                e_.setdefault(hi_, set()).add(lo_)
        curve[label] = {"n_cells": len(sub), "n_dominating": d_,
                        "share": round(d_ / max(n_pairs, 1), 4),
                        "n_separated_in_every_cell": s_,
                        "tiers": _tiers(systems, e_),
                        "distinct_orderings": len({per_cell_orders[k] for k in keys})}

    resolvable = [p for p, v in pairs.items() if v["resolved_in_the_published_cell"]]
    # Every derived quantity above is a function of this table, so releasing it is what makes the
    # share reconstructible rather than asserted.
    acc = {s: {str(c): round(float(hits[(s, c)].mean()), 4) for c in cells} for s in systems}
    return {"system_accuracy_by_cell": acc, "n_items": int(n),
            "published_order": published, "published_cell": str(published_cell),
            "n_systems": len(systems), "n_cells": len(cells), "n_pairs": n_pairs,
            "n_dominating": n_dom, "n_separated_in_every_cell": n_sep, "n_contested": n_con,
            "n_contested_after_correction": n_con_corr,
            "contested_after_correction": corrected,
            "multiplicity": {"family_size": m, "alpha": ALPHA,
                             "n_tests_surviving": len(surviving),
                             "n_reversals_surviving": sum(1 for _, _, mg, _ in surviving if mg < 0),
                             "largest_surviving_reversal_p":
                                 round(max((pv for _, _, mg, pv in surviving if mg < 0),
                                           default=float("nan")), 5)},
            "tiers_distinguished": tiers, "tiers_ci95": tier_ci,
            "distinct_orderings_across_the_grid": distinct,
            "robustness": round(n_dom / max(n_pairs, 1), 4),
            "robustness_ci95": share_ci,
            "separated_share": round(n_sep / max(n_pairs, 1), 4),
            "reversed_by_some_cell": n_pairs - n_dom,
            "reversed_with_an_interval": n_con,
            # A pair that neither dominates nor is contested is unresolved: the published ordering
            # fails somewhere on a margin no cell can separate. That is the third verdict, and the
            # three counts partition the pairs. The quantity below it is a different population --
            # pairs that dominate without being separated everywhere -- and was once named as though
            # it were this one, which put a false sentence in the appendix on three boards of four.
            "n_unresolved": n_pairs - n_dom - n_con,
            "unresolved": sorted(k for k, v in pairs.items()
                                 if not v["dominates"] and not v["contested"]),
            "dominating_but_not_separated_everywhere": n_dom - n_sep,
            "n_resolved_in_the_published_cell": len(resolvable),
            "robustness_among_resolved": round(
                sum(pairs[p]["dominates"] for p in resolvable) / max(len(resolvable), 1), 4),
            "sub_grids": curve, "per_cell_orders": {k: list(v) for k, v in per_cell_orders.items()},
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
    published_cell = ("canonical", 1)
    sub_grids = {"criteria only, at the published budget": [(m, published_cell[1]) for m in MODES],
                 "budgets only, at the published criterion": [(published_cell[0], k) for k in KS],
                 "the product": cells}
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
    return analyse(hits, systems, cells, published_cell, sub_grids)


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
              f"= {r['robustness']} {r['robustness_ci95']}")
        print(f"  and separated in every cell:   {r['n_separated_in_every_cell']}/{r['n_pairs']}")
        print(f"  reversed with an interval:     {r['reversed_with_an_interval']}")
        print(f"  reversed on sign alone:        "
              f"{r['reversed_by_some_cell'] - r['reversed_with_an_interval']}")
        print(f"  unresolved, neither way:       {r['n_unresolved']}")
        print(f"  tiers it still distinguishes:  {r['tiers_distinguished']} {r['tiers_ci95']} "
              f"of {r['n_systems']} published places")
        print(f"  distinct orderings on the grid:{r['distinct_orderings_across_the_grid']} "
              f"of {r['n_cells']} cells")
        print(f"  among pairs its own cell resolves: {r['robustness_among_resolved']} "
              f"({r['n_resolved_in_the_published_cell']} pairs)")
        for lab, v in r["sub_grids"].items():
            print(f"    {lab:44} {v['n_cells']:2} cells -> {v['n_dominating']}/{r['n_pairs']} "
                  f"= {v['share']}, {v['tiers']} tiers, {v['distinct_orderings']} orderings")

    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

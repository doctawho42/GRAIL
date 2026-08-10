#!/usr/bin/env python3
"""Which published margins were resolvable at all, and which were never distinguishable?

A criterion that permutes a leaderboard is only interesting if the leaderboard was measuring
something. The seven retrosynthesis systems sit a median 0.013 apart at top-1 on 4,999 reactions,
and the question a reader is entitled to ask is whether 0.013 is a difference or a coin flip.

It is answerable without retraining anything, because the predictions are frozen. For each pair of
systems the difference in top-1 accuracy is a paired quantity over the reactions they share, so a
paired bootstrap gives an interval on the margin itself. A margin whose interval covers zero was
never resolvable by that benchmark at that sample size, whatever any table reports.

Reported per leaderboard:

    the margin between every pair, its paired interval, and whether the pair was ever separable
    the resolution floor: the smallest margin the benchmark can certify at this sample size
    how many of the published adjacent margins fall below it

This is a statement about the benchmark, not about the criterion. It sharpens the paper's claim
instead of weakening it: where a margin is below the floor, an undeclared convention is one of
several things that will permute the order, and the permutation is not evidence that either system
is better.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retro_leaderboard import MODES, _code_version, build_keys, set_key

N_BOOT, SEED = 10000, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "grail_metabolism" / "data" / "evalretro"))
    ap.add_argument("--clusters", default="cluster0,cluster1")
    ap.add_argument("--mode", default="canonical", choices=MODES,
                    help="the criterion the published tables are read under")
    ap.add_argument("--k", type=int, default=1,
                    help="the budget the published tables lead with")
    ap.add_argument("--max-rank", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "resolution_floor.json"))
    args = ap.parse_args()

    ingest = json.loads((ROOT / "results/evalretro_ingest.json").read_text())
    rep = {"config": {**_code_version(), "criterion": args.mode, "k": args.k, "n_boot": N_BOOT, "seed": SEED,
                      "note": "paired bootstrap over reactions; predictions frozen, nothing trained"},
           "leaderboards": {}}

    for cname in args.clusters.split(","):
        meta = ingest["clusters"][cname]
        systems, rows = meta["systems"], list(csv.DictReader(open(ROOT / meta["test_csv"])))
        preds = {n: json.loads((Path(args.dir) / "normalised" / f"{n}.json").read_text())
                 for n in systems}
        comps = {c for r in rows for c in r["REACTANT"].split(".") if c}
        for n in systems:
            for p in preds[n]:
                for s in p["preds"][: args.max_rank]:
                    comps.update(c for c in s.split(".") if c)
        cache_path = Path(args.dir) / f"keys_{cname}_r{args.max_rank}.json"
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        todo = [c for c in sorted(comps) if c not in cached]
        if todo:
            cached.update(build_keys(todo, args.workers))
            cache_path.write_text(json.dumps(cached))
        keys = cached

        hit = {n: np.zeros(len(rows)) for n in systems}
        for j, row in enumerate(rows):
            truth = set_key(row["REACTANT"], args.mode, keys)
            if truth is None:
                continue
            for n in systems:
                for s in preds[n][j]["preds"][: args.k]:
                    if set_key(s, args.mode, keys) == truth:
                        hit[n][j] = 1.0
                        break

        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))
        acc = {n: float(hit[n].mean()) for n in systems}
        order = sorted(systems, key=lambda n: -acc[n])
        print(f"\n{cname}: {len(systems)} systems, {len(rows)} reactions, {args.mode}, "
              f"top-{args.k}")
        for n in order:
            print(f"    {n:16} {acc[n]:.4f}")

        pairs, unresolved = {}, 0
        for a, b in itertools.combinations(order, 2):
            d = hit[a] - hit[b]
            bt = d[idx].mean(axis=1)
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            sep = lo * hi > 0
            unresolved += not sep
            pairs[f"{a} vs {b}"] = {"margin": round(float(d.mean()), 4),
                                    "ci95": [round(lo, 4), round(hi, 4)],
                                    "separable": bool(sep)}
        # The floor: the smallest margin this benchmark separates from zero at this sample size.
        sepd = [abs(v["margin"]) for v in pairs.values() if v["separable"]]
        floor = min(sepd) if sepd else None
        adj = [f"{order[i]} vs {order[i+1]}" for i in range(len(order) - 1)]
        adj_unres = [k for k in adj if not pairs[k]["separable"]]
        print(f"  {unresolved} of {len(pairs)} pairs are not separable at 95%")
        print(f"  resolution floor: {floor}")
        print(f"  adjacent pairs in the published order that are not separable: "
              f"{len(adj_unres)} of {len(adj)}")
        for k in adj_unres:
            print(f"      {k:36} {pairs[k]['margin']:+.4f} {pairs[k]['ci95']}")

        rep["leaderboards"][cname] = {
            "systems": order, "n_reactions": len(rows), "accuracy": {n: round(acc[n], 4) for n in order},
            "pairs": pairs, "n_pairs": len(pairs), "not_separable": unresolved,
            "resolution_floor": floor, "adjacent_pairs": len(adj),
            "adjacent_not_separable": len(adj_unres), "adjacent_unresolved": adj_unres}

    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

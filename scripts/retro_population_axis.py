#!/usr/bin/env python3
"""Does the population reorder a leaderboard, on files their authors released?

The population axis is this paper's thinnest, and it was thin for a reason: varying it needs two
populations of one split with everything else pinned, and we had only our own. The eleven released
retrosynthesis files supply a second instance, and a better one, because nothing in it is ours.

The eleven sit on three test sets that share about a tenth of their reactions. That overlap is
usually a nuisance. Here it is the experiment: the reactions two clusters share form a population
nested inside each of their own, so for any pair of systems in one cluster the gap can be measured
on the cluster's whole set and on the shared subset, with the method, the criterion and the budget
all held fixed. Only which reactions are in changes.

It also answers a question no published table can. The clusters have no system in common, so no
number anywhere compares a system in one against a system in another; on the shared reactions all
of them are comparable at once, for the first time.

Two things are checked before any of that is believed:

  the ground truth agrees   the same product in two files must carry the same recorded reaction.
                            Where it does not, the two files disagree about the answer and not
                            merely about the sample, which is a different defect and is counted
                            and excluded rather than averaged over.

  the subset is nested      every shared reaction is one of the cluster's own, so the comparison
                            is a restriction and not an independent draw, and is reported as one.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from retro_leaderboard import (KS, MODES, _code_version, build_keys, set_key)

N_BOOT, SEED = 10000, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "grail_metabolism" / "data" / "evalretro"))
    ap.add_argument("--clusters", default="cluster0,cluster1")
    ap.add_argument("--max-rank", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "retro_population_axis.json"))
    args = ap.parse_args()
    names = args.clusters.split(",")

    ingest = json.loads((ROOT / "results/evalretro_ingest.json").read_text())
    rows, preds, systems = {}, {}, {}
    for c in names:
        meta = ingest["clusters"][c]
        systems[c] = meta["systems"]
        rows[c] = list(csv.DictReader(open(ROOT / meta["test_csv"])))
        for n in systems[c]:
            p = json.loads((Path(args.dir) / "normalised" / f"{n}.json").read_text())
            if len(p) != len(rows[c]) or any(a["product"] != b["PRODUCT"]
                                             for a, b in zip(p, rows[c])):
                raise SystemExit(f"{n}: predictions are not aligned to {c}")
            preds[n] = p
        print(f"{c}: {len(systems[c])} systems, {len(rows[c])} reactions", flush=True)

    every = sorted({c for cl in names for r in rows[cl]
                    for c in (r["REACTANT"] + "." + r["PRODUCT"]).split(".") if c}
                   | {c for cl in names for n in systems[cl] for p in preds[n]
                      for s in p["preds"][: args.max_rank] for c in s.split(".") if c})
    # The per-cluster runs already keyed most of these, and tautomer canonicalisation is the
    # expensive step in this codebase; seeding from their caches makes this run minutes rather
    # than an hour, and the keys are the same function of the same string either way.
    cache_path = Path(args.dir) / f"keys_population_r{args.max_rank}.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if not cached:
        for c in names:
            seed = Path(args.dir) / f"keys_{c}_r{args.max_rank}.json"
            if seed.exists():
                cached.update(json.loads(seed.read_text()))
                print(f"  seeded {len(cached)} keys from {seed.name}", flush=True)
    todo = [c for c in every if c not in cached]
    print(f"  {len(every)} distinct components; keying {len(todo)}", flush=True)
    if todo:
        cached.update(build_keys(todo, args.workers))
        cache_path.write_text(json.dumps(cached))
    keys = cached

    def pkey(smiles):
        k = keys.get(smiles)
        return k["canonical"] if k else None

    # Which reactions the two clusters share, matched on the product they are keyed by.
    index = {c: {} for c in names}
    for c in names:
        for j, r in enumerate(rows[c]):
            k = pkey(r["PRODUCT"])
            if k is not None:
                index[c].setdefault(k, j)
    shared_products = sorted(set.intersection(*(set(index[c]) for c in names)))

    # The gate: the same product must carry the same recorded reaction in both files. A product
    # that does not is a disagreement about the answer, not about the sample.
    agree, disagree = [], []
    for k in shared_products:
        truths = {c: set_key(rows[c][index[c][k]]["REACTANT"], "tautomer", keys) for c in names}
        vals = list(truths.values())
        (agree if all(v is not None and v == vals[0] for v in vals) else disagree).append(k)
    print(f"\n  {len(shared_products)} products in both; {len(agree)} carry the same recorded "
          f"reaction, {len(disagree)} do not and are dropped", flush=True)
    if not agree:
        raise SystemExit("the two files share products but agree on no reaction")

    def score(cluster, product_keys, sysnames):
        """hit[system][mode][k] over the given reactions of one cluster."""
        idx = [index[cluster][k] for k in product_keys]
        hit = {n: {m: {k: np.zeros(len(idx)) for k in KS} for m in MODES} for n in sysnames}
        for j2, j in enumerate(idx):
            truth = {m: set_key(rows[cluster][j]["REACTANT"], m, keys) for m in MODES}
            for n in sysnames:
                pk = {m: [] for m in MODES}
                for s in preds[n][j]["preds"][: args.max_rank]:
                    for m in MODES:
                        pk[m].append(set_key(s, m, keys))
                for m in MODES:
                    if truth[m] is None:
                        continue
                    where = next((i for i, v in enumerate(pk[m]) if v == truth[m]), None)
                    for k in KS:
                        hit[n][m][k][j2] = 1.0 if where is not None and where < k else 0.0
        return hit

    own = {c: score(c, list(index[c]), systems[c]) for c in names}
    sub = {c: score(c, agree, systems[c]) for c in names}
    # The shared reactions are nested inside each cluster's own set, so a gap measured on the
    # subset and a gap measured on the whole are not independent and their difference has no
    # honest interval. The complement -- the cluster's own reactions that the other cluster does
    # not have -- is disjoint from the subset, so that contrast does. The reorderings are described
    # against the whole set, because that is the number a reader of the published table has; the
    # interaction is certified against the complement, because that is the one that can be.
    rest = {c: [k for k in index[c] if k not in set(agree)] for c in names}
    comp = {c: score(c, rest[c], systems[c]) for c in names}
    print(f"  complement sizes: " + ", ".join(f"{c}={len(rest[c])}" for c in names), flush=True)

    rng = np.random.default_rng(SEED)
    draws = {c: (rng.integers(0, len(agree), (N_BOOT, len(agree))),
                 rng.integers(0, len(rest[c]), (N_BOOT, len(rest[c])))) for c in names}

    def interaction(c, a_, b_, m, k):
        """gap on the shared reactions minus gap on the cluster's remaining ones."""
        ds = sub[c][a_][m][k] - sub[c][b_][m][k]
        dc = comp[c][a_][m][k] - comp[c][b_][m][k]
        point = float(ds.mean() - dc.mean())
        si, ci = draws[c]
        bt = ds[si].mean(axis=1) - dc[ci].mean(axis=1)
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        pv = 2.0 * min((bt <= 0).mean(), (bt >= 0).mean())
        return {"delta": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "p": round(max(float(pv), 1.0 / N_BOOT), 6)}

    # The axis: one pair of systems, one criterion, one budget, two populations.
    rowsout, flips, ties = [], 0, 0
    for c in names:
        for a, b in itertools.combinations(systems[c], 2):
            for m in MODES:
                for k in KS:
                    g_own = own[c][a][m][k].mean() - own[c][b][m][k].mean()
                    g_sub = sub[c][a][m][k].mean() - sub[c][b][m][k].mean()
                    if g_own == 0 or g_sub == 0:
                        ties += 1
                        continue
                    flip = bool((g_own > 0) != (g_sub > 0))
                    flips += int(flip)
                    inter = interaction(c, a, b, m, k)
                    rowsout.append({"cluster": c, "pair": f"{a} vs {b}", "criterion": m, "k": k,
                                    "gap_own_set": round(float(g_own), 4),
                                    "gap_shared": round(float(g_sub), 4),
                                    "reordered": bool(flip), "interaction": inter})
    print(f"\n  {len(rowsout)} comparisons, method and criterion and budget all held fixed")
    print(f"  the ordering changes with the population in {flips} of them")
    for r in rowsout:
        if r["reordered"]:
            print(f"    {r['cluster']} {r['criterion']:10} k={r['k']:<3} {r['pair']:32} "
                  f"{r['gap_own_set']:+.4f} -> {r['gap_shared']:+.4f}")

    # A null is only a result if it states what it could have detected. The bootstrap gives each
    # interaction's standard error directly, so the smallest effect this design would have rejected
    # at the family's own Holm threshold, with 80% power, is a property of the design and not of
    # the data. Reported beside the null it turns "we found nothing" into a bound.
    from math import sqrt
    try:
        from scipy.stats import norm
        zc = float(norm.ppf(1 - 0.05 / (2 * max(len(rowsout), 1))))
        zp = float(norm.ppf(0.80))
    except Exception:
        zc, zp = 3.9, 0.8416
    ses = []
    for c in names:
        for a_, b_ in itertools.combinations(systems[c], 2):
            for m in MODES:
                for k in KS:
                    ds = sub[c][a_][m][k] - sub[c][b_][m][k]
                    dc = comp[c][a_][m][k] - comp[c][b_][m][k]
                    si, ci = draws[c]
                    bt = ds[si].mean(axis=1) - dc[ci].mean(axis=1)
                    ses.append(float(bt.std(ddof=1)))
    ses.sort()
    mde = [(zc + zp) * se for se in ses]
    rep_mde = {"median": round(mde[len(mde) // 2], 4), "smallest": round(mde[0], 4),
               "largest": round(mde[-1], 4), "power": 0.80,
               "alpha_per_test": round(0.05 / max(len(rowsout), 1), 8)}
    print(f"\n  minimum detectable interaction at the family's Holm threshold, 80% power: "
          f"median {rep_mde['median']:.4f}, range {rep_mde['smallest']:.4f} to "
          f"{rep_mde['largest']:.4f}")

    # Holm across the whole family, at the correction this paper applies to its other one.
    ordered = sorted(rowsout, key=lambda r: r["interaction"]["p"])
    n_fam, survivors = len(ordered), []
    for i, r in enumerate(ordered):
        thr = 0.05 / (n_fam - i)
        r["holm_threshold"] = round(thr, 8)
        if r["interaction"]["p"] <= thr:
            r["survives_holm"] = True
            survivors.append(r)
        else:
            r["survives_holm"] = False
            break
    for r in ordered[len(survivors):]:
        r.setdefault("survives_holm", False)
        r.setdefault("holm_threshold", round(0.05 / max(n_fam - ordered.index(r), 1), 8))
    excl = [r for r in rowsout if r["interaction"]["ci95"][0] * r["interaction"]["ci95"][1] > 0]
    print(f"\n  of {n_fam} interactions, {len(excl)} have intervals excluding zero and "
          f"{len(survivors)} survive Holm at 0.05 across the family")
    for r in survivors[:8]:
        i = r["interaction"]
        print(f"    {r['cluster']} {r['criterion']:10} k={r['k']:<3} {r['pair']:32} "
              f"{i['delta']:+.4f} {i['ci95']}")

    # The by-product: ten systems on one population, which no released table can state.
    joint = {}
    for c in names:
        for n in systems[c]:
            joint[n] = {m: {k: round(float(sub[c][n][m][k].mean()), 4) for k in KS} for m in MODES}
    order = sorted(joint, key=lambda n: -joint[n]["canonical"][1])
    print(f"\n  all {len(joint)} systems on the {len(agree)} shared reactions, top-1 canonical")
    for n in order:
        print(f"    {n:16} {joint[n]['canonical'][1]:.4f}   "
              f"(its own cluster's set: {own[[c for c in names if n in systems[c]][0]][n]['canonical'][1].mean():.4f})")

    rep = {"config": {**_code_version(), "clusters": names, "max_rank": args.max_rank,
                      "criteria": list(MODES), "budgets": list(KS),
                      "population": "the reactions the clusters share, nested inside each",
                      "note": "a restriction of each cluster's own set, not an independent draw; "
                              "products whose recorded reaction differs between files are dropped"},
           "products_in_both": len(shared_products), "agreeing_reactions": len(agree),
           "dropped_disagreeing": len(disagree),
           "comparisons": len(rowsout), "reordered": int(flips), "ties_skipped": int(ties),
           "interactions_excluding_zero": len(excl), "holm_survivors": len(survivors),
           "minimum_detectable_interaction": rep_mde,
           "rows": rowsout, "joint_leaderboard": joint,
           "systems_per_cluster": systems}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Does the population reorder a leaderboard, with the method, criterion and budget held fixed?

Three of the four undeclared choices this paper names are varied with everything else fixed. The
fourth -- which population a familiar dataset name refers to -- is demonstrated rather than varied:
eleven published retrosynthesis files turn out to sit on three test sets, which shows the hazard
exists but is a fact about someone else's release rather than a measurement of ours.

It is measurable here, because the same three methods are scored under the same five criteria, by
the same estimator, on two populations of this paper's own split: the 150 substrates all five
methods share and the full 1,170. Nothing varies between them except which substrates are in.

Reported per method pair and criterion:

    gap on each population, the change between them, and whether the ordering survives

A reordering is a description and not a result, and this script used to stop at the description.
The shared subset is nested inside the full split, so a gap on one and a gap on the other are not
independent and their difference carries no honest interval. The complement -- the substrates the
full split has and the shared subset does not -- is disjoint from the subset, so that contrast does.
The reorderings are reported against the full split, because that is the number a reader has; the
interaction is certified against the complement, because that is the one that can be, and it is
corrected across the whole family at the same Holm threshold this paper applies elsewhere.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from set_metrics_by_criterion import MODES, per_substrate

SRC = ROOT / "results" / "set_metrics_by_criterion.json"
METRICS = ("recall", "f1", "precision", "jaccard")
N_BOOT, SEED = 10000, 0


def _vectors():
    """Per-substrate scores for the methods both populations carry, on the subset and its
    complement. Read from the same prediction files the artifact was built from, because the
    artifact stores only point estimates and an interval cannot be recovered from those."""
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    grail = {r["sub"]: r["deployed_top15"] for r in
             json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]}
    methods = {
        "GRAIL": grail,
        "SyGMa": json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text()),
        "MetaPredictor": json.loads(
            (ROOT / "artifacts" / "tier2_1170" / "metapredictor_preds.json").read_text()),
    }
    five = dict(methods)
    t2 = ROOT / "artifacts" / "tier2"
    for n, f in (("BioTransformer", "biotransformer_preds.json"),
                 ("MetaTrans", "metatrans_preds.json")):
        five[n] = json.loads((t2 / f).read_text())
    full = sorted(set.intersection(*(set(m) for m in methods.values())) & set(truth))
    full = [s for s in full if truth[s]]
    shared = sorted(set.intersection(*(set(m) for m in five.values())) & set(truth))
    shared = [s for s in shared if truth[s]]
    rest = [s for s in full if s not in set(shared)]
    out = {}
    for mode in MODES:
        table = json.loads((ROOT / "results" / "key_tables" / f"{mode}.json").read_text())
        out[mode] = {"shared": {n: per_substrate(p, truth, shared, table)
                                for n, p in methods.items()},
                     "rest": {n: per_substrate(p, truth, rest, table) for n, p in methods.items()}}
    return out, len(shared), len(rest)


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
    ap.add_argument("--out", default=str(ROOT / "results" / "population_axis.json"))
    args = ap.parse_args()

    d = json.loads(SRC.read_text())["populations"]
    pops = sorted(d, key=lambda p: d[p]["n_substrates"])
    if len(pops) < 2:
        raise SystemExit("two populations are needed and the artifact holds fewer")
    small, big = pops[0], pops[-1]
    methods = sorted(set(d[small]["methods"]) & set(d[big]["methods"]))
    modes = sorted(set(d[small]["by_mode"]) & set(d[big]["by_mode"]))
    print(f"{small} (n={d[small]['n_substrates']}) against {big} (n={d[big]['n_substrates']})")
    print(f"  methods on both: {methods}\n  criteria on both: {modes}", flush=True)

    vec, n_shared, n_rest = _vectors()
    print(f"  certifying against the complement: {n_shared} shared, {n_rest} remaining", flush=True)
    rng = np.random.default_rng(SEED)
    si = rng.integers(0, n_shared, (N_BOOT, n_shared))
    ci = rng.integers(0, n_rest, (N_BOOT, n_rest))

    def interaction(mode, a_, b_, metric):
        ds = vec[mode]["shared"][a_][metric] - vec[mode]["shared"][b_][metric]
        dc = vec[mode]["rest"][a_][metric] - vec[mode]["rest"][b_][metric]
        bt = ds[si].mean(axis=1) - dc[ci].mean(axis=1)
        pv = 2.0 * min((bt <= 0).mean(), (bt >= 0).mean())
        return {"delta": round(float(ds.mean() - dc.mean()), 4),
                "ci95": [round(float(np.quantile(bt, .025)), 4),
                         round(float(np.quantile(bt, .975)), 4)],
                "p": round(max(float(pv), 1.0 / N_BOOT), 6)}

    rows, flips, ties = [], 0, 0
    for metric in METRICS:
        for mode in modes:
            for a, b in itertools.combinations(methods, 2):
                def g(pop):
                    A = d[pop]["by_mode"][mode][a][metric]["point"]
                    B = d[pop]["by_mode"][mode][b][metric]["point"]
                    return A - B
                gs, gb = g(small), g(big)
                if gs == 0 or gb == 0:
                    ties += 1
                    continue
                flip = bool((gs > 0) != (gb > 0))
                flips += int(flip)
                rows.append({"metric": metric, "criterion": mode, "pair": f"{a} vs {b}",
                             f"gap_{small}": round(gs, 4), f"gap_{big}": round(gb, 4),
                             "change": round(abs(gb - gs), 4), "reordered": flip,
                             "interaction": interaction(mode, a, b, metric)})

    print(f"\n  {len(rows)} comparisons, method and criterion and budget all held fixed")
    print(f"  the ordering changes with the population in {flips} of them")
    for r in rows:
        if r["reordered"]:
            print(f"    {r['metric']:9} {r['criterion']:18} {r['pair']:28} "
                  f"{r[f'gap_{small}']:+.4f} -> {r[f'gap_{big}']:+.4f}")

    # What this design could have detected, so the null is a bound and not an absence.
    try:
        from scipy.stats import norm
        zc = float(norm.ppf(1 - 0.05 / (2 * max(len(rows), 1))))
        zp = float(norm.ppf(0.80))
    except Exception:
        zc, zp = 3.5, 0.8416
    ses = []
    for r in rows:
        a_, b_ = r["pair"].split(" vs ")
        ds = vec[r["criterion"]]["shared"][a_][r["metric"]] - vec[r["criterion"]]["shared"][b_][r["metric"]]
        dc = vec[r["criterion"]]["rest"][a_][r["metric"]] - vec[r["criterion"]]["rest"][b_][r["metric"]]
        ses.append(float((ds[si].mean(axis=1) - dc[ci].mean(axis=1)).std(ddof=1)))
    ses.sort()
    mde = sorted((zc + zp) * x for x in ses)
    mde_rep = {"median": round(mde[len(mde) // 2], 4), "smallest": round(mde[0], 4),
               "largest": round(mde[-1], 4), "power": 0.80}
    print(f"\n  minimum detectable interaction at the family's Holm threshold, 80% power: "
          f"median {mde_rep['median']:.4f}, range {mde_rep['smallest']:.4f} to {mde_rep['largest']:.4f}")

    ordered = sorted(rows, key=lambda r: r["interaction"]["p"])
    survivors = []
    for i, r in enumerate(ordered):
        if r["interaction"]["p"] <= 0.05 / (len(ordered) - i):
            survivors.append(r)
        else:
            break
    excl = [r for r in rows if r["interaction"]["ci95"][0] * r["interaction"]["ci95"][1] > 0]
    print(f"\n  of {len(rows)} interactions against the complement, {len(excl)} have intervals "
          f"excluding zero and {len(survivors)} survive Holm at 0.05")

    med = sorted(r["change"] for r in rows)
    print(f"\n  median change in a pair's gap between the two populations: "
          f"{med[len(med) // 2]:.4f}")
    by_metric = {m: {"comparisons": sum(1 for r in rows if r["metric"] == m),
                     "reordered": sum(1 for r in rows if r["metric"] == m and r["reordered"])}
                 for m in METRICS}
    for m, v in by_metric.items():
        print(f"    {m:10} {v['reordered']} of {v['comparisons']} reorder")

    rep = {"config": {**_code_version(), "source": str(SRC.relative_to(ROOT)),
                      "populations": {p: d[p]["n_substrates"] for p in (small, big)},
                      "methods": methods, "criteria": modes, "metrics": list(METRICS),
                      "note": "the same methods, criteria and estimator on two populations of one "
                              "split; two populations is a weak design and is not offered as a "
                              "general claim"},
           "comparisons": len(rows), "reordered": int(flips), "ties_skipped": int(ties),
           "n_shared": n_shared, "n_complement": n_rest,
           "interactions_excluding_zero": len(excl), "holm_survivors": len(survivors),
           "minimum_detectable_interaction": mde_rep,
           "median_gap_change": round(med[len(med) // 2], 4),
           "by_metric": by_metric, "rows": rows}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

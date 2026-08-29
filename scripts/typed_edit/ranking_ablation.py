#!/usr/bin/env python3
"""What each learned stage contributes to the order, on the configuration that ships.

The pipeline has two learned scorers and the deployed ranking is a reciprocal rank fusion of
their two orderings. Neither stage's contribution to the ORDER has been reported: the paper shows
that the fusion beats the product of the same two scores, and that it beats an untrained
similarity ordering, but not what happens if either scorer is dropped.

Every arm re-ranks one pool, so the pool, the matching rule and the budget are fixed and only the
order varies. No model runs: the pools already carry both component scores per candidate.

    fusion          the deployed reciprocal rank fusion of the two orderings
    filter          the pair filter's score alone
    generator       the rule score alone, which is also the order the pool cap is applied in
    product         filter times generator, the combination the fusion replaced
    random          a seeded permutation, for what the pool alone is worth

An earlier ablation of these stages ran on a 245-substrate subsample under the superseded
ordering; this is the deployed configuration on the populations the paper reports.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

from bank_without_selection import _key as _tautkey  # noqa: E402

KS = (1, 5, 10, 15, 30, 50)
CAP = 100
N_BOOT, SEED = 10000, 0
ARMS = ("fusion", "filter", "generator", "product", "random")


def load(spec):
    pools, refs = {}, {}
    for f in sorted(glob.glob(spec)) or [spec]:
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    return pools, refs


def run(pools, refs, subs, label):
    rng = np.random.default_rng(SEED)
    hits = {a: {k: [] for k in KS} for a in ARMS}
    U = []
    for s in subs:
        real = set(refs[s])
        U.append(len(real))
        keep = sorted(pools[s], key=lambda c: -c["generator"])[:CAP]
        orders = {
            "fusion": rrf_order(keep),
            "filter": sorted(keep, key=lambda c: -c["filter"]),
            "generator": sorted(keep, key=lambda c: -c["generator"]),
            "product": sorted(keep, key=lambda c: -(c["filter"] * c["generator"])),
            "random": [keep[i] for i in rng.permutation(len(keep))],
        }
        # The comparison table drops a prediction equal to the substrate before the budget bites,
        # for every method alike. An arm scored without that convention is not on the same axis:
        # omitting it here moved recall in both directions against Table 2, because two of these
        # 291 substrates carry their own key among their references.
        pk = _tautkey(s)
        for a, order in orders.items():
            keys = [c["key"] for c in order if c.get("key") and c["key"] != pk]
            for k in KS:
                hits[a][k].append(len(real & set(keys[:k])))

    U = np.array(U, dtype=float)
    N = float(U.sum())
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    den = np.maximum(U[idx].sum(axis=1), 1)

    def contrast(a, b, k):
        d = np.array(hits[a][k], dtype=float) - np.array(hits[b][k], dtype=float)
        bt = d[idx].sum(axis=1) / den
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(d.sum() / N), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "separates": bool(lo > 0 or hi < 0)}

    rec = {a: {str(k): round(float(np.array(hits[a][k]).sum() / N), 4) for k in KS} for a in ARMS}
    return {"population": {"n": len(subs), "n_references": N, "source": label},
            "recall_micro": rec,
            "fusion_minus": {a: {str(k): contrast("fusion", a, k) for k in KS}
                             for a in ARMS if a != "fusion"},
            "single_scorers": {"filter_minus_generator":
                               {str(k): contrast("filter", "generator", k) for k in KS}}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/ranking_ablation.json"))
    args = ap.parse_args()

    out = {}
    for label, spec in (("comparison set", "results/widepools_implicit/w*.json"),
                        ("validation draw", "results/val_pools.json")):
        path = spec if "*" in spec else str(ROOT / spec)
        if not (glob.glob(path) or Path(path).exists()):
            print(f"  {label}: no pool at {spec}, skipped", file=sys.stderr)
            continue
        pools, refs = load(path)
        subs = sorted(s for s in pools if refs.get(s))
        print(f"  {label}: {len(subs)} substrates", file=sys.stderr, flush=True)
        out[label] = run(pools, refs, subs, label)

    rep = {"provenance": stamp(__file__), "aggregation": "micro, ratio of sums",
           "cap": CAP, "n_boot": N_BOOT, "seed": SEED, "arms": list(ARMS),
           "note": ("every arm re-ranks one pool built by the deployed configuration, so pool, "
                    "matching and budget are fixed and only the order varies; the pool cap is "
                    "applied by generator score before any arm sees the pool, which is what the "
                    "deployed system does"),
           "by_population": out}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    for label, r in out.items():
        print(f"\n## {label}  (n={r['population']['n']})")
        print(f"{'k':>4}" + "".join(f"{a:>12}" for a in ARMS))
        for k in KS:
            print(f"{k:>4}" + "".join(f"{r['recall_micro'][a][str(k)]:>12.4f}" for a in ARMS))
        print("  fusion minus each arm at k=15:")
        for a in ARMS[1:]:
            c = r["fusion_minus"][a]["15"]
            print(f"    vs {a:<11}{c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
                  f"{'  *' if c['separates'] else ''}")
        c = r["single_scorers"]["filter_minus_generator"]["15"]
        print(f"  filter minus generator at k=15: {c['gap']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]{'  *' if c['separates'] else ''}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Does the learned rule choice beat an unlearned one of the same size?

Stage one scores every template against the substrate and the interactive mode applies the best
thirty. The rule-budget prediction compared thirty against the whole bank, which asks about size.
This asks about choice: four ways to pick thirty rules, with the application, the standardisation,
the filter and the deployed rank fusion identical downstream.

    learned             the generator's top thirty for this substrate, which ships
    prior_applicable    among the rules that CAN fire here, the thirty with the highest
                        training-frequency log-odds; the non-learned baseline
    random_applicable   thirty drawn from the same applicable set
    random              thirty drawn from the whole bank, which mostly do not fire

The comparison that decides is learned against prior_applicable. If it does not separate, the
substrate-specific part of stage one is reproducing a frequency table.
"""
from __future__ import annotations

import argparse
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
N_BOOT, SEED = 10000, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "results/selection_pools_deployed.json"))
    ap.add_argument("--learned", default=str(ROOT / "results/widepools_k30/all.json"))
    ap.add_argument("--out", default=str(ROOT / "results/selection_ablation_deployed.json"))
    args = ap.parse_args()

    d = json.loads(Path(args.pools).read_text())
    learned = json.loads(Path(args.learned).read_text())["pools"]
    refs = d["references"]
    arms = ["learned"] + list(d["arms"])
    pools = {"learned": learned, **d["pools"]}
    subs = sorted(s for s in d["pools"][d["arms"][0]] if refs.get(s) and s in learned)
    print(f"{len(subs)} substrates, arms {arms}", file=sys.stderr)

    hits = {a: {k: [] for k in KS} for a in arms}
    sizes = {a: [] for a in arms}
    U = []
    for s in subs:
        real = set(refs[s])
        U.append(len(real))
        pk = _tautkey(s)
        for a in arms:
            pool = pools[a].get(s, [])
            sizes[a].append(len(pool))
            # the deployed convention: cap by rule score first, then fuse. On a pool smaller than
            # the cap this truncates nothing, but it fixes the order the fusion sees and therefore
            # how its ties break. Fusing the pool as stored instead moves micro recall at k=1 by
            # 0.0015, because 14 of these substrates have two candidates with identical fused
            # scores at the top. Both arms must use one convention or the difference is the
            # convention rather than the choice of rules.
            keep = sorted(pool, key=lambda c: -c["generator"])[:100]
            order = rrf_order(keep) if keep else []
            # and the comparison table's parent-drop convention, so this arm is on the same axis
            # as Table 2, which it is read against
            keys = [c["key"] for c in order if c.get("key") and c["key"] != pk]
            for k in KS:
                hits[a][k].append(len(real & set(keys[:k])))

    U = np.array(U, dtype=float)
    N = float(U.sum())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    den = np.maximum(U[idx].sum(axis=1), 1)

    def contrast(a, b, k):
        x = np.array(hits[a][k], dtype=float) - np.array(hits[b][k], dtype=float)
        bt = x[idx].sum(axis=1) / den
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(x.sum() / N), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "separates": bool(lo > 0 or hi < 0)}

    rec = {a: {str(k): round(float(np.array(hits[a][k]).sum() / N), 4) for k in KS} for a in arms}
    rep = {
        "provenance": stamp(__file__),
        "population": {"n": len(subs), "n_references": N,
                       "source": "the comparison set of results/four_method_291.json"},
        "aggregation": "micro, ratio of sums", "budget": d["budget"], "seed": SEED,
        "mean_applicable_rules": d["mean_applicable_rules"],
        "note": ("four ways to choose the same number of rules, with application, standardisation, "
                 "filter and the deployed rank fusion identical downstream; only the choice varies"),
        "mean_pool": {a: round(float(np.mean(sizes[a])), 1) for a in arms},
        "recall_micro": rec,
        "learned_minus": {a: {str(k): contrast("learned", a, k) for k in KS}
                          for a in arms if a != "learned"},
        "reading": ("learned against prior_applicable is the comparison that decides; the two "
                    "random arms bound what applicability alone and the bank alone are worth"),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n{'k':>4}" + "".join(f"{a:>20}" for a in arms))
    for k in KS:
        print(f"{k:>4}" + "".join(f"{rec[a][str(k)]:>20.4f}" for a in arms))
    print("\nmean pool:", rep["mean_pool"])
    print("\nlearned minus each arm:")
    for a in arms[1:]:
        cells = "  ".join(f"k{k}:{rep['learned_minus'][a][str(k)]['gap']:+.4f}"
                          f"{'*' if rep['learned_minus'][a][str(k)]['separates'] else ' '}"
                          for k in KS)
        print(f"  vs {a:<20}{cells}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

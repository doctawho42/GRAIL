#!/usr/bin/env python3
"""The best measured configuration against MetaTox, with the interval and the caveat attached.

Rank fusion of the two frozen component scores beats their product by 0.109 on this population,
and with the retrained generator it reaches recall@15 = 0.5488 against MetaTox's 0.5143. This
attaches the paired interval, and it attaches the thing that matters more than the interval:

  RRF WAS CHOSEN AFTER LOOKING AT THESE SUBSTRATES. Four combination rules were computed on the
  291 and the best was taken. That is an argmax over the set it is reported on, which is the
  error the first paper of this series is about. The number below is therefore an upper bound on
  what a declared choice would give, and it is not a result until the rule is fixed in advance
  and checked on the validation split.

MetaTox is gated against results/four_method_291.json, so the population and the aggregation are
the committed ones.
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

from bank_without_selection import _dedup, _key  # noqa: E402

FOUR = ROOT / "results" / "four_method_291.json"
METATOX = ROOT / "results" / "metatox_smirks_preds.json"
BUDGETS = [1, 3, 5, 8, 10, 15, 20, 30, 50]
N_BOOT, SEED = 10000, 0


def rrf_order(cands, k=60):
    f = {id(c): i for i, c in enumerate(sorted(cands, key=lambda x: -x["filter"]))}
    g = {id(c): i for i, c in enumerate(sorted(cands, key=lambda x: -x["generator"]))}
    return sorted(cands, key=lambda c: -(1 / (k + f[id(c)]) + 1 / (k + g[id(c)])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="results/widepools_implicit/w*.json")
    ap.add_argument("--label", default="new generator + rank fusion")
    ap.add_argument("--out", default=str(ROOT / "results" / "rrf_vs_metatox.json"))
    args = ap.parse_args()

    pools, refs = {}, {}
    for p in sorted(glob.glob(args.pools)):
        d = json.loads(Path(p).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    subs = sorted(s for s in pools if refs.get(s))
    mtx = json.loads(METATOX.read_text())["predictions"]
    mt = {s: _dedup(mtx.get(s, []), max(BUDGETS)) for s in subs}
    real = {s: set(refs[s]) for s in subs}
    ours = {s: [c["key"] for c in rrf_order(pools[s])] for s in subs}

    U = np.array([len(real[s]) for s in subs], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = U[idx].sum(axis=1)

    by_budget = {}
    for b in BUDGETS:
        hg = np.array([len(set(ours[s][:b]) & real[s]) for s in subs], dtype=float)
        hm = np.array([len(set(mt[s][:b]) & real[s]) for s in subs], dtype=float)
        d = hg - hm
        bt = d[idx].sum(axis=1) / np.maximum(denom, 1)
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        by_budget[str(b)] = {"ours": round(float(hg.sum() / U.sum()), 4),
                             "metatox": round(float(hm.sum() / U.sum()), 4),
                             "gap": round(float(d.sum() / U.sum()), 4),
                             "ci95": [round(lo, 4), round(hi, 4)],
                             "excludes_zero": bool(lo > 0 or hi < 0)}

    four = json.loads(FOUR.read_text())["per_method"]["MetaTox"]["recall"]
    mism = [f"k={b}: {by_budget[str(b)]['metatox']} vs committed {four[str(b)]}"
            for b in BUDGETS if str(b) in four
            and abs(by_budget[str(b)]["metatox"] - four[str(b)]) > 1e-9]

    rep = {"provenance": stamp(__file__), "arm": args.label,
           "population": {"n": len(subs), "source": "the 291 of results/four_method_291.json"},
           "aggregation": "micro, ratio of sums",
           "caveat": "the fusion rule was selected on this population after computing four "
                     "alternatives on it; this is an upper bound on a declared choice, not a "
                     "result. Fix the rule in advance and check it on validation before quoting.",
           "gate": {"reproduces_four_method_291_metatox": not mism, "mismatches": mism},
           "by_budget": by_budget, "n_boot": N_BOOT, "seed": SEED}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n{args.label}, {len(subs)} substrates, micro\n")
    print(f"{'k':>4}{'ours':>9}{'MetaTox':>10}{'gap':>9}   interval")
    for b in BUDGETS:
        r = by_budget[str(b)]
        print(f"{b:>4}{r['ours']:>9.4f}{r['metatox']:>10.4f}{r['gap']:>+9.4f}   "
              f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
              f"{'separated' if r['excludes_zero'] else ''}")
    if mism:
        print(f"\nGATE FAILED: {mism}")
        return 1
    print("\ngate: MetaTox reproduces the committed table at every budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

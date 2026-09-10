#!/usr/bin/env python3
"""The four orderings on the pools the manuscript itself scored, with no model run at all.

results/widepools_implicit/w*.json holds the whole-bank pool for every substrate of the comparison
set, built at top_k=7581 by the released pair (full5000_implicit generator and filter), with each
candidate's generator score, filter score and tautomer key stored. Everything the ordering question
needs is already there: the expensive half is enumeration and keying, and both are done.

That matters because two earlier runs of mine answered the question with the wrong model.
bank_without_selection.py asserts the full5000_priors generator and its CLI offers the single
filter, and I read that as the deployed pair. It is not: results/deployment_table.json, the
artifact behind the manuscript's sweep, records full5000_implicit for both stages of both arms.
The arms below therefore differ from those runs, and the gate is what says so.

Two gates, and neither is decoration.

  MetaTox has to reproduce results/four_method_291.json at every budget, or the population and the
  conventions are not the manuscript's.

  fusion_cap100 is the deployed arm -- whole bank, cap 100, reciprocal rank fusion -- so it has to
  reproduce the manuscript's own GRAIL exhaustive column. If it does not, this is still not the
  deployed configuration and nothing here is about the released system.

What is new here is the ordering axis, not the cap. scripts/typed_edit/pool_cap_cost.py already
sweeps eight cap values on this population and this artifact reproduces it exactly: uncapped reads
0.5038 at a budget of fifteen there and here, and cap 100 reads 0.5353 in both. That agreement is
the cross-check on this file. It also means the caveat that artifact carries applies to any cap
comparison read off this one, in its words: eight caps were computed on the population they are
reported on and the best was read off, so a cap has to be fixed in advance and checked where it
was not chosen before its margin is quoted.

    python scripts/deployed_pool_orderings.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bank_without_selection as B

DEPLOYED_CAP = 100  # H9/P2, the cap the released arm runs under
sys.path.insert(0, str(ROOT / "scripts" / "typed_edit"))
from _rrf import RRF_K, competition_ranks

BUDGETS = [1, 3, 5, 8, 10, 15, 20, 30, 50]
N_BOOT, SEED = 10000, 0
COMPARATORS = {
    "MetaTox":        ("results/metatox_smirks_preds.json", "predictions"),
    "GLORYx":         ("results/gloryx_service_preds.json", "predictions"),
    "MetaPredictor":  ("artifacts/tier2_1170/metapredictor_preds.json", None),
    "SyGMa":          ("results/sygma_fulltest_predictions.json", None),
    "BioTransformer": ("results/biotransformer_fulltest_preds.json", None),
}


def fuse(cs):
    rf = competition_ranks(cs, lambda c: c["filter"])
    rg = competition_ranks(cs, lambda c: c["generator"])
    i = sorted(range(len(cs)), key=lambda j: -(1.0 / (RRF_K + rf[j]) + 1.0 / (RRF_K + rg[j])))
    return [cs[j] for j in i]


def multiply(cs):
    return sorted(cs, key=lambda c: -(c["generator"] * c["filter"]))


def cap(cs, n):
    return cs if n is None or len(cs) <= n else sorted(cs, key=lambda c: -c["generator"])[:n]


ORDERINGS = (("product_uncapped", multiply, None), ("product_cap100", multiply, DEPLOYED_CAP),
             ("fusion_uncapped", fuse, None), ("fusion_cap100", fuse, DEPLOYED_CAP))


def keys_in_order(cs, self_key, limit):
    """Unique tautomer keys in rank order, parent dropped, to the widest budget."""
    out, seen = [], set()
    for c in cs:
        k = c["key"]
        if k == self_key or k in seen:
            continue
        seen.add(k)
        out.append(k)
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "deployed_pool_orderings.json"))
    args = ap.parse_args()

    pools = {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        pools.update(json.loads(Path(f).read_text())["pools"])
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    comp_raw = {n: (json.loads((ROOT / rel).read_text())[k] if k
                    else json.loads((ROOT / rel).read_text()))
                for n, (rel, k) in COMPARATORS.items()}
    subs = sorted(set(pools) & set(truth) & set.intersection(*(set(p) for p in comp_raw.values())))
    if len(subs) != 291:
        raise SystemExit(f"population is {len(subs)}, not 291")
    print(f"population: {len(subs)} substrates, pools already scored by the released pair",
          flush=True)

    from multiprocessing import Pool
    kp = Pool(6)
    refs = {s: set(B._dedup(truth[s], None, kp)) for s in subs}
    self_key = {s: B._key(s) for s in subs}
    lim = max(BUDGETS)
    comp = {n: {s: [k for k in B._dedup(p.get(s, []), None, kp) if k != self_key[s]][:lim]
                for s in subs} for n, p in comp_raw.items()}
    kp.close()

    U = np.array([len(refs[s]) for s in subs], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def score(order):
        return {b: (np.array([len(set(order[s][:b]) & refs[s]) for s in subs], dtype=float),
                    np.array([B.recall_at(order[s], refs[s], b) for s in subs]))
                for b in BUDGETS}

    csc = {n: score(m) for n, m in comp.items()}
    rep = {"config": {**B._code_version(), "population": len(subs), "budgets": BUDGETS,
                      "pools": "results/widepools_implicit/w*.json",
                      "checkpoints": "full5000_implicit generator and filter, per "
                                     "results/deployment_table.json",
                      "aggregation": "micro, ratio of sums; macro alongside"},
           "n_references": float(U.sum()),
           "comparators": {n: {str(b): {"micro": round(float(csc[n][b][0].sum() / U.sum()), 4),
                                        "macro": round(float(csc[n][b][1].mean()), 4)}
                               for b in BUDGETS} for n in csc},
           "grail_arms": {}}

    for name, fn, pcap in ORDERINGS:
        ranked = {s: keys_in_order(fn(cap(pools[s], pcap)), self_key[s], lim) for s in subs}
        sc = score(ranked)
        arm = {"pool_cap": pcap,
               "mean_pool": round(float(np.mean([len(cap(pools[s], pcap)) for s in subs])), 1),
               "by_budget": {}}
        for b in BUDGETS:
            row = {"micro": round(float(sc[b][0].sum() / U.sum()), 4),
                   "macro": round(float(sc[b][1].mean()), 4), "vs": {}}
            for n in csc:
                d = sc[b][0] - csc[n][b][0]
                bt = d[idx].sum(axis=1) / denom
                lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
                row["vs"][n] = {"gap": round(float(d.sum() / U.sum()), 4),
                                "ci95": [round(lo, 4), round(hi, 4)],
                                "excludes_zero": bool(lo > 0 or hi < 0)}
            arm["by_budget"][str(b)] = row
        rep["grail_arms"][name] = arm

    # Gate 1: the population and conventions are the manuscript's.
    four = json.loads((ROOT / "results/four_method_291.json").read_text())["per_method"]["MetaTox"]["recall"]
    m1 = [f"k={b}: {rep['comparators']['MetaTox'][str(b)]['micro']} vs {four[str(b)]}"
          for b in BUDGETS if str(b) in four
          and abs(rep["comparators"]["MetaTox"][str(b)]["micro"] - four[str(b)]) > 1e-9]
    # Gate 2: fusion_cap100 is the deployed arm, so it is the manuscript's own column.
    dep = json.loads((ROOT / "results/deployment_table.json").read_text())["recall_micro"]
    col = dep.get("whole bank", {})
    m2 = [f"k={b}: {rep['grail_arms']['fusion_cap100']['by_budget'][str(b)]['micro']} vs {col[str(b)]}"
          for b in BUDGETS if str(b) in col
          and abs(rep["grail_arms"]["fusion_cap100"]["by_budget"][str(b)]["micro"] - col[str(b)]) > 5e-4]
    rep["gates"] = {"metatox_reproduces_four_method_291": not m1, "metatox_mismatches": m1,
                    "fusion_cap100_reproduces_the_deployed_column": not m2, "deployed_mismatches": m2}
    print(f"  gate: MetaTox reproduces the committed artifact: {not m1}")
    for m in m1: print(f"    {m}")
    print(f"  gate: fusion_cap100 reproduces the manuscript's exhaustive column: {not m2}")
    for m in m2: print(f"    {m}")

    hdr = f"{'k':>3} | " + " | ".join(f"{n[:9]:>9}" for n in csc) + " || " + " | ".join(
        f"{a[:15]:>15}" for a in rep["grail_arms"])
    print("\n" + hdr); print("-" * len(hdr))
    for b in BUDGETS:
        r = f"{b:>3} | " + " | ".join(f"{rep['comparators'][n][str(b)]['micro']:>9.4f}" for n in csc)
        r += " || " + " | ".join(f"{rep['grail_arms'][a]['by_budget'][str(b)]['micro']:>15.4f}"
                                 for a in rep["grail_arms"])
        print(r)

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

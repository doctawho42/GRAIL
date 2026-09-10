#!/usr/bin/env python3
"""How much of what the bank already found never reaches the top of the list.

The comparison divides by output budget: the released arm leads every published comparator at
thirty and fifty with the intervals excluding zero, and trails at ten and below. The question is
what it would take to lead everywhere, and the answer is not a property of the rule bank.

Everything here is read off results/widepools_implicit/w*.json, the whole-bank pools the manuscript
itself scored with the released pair. Three quantities per budget:

  deployed    what the released ranking delivers
  best rival  the strongest published comparator at that budget
  oracle      what a perfect ranking of the same pool would deliver, which is the number of
              references that pool holds, truncated at the budget

The oracle is a per-substrate upper bound and not an argmax over arms, so it is a ceiling on
ranking rather than a comparison anyone can win. What makes it worth computing is the ratio: the
share of that headroom needed to pass the best rival, which is what a ranking change has to buy.

    python scripts/ranking_headroom.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bank_without_selection as B  # noqa: E402

BUDGETS = [1, 3, 5, 8, 10, 15, 20, 30, 50]
ORDERINGS = ROOT / "results" / "deployed_pool_orderings.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "ranking_headroom.json"))
    args = ap.parse_args()

    pools = {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        pools.update(json.loads(Path(f).read_text())["pools"])
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    subs = sorted(set(pools) & set(truth))

    from multiprocessing import Pool
    kp = Pool(6)
    refs = {s: set(B._dedup(truth[s], None, kp)) for s in subs}
    self_key = {s: B._key(s) for s in subs}
    kp.close()

    pk = {s: {c["key"] for c in pools[s] if c["key"] != self_key[s]} for s in subs}
    U = sum(len(refs[s]) for s in subs)
    in_pool = sum(len(pk[s] & refs[s]) for s in subs)

    dep = json.loads(ORDERINGS.read_text())
    arm = dep["grail_arms"]["fusion_cap100"]["by_budget"]
    comps = dep["comparators"]

    rep = {"config": {**B._code_version(), "n_substrates": len(subs), "n_references": U,
                      "pools": "results/widepools_implicit/w*.json",
                      "arm": "fusion_cap100, the released exhaustive arm",
                      "aggregation": "micro, ratio of sums"},
           "references_in_the_pool": {"n": in_pool, "share": round(in_pool / U, 4),
                                      "meaning": "the ceiling on any re-ranking of this pool: "
                                                 "no new rules and no new chemistry"},
           "by_budget": {}}

    print(f"  {len(subs)} substrates, {U} references")
    print(f"  references anywhere in the pool: {in_pool}/{U} = {in_pool/U:.4f}\n")
    print(f"{'k':>3} | {'deployed':>8} | {'best rival':>10} | {'oracle':>7} | "
          f"{'headroom':>8} | {'needed':>7} | share of headroom")
    for k in BUDGETS:
        orac = sum(min(k, len(pk[s] & refs[s])) for s in subs) / U
        d = arm[str(k)]["micro"]
        rival = max(comps, key=lambda n: comps[n][str(k)]["micro"])
        br = comps[rival][str(k)]["micro"]
        head, need = orac - d, br - d
        share = None if need <= 0 else round(need / head, 4) if head > 0 else None
        rep["by_budget"][str(k)] = {
            "deployed": d, "best_rival": rival, "best_rival_micro": br,
            "oracle": round(orac, 4), "ranking_headroom": round(head, 4),
            "needed_to_pass_the_best_rival": round(need, 4),
            "share_of_headroom_needed": share,
            "already_ahead": bool(need <= 0)}
        tail = "already ahead" if need <= 0 else f"{100*share:.0f}%"
        print(f"{k:>3} | {d:>8.4f} | {br:>10.4f} | {orac:>7.4f} | {head:>+8.4f} | "
              f"{need:>+7.4f} | {tail}")

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

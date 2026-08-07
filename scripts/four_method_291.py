#!/usr/bin/env python3
"""Four methods on one population, spanning an order of magnitude in output size.

The paper's central empirical claim is that a leaderboard in this field is driven by how many
candidates a method emits, and until now the evidence rested on one outlier: SyGMa at 81 against
everyone else's 8 to 14. One point is a coincidence waiting to be named. The SMIRKS-variant MetaTox
predictions add a second high-output method at 36, on 291 substrates where all four methods and the
references are present, which turns the claim into something with a slope rather than a gap.

Everything is scored through the paper's own matcher and its five criteria, on frozen predictions,
so only the match rule and the budget vary and never the chemistry. Two hazards are checked rather
than assumed: a method that returns its own substrate among its predictions inflates nothing here
but has silently inflated pools in this project before, and a method whose predictions are not
deduplicated gets a larger emitted count for free.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _tautomer_inchikey as _tk

KEYS = ROOT / "results" / "key_tables" / "inchikey_tautomer.json"
KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)


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


def load_pools() -> dict:
    grail = {r["sub"]: [c["smiles"] for c in r["candidates"]]
             for r in json.loads((ROOT / "results/scored_predictions.json").read_text())["rows"]}
    return {
        "GRAIL": grail,
        "MetaPredictor": json.loads((ROOT / "artifacts/tier2_1170/metapredictor_preds.json").read_text()),
        "MetaTox": json.loads((ROOT / "results/metatox_smirks_preds.json").read_text())["predictions"],
        "SyGMa": json.loads((ROOT / "results/sygma_fulltest_predictions.json").read_text()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "four_method_291.json"))
    args = ap.parse_args()

    cache = json.loads(KEYS.read_text())
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    pools = load_pools()
    subs = sorted(set(truth) & set.intersection(*(set(p) for p in pools.values())))
    print(f"population: {len(subs)} substrates carrying references and all four prediction sets")

    def key(s: str):
        k = cache.get(s)
        if k is None:
            try:
                k = _tk(s)
            except Exception:
                k = None
            cache[s] = k
        return k

    # hazard 1: a method returning the substrate itself. It cannot score, but it consumes a slot.
    parent_in_own = {}
    for name, pool in pools.items():
        n = sum(1 for s in subs if key(s) in {key(p) for p in pool.get(s, [])[:60]})
        parent_in_own[name] = n
    print("  substrates where a method returns its own parent:",
          {k: v for k, v in parent_in_own.items()})

    per = {}
    for name, pool in pools.items():
        U = 0
        hits = {k: 0 for k in KS}
        emitted = {k: 0 for k in KS}
        raw, dedup = 0, 0
        for s in subs:
            refs = {key(y) for y in truth.get(s, [])} - {None}
            if not refs:
                continue
            U += len(refs)
            seq, seen = [], set()
            for p in pool.get(s, []):
                k = key(p)
                if k is None or k in seen or k == key(s):
                    continue           # drop the parent and any duplicate before the budget bites
                seen.add(k)
                seq.append(k)
            raw += len(pool.get(s, []))
            dedup += len(seq)
            for k in KS:
                hits[k] += len(refs & set(seq[:k]))
                emitted[k] += min(k, len(seq))
        per[name] = {"references": U,
                     "raw_predictions": raw, "after_dedup_and_parent_drop": dedup,
                     "mean_emitted_uncapped": round(dedup / len(subs), 2),
                     "recall": {k: round(hits[k] / max(U, 1), 4) for k in KS},
                     "mean_emitted": {k: round(emitted[k] / len(subs), 2) for k in KS}}

    order = sorted(per, key=lambda m: -per[m]["mean_emitted_uncapped"])
    print(f"\n  {'method':15} {'emitted':>8}  " + "  ".join(f"r@{k:<3}" for k in KS))
    for name in order:
        v = per[name]
        print(f"  {name:15} {v['mean_emitted_uncapped']:>8}  "
              + "  ".join(f"{v['recall'][k]:<5}" for k in KS))

    seen_orders = {}
    for k in KS:
        o = tuple(sorted(per, key=lambda m: -per[m]["recall"][k]))
        seen_orders.setdefault(o, []).append(k)
    print(f"\n  distinct orderings across the budget sweep: {len(seen_orders)}")
    for o, ks in seen_orders.items():
        print(f"    {' > '.join(o)}   at k in {ks}")

    rep = {"config": {**_code_version(), "n_substrates": len(subs), "match": "inchikey_tautomer",
                      "aggregation": "micro, ratio of sums", "k_sweep": list(KS),
                      "population": "the 291 MetaTox submission substrates, all of which carry "
                                    "references and predictions from every method here",
                      "parent_and_duplicates": "dropped before the budget is applied"},
           "parent_returned_by_method": parent_in_own,
           "per_method": per,
           "orderings": {" > ".join(o): ks for o, ks in seen_orders.items()}}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

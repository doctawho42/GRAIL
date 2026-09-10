#!/usr/bin/env python3
"""Is the dehydrogenation class lost in the ranking, or before it?

Of the transformation classes the comparison splits into, dehydrogenation is where this work
trails furthest: the exhaustive arm recovers a fraction of what the strongest comparator does, on
references whose product differs from the substrate by two hydrogens and nothing else. A minimal
delta is exactly where a pipeline can lose a structure for reasons that have nothing to do with
chemistry, so the loss is located before it is explained.

Three places it can happen, and they are distinguishable.

  The reference is in the ranked pool but below the budget. That is ranking, and it is the
  reading the per-class table already implies.

  The reference is nowhere in the pool. Then no template reached it, or one reached it and the
  candidate was rewritten on the way out, and those are different repairs.

  The reference is in the pool under a different key than the one it is scored by. That is the
  matcher and not the model.

The class is read from the same formula-delta vocabulary the per-class table uses, imported rather
than restated, so a change to that vocabulary changes both together.

    python scripts/dehydrogenation_diagnostic.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bank_without_selection as B  # noqa: E402
from error_by_chemistry import classify  # noqa: E402
from _rrf import rrf_order  # noqa: E402

CAP = 100


def deployed_order(pool, self_key):
    """The released ranking: dedup by key in product order, cap by generator, fuse by rank."""
    cands = sorted(pool, key=lambda c: -(c["filter"] * c["generator"]))
    seen, dedup = set(), []
    for c in cands:
        if not c["key"] or c["key"] in seen:
            continue
        seen.add(c["key"])
        dedup.append(c)
    keep = sorted(dedup, key=lambda c: -c["generator"])[:CAP]
    return [c["key"] for c in rrf_order(keep) if c["key"] != self_key]

CLASS = "dehydrogenation"
BUDGETS = (5, 15, 30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="klass", default=CLASS)
    ap.add_argument("--out", default=str(ROOT / "results" / "dehydrogenation_diagnostic.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    pools, refs_raw = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"])
        refs_raw.update(blob["references"])
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    subs = sorted(set(pools) & set(truth) & set(refs_raw))

    from multiprocessing import Pool
    kp = Pool(6)
    # The reference structures, and the keys they are scored under. Both are needed: the class is
    # a property of the structure and the hit is a property of the key.
    keyed = {s: B._dedup(truth[s], None, kp) for s in subs}
    kp.close()

    rows = []
    for s in subs:
        sm = Chem.MolFromSmiles(s)
        if sm is None:
            continue
        for struct, key in zip(truth[s], keyed[s]):
            mm = Chem.MolFromSmiles(struct)
            if mm is None:
                continue
            name, _ = classify(sm, mm)
            if name != args.klass:
                continue
            pool_keys = [c["key"] for c in pools[s]]
            in_key = key in pool_keys
            canon = Chem.MolToSmiles(mm)
            by_smiles = any(c["smiles"] == canon for c in pools[s])
            # Where in the released ranking does it land? A reference in the pool but past the
            # budget is lost to the order; one inside the budget is a hit. This is what turns the
            # "ranking" reading from an inference into a count.
            order = deployed_order(pools[s], B._key(s))
            rank = order.index(key) + 1 if key in order else None
            rows.append({"substrate": s, "reference": struct, "key": key,
                         "in_pool_by_key": in_key,
                         "in_pool_by_structure": by_smiles,
                         "deployed_rank": rank,
                         "pool_size": len(pools[s])})

    n = len(rows)
    by_key = sum(r["in_pool_by_key"] for r in rows)
    by_struct_only = sum(r["in_pool_by_structure"] and not r["in_pool_by_key"] for r in rows)
    absent = sum(not r["in_pool_by_key"] and not r["in_pool_by_structure"] for r in rows)
    in15 = sum(1 for r in rows if r["deployed_rank"] and r["deployed_rank"] <= 15)
    past15 = sum(1 for r in rows if r["deployed_rank"] and r["deployed_rank"] > 15)
    capped = sum(1 for r in rows if r["in_pool_by_key"] and r["deployed_rank"] is None)

    rep = {"config": {**B._code_version(), "class": args.klass, "population": len(subs),
                      "pools": "results/widepools_implicit/w*.json"},
           "references_in_the_class": n,
           "in_the_pool_under_the_key_they_are_scored_by": by_key,
           "in_the_pool_as_a_structure_but_not_under_that_key": by_struct_only,
           "not_in_the_pool_at_all": absent,
           "in_the_pool_and_inside_the_budget": in15,
           "in_the_pool_but_ranked_past_the_budget": past15,
           "in_the_pool_but_dropped_by_the_cap": capped,
           "reading": ("ranking: the references are enumerated and ranked below the budget"
                       if by_key > n / 2 else
                       "not ranking: most of this class never reaches the ranked pool, so the "
                       "repair is in enumeration or in what the pipeline does to a candidate "
                       "on the way out, not in the order"),
           "rows": rows}

    print(f"  class {args.klass!r}: {n} references over {len(subs)} substrates")
    print(f"    in the pool under the scoring key      {by_key}")
    print(f"    in the pool as a structure, other key  {by_struct_only}")
    print(f"    not in the pool at all                 {absent}")
    print(f"    -- of the {by_key} in the pool: {in15} inside top-15, "
          f"{past15} ranked past it, {capped} dropped by the cap")
    print(f"\n  {rep['reading']}")
    print(f"  standardiser rewrite of a reference key: {by_struct_only} of {n} "
          f"(zero means the matcher is not the cause)")

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

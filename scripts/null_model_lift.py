#!/usr/bin/env python3
r"""An aggregate the output budget does not move, and it costs nothing to compute.

The paper spends most of its length showing that a leaderboard is driven by how many candidates a
method emits, and then recommends declaring the count. A recommendation is not a fix. The
enumerability that makes the coverage factor an exhaustive measurement also supplies the fix for
free, because a method whose candidate pool can be enumerated has an exactly computable chance
baseline: draw $k$ of its own $N$ candidates uniformly and the expected number of references
recovered is $k M / N$, where $M$ is how many of them the pool contains at all.

Dividing by that baseline is the enrichment factor of virtual screening, and it is not a new
proposal so much as one this literature does not use. What it buys here is that a method emitting
eighty-one candidates is measured against a correspondingly higher chance level, so the quantity the
paper documents as the confound is divided out rather than declared.

It is also not a bolt-on. Writing the decomposition's third factor as $H / C_{\mathrm{bud}}$ and the
baseline as $k/N$, the lift is exactly that factor over its chance value -- so this is what the
ranking term means once the size of the pool it ranks is taken into account. That has a consequence
the raw factor hides, and it is not in our favour: a pool that fits inside the budget has a chance
level of one, so a ranking factor near one is free there and earned only where the budget binds.

That was the motivation, and half of it does not survive the measurement. Lift does NOT stabilise
the ordering: sweeping $k$ from 1 to 40, recall gives two orderings of the three methods and lift
also gives two. As a remedy for the budget confound this fails, and it is reported as a failure.

What it does buy is a different split, and one that runs against us. Recall rewards a pool and an
ordering together; lift isolates the ordering. SyGMa's ranking recovers 2.9 times what a uniform
draw from its own 81-candidate pool would, while GRAIL reaches 1.06 and MetaPredictor 1.01 -- both
within a rounding of chance, because their pools are smaller than the budget and there is almost
nothing left to order. So a ranking factor near one is not evidence of ranking; it is evidence that
the budget never binds, and the two are indistinguishable in recall@$k$ but not here.

The gate is that each method's micro recall@15 is reproduced from its own released pool before any
of this is read.
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
sys.path.insert(0, str(ROOT / "scripts"))

from grail_metabolism.metrics import _tautomer_inchikey as _tk

KEYS = ROOT / "results" / "key_tables" / "inchikey_tautomer.json"
# Micro recall@15 on the clean test split, recomputed from each method's released pool.
#
# GRAIL's gate is NOT results/recall_factorization.json's stored 0.2607. That figure is the deployed
# output's hit count after two monotonicity clamps pulled it down to a ceiling measured in a
# different hydrogen convention; a fresh run of the deployed pipeline returns 689 hits, 0.2653, and
# results/set_metrics_by_criterion.json -- computed from the same stored output by another script --
# agrees at 0.3395 macro. The gate is the reproducible number, not the published one.
GATES = {"GRAIL": ("deployed pipeline re-run, results/clamp_verification.json", 0.2653),
         "SyGMa": ("results/decompose_sygma.json", 0.5137)}
KS = tuple(range(1, 41))


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
    """Each method's own ranked candidate list per substrate, untruncated, as released."""
    grail = {r["sub"]: [c["smiles"] for c in r["candidates"]]
             for r in json.loads((ROOT / "results/scored_predictions.json").read_text())["rows"]}
    sygma = json.loads((ROOT / "results/sygma_fulltest_predictions.json").read_text())
    meta = json.loads((ROOT / "artifacts/tier2_1170/metapredictor_preds.json").read_text())
    return {"GRAIL": grail, "SyGMa": sygma, "MetaPredictor": meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "null_model_lift.json"))
    args = ap.parse_args()

    cache = json.loads(KEYS.read_text())
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    pools = load_pools()

    def key(smiles: str):
        k = cache.get(smiles)
        if k is None:
            try:
                k = _tk(smiles)
            except Exception:
                k = None
            cache[smiles] = k
        return k

    subs = sorted(set(truth) & set.intersection(*(set(p) for p in pools.values())))
    print(f"substrates common to every released pool: {len(subs)}")

    per = {}
    for name, pool in pools.items():
        U = M = 0
        hits = {k: 0 for k in KS}
        expected = {k: 0.0 for k in KS}
        sizes = []
        for s in subs:
            refs = {key(y) for y in truth.get(s, [])}
            refs.discard(None)
            if not refs:
                continue
            cands = [key(c) for c in pool.get(s, [])]
            n = len(cands)
            sizes.append(n)
            U += len(refs)
            in_pool = refs & set(cands)
            M += len(in_pool)
            if n == 0:
                continue
            for k in KS:
                hits[k] += len(refs & set(cands[:k]))
                expected[k] += min(k, n) * len(in_pool) / n
        per[name] = {"U": U, "references_in_pool": M,
                     "mean_pool": round(sum(sizes) / max(len(sizes), 1), 2),
                     "recall": {k: round(hits[k] / max(U, 1), 4) for k in KS},
                     "null": {k: round(expected[k] / max(U, 1), 4) for k in KS},
                     "lift": {k: round(hits[k] / max(expected[k], 1e-9), 3) for k in KS}}

    print(f"\n  {'method':14} {'pool':>6} {'recall@15':>10} {'null@15':>9} {'lift@15':>8}")
    for name, v in per.items():
        print(f"  {name:14} {v['mean_pool']:>6} {v['recall'][15]:>10} "
              f"{v['null'][15]:>9} {v['lift'][15]:>8}")

    print("\ngate: micro recall@15 against the committed artifacts")
    for name, (src, want) in GATES.items():
        got = per[name]["recall"][15]
        print(f"  {name:14} {got} against {want} from {src}")
        if abs(got - want) > 5e-4:
            raise SystemExit(f"{name} does not reproduce its committed recall; the released pool is "
                             "not being read in the order the paper scores it")

    def orders(field):
        seen = {}
        for k in KS:
            o = tuple(sorted(per, key=lambda m: -per[m][field][k]))
            seen.setdefault(o, []).append(k)
        return seen

    by_recall, by_lift = orders("recall"), orders("lift")
    print(f"\nacross k={KS[0]} to {KS[-1]}:")
    print(f"  under recall: {len(by_recall)} distinct orderings")
    for o, ks in by_recall.items():
        print(f"    {' > '.join(o)}  at k in {ks[0]}..{ks[-1]}")
    print(f"  under lift:   {len(by_lift)} distinct orderings")
    for o, ks in by_lift.items():
        print(f"    {' > '.join(o)}  at k in {ks[0]}..{ks[-1]}")

    rep = {"config": {**_code_version(), "n_substrates": len(subs), "k_sweep": list(KS),
                      "match": "inchikey_tautomer", "aggregation": "micro, ratio of sums",
                      "null": "uniform draw of min(k,N) from the method's own pool of N",
                      "gate": "micro recall@15 reproduces each committed artifact"},
           "per_method": per,
           "orderings": {"under_recall": {" > ".join(o): ks for o, ks in by_recall.items()},
                         "under_lift": {" > ".join(o): ks for o, ks in by_lift.items()}}}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""What the tautomer budget costs, in time and in the invariance the matching depends on.

The generator's cold time is 94 to 99 per cent standardisation. Applying all 7,580 templates to
a 109-heavy-atom substrate takes 5.2 seconds and produces 20,655 products; standardising them is
where the hours go, and `TautomerEnumerator` is set to 1,000 tautomers and 1,000 transforms per
product.

Lowering that budget is not free, and the thing it can break is not the speed. Matching runs
both the prediction and the reference through the same canonicaliser, so the comparison is
tautomer-invariant exactly as long as the canonicaliser still sends every tautomer of one
molecule to one representative. A truncated enumeration can stop before it gets there, and two
tautomers of the same metabolite would then carry different keys and stop matching.

So the measurement is invariance, not agreement with the current setting: for each product,
several of its own tautomers are generated and then canonicalised at each budget, and a budget
passes on that product only if all of them land on one key.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

BUDGETS = (1000, 500, 200, 100, 50, 20, 10, 5)   # 1000 is what ships
N_VARIANTS = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "results/wide_pools.json"))
    ap.add_argument("--n", type=int, default=250, help="products sampled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/tautomer_budget.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize

    from grail_metabolism.utils import preparation as prep

    blob = json.loads(Path(args.pools).read_text())
    every = [c["smiles"] for pool in blob["pools"].values() for c in pool]
    rng = random.Random(args.seed)
    products = rng.sample(every, min(args.n, len(every)))
    print(f"{len(products)} products sampled from {len(every)}", file=sys.stderr, flush=True)

    # Each product's own tautomers, generated once at a generous fixed budget. These are the
    # variants a rule engine can emit in place of the annotated reference, and the property
    # under test is that they all still collapse together.
    gen = rdMolStandardize.TautomerEnumerator()
    gen.SetMaxTautomers(64)
    families = []
    for p in products:
        m = Chem.MolFromSmiles(p)
        if m is None:
            continue
        try:
            variants = [Chem.MolToSmiles(t) for t in gen.Enumerate(m)][:N_VARIANTS]
        except Exception:
            continue
        if p not in variants:
            variants.append(p)
        if len(variants) >= 2:
            families.append(variants)
    print(f"{len(families)} products with at least two tautomers", file=sys.stderr, flush=True)

    def key(smiles):
        try:
            std = prep.standardize_mol(Chem.MolFromSmiles(smiles))
            return inchi.MolToInchiKey(std)
        except Exception:
            return None

    rows = {}
    for b in BUDGETS:
        prep._TAUTOMER_ENUMERATOR.SetMaxTautomers(b)
        prep._TAUTOMER_ENUMERATOR.SetMaxTransforms(b)
        invariant, evaluated, t = 0, 0, 0.0
        per_call = []
        for fam in families:
            t0 = time.perf_counter()
            keys = {key(v) for v in fam}
            dt = time.perf_counter() - t0
            t += dt
            per_call.append(dt / len(fam))
            if None in keys:
                continue
            evaluated += 1
            invariant += (len(keys) == 1)
        per_call.sort()
        rows[str(b)] = {
            "invariance": round(invariant / evaluated, 4) if evaluated else None,
            "families_evaluated": evaluated,
            "ms_per_standardisation": round(1000 * t / sum(len(f) for f in families), 2),
            "p99_ms_per_family_member": round(1000 * per_call[int(0.99 * len(per_call))], 2),
            "total_s": round(t, 1)}
        r = rows[str(b)]
        print(f"  max_tautomers={b:>5}  invariance={r['invariance']:.4f}  "
              f"{r['ms_per_standardisation']:>8.2f} ms/standardisation  "
              f"p99 {r['p99_ms_per_family_member']:.0f} ms", file=sys.stderr, flush=True)
        Path(args.out).write_text(json.dumps(
            {"provenance": stamp(__file__),
             "status": "EXPLORATORY. A curve, not a hypothesis. Any budget adopted from it must "
                       "be fixed by a stated rule and checked where it was not chosen.",
             "shipped_budget": 1000, "n_products": len(products),
             "n_families": len(families), "variants_per_product": N_VARIANTS,
             "seed": args.seed,
             "invariance_means": "all tautomers of one product canonicalise to one InChIKey, "
                                 "which is the property tautomer-invariant matching rests on",
             "by_budget": rows}, indent=1))
    prep._TAUTOMER_ENUMERATOR.SetMaxTautomers(1000)
    prep._TAUTOMER_ENUMERATOR.SetMaxTransforms(1000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

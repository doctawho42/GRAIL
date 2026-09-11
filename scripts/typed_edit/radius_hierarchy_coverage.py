"""Free first stage of queue point 3: does a radius hierarchy raise coverage, and at what inflation?

98 of the uncovered references are known-type misses -- the type is in the bank, but the deployed
radius-1 template is too specific to fire on this substrate. A hierarchy would mine at radii 0, 1
and 2 and, on application, take the most specific template that fires, dropping to general ones
only where nothing fires. The relaxation ladder already hinted this recovers little (3 of 98) and
inflates the pool; this measures the radius version directly, and reports coverage AND mean pool
size together, because on precision 0.032 a coverage gain bought with pool inflation is the trade
point 3 itself warns about.

Compact by design: a train subsample mines each radius, applied to a test subsample under a heavy
cap, so the direction is read in ~20 min rather than the hours a full remine would take.

    python scripts/typed_edit/radius_hierarchy_coverage.py --train-pairs 400 --test 40
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")
from engine_knobs import apply_with  # noqa: E402
from mine_rules import TRAIN_SDF, TRAIN_TRIPLES_CLEAN, process_pair  # noqa: E402
from grail_metabolism.metrics import _tautomer_inchikey  # noqa: E402
from _provenance import stamp  # noqa: E402

RADII = (0, 1, 2)


def keyset(smiles_list):
    out = set()
    for s in smiles_list:
        try:
            out.add(_tautomer_inchikey(s))
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pairs", type=int, default=400)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--heavy-cap", type=int, default=38)
    ap.add_argument("--out", default=str(ROOT / "results" / "radius_hierarchy_coverage.json"))
    args = ap.parse_args()

    # train positive pairs, subsampled -- mine each radius from the SAME pairs so the only
    # difference between the three banks is the radius
    from measure_coverage import load_positive_split
    split = load_positive_split("train", TRAIN_SDF, TRAIN_TRIPLES_CLEAN)
    pairs = sorted(split.positive_pairs)[: args.train_pairs]
    print(f"mining {len(pairs)} train pairs at radii {RADII}", flush=True)

    banks = {r: set() for r in RADII}
    t0 = time.time()
    for i, (sub, prod) in enumerate(pairs, 1):
        if i % 100 == 0:
            print(f"  mine {i}/{len(pairs)} ({time.time()-t0:.0f}s) "
                  + " ".join(f"r{r}={len(banks[r])}" for r in RADII), flush=True)
        for r in RADII:
            o = process_pair(sub, prod, radius=r)
            if o.smirks:
                banks[r].add(o.smirks)
    for r in RADII:
        print(f"  radius {r}: {len(banks[r])} distinct rules", flush=True)

    # test substrates with references, capped
    wide, refs_raw = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        b = json.loads(Path(f).read_text())
        wide.update(b["pools"])
        refs_raw.update(b["references"])
    subs = [s for s in sorted(set(wide) & set(refs_raw))
            if refs_raw[s] and (m := Chem.MolFromSmiles(s)) and m.GetNumHeavyAtoms() <= args.heavy_cap][: args.test]

    rep = {"provenance": stamp(__file__), "train_pairs": len(pairs), "test_substrates": len(subs),
           "bank_sizes": {str(r): len(banks[r]) for r in RADII}, "by_radius": {}}
    # apply each radius bank alone, and the hierarchy (most specific that fires), measuring
    # coverage and mean pool size (inflation)
    rlist = {r: sorted(banks[r]) for r in RADII}
    tot_refs = sum(len(refs_raw[s]) for s in subs)
    t0 = time.time()
    for label in ("r0", "r1", "r2", "hierarchy"):
        hit = 0
        pool_sizes = []
        for s in subs:
            m = Chem.MolFromSmiles(s)
            refs = set(refs_raw[s])
            if label == "hierarchy":
                # most specific first: r2, then r1, then r0 only where nothing fired yet
                keys = set()
                for r in (2, 1, 0):
                    if keys:
                        break
                    keys = keyset(apply_with(m, rlist[r], False, "canonical", False))
                # union across radii would be the loose reading; "most specific that fires" is the
                # deployed-intent reading -- fall to general only when specific gives nothing
            else:
                r = int(label[1])
                keys = keyset(apply_with(m, rlist[r], False, "canonical", False))
            pool_sizes.append(len(keys))
            hit += len(refs & keys)
        rep["by_radius"][label] = {
            "coverage": round(hit / tot_refs, 4) if tot_refs else None,
            "mean_pool": round(sum(pool_sizes) / len(pool_sizes), 1) if pool_sizes else 0}
        print(f"  {label}: coverage {rep['by_radius'][label]['coverage']} "
              f"mean_pool {rep['by_radius'][label]['mean_pool']} ({time.time()-t0:.0f}s)", flush=True)

    b1 = rep["by_radius"]["r1"]
    rep["reading"] = (
        f"radius 1 is the deployed granularity at coverage {b1['coverage']} / pool {b1['mean_pool']}. "
        "A hierarchy helps only if it raises coverage without the pool blowing up; radius 0 alone is "
        "the inflation ceiling. Read coverage and pool together -- a coverage gain at many times the "
        "pool is the precision cost point 3 warns of, on a system already at 0.032.")
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

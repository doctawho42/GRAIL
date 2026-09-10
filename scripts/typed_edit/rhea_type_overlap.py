"""Are the missing metabolic types present in Rhea, the curated enzymatic reaction database?

The synthetic control (uspto_type_overlap) asked this of 42,554 USPTO retro templates and found
14 of the 312 missing types, carrying 18 of 337 misses. The obvious next question, which a JCIM
referee asks first, is the enzymatic one: Rhea is ~17k curated enzyme reactions (both directions on
disk, ~36k rows). Its reactions are not atom-mapped, so they are typed the same way the coverage
gap itself is -- pair_to_type, the mining MCS route -- rather than through canonical_type on a
SMIRKS, so Rhea and the gap are typed on one footing.

Cofactors do not need filtering by hand: pair_to_type requires the MCS to cover 40% of the smaller
molecule, so a large product paired with water or ammonia returns None on its own. The main
transformation is the largest-reactant / largest-product pair, which is what a metabolic
substrate->product is; typing only that pair keeps this to one MCS per reaction.

    python scripts/typed_edit/rhea_type_overlap.py --rhea <path> --out results/rhea_type_overlap.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

import coverage_gap_types as CG  # pair_to_type and its MCS helpers  # noqa: E402


def key(t):
    return json.dumps(t, sort_keys=True)


def largest(smiles_group):
    best, best_n = None, -1
    for s in smiles_group.split("."):
        s = s.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        n = m.GetNumHeavyAtoms()
        if n > best_n:
            best_n, best = n, m
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhea", required=True)
    ap.add_argument("--shards", default=str(ROOT / "results/gaptypes/a*.json"))
    ap.add_argument("--out", default=str(ROOT / "results/rhea_type_overlap.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    novel = []
    for f in sorted(glob.glob(args.shards)):
        novel += json.loads(Path(f).read_text())["phase_a"].get("novel_pairs", [])
    missing = Counter(key(r["type"]) for r in novel)
    print(f"{len(novel)} novel-type misses over {len(missing)} distinct types", flush=True)

    rows = [ln.rstrip("\n").split("\t") for ln in Path(args.rhea).read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    rhea_types, typed, skipped, t0 = set(), 0, 0, time.time()
    for i, parts in enumerate(rows, 1):
        if i % 2000 == 0:
            print(f"  {i}/{len(rows)} ({time.time()-t0:.0f}s) typed={typed} distinct={len(rhea_types)}",
                  flush=True)
        if len(parts) < 2 or ">>" not in parts[1]:
            continue
        lhs, rhs = parts[1].split(">>", 1)
        r, p = largest(lhs), largest(rhs)
        if r is None or p is None:
            skipped += 1
            continue
        try:
            t = CG.pair_to_type(r, p)
        except Exception:
            t = None
        if t is not None:
            rhea_types.add(key(t))
            typed += 1
        else:
            skipped += 1

    hit = [t for t in missing if t in rhea_types]
    mass = sum(missing[t] for t in hit)
    # the bank must share types with Rhea, or the vocabularies do not meet
    bank_rules = [ln.split()[0] for ln in
                  (ROOT / "grail_metabolism/resources/extended_smirks.txt").read_text().splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")]
    from grail_metabolism.model.reaction_types import canonical_type
    bank_types = {key(t) for t in (canonical_type(r) for r in bank_rules) if t is not None}

    rep = {"provenance": stamp(__file__),
           "rhea": {"rows": len(rows), "typed": typed, "skipped": skipped,
                    "distinct_types": len(rhea_types),
                    "note": "largest-reactant/largest-product pair per reaction, typed by the MCS "
                            "route; a pair whose MCS covers <40% of the smaller molecule is None"},
           "missing": {"misses": len(novel), "distinct_types": len(missing)},
           "overlap": {"missing_types_in_rhea": len(hit),
                       "misses_those_types_carry": mass,
                       "share_of_the_novel_gap": round(mass / len(novel), 4) if novel else None},
           "sanity": {"bank_types_rhea_also_has": len(bank_types & rhea_types),
                      "note": "a zero here would mean the two type vocabularies do not meet"},
           "examples_recoverable": [json.loads(t) for t in hit[:8]]}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nRhea: {typed} reactions typed, {len(rhea_types)} distinct types")
    print(f"of the {len(missing)} missing types, Rhea has {len(hit)}, carrying {mass} of "
          f"{len(novel)} misses ({mass/len(novel):.1%})")
    print(f"sanity: bank and Rhea share {len(bank_types & rhea_types)} types "
          f"({len(bank_types & rhea_types)/len(bank_types):.1%} of the bank's)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

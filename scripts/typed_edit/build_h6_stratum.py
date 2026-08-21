#!/usr/bin/env python3
"""The membership files H6 is registered on: which test substrates carry a symmetry.

H6 is the negative control of the set. Pooling probability over automorphism classes must NOT
change recall on a substrate whose graph has no symmetry, because on such a substrate the
pooling is the identity, and it must improve recall where there is symmetry to pool over. A
control of that shape needs its two arms written down before the run, or the arms can be drawn
after it.

Symmetry classes come from RDKit's canonical ranking with ties left unbroken, which returns the
refinement classes of the graph. Every atom in its own class means the automorphism group is
trivial; that direction is exact, which is the direction the control arm needs. Two atoms
sharing a class means a symmetry in all but pathological cases that refinement cannot separate,
and those go in the other arm, where a false member weakens the prediction rather than
flattering it.

Chirality is excluded from the ranking because the pipeline strips stereochemistry before the
rules fire: ranking with it would call a substrate asymmetric that the pipeline sees as
symmetric.

The registered split is binary. The orbit sizes are recorded beside it, because how much
symmetry a substrate carries is the obvious covariate and computing it later, from a file that
only holds names, would not be possible.

    python scripts/typed_edit/build_h6_stratum.py --self-test
    python scripts/typed_edit/build_h6_stratum.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rdkit import Chem, RDLogger  # noqa: E402

from scripts.run_benchmark import load_test_map  # noqa: E402

RDLogger.DisableLog("rdApp.*")

STRATA = ROOT / "strata"


def orbits(mol):
    """Symmetry classes of the heavy-atom graph, as (n_atoms, class sizes descending)."""
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
    sizes = sorted(Counter(ranks).values(), reverse=True)
    return len(ranks), sizes


def describe(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_atoms, sizes = orbits(mol)
    largest = sizes[0] if sizes else 0
    return {"substrate": smiles, "heavy_atoms": n_atoms, "n_classes": len(sizes),
            "largest_orbit": largest,
            "atoms_in_a_shared_class": sum(s for s in sizes if s > 1),
            "trivial": largest == 1}


def self_test() -> int:
    """Cases whose symmetry is known by inspection, including both directions."""
    cases = [("CCO", True, 1), ("CC", False, 2), ("c1ccccc1", False, 6),
             # para-xylene: the four unsubstituted ring carbons are one orbit, not two
             ("Cc1ccc(C)cc1", False, 4), ("OCC(N)C(=O)O", True, 1),
             ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", False, 2)]
    ok = True
    for smi, trivial, largest in cases:
        d = describe(smi)
        if d is None:
            print(f"FAIL: {smi} did not parse"); ok = False; continue
        if d["trivial"] != trivial or d["largest_orbit"] != largest:
            print(f"FAIL: {smi} -> trivial={d['trivial']} largest={d['largest_orbit']}, "
                  f"expected trivial={trivial} largest={largest}")
            ok = False
        else:
            print(f"  {smi:30} trivial={d['trivial']!s:<5} largest orbit {d['largest_orbit']}")
    # the two arms have to partition, on any input
    rows = [describe(s) for s, _, _ in cases]
    if sum(r["trivial"] for r in rows) + sum(not r["trivial"] for r in rows) != len(rows):
        print("FAIL: the arms do not partition"); ok = False
    print("self-test: OK" if ok else "self-test: FAILURES ABOVE")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "h6_stratum.json"))
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    subs = list(load_test_map(None, 42))
    rows, unparseable = [], []
    for s in subs:
        d = describe(s)
        (rows.append(d) if d else unparseable.append(s))

    trivial = sorted(r["substrate"] for r in rows if r["trivial"])
    nontrivial = sorted(r["substrate"] for r in rows if not r["trivial"])
    STRATA.mkdir(exist_ok=True)
    (STRATA / "trivial_automorphism.txt").write_text("\n".join(trivial) + "\n")
    (STRATA / "nontrivial_automorphism.txt").write_text("\n".join(nontrivial) + "\n")

    orb = Counter(r["largest_orbit"] for r in rows)
    report = {
        "definition": {
            "classes": "Chem.CanonicalRankAtoms(breakTies=False, includeChirality=False)",
            "trivial": "every heavy atom in its own class, so the automorphism group is trivial",
            "stereochemistry": "excluded, because the pipeline strips it before the rules fire",
        },
        "n_substrates": len(rows),
        "unparseable": unparseable,
        "n_trivial": len(trivial),
        "n_nontrivial": len(nontrivial),
        "share_trivial": round(len(trivial) / max(len(rows), 1), 4),
        "largest_orbit_histogram": {str(k): v for k, v in sorted(orb.items())},
        "median_heavy_atoms": sorted(r["heavy_atoms"] for r in rows)[len(rows) // 2],
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    print(f"wrote {args.out}, {len(trivial)} trivial and {len(nontrivial)} not",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

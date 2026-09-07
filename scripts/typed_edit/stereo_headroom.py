#!/usr/bin/env python3
"""How much of the annotation a configuration-aware criterion could distinguish at all.

The manuscript states that its sweep over matching criteria bounds what a stereochemistry-aware
criterion would cost. It cannot, for the reason the sweep itself gives: no reference in this
corpus carries stereochemistry, so the axis has no variation on it to measure, and a sweep whose
settings cannot disagree about configuration bounds nothing about configuration.

What this docstring said before, and what the manuscript printed from it, was that two of the five
criteria differ only in the stereochemistry layer and return the identical verdict at every budget.
Both halves are wrong. The full InChIKey and its first block differ at an output budget of
twenty and again in reference count, and the SI traces that difference to a thiol and thiolate
merging under the skeleton hash, which is protonation. The first block is charge- and
isotope-blind as well as stereo-blind, so the two do not differ only in that layer. The pairs that
are identical at every budget are canonical SMILES equality with a Tanimoto of one, and the
stereo-blind first block with the tautomer-aware key.

What can be bounded is the size of the question. A configuration-aware criterion can only
distinguish a prediction from a reference where the reference has a configuration to carry, so the
bound is the count of annotated metabolites that acquire a stereogenic element their substrate did
not have: a tetrahedral centre, potential or assigned, or a stereogenic double bond. That count is
an upper bound on how many references such a criterion could ever separate, and it is measured
here rather than argued.

Nothing here assigns a configuration. The corpus carries none and this work predicts none; the
measurement is of where the question arises, which is the honest thing the sweep was being asked
for.

    python scripts/typed_edit/stereo_headroom.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
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


def _elements(smiles):
    """(tetrahedral centres, stereogenic double bonds) of one structure, potential included.

    Potential and not only assigned, because the corpus stores no configuration: an assigned-only
    count over structures that carry no assignment would be zero everywhere and would measure the
    corpus's notation rather than its chemistry.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    centres = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
    doubles = sum(1 for b in mol.GetBonds()
                  if b.GetStereo() != Chem.BondStereo.STEREONONE
                  or (b.GetBondType() == Chem.BondType.DOUBLE
                      and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1
                      and not b.GetIsAromatic() and not b.IsInRing()))
    return centres, doubles


def _worker(pair):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    sub, prod = pair
    a, b = _elements(sub), _elements(prod)
    if a is None or b is None:
        return None
    return (b[0] - a[0], b[1] - a[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "stereo_headroom.json"))
    args = ap.parse_args()

    refs = json.loads((ROOT / "results/test_references.json").read_text())
    pairs = [(s, m) for s, mets in refs.items() for m in mets]
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(pairs)} annotated pairs on {workers} workers", flush=True)

    ctx = multiprocessing.get_context("spawn")
    gained_centre = gained_double = gained_either = typed = 0
    hist = Counter()
    t0 = time.perf_counter()
    with ctx.Pool(workers) as pool:
        for n, row in enumerate(pool.imap_unordered(_worker, pairs, 64), 1):
            if row is not None:
                typed += 1
                dc, dd = row
                gained_centre += dc > 0
                gained_double += dd > 0
                gained_either += (dc > 0 or dd > 0)
                hist[min(max(dc, 0), 5)] += 1
            if n % 500 == 0 or n == len(pairs):
                print(f"  {n}/{len(pairs)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    report = {
        "provenance": stamp(__file__),
        "question": ("how many annotated metabolites acquire a stereogenic element their substrate "
                     "does not have, which bounds from above what a configuration-aware matching "
                     "criterion could distinguish on this annotation"),
        "population": {"pairs": len(pairs), "typed": typed,
                       "source": "results/test_references.json, the evaluated test set"},
        "counting": ("tetrahedral centres by FindMolChiralCenters with unassigned included, and "
                     "stereogenic double bonds; potential rather than assigned, because the corpus "
                     "stores no configuration and an assigned-only count would be zero everywhere"),
        "references_gaining_a_tetrahedral_centre": gained_centre,
        "references_gaining_a_stereogenic_double_bond": gained_double,
        "references_gaining_either": gained_either,
        "share_gaining_either": round(gained_either / max(typed, 1), 4),
        "centres_gained_histogram": {str(k): hist[k] for k in sorted(hist)},
        "reading": (
            "This is a ceiling on the question and not an estimate of the cost. A criterion that "
            "distinguished configurations could separate a prediction from a reference only where "
            "the reference has a configuration to carry; on the rest it would return exactly what "
            "the constitutional criteria return. Nothing here assigns a configuration."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\ntyped {typed} of {len(pairs)} annotated pairs")
    print(f"  gaining a tetrahedral centre    : {gained_centre}")
    print(f"  gaining a stereogenic double bond: {gained_double}")
    print(f"  gaining either                  : {gained_either} "
          f"({gained_either / max(typed,1):.1%})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

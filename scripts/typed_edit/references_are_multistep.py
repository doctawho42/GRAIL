#!/usr/bin/env python3
"""How much of the annotation is a single enzymatic step, measured on the references themselves.

The manuscript establishes that a fifth of the mined templates perform more than one enzymatic
edit, and draws the right consequence: the coverage bound is a one-template ceiling and not a
one-enzyme one. That measurement is over templates. The question a reader of a recall figure has is
about the references: how much of the annotation this work is scored against is a single step at
all, since a corpus that records a metabolite several steps from its parent asks a predictor for
something no single template application can be.

The same two instruments the templates were measured with are applied to every annotated
substrate--metabolite pair of the evaluated test set: the number of connected loci the reaction
centre falls into, and the count of core-incident edits, called composite at the threshold the
register fixed before either instrument was written.

Nothing here says a composite reference is unreachable. A template mined from a composite pair
performs the whole transformation in one application, which is exactly why the bank reaches many of
them. What it says is what a recall figure means: a hit on a composite reference is one template
standing in for what an enzyme system did in several turnovers.

    python scripts/typed_edit/references_are_multistep.py
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


def _worker(pair):
    """(loci, core-incident edits) for one annotated pair, or None where it cannot be typed.

    The steps are the ones composite_instruments.py takes over a template's own source pair, in
    the same order and with the same settings, so the two measurements are comparable.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFMCS

    RDLogger.DisableLog("rdApp.*")
    from composite_instruments import _core_incident_edits, _loci
    from scripts.mine_rules import MCS_TIMEOUT_SECONDS, find_reaction_center

    sub, prod = (Chem.MolFromSmiles(x) for x in pair)
    if sub is None or prod is None:
        return None
    try:
        mcs = rdFMCS.FindMCS([sub, prod], timeout=MCS_TIMEOUT_SECONDS, matchValences=False,
                             ringMatchesRingOnly=True, completeRingsOnly=True,
                             bondCompare=rdFMCS.BondCompare.CompareAny,
                             atomCompare=rdFMCS.AtomCompare.CompareElements)
        if mcs.canceled or mcs.numAtoms == 0:
            return None
        core = Chem.MolFromSmarts(mcs.smartsString)
        sm, pm = sub.GetSubstructMatch(core), prod.GetSubstructMatch(core)
        if not sm or not pm:
            return None
        centre, _ = find_reaction_center(sub, prod, sm, pm)
        if not centre:
            return None
        return _loci(sub, set(centre)), _core_incident_edits(sub, prod, sm, pm)
    except Exception:
        return None


def main() -> int:
    from composite_instruments import E_THRESHOLD

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "references_are_multistep.json"))
    args = ap.parse_args()

    refs = json.loads((ROOT / "results/test_references.json").read_text())
    pairs = [(s, m) for s, mets in refs.items() for m in mets]
    if args.pairs:
        pairs = pairs[: args.pairs]

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(pairs)} annotated pairs on {workers} workers", flush=True)
    ctx = multiprocessing.get_context("spawn")
    loci_hist, edit_hist = Counter(), Counter()
    union = both = 0
    typed, t0 = 0, time.perf_counter()
    with ctx.Pool(workers) as pool:
        for n, row in enumerate(pool.imap_unordered(_worker, pairs, 16), 1):
            if row is not None:
                typed += 1
                loci_hist[row[0]] += 1
                edit_hist[min(row[1], 20)] += 1
                i1, i2 = row[0] > 1, row[1] >= 5
                union += (i1 or i2)
                both += (i1 and i2)
            if n % 500 == 0 or n == len(pairs):
                print(f"  {n}/{len(pairs)} ({time.perf_counter() - t0:.0f}s) typed {typed}",
                      flush=True)

    multi_loci = sum(v for k, v in loci_hist.items() if k > 1)
    composite = sum(v for k, v in edit_hist.items() if k >= E_THRESHOLD)


    report = {
        "provenance": stamp(__file__),
        "population": {"pairs": len(pairs), "typed": typed,
                       "source": "results/test_references.json, the evaluated test set"},
        "instruments": ("loci of the reaction centre, and core-incident edits called composite at "
                        f"E >= {E_THRESHOLD}; the same two the mined templates were measured with"),
        "threshold_E": E_THRESHOLD,
        "references_whose_centre_falls_in_more_than_one_locus": multi_loci,
        "share_by_loci": round(multi_loci / max(typed, 1), 4),
        "references_composite_by_edit_count": composite,
        "share_by_edits": round(composite / max(typed, 1), 4),
        "loci_histogram": {str(k): loci_hist[k] for k in sorted(loci_hist)},
        "edit_histogram": {str(k): edit_hist[k] for k in sorted(edit_hist)},
        "references_composite_by_either_instrument": union,
        "share_by_either": round(union / max(typed, 1), 4),
        "references_composite_by_both": both,
        "reading": (
            "A composite reference is one an enzyme system reached in more than one turnover and "
            "the corpus records as a single substrate-product pair. It is not unreachable: a "
            "template mined from such a pair performs the whole transformation at once. What it "
            "settles is what a recall figure means, and it is the reference-side counterpart of "
            "the template-side measurement the manuscript already reports."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\ntyped {typed} of {len(pairs)} annotated pairs")
    print(f"  centre in more than one locus : {multi_loci} ({multi_loci / max(typed,1):.1%})")
    print(f"  composite at E >= {E_THRESHOLD}          : {composite} "
          f"({composite / max(typed,1):.1%})")
    print(f"  either instrument             : {union} ({union / max(typed,1):.1%})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

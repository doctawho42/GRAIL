#!/usr/bin/env python3
"""What the corpus's drawing of the substrate costs the comparator it was never measured on.

The corpus stores its structures as InChI round-trip fixed points, so an amide is stored as its
imidic acid, and this work measures what that costs SyGMa. It could not measure it for
MetaPredictor, and the Supporting Information said so and left it. That asymmetry ran in this
work's favour: MetaPredictor is one of the two comparators the wide-budget lead does not survive
against, so a correction it never received is a correction that could only have widened a gap
this work does not claim.

MetaPredictor is a source checkout with its weights on disk, so it can be re-run. It is, on the
substrates the drawing actually changes and only those: the rest are the same molecule under both
drawings and contribute a paired difference of exactly zero, so running them would cost hours and
add nothing. Their stored predictions are used unchanged, the re-run ones are spliced in, and the
paired difference is computed over the whole comparison set as well as over the substrates that
moved.

    python scripts/typed_edit/metapredictor_drawing.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

KS = (5, 10, 15, 20, 30, 50)
CAP = 100
N_BOOT, SEED = 10000, 0
STORED = ROOT / "artifacts/tier2_1170/metapredictor_preds.json"
REDRAWN = ROOT / "results/metapredictor_natural_drawing_preds.json"
MAP = ROOT / "results/metatox_input/substrate_map.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "metapredictor_drawing.json"))
    args = ap.parse_args()

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from bank_without_selection import _dedup, _key as tautkey

    refs = {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        blob = json.loads(Path(f).read_text())
        refs.update(blob["references"])
    subs = sorted(s for s in refs if refs[s])

    natural = {r["substrate_smiles"]: r["submission_smiles"]
               for r in csv.DictReader(open(MAP))}
    moved = [s for s in subs if natural.get(s, s) != s]

    stored = json.loads(STORED.read_text())
    redrawn = json.loads(REDRAWN.read_text())
    absent = [s for s in moved if s not in redrawn]
    if absent:
        print(f"FAIL: {len(absent)} of the {len(moved)} substrates the drawing changes were not "
              f"re-run; a partial splice would compare two populations", file=sys.stderr)
        return 1

    parent = {s: tautkey(s) for s in subs}
    real = {s: set(refs[s]) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)

    def order(preds, s):
        return [k for k in _dedup(preds.get(s, []), CAP + 5) if k and k != parent[s]]

    a = {s: order(stored, s) for s in subs}
    # The re-run list replaces the stored one only where the drawing changed the molecule.
    b = {s: (order(redrawn, s) if s in redrawn else a[s]) for s in subs}
    unchanged_but_differing = [s for s in subs
                               if s not in redrawn and a[s] != b[s]]
    if unchanged_but_differing:
        print("FAIL: a substrate outside the re-run set differs between arms", file=sys.stderr)
        return 1

    rows = {}
    for population, tag in ((subs, "the whole comparison set"),
                            (moved, "only the substrates the drawing changes")):
        pop_U = np.array([len(real[s]) for s in population], dtype=float)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(population), (N_BOOT, len(population)))
        denom = np.maximum(pop_U[idx].sum(axis=1), 1)
        cell = {"n_substrates": len(population), "n_references": int(pop_U.sum()),
                "by_budget": {}}
        for k in KS:
            hb = np.array([len(set(b[s][:k]) & real[s]) for s in population], dtype=float)
            ha = np.array([len(set(a[s][:k]) & real[s]) for s in population], dtype=float)
            d = hb - ha
            bt = d[idx].sum(axis=1) / denom
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            cell["by_budget"][str(k)] = {
                "recall_as_stored": round(float(ha.sum() / pop_U.sum()), 4),
                "recall_as_drawn": round(float(hb.sum() / pop_U.sum()), 4),
                "difference": round(float(d.sum() / pop_U.sum()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
        rows[tag] = cell

    report = {
        "provenance": stamp(__file__),
        "difference": ("MetaPredictor's recall with the substrate as a chemist draws it, minus "
                       "its recall with the substrate as the corpus stores it"),
        "design": ("only the substrates whose two drawings are different molecules were re-run; "
                   "the rest keep their frozen predictions and contribute exactly zero to the "
                   "paired difference, which is why re-running them would add nothing"),
        "substrates_re_run": len(moved),
        "budgets": list(KS),
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "by_population": rows,
        "reading": (
            "The correction this comparator never received is now measured rather than bounded "
            "by a proxy, which matters because the one proxy available disagrees in direction "
            "with the other comparator for which both drawings could be run."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"{len(moved)} of {len(subs)} substrates re-run")
    for tag, cell in rows.items():
        print(f"\n{tag} ({cell['n_substrates']} substrates, {cell['n_references']} refs)")
        for k in KS:
            c = cell["by_budget"][str(k)]
            print(f"  k={k:<3d} stored {c['recall_as_stored']:.4f}  drawn "
                  f"{c['recall_as_drawn']:.4f}  {c['difference']:+.4f} "
                  f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
                  f"{'  separates' if c['excludes_zero'] else ''}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

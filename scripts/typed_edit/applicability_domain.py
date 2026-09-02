#!/usr/bin/env python3
"""What kind of molecules these numbers were measured on.

The introduction motivates the problem with environmental chemicals and the corpus is assembled
from four drug-centric sources. The annotation has been characterised by heavy-atom delta and
element composition; the substrates themselves never were, so a reader deciding whether any of
this transfers to their own compounds has nothing to compare against.

This is that description: molecular weight, calculated logP, heavy atoms, rings and rotatable
bonds over the evaluated test set and the comparison set, with the Bemis-Murcko scaffolds counted
so the diversity is a number rather than an impression. Nothing here is a claim about
applicability; it is the distribution a reader needs in order to make one.

    python scripts/typed_edit/applicability_domain.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def describe(smiles_list):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    rows, scaffolds, unparsed = {k: [] for k in
                                 ("molecular weight", "calculated logP", "heavy atoms",
                                  "rings", "rotatable bonds")}, Counter(), 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            unparsed += 1
            continue
        rows["molecular weight"].append(Descriptors.MolWt(mol))
        rows["calculated logP"].append(Crippen.MolLogP(mol))
        rows["heavy atoms"].append(mol.GetNumHeavyAtoms())
        rows["rings"].append(rdMolDescriptors.CalcNumRings(mol))
        rows["rotatable bonds"].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        try:
            scaffolds[MurckoScaffold.MurckoScaffoldSmiles(mol=mol)] += 1
        except Exception:
            pass
    out = {}
    for name, values in rows.items():
        arr = np.array(values, dtype=float)
        out[name] = {"n": int(arr.size),
                     "mean": round(float(arr.mean()), 2) if arr.size else None,
                     "quantiles": {str(q): round(float(np.quantile(arr, q)), 2)
                                   for q in QUANTILES} if arr.size else {}}
    singletons = sum(1 for c in scaffolds.values() if c == 1)
    return {
        "n_substrates": len(smiles_list),
        "unparsed": unparsed,
        "descriptors": out,
        "bemis_murcko_scaffolds": len(scaffolds),
        # An acyclic molecule has an empty Murcko scaffold, so it is counted separately rather
        # than collapsed with the ring systems into one meaningless bin.
        "acyclic_substrates": scaffolds.get("", 0),
        "scaffolds_carried_by_one_substrate": singletons,
        "largest_scaffold_share": (round(max(scaffolds.values()) / max(len(smiles_list), 1), 4)
                                   if scaffolds else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "applicability_domain.json"))
    args = ap.parse_args()

    from vs_metatox import population

    refs = json.loads((ROOT / "results/test_references.json").read_text())
    evaluated = sorted(s for s, v in refs.items() if v)
    comparison, _, _ = population()
    comparison = sorted(comparison)

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([ROOT / "results/test_references.json"]),
        "question": ("what kind of molecules the evaluated test set and the comparison set hold, "
                     "so a reader can judge whether these numbers speak to their own compounds"),
        "populations": {"the evaluated test set": describe(evaluated),
                        "the comparison set": describe(comparison)},
        "reading": (
            "The corpus is assembled from four drug-centric sources and the introduction "
            "motivates the problem with environmental chemicals as well. The distribution here "
            "is what a reader should hold that motivation against; it is a description and not "
            "a claim of applicability."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    for name, row in report["populations"].items():
        print(f"\n{name}: {row['n_substrates']} substrates, "
              f"{row['bemis_murcko_scaffolds']} Bemis-Murcko scaffolds "
              f"({row['scaffolds_carried_by_one_substrate']} carried by one substrate)")
        for d, cell in row["descriptors"].items():
            q = cell["quantiles"]
            print(f"  {d:18s} median {q.get('0.5')}, "
                  f"5-95% {q.get('0.05')} to {q.get('0.95')}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Does the same undeclared convention reach past evaluation and into model inputs and rewards?

The engine finding is about applying a rule bank. But a SMARTS pattern library is queried in two
other places a molecular pipeline cares about, and neither is an evaluation: substructure-derived
FEATURES, and substructure-derived REWARDS -- the filter catalogues that generative chemistry reports
as "fraction passing" and optimises against directly.

If those are convention-dependent too, the finding stops being about a leaderboard and becomes about
a training signal. If they are not, that is worth knowing just as plainly, and this script is written
so the negative is as legible as the positive: it reports every descriptor and every catalogue,
including the ones that do not move.

The comparison is the same one throughout: one molecule, two representations that any pipeline might
hand onward -- as parsed, and after Chem.AddHs -- and no other difference. Substrates are the clean
test split, so nothing new is downloaded and the population is one the paper already names.

What this does NOT establish, and the script's output should not be read as establishing: how often
a real pipeline hands an expanded molecule to a pattern library. Convention-dependence is a property
of the function; incidence is a claim about practice and needs its own evidence. That evidence was
then collected, and it cuts most of this measurement down, so it belongs beside the numbers rather
than after them:

  - A sweep of 129 site-packages roots on one machine finds 39 executable AddHs call sites in
    third-party library code across 12 packages. Exactly ONE reaches a fingerprint or descriptor
    with no intervening RemoveHs and no SMILES round-trip: chemprop, where --adding_h and
    --features_generator are legal together and nothing validates the combination.
  - No third-party package on that machine calls FilterCatalog at all, so the catalogue rows below
    have no demonstrated consumer.
  - The catalogue flips do not survive attribution either. ZINC's come almost entirely from a single
    entry named Non-Hydrogen_atoms, BRENK's entirely from Aliphatic_long_chain, NIH's from five
    count-and-degree predicates. They are the same phenomenon this repository measures in
    retrosynthesis templates -- a primitive that counts explicit structure -- and not an independent
    finding.
  - The zero rows are a defence rather than luck: RDKit's Crippen expands internally, and mordred
    normalises in both directions by construction.
  - A SMILES round-trip anywhere between the expansion and the featurizer neutralises all of it, and
    most pipelines have one.

So the defensible reading of this artifact is narrow: these functions are convention-dependent, at
these magnitudes, and one released package can be driven into it by a flag combination it does not
forbid.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.EState import EState_VSA
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")
CATALOGUES = ("PAINS", "BRENK", "NIH", "ZINC")


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


def _catalog(name: str) -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog(params)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "featurizer_convention.json"))
    args = ap.parse_args()

    truth = json.loads((ROOT / "results/test_references.json").read_text())
    smiles = sorted(truth)
    if args.limit:
        smiles = smiles[: args.limit]

    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    catalogs = {c: _catalog(c) for c in CATALOGUES}

    counts = {k: 0 for k in ("maccs", "morgan", "crippen_logp", "crippen_mr", "estate", "tpsa")}
    bits = {"maccs_bits_moved": 0, "morgan_bits_moved": 0}
    flips = {c: 0 for c in CATALOGUES}
    n = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n += 1
        expanded = Chem.AddHs(Chem.Mol(mol))

        a, b = MACCSkeys.GenMACCSKeys(mol), MACCSkeys.GenMACCSKeys(expanded)
        moved = sum(1 for i in range(a.GetNumBits()) if a[i] != b[i])
        bits["maccs_bits_moved"] += moved
        counts["maccs"] += moved > 0

        fa, fb = morgan.GetFingerprint(mol), morgan.GetFingerprint(expanded)
        mb = len(set(fa.GetOnBits()) ^ set(fb.GetOnBits()))
        bits["morgan_bits_moved"] += mb
        counts["morgan"] += mb > 0

        counts["crippen_logp"] += abs(Crippen.MolLogP(mol) - Crippen.MolLogP(expanded)) > 1e-6
        counts["crippen_mr"] += abs(Crippen.MolMR(mol) - Crippen.MolMR(expanded)) > 1e-6
        counts["tpsa"] += abs(Descriptors.TPSA(mol) - Descriptors.TPSA(expanded)) > 1e-6
        ea, eb = EState_VSA.EState_VSA_(mol), EState_VSA.EState_VSA_(expanded)
        counts["estate"] += any(abs(x - y) > 1e-6 for x, y in zip(ea, eb))

        for name, cat in catalogs.items():
            flips[name] += cat.HasMatch(mol) != cat.HasMatch(expanded)

    rep = {"config": {**_code_version(), "n_molecules": n, "rdkit": Chem.rdBase.rdkitVersion,
                      "population": "results/test_references.json, the clean test substrates",
                      "comparison": "as parsed against Chem.AddHs, nothing else changed"},
           "share_of_molecules_whose_value_changes":
               {k: round(v / max(n, 1), 4) for k, v in counts.items()},
           "mean_bits_moved": {k: round(v / max(n, 1), 2) for k, v in bits.items()},
           "filter_verdict_flips":
               {k: {"molecules": v, "share": round(v / max(n, 1), 4)} for k, v in flips.items()}}

    print(f"{n} molecules, RDKit {rep['config']['rdkit']}\n")
    print("  share of molecules whose value changes under the expansion:")
    for k, v in rep["share_of_molecules_whose_value_changes"].items():
        print(f"    {k:14} {100 * v:6.2f}%")
    print(f"\n  mean bits moved: MACCS {rep['mean_bits_moved']['maccs_bits_moved']} of 167, "
          f"Morgan {rep['mean_bits_moved']['morgan_bits_moved']} of 1024")
    print("\n  substructure filter verdicts that flip:")
    for k, v in rep["filter_verdict_flips"].items():
        print(f"    {k:6} {v['molecules']:>5} molecules  {100 * v['share']:6.2f}%")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

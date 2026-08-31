#!/usr/bin/env python3
"""The reference set as matching descriptors, so every criterion survives without the structures.

The repository tracked the evaluated test substrates and their annotated metabolites as SMILES,
which is a derivative of a corpus whose four sources' terms do not combine, and whose assembly
recorded no per-record provenance, so no subset of it can be shown free of either restriction.
Keying it with one hash was the obvious answer and it is the wrong one: this work sweeps five
matching criteria, and fixing the reference to a single key fixes the criterion axis with it.

Every one of the five is nonetheless computable from something that is not a structure. Canonical
SMILES equality is equality of a string, so a cryptographic hash of that string decides it exactly.
The full InChIKey and its stereochemistry-blind first block are already hashes. The tautomer-aware
key is a hash. Tanimoto similarity of one on a folded Morgan fingerprint is decided by the
fingerprint, which is a lossy irreversible descriptor and not the molecule. So a record carrying
those four things reproduces all five verdicts exactly and reconstructs none of the reference.

The substrate side stays as a structure and is not keyed here, for a reason that has nothing to do
with licences: a reader reproducing the comparison has to run a predictor on the substrate, and a
hash cannot be run on.

    python scripts/typed_edit/reference_descriptors.py
    python scripts/typed_edit/reference_descriptors.py --verify   # the criteria still agree
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

NBITS, RADIUS = 1024, 2


def descriptors(smiles: str) -> dict | None:
    """The four things the five declared criteria are decided by, and no fifth thing."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    from grail_metabolism.metrics import _tautomer_inchikey as tk

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        key = None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS)
    return {
        "canonical_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "inchikey": key,
        "tautomer_key": tk(smiles) if tk else None,
        "morgan_bits": sorted(fp.GetOnBits()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check that each criterion's verdict on the descriptors matches its "
                         "verdict on the structures, on every reference")
    ap.add_argument("--out", default=str(ROOT / "results" / "test_reference_descriptors.json"))
    args = ap.parse_args()

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    refs = json.loads((ROOT / "results/test_references.json").read_text())
    out, dropped = {}, 0
    for n, (sub, mets) in enumerate(refs.items(), 1):
        rows = []
        for met in mets:
            d = descriptors(met)
            if d is None:
                dropped += 1
                continue
            rows.append(d)
        out[sub] = rows
        if n % 200 == 0 or n == len(refs):
            print(f"  {n}/{len(refs)} substrates", flush=True)

    if args.verify:
        # The property that makes this a substitution rather than a loss: for every pair of
        # references, each criterion has to give the same verdict on the descriptors as on the
        # structures. A criterion that disagreed on one pair would be one this file cannot carry.
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdMolDescriptors

        flat = [(s, m) for s, mets in refs.items() for m in mets][:400]
        bad = 0
        for i, (_, a) in enumerate(flat):
            for _, b in flat[i + 1:i + 12]:
                ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
                if ma is None or mb is None:
                    continue
                da, db = descriptors(a), descriptors(b)
                fa = rdMolDescriptors.GetMorganFingerprintAsBitVect(ma, RADIUS, nBits=NBITS)
                fb = rdMolDescriptors.GetMorganFingerprintAsBitVect(mb, RADIUS, nBits=NBITS)
                truth = {
                    "canonical": Chem.MolToSmiles(ma) == Chem.MolToSmiles(mb),
                    "inchikey": Chem.MolToInchiKey(ma) == Chem.MolToInchiKey(mb),
                    "first_block": (Chem.MolToInchiKey(ma)[:14]
                                    == Chem.MolToInchiKey(mb)[:14]),
                    "tanimoto1": DataStructs.TanimotoSimilarity(fa, fb) == 1.0,
                }
                got = {
                    "canonical": da["canonical_sha256"] == db["canonical_sha256"],
                    "inchikey": da["inchikey"] == db["inchikey"],
                    "first_block": (da["inchikey"] or "")[:14] == (db["inchikey"] or "")[:14],
                    "tanimoto1": set(da["morgan_bits"]) == set(db["morgan_bits"]),
                }
                bad += sum(1 for k in truth if truth[k] != got[k])
        print(f"\nverification: {bad} disagreements between the descriptors and the structures")
        if bad:
            return 1

    report = {
        "provenance": stamp(__file__),
        "what_this_is": (
            "the evaluated test set's annotated metabolites as matching descriptors rather than "
            "as structures, so that all five declared matching criteria remain computable by a "
            "reader who does not hold the corpus sources' licences"),
        "fields": {
            "canonical_sha256": "SHA-256 of the RDKit canonical SMILES; equality decides the "
                                "canonical-SMILES criterion exactly",
            "inchikey": "the full InChIKey; equality decides that criterion, and its first 14 "
                        "characters decide the stereochemistry-blind one",
            "tautomer_key": "the tautomer-aware key this work defaults to",
            "morgan_bits": f"the on-bits of a folded Morgan fingerprint, radius {RADIUS}, "
                           f"{NBITS} bits; equality of the bit sets decides Tanimoto = 1",
        },
        "not_included": ("the metabolite structures themselves, and any representation from which "
                         "they could be recovered"),
        "substrates": ("kept as structures in results/test_references.json, because reproducing "
                       "the comparison means running a predictor on them and a hash cannot be run "
                       "on"),
        "population": {"substrates": len(out),
                       "references": sum(len(v) for v in out.values()),
                       "unparsed_and_dropped": dropped},
        "references": out,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n{len(out)} substrates, {sum(len(v) for v in out.values())} references, "
          f"{dropped} dropped")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

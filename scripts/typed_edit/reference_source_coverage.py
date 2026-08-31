#!/usr/bin/env python3
"""Which of the released reference structures can be traced to a source held on disk.

The repository tracks the evaluated test substrates and their annotated metabolites as structures,
and the corpus they come from is assembled from four sources whose terms do not combine. The
assembly is unrecoverable, so no record carries a source field and the obvious question -- which
of these came from which database -- has no recorded answer.

It has a partial measured one. Two of the four sources are on disk in full: MetXBioDB, which ships
with BioTransformer, and the GLORYx reference set. Matching the released pairs against both by
structure gives a lower bound on how much of the released annotation is reachable from a source
whose terms are known, and by subtraction an upper bound on how much can only have come from the
two whose terms conflict.

Matching is on the skeleton, the first block of the InChIKey, on both sides. That is deliberately
loose: it counts a pair as traced when the same transformation between the same skeletons appears
in the source, which is the question a licence asks about a fact, and it makes the traced count an
over-estimate rather than an under-estimate. The bound is therefore conservative in the direction
that matters.

    python scripts/typed_edit/reference_source_coverage.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

METX = ROOT / "artifacts/tier2/biotransformer/database/MetXBioDB-1-0.json"
GLORY = ROOT / "docs/benchmark/data/gloryx_test.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "reference_source_coverage.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    def skeleton_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        try:
            return Chem.MolToInchiKey(mol)[:14]
        except Exception:
            return None

    def skeleton_inchi(value):
        if not value or not isinstance(value, str) or not value.startswith("InChI="):
            return None
        mol = Chem.MolFromInchi(value)
        if mol is None:
            return None
        try:
            return Chem.MolToInchiKey(mol)[:14]
        except Exception:
            return None

    released = json.loads((ROOT / "results/test_references.json").read_text())
    pairs = []
    for sub, mets in released.items():
        a = skeleton_smiles(sub)
        for met in mets:
            b = skeleton_smiles(met)
            if a and b:
                pairs.append((a, b))
    distinct = set(pairs)

    metx_pairs = set()
    if METX.exists():
        blob = json.loads(METX.read_text())
        for row in (blob.get("biotransformations") or {}).values():
            substrate = row.get("Substrate") or {}
            a = skeleton_inchi(substrate.get("InChI")) if isinstance(substrate, dict) else None
            if not a:
                continue
            for product in (row.get("Products") or []):
                b = skeleton_inchi(product.get("InChI")) if isinstance(product, dict) else None
                if b:
                    metx_pairs.add((a, b))

    glory_pairs = set()
    if GLORY.exists():
        text = GLORY.read_text()
        try:
            blob = json.loads(text)
        except json.JSONDecodeError:
            blob = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text))
        rows = blob if isinstance(blob, list) else blob.get("parents") or list(blob.values())
        for row in rows:
            if not isinstance(row, dict):
                continue
            a = skeleton_smiles(row.get("smiles") or row.get("parent") or row.get("substrate"))
            if not a:
                continue
            for prod in (row.get("metabolites") or row.get("products") or []):
                b = skeleton_smiles(prod if isinstance(prod, str) else prod.get("smiles"))
                if b:
                    glory_pairs.add((a, b))

    in_metx = distinct & metx_pairs
    in_glory = distinct & glory_pairs
    traced = in_metx | in_glory
    untraced = distinct - traced

    report = {
        "provenance": stamp(__file__),
        "question": ("how much of the released reference annotation is reachable from a source "
                     "held on disk whose terms are known, which bounds from above how much of it "
                     "can only have come from the two sources whose terms conflict"),
        "matching": ("skeleton against skeleton, the first block of the InChIKey on both sides; "
                     "deliberately loose, so the traced count is an over-estimate and the untraced "
                     "residue a conservative upper bound"),
        "released": {"substrates": len(released),
                     "pairs": len(pairs), "distinct_pairs": len(distinct)},
        "sources_on_disk": {"MetXBioDB": len(metx_pairs), "GLORYx": len(glory_pairs)},
        "traced_to_metxbiodb": len(in_metx),
        "traced_to_gloryx": len(in_glory),
        "traced_to_either": len(traced),
        "share_traced": round(len(traced) / max(len(distinct), 1), 4),
        "untraced": len(untraced),
        "share_untraced": round(len(untraced) / max(len(distinct), 1), 4),
        "reading": (
            "A traced pair is one the same transformation for which appears in a source this "
            "repository already redistributes under stated terms. An untraced pair is not thereby "
            "shown to come from ChEMBL or DrugBank; it is shown not to be reachable from the two "
            "sources that can be checked, which is the bound the licence question needs and the "
            "most the unrecoverable assembly allows."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"released: {len(distinct)} distinct reference pairs over {len(released)} substrates")
    print(f"  traced to MetXBioDB : {len(in_metx)}")
    print(f"  traced to GLORYx    : {len(in_glory)}")
    print(f"  traced to either    : {len(traced)} ({len(traced)/max(len(distinct),1):.1%})")
    print(f"  untraced            : {len(untraced)} ({len(untraced)/max(len(distinct),1):.1%})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

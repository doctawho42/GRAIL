#!/usr/bin/env python3
"""Parse MetaPredictor raw beam output -> {substrate: [ranked preds]} JSON.

MetaPredictor (zhukeyun/Meta-Predictor) runs a two-stage OpenNMT ensemble:
  stage 1 (SoM identifier)   : parent  -> 8 SoM-prompted seqs   (n_best 8)
  stage 2 (metabolite pred.) : each SoM -> 2 metabolite SMILES   (n_best 2)
=> exactly 16 ranked metabolite lines per VALID parent, in beam order, written
to metabolite.txt (space-tokenised SMILES).

We deliberately bypass the repo's process_predictions.py, which collapses each
parent's predictions into a Python set() -- destroying beam rank and applying a
method-specific size/added-atom filter. For an apples-to-apples recall@k
rank-flip we hold each method's RANKED RAW predictions fixed and let
run_match_sensitivity apply the single uniform canonical dedup. So here we only:
  - join tokens, canonicalise (RDKit, isomeric) ,
  - drop blanks / multi-fragment ('.') / the parent itself,
  - dedup preserving first-seen (rank) order.

Alignment: prepare_input_file.py SKIPS invalid SMILES (check_smile =
MolFromSmiles(sanitize=False) is not None), so a dropped parent contributes NO
lines to metabolite.txt. We reproduce that exact skip, advancing the cursor by
16 only for valid parents -- identical to process_predictions.py's own logic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

PER_PARENT = 16  # n_best 8 (SoM) x n_best 2 (metabolite)


def check_smile(smi: str) -> bool:
    return Chem.MolFromSmiles(smi, sanitize=False) is not None


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, isomericSmiles=True)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="mol_id,smiles rows (header-less), input order")
    ap.add_argument("--metabolite-txt", required=True, help="stage-2 output, 16 tokenised lines per valid parent")
    ap.add_argument("--sub-index-map", required=True, help="sub_N -> substrate SMILES (the run_match_sensitivity keys)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    id2smiles = json.loads(Path(args.sub_index_map).read_text())  # sub_N -> substrate SMILES
    drug_lines = [ln for ln in Path(args.input_csv).read_text().split("\n") if ln.strip()]
    pred_lines = Path(args.metabolite_txt).read_text().split("\n")
    # Remove ONLY the final-newline artifact, not every trailing blank: onmt emits
    # exactly n_best lines per source (some may legitimately be empty beams), so
    # over-stripping would truncate the last parent's 16-line block and drift the
    # cursor. One pop restores an exact 16 x n_valid line count for the integrity check.
    if pred_lines and pred_lines[-1] == "":
        pred_lines.pop()

    n_valid = sum(1 for ln in drug_lines if check_smile(ln.split(",", 1)[1]))
    expected = n_valid * PER_PARENT
    if len(pred_lines) != expected:
        print(f"WARNING: metabolite.txt has {len(pred_lines)} lines, expected "
              f"{expected} (={n_valid} valid parents x {PER_PARENT}). Alignment may be off.",
              file=sys.stderr)

    out: Dict[str, List[str]] = {}
    idx = 0
    n_with_pred = 0
    sizes = []
    for ln in drug_lines:
        mol_id, smiles = ln.split(",", 1)
        if not check_smile(smiles):
            print(f"skip invalid parent {mol_id}: {smiles}", file=sys.stderr)
            continue
        block = pred_lines[idx: idx + PER_PARENT]
        idx += PER_PARENT
        parent_c = canon(smiles)
        ranked, seen = [], set()
        for raw in block:
            smi = "".join(raw.strip().split())  # un-tokenise
            if not smi or "." in smi:           # blank beam / multi-fragment
                continue
            c = canon(smi)
            if c is None or c == parent_c or c in seen:
                continue
            seen.add(c)
            ranked.append(c)
        key = id2smiles[mol_id]                  # the substrate SMILES run_match_sensitivity keys on
        out[key] = ranked
        if ranked:
            n_with_pred += 1
            sizes.append(len(ranked))

    Path(args.out).write_text(json.dumps(out))
    med = sorted(sizes)[len(sizes) // 2] if sizes else 0
    print(f"wrote {len(out)} substrates ({n_with_pred} with >=1 prediction), "
          f"median output size {med}, max {max(sizes) if sizes else 0} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Parse MetaTrans 6-model beam output -> {substrate: [ranked preds]} JSON, with an ensemble-vote
proxy ranking.

MetaTrans (KavrakiLab, Litsa et al. 2020) is a 6-model transformer ensemble; each model emits
`beam` (=5) hypotheses per valid parent, in beam order, to predictions/model{1..6}_beam5.txt.
The repo's process_predictions.py collapses all 6*beam predictions per parent into a Python
set() -- destroying rank (open issue #3 is exactly "scores are missing"). For an apples-to-apples
recall@k rank-flip we need a RANKED list, so we derive a proxy ranking:

  rank key = (-vote_count, best_beam_pos, first_seen)
    vote_count    = # of the 6*beam model/beam slots that produced this canonical prediction
                    (ensemble agreement -- the natural confidence signal for a union ensemble)
    best_beam_pos = the lowest beam position (0=top) at which any model produced it

Alignment mirrors process_predictions.py exactly: prepare_input_file.py SKIPS parents whose
check_smile (Chem.MolFromSmiles is not None, sanitized) fails, contributing NO lines; we advance
the per-model cursor by `beam` only for valid parents. Predictions are un-tokenised, canonicalised
(RDKit isomeric), and the parent + blanks + multi-fragment '.' are dropped -- matching the uniform
treatment the other Tier-2 methods get before run_match_sensitivity applies its canonical dedup.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def check_smile(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None  # matches MetaTrans utils.check_smile (sanitized)


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, isomericSmiles=True)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="mol_id,smiles rows (header-less), input order")
    ap.add_argument("--pred-dir", required=True, help="dir with model{1..6}_beam{B}.txt")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--models", type=int, default=6)
    ap.add_argument("--sub-index-map", required=True, help="sub_N -> substrate SMILES (run_match_sensitivity keys)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    id2smiles = json.loads(Path(args.sub_index_map).read_text())
    drug_lines = [ln for ln in Path(args.input_csv).read_text().split("\n") if ln.strip()]

    # load each model's predictions (un-tokenised on read; keep blank beams to preserve line count)
    model_lines: List[List[str]] = []
    for m in range(1, args.models + 1):
        p = Path(args.pred_dir) / f"model{m}_beam{args.beam}.txt"
        raw = p.read_text().split("\n")
        if raw and raw[-1] == "":
            raw.pop()  # drop only the final-newline artifact (exact beam*n_valid count)
        model_lines.append(["".join(ln.strip().split()) for ln in raw])

    n_valid = sum(1 for ln in drug_lines if check_smile(ln.split(",", 1)[1]))
    for m, ml in enumerate(model_lines, 1):
        if len(ml) != n_valid * args.beam:
            print(f"WARNING: model{m} has {len(ml)} lines, expected {n_valid*args.beam} "
                  f"(={n_valid} valid x beam {args.beam}); alignment may be off.", file=sys.stderr)

    out: Dict[str, List[str]] = {}
    idx = 0
    n_with_pred, sizes = 0, []
    for ln in drug_lines:
        mol_id, smiles = ln.split(",", 1)
        if not check_smile(smiles):
            print(f"skip invalid parent {mol_id}: {smiles}", file=sys.stderr)
            continue
        parent_c = canon(smiles)
        votes: "OrderedDict[str, list]" = OrderedDict()  # canon -> [vote_count, best_beam_pos]
        for ml in model_lines:
            for pos in range(args.beam):
                smi = ml[idx + pos] if idx + pos < len(ml) else ""
                if not smi or "." in smi:
                    continue
                c = canon(smi)
                if c is None or c == parent_c:
                    continue
                if c not in votes:
                    votes[c] = [0, pos]
                votes[c][0] += 1
                votes[c][1] = min(votes[c][1], pos)
        idx += args.beam
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
        preds = [c for c, _ in ranked]
        key = id2smiles[mol_id]
        out[key] = preds
        if preds:
            n_with_pred += 1
            sizes.append(len(preds))

    Path(args.out).write_text(json.dumps(out))
    med = sorted(sizes)[len(sizes) // 2] if sizes else 0
    print(f"wrote {len(out)} substrates ({n_with_pred} with >=1 prediction), "
          f"median output size {med}, max {max(sizes) if sizes else 0} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

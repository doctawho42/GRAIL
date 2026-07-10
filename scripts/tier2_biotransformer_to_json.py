#!/usr/bin/env python3
"""Convert BioTransformer's -ocsv output into the run_match_sensitivity.py --extra format.

BioTransformer writes one row per predicted metabolite with columns `SMILES` (the
metabolite) and `Precursor ID` / `Precursor SMILES` (the parent). We group metabolites by
parent, keep BioTransformer's file order as the ranking, drop the parent itself and
canonical-duplicate metabolites, and map each parent back to our input substrate SMILES
(via the SDF names sub_N -> sub_index_map.json, falling back to canonical Precursor SMILES).

Output: JSON {substrate_smiles: [ranked_predicted_smiles, ...]} covering ALL input
substrates (empty list if BioTransformer produced none) so recall is scored fairly.

Usage:
  python scripts/tier2_biotransformer_to_json.py \
      --csv artifacts/tier2/bt_out.csv \
      --index-map artifacts/tier2/sub_index_map.json \
      --substrates artifacts/tier2/substrates.json \
      --out artifacts/tier2/biotransformer_preds.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def _canon(s: str) -> str:
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="artifacts/tier2/bt_out.csv")
    ap.add_argument("--index-map", default="artifacts/tier2/sub_index_map.json")
    ap.add_argument("--substrates", default="artifacts/tier2/substrates.json")
    ap.add_argument("--out", default="artifacts/tier2/biotransformer_preds.json")
    args = ap.parse_args()

    index_map = json.loads(Path(args.index_map).read_text())        # sub_N -> substrate SMILES
    substrates = json.loads(Path(args.substrates).read_text())      # the 150 input SMILES
    canon_to_sub = {_canon(s): s for s in substrates}               # fallback map by structure

    # group metabolite SMILES per substrate, in file order
    by_sub: dict[str, list[str]] = {s: [] for s in substrates}
    rows = list(csv.DictReader(open(args.csv)))
    for r in rows:
        met = (r.get("SMILES") or "").strip()
        if not met:
            continue
        pid = (r.get("Precursor ID") or "").strip()
        sub = index_map.get(pid)
        if sub is None:  # fall back to matching the precursor structure
            sub = canon_to_sub.get(_canon((r.get("Precursor SMILES") or "").strip()))
        if sub is None:
            continue
        by_sub.setdefault(sub, []).append(met)

    # per substrate: drop the parent + canonical-duplicate metabolites, keep order
    out: dict[str, list[str]] = {}
    for sub, mets in by_sub.items():
        parent_c = _canon(sub)
        seen, ranked = set(), []
        for m in mets:
            c = _canon(m)
            if c == parent_c or c in seen:
                continue
            seen.add(c)
            ranked.append(m)
        out[sub] = ranked

    Path(args.out).write_text(json.dumps(out))
    n_nonempty = sum(1 for v in out.values() if v)
    sizes = sorted(len(v) for v in out.values())
    med = sizes[len(sizes) // 2] if sizes else 0
    print(f"wrote {len(out)} substrates ({n_nonempty} with >=1 prediction), "
          f"median output size {med}, max {max(sizes) if sizes else 0} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

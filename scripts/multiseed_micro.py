"""Per-seed macro + micro recall@15 (tautomer) from the exported deployed predictions.

The multi-seed driver reports the ensemble-eval macro recall; this recomputes BOTH the macro
(per-substrate mean) and the micro (ratio-of-sums, = the manuscript's pooled 0.261 frame) from
each seed's `predictions/test_predictions.csv`, using the same tautomer-InChIKey matching as
metrics.py / factorize_recall, so the headline can be reported as mean±std over seeds in both
frames. Validates by reproducing the ensemble macro (seed0 0.3437, seed1 0.3315).

Usage: python scripts/multiseed_micro.py artifacts/multiseed_full5000_seed{0,1,2}/predictions/test_predictions.csv
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _tautomer_inchikey

K = 15


def seed_recalls(csv_path: str):
    """Return (macro_recall@15, micro_recall@15) under tautomer-InChIKey matching."""
    hits_sum = 0
    true_sum = 0
    per_sub_recall: List[float] = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pred_smiles = [s for s in (row.get("predicted") or "").split("|") if s][:K]
            real_smiles = [s for s in (row.get("real") or "").split("|") if s]
            if not real_smiles:
                continue
            true_tk = {_tautomer_inchikey(s) for s in real_smiles}
            pred_tk = {_tautomer_inchikey(s) for s in pred_smiles}
            h = len(pred_tk & true_tk)
            u = len(true_tk)
            hits_sum += h
            true_sum += u
            per_sub_recall.append(h / u)
    macro = st.mean(per_sub_recall) if per_sub_recall else 0.0
    micro = hits_sum / true_sum if true_sum else 0.0
    return macro, micro, len(per_sub_recall)


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print("usage: multiseed_micro.py <test_predictions.csv> [...]")
        return 2
    macros, micros = [], []
    print(f"{'seed/file':<60} {'macro@15':>9} {'micro@15':>9} {'n':>6}")
    for p in paths:
        if not Path(p).exists():
            print(f"{p:<60} (missing)")
            continue
        macro, micro, n = seed_recalls(p)
        macros.append(macro)
        micros.append(micro)
        print(f"{p.split('/')[-3] if '/' in p else p:<60} {macro:>9.4f} {micro:>9.4f} {n:>6}")
    if len(macros) >= 2:
        print("\n=== mean ± s.d. over seeds ===")
        print(f"  macro recall@15: {st.mean(macros):.4f} ± {st.pstdev(macros):.4f}   values={[round(v,4) for v in macros]}")
        print(f"  micro recall@15: {st.mean(micros):.4f} ± {st.pstdev(micros):.4f}   values={[round(v,4) for v in micros]}")
        print("  (published single deployed checkpoint: macro 0.330 / micro 0.261, factorize_recall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Quantify whether the residual structure-leak inflates GRAIL's recall.

The clean splits are substrate-disjoint but not molecule-disjoint: ~10% of test substrates appear
as a TRAIN metabolite (see scripts/audit_leakage.py). This partitions the deployed test predictions
into LEAK-FREE (test substrate never seen as any train molecule) vs LEAKED, and reports macro+micro
recall@15 (tautomer) on each. If recall is comparable, the leak is benign and the headline holds on
the leak-free subset; if the leaked subset is much higher, the leak inflates the number.

Usage: python scripts/leak_impact.py [predictions.csv]
  default predictions: artifacts/full5000_single/predictions/test_predictions.csv (the headline checkpoint)
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _tautomer_inchikey
from scripts.fix_splits import (
    CLEAN_TRIPLES,
    SPLIT_SPECS,
    build_canonical_triples,
    canonicalize_smiles,
    molecule_set,
)

K = 15
DEFAULT_PREDS = ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"


def partition_recall(rows, in_leak_set):
    """Return (macro, micro, n) over rows whose canonical substrate's leak-membership == in_leak_set-selected."""
    per_sub = []
    hits = trues = 0
    for canon_sub, pred_smiles, real_smiles in rows:
        true_tk = {_tautomer_inchikey(s) for s in real_smiles}
        if not true_tk:
            continue
        pred_tk = {_tautomer_inchikey(s) for s in pred_smiles[:K]}
        h = len(pred_tk & true_tk)
        u = len(true_tk)
        hits += h
        trues += u
        per_sub.append(h / u)
    macro = st.mean(per_sub) if per_sub else 0.0
    micro = hits / trues if trues else 0.0
    return macro, micro, len(per_sub)


def factorization_partition(fact_path, train_mols):
    """Full-1170 leak partition using recall_factorization.json's precomputed per-substrate H/U
    (the deployed 0.330 macro / 0.261 micro headline pipeline). macro = mean(H/U); micro = sumH/sumU."""
    import json
    rows = json.loads(Path(fact_path).read_text())["per_substrate"]
    parts = {"all": [], "clean": [], "leaked": []}
    for r in rows:
        canon = canonicalize_smiles(r["sub"]) or r["sub"]
        rec = (r["H"] / r["U"]) if r["U"] else 0.0
        parts["all"].append((r["H"], r["U"], rec))
        (parts["leaked"] if canon in train_mols else parts["clean"]).append((r["H"], r["U"], rec))
    print(f"\n=== FULL-1170 (recall_factorization.json precomputed H/U — the 0.330/0.261 headline) ===")
    print(f"{'partition':<28} {'n':>6} {'macro@15':>9} {'micro@15':>9}")
    for name, key in [("ALL", "all"), ("LEAK-FREE (sub not in train)", "clean"), ("LEAKED (sub in train mols)", "leaked")]:
        p = parts[key]
        macro = st.mean([r for _, _, r in p]) if p else 0.0
        micro = sum(h for h, _, _ in p) / sum(u for _, u, _ in p) if sum(u for _, u, _ in p) else 0.0
        print(f"{name:<28} {len(p):>6} {macro:>9.4f} {micro:>9.4f}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--factorization":
        train_triples, _ = build_canonical_triples("train", SPLIT_SPECS["train"][0], CLEAN_TRIPLES["train"])
        train_mols = molecule_set(train_triples)
        factorization_partition(ROOT / "results" / "recall_factorization.json", train_mols)
        return 0
    preds_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PREDS
    train_triples, _ = build_canonical_triples("train", SPLIT_SPECS["train"][0], CLEAN_TRIPLES["train"])
    train_mols = molecule_set(train_triples)

    leaked, clean = [], []
    with open(preds_path) as fh:
        for row in csv.DictReader(fh):
            sub = row["substrate"]
            canon = canonicalize_smiles(sub) or sub
            pred = [s for s in (row.get("predicted") or "").split("|") if s]
            real = [s for s in (row.get("real") or "").split("|") if s]
            entry = (canon, pred, real)
            (leaked if canon in train_mols else clean).append(entry)

    all_rows = leaked + clean
    m_all, u_all, n_all = partition_recall(all_rows, None)
    m_clean, u_clean, n_clean = partition_recall(clean, None)
    m_leak, u_leak, n_leak = partition_recall(leaked, None)

    print(f"predictions: {preds_path}")
    print(f"{'partition':<28} {'n':>6} {'macro@15':>9} {'micro@15':>9}")
    print(f"{'ALL test substrates':<28} {n_all:>6} {m_all:>9.4f} {u_all:>9.4f}")
    print(f"{'LEAK-FREE (sub not in train)':<28} {n_clean:>6} {m_clean:>9.4f} {u_clean:>9.4f}")
    print(f"{'LEAKED (sub in train mols)':<28} {n_leak:>6} {m_leak:>9.4f} {u_leak:>9.4f}")
    print(f"\ninterpretation: if LEAK-FREE macro ({m_clean:.4f}) ~= ALL ({m_all:.4f}), the residual "
          f"structure-leak does NOT materially inflate the headline; the paper's recall holds on the "
          f"{n_clean} substrates with no training-structure exposure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

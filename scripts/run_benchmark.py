#!/usr/bin/env python3
"""Field-standard benchmark on the real test split.

Reports, with InChIKey matching (the MetaTrans/GLORYx/LAGOM convention):
  1. GRAIL rule-bank RECALL CEILING  -- the max achievable recall of the default
     rule bank (any rule may fire); the hard upper bound for a rule-based method.
  2. A SyGMa baseline (phase1+phase2) -- recall@k, precision@k, mean output size --
     the closest rule-based comparator to GRAIL's architecture.

Usage:
  python scripts/run_benchmark.py [--sample N] [--seed 42] [--ks 5 10 12 15]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from grail_metabolism.metrics import _inchikey
from grail_metabolism.utils.preparation import (
    apply_rules_to_molecule,
    load_default_rules,
    load_phase2_rules,
)

DATA = ROOT / "grail_metabolism" / "data"


# ---- test-set loading (lightweight, canonical SMILES; no tautomer standardization) ----
def _read_positives(path: Path):
    pos = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 3 and parts[2] == "1":
                pos.append((int(parts[0]), int(parts[1])))
    return pos


def _load_ids_to_smiles(sdf: Path, ids: Set[int]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
    for fallback, mol in enumerate(supplier, start=1):
        if mol is None:
            continue
        try:
            idx = int(mol.GetProp("Index")) if mol.HasProp("Index") else fallback
        except Exception:
            idx = fallback
        if idx not in ids:
            continue
        smi = mol.GetProp("SMILES") if mol.HasProp("SMILES") else Chem.MolToSmiles(mol)
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            out[idx] = Chem.MolToSmiles(m)
        if len(out) == len(ids):
            break
    return out


def load_test_map(sample: Optional[int], seed: int) -> Dict[str, Set[str]]:
    triples_path = DATA / "test_triples_clean.txt"
    if not triples_path.exists():
        triples_path = DATA / "test_triples.txt"
    positives = _read_positives(triples_path)
    ids = {a for a, _ in positives} | {b for _, b in positives}
    print(f"Loading {len(ids)} molecules from {DATA/'test.sdf'} ...", flush=True)
    id2smi = _load_ids_to_smiles(DATA / "test.sdf", ids)
    substrate_map: Dict[str, Set[str]] = {}
    for sub_id, prod_id in positives:
        s, p = id2smi.get(sub_id), id2smi.get(prod_id)
        if s and p:
            substrate_map.setdefault(s, set()).add(p)
    subs = sorted(substrate_map)
    if sample and sample < len(subs):
        random.Random(seed).shuffle(subs)
        subs = subs[:sample]
        substrate_map = {s: substrate_map[s] for s in subs}
    print(f"Test substrates: {len(substrate_map)}, true pairs: {sum(len(v) for v in substrate_map.values())}", flush=True)
    return substrate_map


def ik_set(smiles_iter) -> Set[str]:
    return {_inchikey(s) for s in smiles_iter}


# ---- 1. GRAIL rule-bank recall ceiling ----
def grail_ceiling(test_map: Dict[str, Set[str]], rules: List[str]) -> Dict[str, float]:
    recovered = 0
    total = 0
    subs_with_hit = 0
    cand_sizes = []
    start = time.perf_counter()
    items = list(test_map.items())
    for i, (sub, true_prods) in enumerate(items, start=1):
        if i == 1 or i % 50 == 0 or i == len(items):
            print(f"  [ceiling] {i}/{len(items)} ({time.perf_counter()-start:.0f}s)", flush=True)
        mol = Chem.MolFromSmiles(sub)
        true_ik = ik_set(true_prods)
        if mol is None:
            total += len(true_ik)
            continue
        products = apply_rules_to_molecule(mol, rules, normalization_mode="canonical")
        gen_ik = ik_set(products.keys())
        cand_sizes.append(len(gen_ik))
        hit = len(true_ik & gen_ik)
        recovered += hit
        total += len(true_ik)
        if hit:
            subs_with_hit += 1
    return {
        "recall_ceiling": recovered / total if total else 0.0,
        "fraction_substrates_with_any_hit": subs_with_hit / len(items) if items else 0.0,
        "mean_candidates_per_substrate": sum(cand_sizes) / len(cand_sizes) if cand_sizes else 0.0,
        "true_recovered": recovered,
        "true_total": total,
        "n_rules": len(rules),
    }


def grail_ceiling_delta(test_map: Dict[str, Set[str]], base_rules: List[str], phase2_rules: List[str]) -> Dict[str, float]:
    """Recall ceiling of the base bank vs base+phase2, on the SAME substrates (one pass)."""
    base_rec = comb_rec = total = base_hit = comb_hit = 0
    start = time.perf_counter()
    items = list(test_map.items())
    for i, (sub, true_prods) in enumerate(items, start=1):
        if i == 1 or i % 50 == 0 or i == len(items):
            print(f"  [phase2-delta] {i}/{len(items)} ({time.perf_counter()-start:.0f}s)", flush=True)
        true_ik = ik_set(true_prods)
        total += len(true_ik)
        mol = Chem.MolFromSmiles(sub)
        if mol is None:
            continue
        base_ik = ik_set(apply_rules_to_molecule(mol, base_rules, normalization_mode="canonical").keys())
        p2_ik = ik_set(apply_rules_to_molecule(mol, phase2_rules, normalization_mode="canonical").keys())
        comb_ik = base_ik | p2_ik
        bh, ch = len(true_ik & base_ik), len(true_ik & comb_ik)
        base_rec += bh
        comb_rec += ch
        base_hit += 1 if bh else 0
        comb_hit += 1 if ch else 0
    n = len(items)
    return {
        "base_recall_ceiling": base_rec / total if total else 0.0,
        "base_plus_phase2_recall_ceiling": comb_rec / total if total else 0.0,
        "ceiling_lift": (comb_rec - base_rec) / total if total else 0.0,
        "extra_metabolites_recovered": comb_rec - base_rec,
        "base_fraction_with_hit": base_hit / n if n else 0.0,
        "base_plus_phase2_fraction_with_hit": comb_hit / n if n else 0.0,
        "true_total": total,
        "n_phase2_rules": len(phase2_rules),
    }


# ---- 2. SyGMa baseline ----
def sygma_baseline(test_map: Dict[str, Set[str]], ks: List[int]) -> Optional[Dict[str, float]]:
    try:
        import sygma
    except ImportError:
        print("sygma not installed; skipping baseline (pip install sygma)", flush=True)
        return None
    scenario = sygma.Scenario([[sygma.ruleset["phase1"], 1], [sygma.ruleset["phase2"], 1]])
    recall_at = {k: 0.0 for k in ks}
    prec_at = {k: 0.0 for k in ks}
    out_sizes = []
    n = 0
    start = time.perf_counter()
    items = list(test_map.items())
    for i, (sub, true_prods) in enumerate(items, start=1):
        if i == 1 or i % 50 == 0 or i == len(items):
            print(f"  [sygma] {i}/{len(items)} ({time.perf_counter()-start:.0f}s)", flush=True)
        mol = Chem.MolFromSmiles(sub)
        if mol is None:
            continue
        try:
            tree = scenario.run(mol)
            tree.calc_scores()  # required before to_smiles(), which sorts by score
            ranked = tree.to_smiles()  # list of [smiles, score], parent first
        except Exception:
            continue
        preds = []
        for entry in ranked:
            smi = entry[0]
            ikey = _inchikey(smi)
            if ikey not in preds:
                preds.append(ikey)
        # drop the parent (rank 0 is usually the parent molecule)
        parent_ik = _inchikey(sub)
        preds = [p for p in preds if p != parent_ik]
        out_sizes.append(len(preds))
        true_ik = ik_set(true_prods)
        n += 1
        for k in ks:
            topk = set(preds[:k])
            hit = len(topk & true_ik)
            recall_at[k] += hit / len(true_ik) if true_ik else 0.0
            prec_at[k] += hit / k
    if n == 0:
        return None
    return {
        "recall_at": {str(k): recall_at[k] / n for k in ks},
        "precision_at": {str(k): prec_at[k] / n for k in ks},
        "mean_output_size": sum(out_sizes) / len(out_sizes) if out_sizes else 0.0,
        "n_substrates": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="evaluate on a random N-substrate sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 12, 15])
    ap.add_argument("--with-phase2", action="store_true", help="add curated phase II bank to GRAIL ceiling")
    ap.add_argument("--phase2-delta", action="store_true", help="report base vs base+phase2 ceiling on the same substrates (one pass)")
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "benchmark_report.json"))
    args = ap.parse_args()

    test_map = load_test_map(args.sample, args.seed)

    if args.phase2_delta:
        base = load_default_rules()
        p2 = [r for r in load_phase2_rules() if r not in set(base)]
        print(f"\n== phase II ceiling delta (base={len(base)} rules, +{len(p2)} phase II) ==", flush=True)
        delta = grail_ceiling_delta(test_map, base, p2)
        print(json.dumps(delta, indent=2), flush=True)
        out_path = Path(args.out.replace(".json", "_phase2_delta.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"n_test_substrates": len(test_map), "matching": "inchikey", "phase2_delta": delta}, indent=2))
        print(f"\nWrote {out_path}", flush=True)
        return 0

    rules = load_default_rules()
    if args.with_phase2:
        extra = [r for r in load_phase2_rules() if r not in set(rules)]
        rules = rules + extra
        print(f"Added {len(extra)} phase II rules -> {len(rules)} total", flush=True)

    print("\n== GRAIL rule-bank recall ceiling ==", flush=True)
    ceiling = grail_ceiling(test_map, rules)
    print(json.dumps(ceiling, indent=2), flush=True)

    print("\n== SyGMa baseline ==", flush=True)
    sygma_metrics = sygma_baseline(test_map, args.ks)
    if sygma_metrics:
        print(json.dumps(sygma_metrics, indent=2), flush=True)

    report = {
        "n_test_substrates": len(test_map),
        "matching": "inchikey",
        "with_phase2_rules": bool(args.with_phase2),
        "grail_rule_bank_ceiling": ceiling,
        "sygma_baseline": sygma_metrics,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

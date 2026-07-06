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

from grail_metabolism.eval.diversity import (
    circles_count,
    mean_pairwise_tanimoto,
    n_unique_scaffolds,
)
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


def grail_ceiling_depth(
    test_map: Dict[str, Set[str]],
    rules: List[str],
    depth: int,
    beam: int = 25,
    node_budget: int = 4000,
) -> Dict[str, object]:
    """Recall ceiling when rules may be applied up to `depth` times (no policy).

    A TRUE unbounded depth-2 ceiling is computationally infeasible (all 7581 rules applied
    to every one of ~195 depth-1 products per substrate). `beam` caps the frontier breadth
    per level (deterministically, by canonical SMILES), so this is a breadth-capped LOWER
    BOUND on the real multi-step ceiling: if even this lifts recall over the depth-1
    ceiling, multi-step rule application genuinely helps. `node_budget` is a global per-
    substrate safety cap on total expansions.
    """
    recovered = total = subs_with_hit = 0
    cand_sizes: List[int] = []
    start = time.perf_counter()
    items = list(test_map.items())
    for i, (sub, true_prods) in enumerate(items, start=1):
        if i == 1 or i % 25 == 0 or i == len(items):
            print(f"  [ceiling d={depth} beam={beam}] {i}/{len(items)} ({time.perf_counter()-start:.0f}s)", flush=True)
        true_ik = ik_set(true_prods)
        total += len(true_ik)
        if Chem.MolFromSmiles(sub) is None:
            continue
        root_ik = _inchikey(sub)
        visited_ik = {root_ik}
        frontier = [sub]
        expansions = 0
        for _ in range(depth):
            next_candidates: Set[str] = set()
            for m_smi in frontier:
                if expansions >= node_budget:
                    break
                m = Chem.MolFromSmiles(m_smi)
                if m is None:
                    continue
                expansions += 1
                for p_smi in apply_rules_to_molecule(m, rules, normalization_mode="canonical").keys():
                    p_ik = _inchikey(p_smi)
                    if p_ik not in visited_ik:
                        visited_ik.add(p_ik)
                        next_candidates.add(p_smi)
            if not next_candidates or expansions >= node_budget:
                break
            # breadth cap: keep a deterministic subset as the next frontier
            frontier = sorted(next_candidates)[:beam] if beam and len(next_candidates) > beam else sorted(next_candidates)
        gen_ik = visited_ik - {root_ik}
        cand_sizes.append(len(gen_ik))
        hit = len(true_ik & gen_ik)
        recovered += hit
        if hit:
            subs_with_hit += 1
    return {
        "depth": depth,
        "beam": beam,
        "node_budget": node_budget,
        "recall_ceiling_lower_bound": recovered / total if total else 0.0,
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


_GAP_CLASSES = [
    ("hydroxylation/oxidation (+O, +16)", 15.995),
    ("dioxidation (+2O, +32)", 31.990),
    ("methylation (+CH2, +14)", 14.016),
    ("demethylation (-CH2, -14)", -14.016),
    ("desaturation (-H2, -2)", -2.016),
    ("reduction (+H2, +2)", 2.016),
    ("hydration/hydrolysis (+H2O, +18)", 18.011),
    ("dehydration (-H2O, -18)", -18.011),
    ("dihydrodiol (+H2O2-ish, +34)", 34.005),
    ("decarboxylation (-CO2, -44)", -43.990),
    ("acetylation (+C2H2O, +42)", 42.011),
    ("glucuronidation (+C6H8O6, +176)", 176.032),
    ("sulfation (+SO3, +80)", 79.957),
    ("glycine conjugation (+57)", 57.021),
    ("taurine conjugation (+107)", 107.004),
    ("glutathione conjugation (+305)", 305.068),
    ("cysteine conjugation (+119)", 119.004),
    ("mercapturate/NAC (+161)", 161.015),
    ("oxidative deamination (-NH, +O-1, +1)", 0.984),
    ("isomerization/rearrangement (~0)", 0.0),
]


def classify_gap(delta_mw: Optional[float]) -> str:
    if delta_mw is None:
        return "other (invalid MW)"
    for label, target in _GAP_CLASSES:
        if abs(delta_mw - target) < 1.5:
            return label
    return f"other (deltaMW~{round(delta_mw)})"


def gap_analysis(test_map: Dict[str, Set[str]], rules: List[str]) -> Dict[str, object]:
    """Bucket the TRUE metabolites the base bank fails to reach, by substrate->metabolite
    mass shift, to reveal which transformation classes are missing from the bank."""
    from collections import Counter
    from rdkit.Chem import Descriptors

    buckets: Counter = Counter()
    examples: Dict[str, list] = {}
    n_uncovered = 0
    n_total = 0
    start = time.perf_counter()
    items = list(test_map.items())
    for i, (sub, true_prods) in enumerate(items, start=1):
        if i == 1 or i % 50 == 0 or i == len(items):
            print(f"  [gap] {i}/{len(items)} ({time.perf_counter()-start:.0f}s)", flush=True)
        mol = Chem.MolFromSmiles(sub)
        if mol is None:
            continue
        sub_mw = Descriptors.MolWt(mol)
        gen_ik = ik_set(apply_rules_to_molecule(mol, rules, normalization_mode="canonical").keys())
        for prod in true_prods:
            n_total += 1
            if _inchikey(prod) in gen_ik:
                continue
            n_uncovered += 1
            pm = Chem.MolFromSmiles(prod)
            delta = (Descriptors.MolWt(pm) - sub_mw) if pm is not None else None
            label = classify_gap(delta)
            buckets[label] += 1
            if len(examples.setdefault(label, [])) < 3:
                examples[label].append({"sub": sub, "met": prod, "deltaMW": round(delta, 1) if delta is not None else None})
    ranked = buckets.most_common()
    return {
        "n_true_total": n_total,
        "n_uncovered": n_uncovered,
        "uncovered_fraction": n_uncovered / n_total if n_total else 0.0,
        "top_missing_transformation_classes": [
            {"class": label, "count": count, "fraction_of_uncovered": count / n_uncovered if n_uncovered else 0.0,
             "examples": examples.get(label, [])}
            for label, count in ranked
        ],
    }


# ---- 2. SyGMa baseline ----
def sygma_baseline(test_map: Dict[str, Set[str]], ks: List[int]) -> Optional[Dict[str, object]]:
    """SyGMa phase1+phase2 baseline: recall@k/precision@k (plain-InChIKey matching,
    unchanged) plus the diversity triplet (BASE-04, canonicalization-consistent via
    eval/diversity.py).

    NOTE (Pitfall 4, canonicalization divergence -- documented, not a bug): this
    function's recall_at/precision_at path matches predictions via the PLAIN
    _inchikey (line below, unchanged from before BASE-04), while the
    mean_pairwise_tanimoto/circles@t0.4/circles@t0.7/n_unique_scaffolds diversity
    triplet is computed by feeding the RAW ranked SMILES (pre plain-InChIKey dedup,
    parent dropped by SMILES equality) into eval/diversity.py's functions, each of
    which runs its OWN internal tautomer-InChIKey dedup
    (_dedup_smiles_by_tautomer_ik) before fingerprinting. A tautomer pair in SyGMa's
    raw output can therefore count as TWO distinct predictions in recall_at/
    precision_at but collapse to ONE molecule in the diversity triplet -- this is
    expected and correct (it keeps the triplet consistent with every other baseline
    and with the gflownet path), not a mismatch to "fix".
    """
    try:
        import sygma
    except ImportError:
        print("sygma not installed; skipping baseline (pip install -e .[baselines])", flush=True)
        return None
    scenario = sygma.Scenario([[sygma.ruleset["phase1"], 1], [sygma.ruleset["phase2"], 1]])
    recall_at = {k: 0.0 for k in ks}
    prec_at = {k: 0.0 for k in ks}
    out_sizes = []
    n = 0
    # Diversity-triplet accumulators (BASE-04): running sum over substrates,
    # divided by n at the end -- mean over substrates, NOT the last substrate's
    # numbers (mirrors how recall_at/prec_at are aggregated above).
    div_mean_pairwise_tanimoto = 0.0
    div_circles_t04 = 0.0
    div_circles_t07 = 0.0
    div_n_unique_scaffolds = 0.0
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

        # --- BASE-04 diversity triplet: RAW ranked SMILES (NOT the plain-IK
        # -deduped `preds` above), parent dropped by SMILES equality; each
        # eval/diversity.py function dedups by tautomer-InChIKey internally.
        raw_ranked_smiles = [entry[0] for entry in ranked if entry[0] != sub]
        div_mean_pairwise_tanimoto += mean_pairwise_tanimoto(raw_ranked_smiles)
        div_circles_t04 += circles_count(raw_ranked_smiles, threshold=0.4)
        div_circles_t07 += circles_count(raw_ranked_smiles, threshold=0.7)
        div_n_unique_scaffolds += n_unique_scaffolds(raw_ranked_smiles)
    if n == 0:
        return None
    return {
        "recall_at": {str(k): recall_at[k] / n for k in ks},
        "precision_at": {str(k): prec_at[k] / n for k in ks},
        "mean_output_size": sum(out_sizes) / len(out_sizes) if out_sizes else 0.0,
        "n_substrates": n,
        # BASE-04: the four scalar key NAMES here (mean_pairwise_tanimoto,
        # circles@t0.4, circles@t0.7, n_unique_scaffolds) match
        # run_gflownet.py:_diversity_block's diversity-triplet keys verbatim,
        # for side-by-side reading. NOTE: this dict is flat, unlike
        # _diversity_block's shape, and aggregate_seeds.py does not read
        # sygma's JSON -- so "compatible" here is limited to key-name parity,
        # not a shared schema/consumer.
        "mean_pairwise_tanimoto": div_mean_pairwise_tanimoto / n,
        "circles@t0.4": div_circles_t04 / n,
        "circles@t0.7": div_circles_t07 / n,
        "n_unique_scaffolds": div_n_unique_scaffolds / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="evaluate on a random N-substrate sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 12, 15])
    ap.add_argument("--with-phase2", action="store_true", help="add curated phase II bank to GRAIL ceiling")
    ap.add_argument("--phase2-delta", action="store_true", help="report base vs base+phase2 ceiling on the same substrates (one pass)")
    ap.add_argument("--gap-analysis", action="store_true", help="bucket the metabolites the bank misses by mass-shift class")
    ap.add_argument("--depth", type=int, default=1, help="multi-step ceiling: apply rules up to this depth (breadth-capped lower bound)")
    ap.add_argument("--beam", type=int, default=25, help="frontier breadth cap per level for --depth>1")
    ap.add_argument("--node-budget", type=int, default=4000, help="per-substrate expansion cap for --depth>1")
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "benchmark_report.json"))
    args = ap.parse_args()

    test_map = load_test_map(args.sample, args.seed)

    if args.depth and args.depth > 1:
        rules = load_default_rules()
        print(f"\n== depth-{args.depth} ceiling (breadth-capped lower bound, beam={args.beam}, base={len(rules)} rules) ==", flush=True)
        d1 = grail_ceiling(test_map, rules)
        dD = grail_ceiling_depth(test_map, rules, depth=args.depth, beam=args.beam, node_budget=args.node_budget)
        lift = float(dD["recall_ceiling_lower_bound"]) - float(d1["recall_ceiling"])
        print(json.dumps({"depth1_ceiling": d1, f"depth{args.depth}_ceiling_lower_bound": dD, "lift_over_depth1": lift}, indent=2), flush=True)
        out_path = Path(args.out.replace(".json", f"_depth{args.depth}.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "n_test_substrates": len(test_map), "matching": "inchikey",
            "depth1_ceiling": d1, f"depth{args.depth}_ceiling_lower_bound": dD,
            "lift_over_depth1": lift,
        }, indent=2))
        print(f"\nWrote {out_path}", flush=True)
        return 0

    if args.gap_analysis:
        rules = load_default_rules()
        print(f"\n== gap analysis (uncovered true metabolites by mass-shift class, base={len(rules)} rules) ==", flush=True)
        gap = gap_analysis(test_map, rules)
        print(json.dumps(gap, indent=2), flush=True)
        out_path = Path(args.out.replace(".json", "_gap.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"n_test_substrates": len(test_map), "matching": "inchikey", "gap_analysis": gap}, indent=2))
        print(f"\nWrote {out_path}", flush=True)
        return 0

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

#!/usr/bin/env python3
"""Stage-2 Milestone M0: within-rule vs cross-rule headroom decomposition.

Decomposes the recall@15 headroom (actual->oracle-full) into two components:
  - within-rule:  if only sibling-regioisomers of each rule were sorted optimally
                  (true hits first), keeping cross-rule interleaving fixed.
  - cross-rule:   the remaining gain from optimally reordering rule groups.

This answers: is the reranker's site-contrastive emphasis (regioselectivity)
aimed at the right bottleneck?

Four orderings at each pool size (50, 100):
  actual:             generator order as-is
  oracle-within-rule: within each rule group, move hits to earliest positions
                      of that group (siblings reranked, cross-rule unchanged)
  oracle-cross-rule:  move rule groups containing >=1 hit to front
                      (within-group order = generator order)
  oracle-full:        all hits first in pool (upper bound = rerank ceiling)

Headroom split:
  within-rule headroom = oracle-within-rule - actual
  cross-rule headroom  = oracle-full - oracle-within-rule
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rdkit import Chem

from grail_metabolism.config import DatasetConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.utils.preparation import safe_run_reactants, _normalize_smiles_cached
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_generator


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    return model


def _ikset(smiles_list):
    out = set()
    for s in smiles_list:
        try:
            out.add(_tautomer_inchikey(s))
        except Exception:
            out.add(s)
    return out


def _to_ik(s: str) -> str:
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return s


def _generate_with_rule_id(gen, sub: str, top_k: int, threshold):
    """Like generate_scored_with_details but WITHOUT firing_atoms computation.

    Returns list of (ik, gen_score, rule_id) sorted by gen_score desc.
    Uses tautomer-InChIKey for identity (deduped), keeping best-score rule_id.
    """
    mol, scores, ranked_indices = gen._prepare_generation(sub, top_k, threshold)
    if mol is None:
        return []

    # ik -> {"scores": [score,...], "best": (best_score, rule_id)}
    data: dict = {}
    for index in ranked_indices:
        if index >= len(gen.rule_reactions):
            continue
        reaction = gen.rule_reactions[index]
        if reaction is None:
            continue
        rule_score = float(scores[index])
        seen_normalized: set = set()
        for product_tuple in safe_run_reactants(reaction, mol):
            for product in product_tuple:
                try:
                    smiles = Chem.MolToSmiles(product)
                except Exception:
                    continue
                for fragment in smiles.split("."):
                    fragment = fragment.strip()
                    if not fragment:
                        continue
                    try:
                        normalized = _normalize_smiles_cached(fragment, gen.gen_normalization)
                    except Exception:
                        continue
                    if normalized in seen_normalized:
                        continue
                    seen_normalized.add(normalized)
                    # Compute InChIKey for dedup
                    ik = _to_ik(normalized)
                    entry = data.setdefault(ik, {"scores": [], "best": (-1e9, index)})
                    entry["scores"].append(rule_score)
                    if rule_score > entry["best"][0]:
                        entry["best"] = (rule_score, index)

    out = []
    for ik, entry in data.items():
        agg = gen._aggregate_candidate_scores(entry["scores"])
        _, rule_id = entry["best"]
        out.append((ik, float(agg), int(rule_id)))
    return sorted(out, key=lambda item: (-item[1], item[0]))


def compute_four_recalls(
    pools: list[list[tuple[str, float, int]]],  # [(ik, gen_score, rule_id), ...]
    trues: list[set[str]],
    pool_n: int,
    k: int = 15,
) -> dict[str, float]:
    """Compute the four recall@k orderings for a given pool size."""

    actuals, within_rules, cross_rules, oracles = [], [], [], []

    for pool_all, true_iks in zip(pools, trues):
        if not true_iks:
            continue

        # Truncate to pool_n
        pool = pool_all[:pool_n]  # list of (ik, gen_score, rule_id)
        if not pool:
            continue

        # ---- actual: generator order as-is ----
        actual_top = [ik for ik, _, _ in pool[:k]]
        actual_hits = sum(1 for ik in actual_top if ik in true_iks)
        actuals.append(actual_hits / len(true_iks))

        # ---- oracle-within-rule ----
        # For each rule_id group: take the LIST POSITIONS of its candidates
        # in the generator order, reassign within-group candidates to those
        # positions sorted by (is_hit desc, gen_score desc).
        # Cross-rule interleaving of positions is UNCHANGED.
        rule_to_positions: dict[int, list[int]] = defaultdict(list)
        for pos, (ik, score, rid) in enumerate(pool):
            rule_to_positions[rid].append(pos)

        within_order = list(pool)  # copy
        for rid, positions in rule_to_positions.items():
            siblings = [pool[p] for p in positions]
            # Sort siblings: hits first, then by gen_score desc
            siblings_sorted = sorted(
                siblings,
                key=lambda x: (0 if x[0] in true_iks else 1, -x[1])
            )
            for pos, new_cand in zip(positions, siblings_sorted):
                within_order[pos] = new_cand

        within_top = [ik for ik, _, _ in within_order[:k]]
        within_hits = sum(1 for ik in within_top if ik in true_iks)
        within_rules.append(within_hits / len(true_iks))

        # ---- oracle-cross-rule ----
        # Keep each rule group's INTERNAL order = generator order.
        # Reorder groups so those with >=1 hit come first.
        # Ties by the group's best gen_score.

        # Collect rule groups in generator order (first occurrence order)
        seen_rules: list[int] = []
        seen_set: set[int] = set()
        for _, _, rid in pool:
            if rid not in seen_set:
                seen_rules.append(rid)
                seen_set.add(rid)

        # For each rule group, collect its candidates in generator order
        rule_candidates: dict[int, list[tuple[str, float, int]]] = defaultdict(list)
        for ik, score, rid in pool:
            rule_candidates[rid].append((ik, score, rid))

        def group_has_hit(rid):
            return any(ik in true_iks for ik, _, _ in rule_candidates[rid])

        def group_best_score(rid):
            return max(score for _, score, _ in rule_candidates[rid])

        # Sort groups: hit groups first (by best score desc), then miss groups (by best score desc)
        hit_groups = sorted([r for r in seen_rules if group_has_hit(r)], key=lambda r: -group_best_score(r))
        miss_groups = sorted([r for r in seen_rules if not group_has_hit(r)], key=lambda r: -group_best_score(r))
        cross_order_flat: list[tuple[str, float, int]] = []
        for rid in hit_groups + miss_groups:
            cross_order_flat.extend(rule_candidates[rid])

        cross_top = [ik for ik, _, _ in cross_order_flat[:k]]
        cross_hits = sum(1 for ik in cross_top if ik in true_iks)
        cross_rules.append(cross_hits / len(true_iks))

        # ---- oracle-full: all hits in pool first ----
        pool_iks_in_order = [ik for ik, _, _ in pool]
        hits_in_pool = [ik for ik in pool_iks_in_order if ik in true_iks]
        oracle_hit_count = min(len(hits_in_pool), k)
        oracles.append(oracle_hit_count / len(true_iks))

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "actual": round(mean(actuals), 4),
        "within_rule": round(mean(within_rules), 4),
        "cross_rule": round(mean(cross_rules), 4),
        "full": round(mean(oracles), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Within-vs-cross-rule headroom decomposition")
    ap.add_argument("--ckpt-dir", type=str,
                    default=str(ROOT / "artifacts" / "full5000_priors" / "checkpoints"))
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-substrates", type=int, default=120)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--prior-strength", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    # Load generator
    ck = Path(args.ckpt_dir)
    gen = _load(ck / "generator.pt", lambda a, r: build_generator(GeneratorConfig(**a), r))
    gen.gen_normalization = "canonical"
    gen.prior_strength = args.prior_strength
    gen_threshold = getattr(gen, "calibrated_threshold", None)

    # Load dataset
    dataset = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf",
        train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf",
        val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf",
        test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True,
        standardize=False,
        max_train_substrates=8,
        max_val_substrates=(args.max_substrates if args.split == "val" else 8),
        max_test_substrates=(args.max_substrates if args.split == "test" else 8),
        sampling_seed=args.sampling_seed,
    )
    print(f"loading {args.split} split...", flush=True)
    bundle = load_dataset_bundle(dataset)
    items = list((bundle.val if args.split == "val" else bundle.test).map.items())
    print(f"{args.split} substrates: {len(items)}", flush=True)

    MAX_POOL = 100

    # Generate all candidates with rule_id (no firing atoms -- faster)
    pools: list[list[tuple[str, float, int]]] = []  # [(ik, score, rule_id), ...]
    trues: list[set[str]] = []

    t = time.perf_counter()
    for i, (sub, prods) in enumerate(items, 1):
        if i == 1 or i % 10 == 0 or i == len(items):
            elapsed = time.perf_counter() - t
            eta = (elapsed / i) * (len(items) - i)
            print(f"  gen {i}/{len(items)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

        pool = _generate_with_rule_id(gen, sub, top_k=MAX_POOL, threshold=gen_threshold)
        pools.append(pool)
        trues.append(_ikset(prods))

    print(f"\nGeneration done in {time.perf_counter()-t:.0f}s. Computing recall decompositions...", flush=True)

    results = {}
    for pool_n in [50, 100]:
        r = compute_four_recalls(pools, trues, pool_n=pool_n, k=15)
        within_headroom = round(r["within_rule"] - r["actual"], 4)
        cross_headroom = round(r["full"] - r["within_rule"], 4)
        r["within_rule_headroom"] = within_headroom
        r["cross_rule_headroom"] = cross_headroom
        results[f"pool{pool_n}"] = r

    # GO/NO-GO decision
    r100 = results["pool100"]
    within_h = r100["within_rule_headroom"]
    cross_h = r100["cross_rule_headroom"]
    total_h = within_h + cross_h

    if total_h > 0:
        within_frac = within_h / total_h
        cross_frac = cross_h / total_h
    else:
        within_frac = cross_frac = 0.0

    if within_frac >= 0.5:
        go_nogo = "GO"
        go_note = (f"within-rule headroom dominates ({within_frac:.0%} of total). "
                   f"Site-contrastive reranker emphasis is correctly aimed at regioselectivity.")
    else:
        go_nogo = "GO_WITH_NOTE"
        go_note = (f"cross-rule headroom dominates ({cross_frac:.0%} of total). "
                   f"Proceed but weight cross-substrate listwise term more than sibling-InfoNCE in Task 6.")

    report = {
        "split": args.split,
        "n_substrates": len(items),
        "prior_strength": args.prior_strength,
        "recall_at_15_decomposition": results,
        "go_nogo": go_nogo,
        "go_nogo_note": go_note,
        "within_rule_frac_of_total_headroom_pool100": round(within_frac, 3),
        "cross_rule_frac_of_total_headroom_pool100": round(cross_frac, 3),
    }

    out = ROOT / "results" / "rank_decomposition.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    # Print table
    print(f"\n{'='*70}", flush=True)
    print(f"WITHIN-vs-CROSS-RULE HEADROOM DECOMPOSITION ({args.split}, n={len(items)})", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'ordering':<25} {'pool=50':>10} {'pool=100':>10}", flush=True)
    print(f"{'-'*47}", flush=True)
    for key, label in [
        ("actual", "actual (generator)"),
        ("within_rule", "oracle-within-rule"),
        ("cross_rule", "oracle-cross-rule"),
        ("full", "oracle-full"),
    ]:
        v50 = results["pool50"][key]
        v100 = results["pool100"][key]
        print(f"{label:<25} {v50:>10.4f} {v100:>10.4f}", flush=True)
    print(f"{'-'*47}", flush=True)
    print(f"{'within-rule headroom':<25} {results['pool50']['within_rule_headroom']:>10.4f} "
          f"{results['pool100']['within_rule_headroom']:>10.4f}", flush=True)
    print(f"{'cross-rule headroom':<25} {results['pool50']['cross_rule_headroom']:>10.4f} "
          f"{results['pool100']['cross_rule_headroom']:>10.4f}", flush=True)
    print(f"\nPool=100 headroom split: within={within_frac:.0%}, cross={cross_frac:.0%}", flush=True)
    print(f"GO/NO-GO: {go_nogo}", flush=True)
    print(f"  {go_note}", flush=True)
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

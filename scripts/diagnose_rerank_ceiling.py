#!/usr/bin/env python3
"""Stage-2 spike #2: can a better PRODUCT-LEVEL RERANKER reach SOTA while keeping the rule env?

Spike #1 (diagnose_ranker.py) showed rank-BY-RULE is capped ~0.38: the per-rule score/prior
cannot rank the correct regioisomer above its wrong-site siblings. This asks the complementary
question: are the true metabolites even PRESENT in the generator's candidate pool (just
mis-ranked), or missing at a reasonable budget? I.e. what recall@15 could a PERFECT product-
level reranker reach, given a generation budget of N candidates (ranked by the generator)?

For each substrate take the generator's top-N candidates, then ORACLE-rerank (true metabolites
first) and score recall@15. The gap between oracle@N and the real ~0.38 is the RERANK HEADROOM:
  - oracle@50 >> 0.38  -> the truth IS in the pool, mis-ranked: a product-level reranker (the
    diagnosed regioselectivity fix) is the path to SOTA while keeping the rule env.
  - oracle@50 ~ 0.38   -> the pool lacks the truth at budget: reranking can't help; the
    generation/coverage-at-budget is the bottleneck.

Inference only, no filter (we are measuring the *ceiling* of reranking the generator pool).
Tautomer-InChIKey match. N=15 is the no-rerank-room baseline; full-bank coverage ceiling 0.718.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from grail_metabolism.config import DatasetConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_generator

KS = [5, 10, 12, 15]
NS = [15, 30, 50, 100, 200]


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, default=str(ROOT / "artifacts" / "full5000_priors" / "checkpoints"))
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-substrates", type=int, default=120)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--prior-strength", type=float, default=8.0)
    ap.add_argument("--max-pool", type=int, default=200)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    gen = _load(Path(args.ckpt_dir) / "generator.pt", lambda a, r: build_generator(GeneratorConfig(**a), r))
    gen.gen_normalization = "canonical"
    gen.prior_strength = args.prior_strength

    dataset = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=8,
        max_val_substrates=(args.max_substrates if args.split == "val" else 8),
        max_test_substrates=(args.max_substrates if args.split == "test" else 8),
        sampling_seed=args.sampling_seed,
    )
    print(f"loading {args.split} split...", flush=True)
    bundle = load_dataset_bundle(dataset)
    items = list((bundle.val if args.split == "val" else bundle.test).map.items())
    gen_threshold = getattr(gen, "calibrated_threshold", None)
    print(f"{args.split} substrates: {len(items)}", flush=True)

    # per-substrate: generate the max pool once, record the ranked candidate InChIKeys + true set
    pools, trues = [], []
    t = time.perf_counter()
    for i, (sub, prods) in enumerate(items, 1):
        if i == 1 or i % 25 == 0 or i == len(items):
            print(f"  gen {i}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)
        scored = gen.generate_scored(sub, top_k=args.max_pool, threshold=gen_threshold)
        ranked_iks = []
        seen = set()
        for s, _ in scored:  # dedup by tautomer-InChIKey, keep generator order
            try:
                k = _tautomer_inchikey(s)
            except Exception:
                k = s
            if k not in seen:
                seen.add(k)
                ranked_iks.append(k)
        pools.append(ranked_iks)
        trues.append(_ikset(prods))

    def oracle_recall_at(pool_n, k):
        vals = []
        for pool, tru in zip(pools, trues):
            if not tru:
                continue
            present = tru.intersection(pool[:pool_n])  # truths inside the top-pool_n pool
            hit = min(len(present), k)                  # oracle puts truths first, cap at k
            vals.append(hit / len(tru))
        return sum(vals) / len(vals) if vals else 0.0

    def pool_coverage(pool_n):
        vals = []
        for pool, tru in zip(pools, trues):
            if not tru:
                continue
            vals.append(len(tru.intersection(pool[:pool_n])) / len(tru))
        return sum(vals) / len(vals) if vals else 0.0

    report = {"split": args.split, "n": len(items), "prior_strength": args.prior_strength,
              "oracle_recall_at_15_by_pool": {str(n): round(oracle_recall_at(n, 15), 3) for n in NS},
              "oracle_recall_by_k_at_pool50": {str(k): round(oracle_recall_at(50, k), 3) for k in KS},
              "pool_coverage_by_pool": {str(n): round(pool_coverage(n), 3) for n in NS},
              "real_rank_by_rule_at15": 0.385, "sygma_at15": 0.558, "coverage_ceiling": 0.718}
    out = ROOT / "results" / "rerank_ceiling.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\n==== oracle-rerank ceiling ({args.split}, n={len(items)}, tautomer) ====", flush=True)
    print("generation budget N (pool size) -> recall@15 if PERFECTLY reranked:", flush=True)
    for n in NS:
        print(f"  N={n:>3}: oracle recall@15 = {report['oracle_recall_at_15_by_pool'][str(n)]:.3f}"
              f"   (pool coverage {report['pool_coverage_by_pool'][str(n)]:.3f})", flush=True)
    print(f"\nreference: real rank-by-rule @15 ~0.385 | SyGMa 0.558 | full coverage ceiling 0.718", flush=True)
    print(f"oracle recall@k at pool=50: " +
          ", ".join(f"@{k}={report['oracle_recall_by_k_at_pool50'][str(k)]:.3f}" for k in KS), flush=True)
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

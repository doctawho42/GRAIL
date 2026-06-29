#!/usr/bin/env python3
"""Stage 2a GO/DEAD gate: a minimal listwise-trained product-level reranker vs the
generator's own ranking.

Builds an IK-deduped budget-100 candidate pool (top_k=200) per substrate from the trained
generator, trains a ``MinimalReranker`` with the per-substrate listwise InfoNCE objective,
then evaluates on VAL: reranker vs generator-alone vs oracle recall@{5,10,12,15}
(tautomer-InChIKey). Pool assembly is cached to .pt per split (the slow part).

Decision: does the reranker's VAL recall@15 BEAT generator-alone (~0.394)?
  reranker toward oracle (0.49-0.57): GO -- listwise objective unlocks cross-rule ranking.
  reranker <= generator-alone:        DEAD -- cross-encoder can't capture cross-rule.

Usage:
  python scripts/run_reranker_gate.py --train-substrates 300 --val-substrates 150 \
      --epochs 15 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from grail_metabolism.config import DatasetConfig, GeneratorConfig
from grail_metabolism.model.grail import _read_checkpoint
from grail_metabolism.model.reranker import BiEncoderReranker, MinimalReranker
from grail_metabolism.utils.seed import seed_everything
from grail_metabolism.utils.transform import SINGLE_NODE_DIM
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_generator
from grail_metabolism.workflows.reranker import (
    BiRerankerTrainer,
    RerankerTrainer,
    evaluate,
    evaluate_bi,
    load_or_build_examples,
    load_or_build_examples_bi,
)

GEN_CKPT = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
CACHE_DIR = ROOT / "artifacts" / "reranker_gate_cache"
RESULTS_PATH = ROOT / "results" / "reranker_gate.json"
RESULTS_PATH_BI = ROOT / "results" / "reranker_gate_bi.json"
KS = (5, 10, 12, 15)


def _load_generator():
    state = _read_checkpoint(GEN_CKPT)
    if state is None or "arch" not in state or "rules" not in state:
        raise SystemExit(f"Generator checkpoint missing arch/rules: {GEN_CKPT}")
    generator = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    generator.load_state_dict(state["state_dict"], strict=False)
    generator.eval()
    return generator, state["rules"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2a reranker GO/DEAD gate")
    parser.add_argument("--train-substrates", type=int, default=300)
    parser.add_argument("--val-substrates", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--max-pool", type=int, default=100)
    parser.add_argument("--rule-embed-dim", type=int, default=32)
    parser.add_argument(
        "--arch", choices=["pair", "bi"], default="bi",
        help="pair = MCS pair-graph + learned rule embedding (legacy); "
             "bi = no-MCS siamese single-graph + rule-prior scalar feature (fair).",
    )
    parser.add_argument(
        "--no-rule-prior", action="store_true",
        help="Ablation: zero out the rule-prior scalar feature in BiEncoderReranker.",
    )
    parser.add_argument(
        "--no-gen-score", action="store_true",
        help="Ablation: zero out the generator-score scalar feature in BiEncoderReranker.",
    )
    args = parser.parse_args()

    t_start = time.time()
    seed_everything(args.seed)
    print(f"[gate] arch={args.arch} seed={args.seed} top_k={args.top_k} max_pool={args.max_pool}", flush=True)

    print("[gate] loading trained generator ...", flush=True)
    t0 = time.time()
    generator, rules = _load_generator()
    print(f"[gate] generator loaded in {time.time()-t0:.1f}s; num_rules={generator.num_rules}", flush=True)

    # Subsample substrates at load time so the SDF standardization stays tractable. Pull a
    # generous multiple of the requested counts because some substrates yield empty pools.
    cfg = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf",
        train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf",
        val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf",
        test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True,
        standardize=False,
        cache_preprocessed=False,
        max_train_substrates=args.train_substrates + 60,
        max_val_substrates=args.val_substrates + 30,
        max_test_substrates=1,
        sampling_seed=args.seed,
    )
    print("[gate] loading dataset bundle (SDF standardization is the slow load) ...", flush=True)
    t0 = time.time()
    bundle = load_dataset_bundle(cfg)
    print(
        f"[gate] bundle loaded in {time.time()-t0:.1f}s; "
        f"train={len(bundle.train.map)} val={len(bundle.val.map)}",
        flush=True,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Bi-encoder caches are NOT pair-graph compatible -- tag the arch into the filename so
    # the two paths never read each other's .pt.
    tag = f"_{args.arch}"
    train_cache = CACHE_DIR / f"train{tag}_s{args.train_substrates}_seed{args.seed}_k{args.top_k}.pt"
    val_cache = CACHE_DIR / f"val{tag}_s{args.val_substrates}_seed{args.seed}_k{args.top_k}.pt"

    build_train = load_or_build_examples_bi if args.arch == "bi" else load_or_build_examples
    build_val = build_train

    print("[gate] assembling TRAIN pools (cached) ...", flush=True)
    t0 = time.time()
    train_examples = build_train(
        generator, bundle.train, args.train_substrates, train_cache,
        top_k=args.top_k, max_pool=args.max_pool,
    )
    print(f"[gate] train examples={len(train_examples)} in {time.time()-t0:.1f}s", flush=True)

    print("[gate] assembling VAL pools (cached) ...", flush=True)
    t0 = time.time()
    val_examples = build_val(
        generator, bundle.val, args.val_substrates, val_cache,
        top_k=args.top_k, max_pool=args.max_pool,
    )
    print(f"[gate] val examples={len(val_examples)} in {time.time()-t0:.1f}s", flush=True)

    train_hits = sum(1 for ex in train_examples if bool(ex.hit_mask.any()))
    val_hits = sum(1 for ex in val_examples if bool(ex.hit_mask.any()))
    print(f"[gate] train substrates with >=1 pool hit: {train_hits}/{len(train_examples)}", flush=True)
    print(f"[gate] val substrates with >=1 pool hit:   {val_hits}/{len(val_examples)}", flush=True)

    print(f"[gate] training reranker (listwise InfoNCE, arch={args.arch}) ...", flush=True)
    t0 = time.time()
    if args.arch == "bi":
        reranker = BiEncoderReranker(
            in_channels=SINGLE_NODE_DIM,
            use_rule_prior=not args.no_rule_prior,
            use_gen_score=not args.no_gen_score,
        )
        trainer = BiRerankerTrainer(reranker, lr=1e-3, seed=args.seed)
        trainer.fit(train_examples, epochs=args.epochs)
        eval_fn = evaluate_bi
    else:
        reranker = MinimalReranker(n_rules=generator.num_rules, rule_embed_dim=args.rule_embed_dim)
        trainer = RerankerTrainer(reranker, lr=1e-3, seed=args.seed)
        trainer.fit(train_examples, epochs=args.epochs)
        eval_fn = evaluate
    print(f"[gate] training done in {time.time()-t0:.1f}s", flush=True)

    print("[gate] evaluating on VAL ...", flush=True)
    metrics = eval_fn(reranker, val_examples, ks=KS, device=trainer.device)

    reranker_r15 = metrics["reranker_recall@15"]
    generator_r15 = metrics["generator_recall@15"]
    oracle_r15 = metrics["oracle_recall@15"]

    # GO vs DEAD reading.
    if reranker_r15 > generator_r15 + 0.01:
        reading = "GO"
    elif reranker_r15 < generator_r15 - 0.01:
        reading = "DEAD-worse"
    else:
        reading = "DEAD-tie"

    result = {
        "seed": args.seed,
        "arch": args.arch,
        "config": {
            "train_substrates_requested": args.train_substrates,
            "val_substrates_requested": args.val_substrates,
            "epochs": args.epochs,
            "top_k": args.top_k,
            "max_pool": args.max_pool,
            "rule_embed_dim": args.rule_embed_dim,
            "use_rule_prior": not getattr(args, "no_rule_prior", False),
            "use_gen_score": not getattr(args, "no_gen_score", False),
        },
        "counts": {
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "train_substrates_with_hits": train_hits,
            "val_substrates_with_hits": val_hits,
        },
        "metrics": metrics,
        "headline": {
            "reranker_recall@15": reranker_r15,
            "generator_alone_recall@15": generator_r15,
            "oracle_recall@15": oracle_r15,
            "generator_baseline_reference": 0.394,
            "reading": reading,
        },
        "wall_seconds": time.time() - t_start,
    }

    # Write the bi run to its own file so the legacy pair-run verdict is preserved.
    results_path = RESULTS_PATH_BI if args.arch == "bi" else RESULTS_PATH
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as handle:
        json.dump(result, handle, indent=2)

    print("\n========== STAGE 2a RERANKER GATE ==========", flush=True)
    print(f"  val substrates evaluated: {int(metrics['n_substrates'])}", flush=True)
    print(f"  mean pool size:           {metrics['mean_pool_size']:.1f}", flush=True)
    for k in KS:
        print(
            f"  recall@{k:<2}  reranker={metrics[f'reranker_recall@{k}']:.4f}  "
            f"generator={metrics[f'generator_recall@{k}']:.4f}  "
            f"oracle={metrics[f'oracle_recall@{k}']:.4f}",
            flush=True,
        )
    print(f"\n  READING: {reading}", flush=True)
    print(
        f"  reranker@15={reranker_r15:.4f}  generator-alone@15={generator_r15:.4f}  "
        f"oracle@15={oracle_r15:.4f}",
        flush=True,
    )
    print(f"  results -> {results_path}", flush=True)
    print(f"  total wall: {result['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

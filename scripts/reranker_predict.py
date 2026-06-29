#!/usr/bin/env python3
"""Reusable GLORYx / clean-test predictor for the no-MCS bi-encoder reranker (Stage 2a).

Loads the trained generator, trains a BiEncoderReranker on the cached 800-substrate bi
training pools (instant load if cache exists), then predicts top-15 reranked SMILES for
each substrate.

Usage:
    python scripts/reranker_predict.py --substrates gloryx --out docs/benchmark/data/grail_reranker_gloryx.json
    python scripts/reranker_predict.py --substrates test --out /tmp/reranker_test.json --max-substrates 300
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.config import DatasetConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.model.grail import _read_checkpoint
from grail_metabolism.model.reranker import BiEncoderReranker
from grail_metabolism.utils.seed import seed_everything
from grail_metabolism.utils.transform import SINGLE_NODE_DIM, from_rdmol
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_generator
from grail_metabolism.workflows.reranker import (
    BiRerankerTrainer,
    build_pool,
    load_or_build_examples_bi,
)

GEN_CKPT = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
CACHE_DIR = ROOT / "artifacts" / "reranker_gate_cache"
GLORYX_JSON = ROOT / "docs" / "benchmark" / "data" / "gloryx_test.json"

# The cached bi-encoder training pools from the gate run (instant load).
TRAIN_CACHE = CACHE_DIR / "train_bi_s800_seed0_k100.pt"

# Training hyperparameters -- match gate run defaults.
EPOCHS = 20
SEED = 0
TOP_K = 100
MAX_POOL = 80
TOP_N_OUT = 15


def _load_generator():
    state = _read_checkpoint(GEN_CKPT)
    if state is None or "arch" not in state or "rules" not in state:
        raise SystemExit(f"Generator checkpoint missing arch/rules: {GEN_CKPT}")
    generator = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    generator.load_state_dict(state["state_dict"], strict=False)
    generator.eval()
    return generator


def _train_reranker(generator, seed: int = SEED, epochs: int = EPOCHS) -> BiEncoderReranker:
    """Load cached train-bi examples and fit the BiEncoderReranker.

    Uses the pre-built 800-substrate bi pool cache (train_bi_s800_seed0_k100.pt) so this
    completes in seconds rather than minutes.
    """
    if not TRAIN_CACHE.exists():
        raise FileNotFoundError(
            f"Training cache not found: {TRAIN_CACHE}\n"
            "Run scripts/run_reranker_gate.py --arch bi first to build and cache the pools."
        )
    print(f"[reranker_predict] loading cached train examples from {TRAIN_CACHE} ...", flush=True)
    t0 = time.time()
    # We need a molframe object to satisfy the API, but load_or_build_examples_bi will use
    # the cache if it exists. Build a minimal dataset bundle for the cache path only.
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
        max_train_substrates=860,
        max_val_substrates=1,
        max_test_substrates=1,
        sampling_seed=SEED,
    )
    bundle = load_dataset_bundle(cfg)
    train_examples = load_or_build_examples_bi(
        generator, bundle.train, 800, TRAIN_CACHE,
        top_k=100, max_pool=100,  # match original cache params
    )
    print(f"[reranker_predict] {len(train_examples)} train examples loaded in {time.time()-t0:.1f}s", flush=True)

    seed_everything(seed)
    reranker = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    trainer = BiRerankerTrainer(reranker, lr=1e-3, seed=seed)
    print(f"[reranker_predict] training BiEncoderReranker for {epochs} epochs ...", flush=True)
    t0 = time.time()
    trainer.fit(train_examples, epochs=epochs)
    print(f"[reranker_predict] training done in {time.time()-t0:.1f}s", flush=True)
    return reranker


def reranker_predict(
    generator,
    reranker: BiEncoderReranker,
    substrates: Sequence[str],
    top_k: int = TOP_K,
    max_pool: int = MAX_POOL,
    top_n: int = TOP_N_OUT,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Score each substrate's candidate pool with the reranker and return top-N reranked SMILES.

    For each substrate:
      1. build_pool(top_k, max_pool) via the generator (compute_sites=False).
      2. from_rdmol graphs for substrate + each candidate.
      3. BiEncoderReranker scores -> rank descending.
      4. Tautomer-InChIKey dedup -> top_n.

    Returns {substrate_smiles: [top_n_smiles, ...]}.
    """
    device = next(reranker.parameters()).device
    reranker.eval()
    prior = generator.rule_prior_logits.detach().cpu()
    num_rules = int(prior.numel())

    results: Dict[str, List[str]] = {}
    t_start = time.time()
    for idx, sub in enumerate(substrates, 1):
        if verbose and (idx == 1 or idx % 5 == 0 or idx == len(substrates)):
            print(
                f"  [predict] {idx}/{len(substrates)} ({time.time()-t_start:.0f}s elapsed)",
                flush=True,
            )
        sub_mol = Chem.MolFromSmiles(sub)
        if sub_mol is None:
            results[sub] = []
            continue
        sub_graph = from_rdmol(sub_mol)
        if sub_graph is None:
            results[sub] = []
            continue

        pool = build_pool(generator, sub, top_k=top_k, max_pool=max_pool)
        if not pool:
            results[sub] = []
            continue

        prod_graphs = []
        rule_priors_list = []
        gen_scores_list = []
        cand_smiles_list = []

        for cand_smiles, gen_score, rule_id in pool:
            cand_mol = Chem.MolFromSmiles(cand_smiles)
            if cand_mol is None:
                continue
            graph = from_rdmol(cand_mol)
            if graph is None:
                continue
            prod_graphs.append(graph)
            rid = int(rule_id) if 0 <= int(rule_id) < num_rules else 0
            rule_priors_list.append(float(prior[rid]) if num_rules else 0.0)
            gen_scores_list.append(float(gen_score))
            cand_smiles_list.append(cand_smiles)

        if not prod_graphs:
            results[sub] = []
            continue

        prod_batch = Batch.from_data_list(prod_graphs)
        rule_prior_t = torch.tensor(rule_priors_list, dtype=torch.float32)
        gen_score_t = torch.tensor(gen_scores_list, dtype=torch.float32)

        with torch.no_grad():
            scores = reranker(sub_graph, prod_batch, rule_prior_t, gen_score_t).cpu()

        # Sort by score descending, tiebreak on pool order.
        order = sorted(range(len(cand_smiles_list)), key=lambda i: (-float(scores[i]), i))

        # Dedup by tautomer-InChIKey; truncate to top_n.
        seen = set()
        top_smiles = []
        for i in order:
            smi = cand_smiles_list[i]
            try:
                key = _tautomer_inchikey(smi)
            except Exception:
                key = smi
            if key in seen:
                continue
            seen.add(key)
            top_smiles.append(smi)
            if len(top_smiles) >= top_n:
                break

        results[sub] = top_smiles
    return results


def _load_gloryx_parents() -> List[str]:
    raw = GLORYX_JSON.read_text()
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
    data = json.loads(fixed)
    return [p["smiles"] for p in data if p.get("smiles")]


def _load_test_substrates(max_substrates: Optional[int], sampling_seed: int = 42) -> List[str]:
    """Load clean-test substrates (up to max_substrates) from the dataset bundle."""
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
        max_train_substrates=1,
        max_val_substrates=1,
        max_test_substrates=(max_substrates + 30) if max_substrates else None,
        sampling_seed=sampling_seed,
    )
    bundle = load_dataset_bundle(cfg)
    subs = list(bundle.test.map.keys())
    if max_substrates:
        subs = subs[:max_substrates]
    return subs


def main() -> None:
    ap = argparse.ArgumentParser(description="BiEncoder reranker predictions for GLORYx or clean-test.")
    ap.add_argument(
        "--substrates", choices=["gloryx", "test"], default="gloryx",
        help="Source of substrates: gloryx (37 parents) or test (clean-test sample).",
    )
    ap.add_argument("--out", required=True, help="Output JSON path {smiles: [preds,...]}.")
    ap.add_argument("--max-substrates", type=int, default=None,
                    help="Limit number of substrates (mainly for --substrates test).")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--max-pool", type=int, default=MAX_POOL)
    ap.add_argument("--top-n", type=int, default=TOP_N_OUT)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    t_total = time.time()

    print("[reranker_predict] loading generator ...", flush=True)
    generator = _load_generator()
    print(f"[reranker_predict] generator loaded; num_rules={generator.num_rules}", flush=True)

    reranker = _train_reranker(generator, seed=args.seed, epochs=args.epochs)

    if args.substrates == "gloryx":
        print("[reranker_predict] loading GLORYx parents ...", flush=True)
        substrates = _load_gloryx_parents()
        if args.max_substrates:
            substrates = substrates[:args.max_substrates]
        print(f"[reranker_predict] {len(substrates)} GLORYx parents", flush=True)
    else:
        print("[reranker_predict] loading clean-test substrates ...", flush=True)
        substrates = _load_test_substrates(args.max_substrates, sampling_seed=42)
        print(f"[reranker_predict] {len(substrates)} test substrates", flush=True)

    print(f"[reranker_predict] predicting top-{args.top_n} for {len(substrates)} substrates ...", flush=True)
    preds = reranker_predict(
        generator, reranker, substrates,
        top_k=args.top_k, max_pool=args.max_pool, top_n=args.top_n,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preds, indent=2))
    print(f"[reranker_predict] wrote {len(preds)} predictions -> {out_path}", flush=True)
    print(f"[reranker_predict] total wall: {time.time()-t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()

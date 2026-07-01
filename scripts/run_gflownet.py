#!/usr/bin/env python3
"""Stage 2b Set-GFlowNet: train + dual-eval matrix.

Trains a Set-GFlowNet forest sampler (``model.set_gflownet.SetGFlowNetTrainer``) whose
forward policy is a Stage-2a bi-encoder reranker (``BiEncoderReranker``), warm-started here
in-process (no reranker checkpoint is persisted to disk by ``run_reranker_gate.py``) and
optionally fine-tuned on a depth-2 bootstrap (``build_intermediate_pairs``). Evaluates on a
dual matrix at a matched output budget K = ``max_size``:

  - ``gflownet``: recall@K from the SINGLE highest-log-reward sampled forest (of
    ``--n-samples`` sampled forests), truncated to K, plus diversity metrics
    (``modes_discovered``, ``mean_pairwise_tanimoto``, ``n_unique_scaffolds``,
    ``set_size_calibration``) computed across ALL sampled forests;
  - ``reranker``: top-K of the same trained reranker scoring the root's candidate pool
    (Stage 2a baseline, no forest rollout);
  - ``beam``: ``model.multistep.MetabolicTree.beam_search`` truncated to K (the existing
    filter+generator multi-step baseline).

Recall counts a set member as a hit by tautomer-InChIKey against the substrate's annotated
metabolites (``metrics._tautomer_inchikey``), mirroring ``workflows/reranker.py``.

M0 gate: the external multi-gen (depth>=2 chain) recall claim is only pursued once the
depth-2 census (Task 1 / ``scripts/census_multistep.py``) has confirmed such chains exist
in the annotated data; the diversity metrics here are reported regardless, since they do
not depend on that finding.

Mirrors ``scripts/run_reranker_gate.py``'s structure (checkpoint loading, DatasetConfig,
pool caching, results JSON tagged by seed/split) so ``scripts/aggregate_seeds.py`` can read
the output unchanged.

Usage:
  python scripts/run_gflownet.py --train-substrates 300 --eval-split val \
      --max-depth 2 --max-size 15 --epochs 10 --n-samples 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from grail_metabolism.config import (
    DatasetConfig,
    FilterConfig,
    GeneratorConfig,
    GFlowNetConfig,
    MultiStepConfig,
)
from grail_metabolism.eval.diversity import (
    mean_pairwise_tanimoto,
    modes_discovered,
    n_unique_scaffolds,
    set_size_calibration,
)
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.model.grail import _read_checkpoint
from grail_metabolism.model.multistep import MetabolicTree
from grail_metabolism.model.reranker import BiEncoderReranker
from grail_metabolism.model.set_gflownet import SetGFlowNetTrainer, set_coverage_logreward
from grail_metabolism.utils.seed import seed_everything
from grail_metabolism.utils.transform import SINGLE_NODE_DIM
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter, build_generator
from grail_metabolism.workflows.reranker import (
    BiRerankerTrainer,
    build_intermediate_pairs,
    build_pool,
    load_or_build_examples_bi,
)

GEN_CKPT = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
FILTER_CKPT = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "filter.pt"
CACHE_DIR = ROOT / "artifacts" / "reranker_gate_cache"
RESULTS_PATH = ROOT / "results" / "gflownet.json"


def _load_generator():
    state = _read_checkpoint(GEN_CKPT)
    if state is None or "arch" not in state or "rules" not in state:
        raise SystemExit(f"Generator checkpoint missing arch/rules: {GEN_CKPT}")
    generator = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    generator.load_state_dict(state["state_dict"], strict=False)
    generator.eval()
    return generator, state["rules"]


def _load_filter():
    """Load the trained filter for the ``beam`` baseline (``MetabolicTree`` needs one).

    Not persisted per-run like the reranker -- the filter IS checkpointed to disk (unlike
    the reranker), so this loads it exactly like ``_load_generator`` loads the generator.
    """
    state = _read_checkpoint(FILTER_CKPT)
    if state is None or "arch" not in state:
        raise SystemExit(f"Filter checkpoint missing arch: {FILTER_CKPT}")
    filt = build_filter(FilterConfig(**state["arch"]))
    filt.load_state_dict(state["state_dict"], strict=False)
    filt.calibrated_threshold = state.get("calibrated_threshold")
    filt.eval()
    return filt


def _make_annotated_ik_fn(molframe):
    def annotated_ik_fn(root: str):
        return {_tautomer_inchikey(p) for p in molframe.map.get(root, [])} - {None}

    return annotated_ik_fn


def _reranker_topk_smiles(reranker, generator, root: str, k: int, top_k: int, max_pool: int, device) -> List[str]:
    """Stage-2a baseline: score the root's candidate pool with the trained reranker,
    return the top-K smiles by predicted logit (matches ``workflows.reranker.evaluate_bi``'s
    ranking, minus the recall bookkeeping)."""
    from torch_geometric.data import Batch

    from grail_metabolism.utils.transform import from_rdmol

    pool = build_pool(generator, root, top_k=top_k, max_pool=max_pool)
    if not pool:
        return []
    sub_mol = Chem.MolFromSmiles(root)
    if sub_mol is None:
        return []
    sub_graph = from_rdmol(sub_mol)
    prior = generator.rule_prior_logits.detach().cpu()
    num_rules = int(prior.numel())
    prod_graphs, rule_priors, gen_scores, smiles = [], [], [], []
    for cand_smiles, gen_score, rule_id in pool:
        cand_mol = Chem.MolFromSmiles(cand_smiles)
        if cand_mol is None:
            continue
        graph = from_rdmol(cand_mol)
        if graph is None:
            continue
        prod_graphs.append(graph)
        rid = int(rule_id) if 0 <= int(rule_id) < num_rules else 0
        rule_priors.append(float(prior[rid]) if num_rules else 0.0)
        gen_scores.append(float(gen_score))
        smiles.append(cand_smiles)
    if not prod_graphs:
        return []
    with torch.no_grad():
        prod_batch = Batch.from_data_list(prod_graphs).to(device)
        scores = reranker(
            sub_graph.to(device), prod_batch,
            torch.tensor(rule_priors, device=device), torch.tensor(gen_scores, device=device),
        ).detach().cpu()
    order = sorted(range(len(smiles)), key=lambda i: (-float(scores[i]), i))
    ranked = [smiles[i] for i in order]
    return ranked[:k]


def _recall_at_k(smiles_list: Sequence[str], annotated_ik: set) -> float:
    if not annotated_ik:
        return 0.0
    hit_iks = {_tautomer_inchikey(s) for s in smiles_list}
    return len(hit_iks & annotated_ik) / len(annotated_ik)


def _diversity_block(sampled_sets: List[frozenset], smiles_of: Dict[str, str], annotated_ik: set) -> Dict[str, float]:
    """Diversity metrics over ALL sampled forests for one substrate (Task 8's eval matrix
    reports these regardless of the M0 multi-gen recall gate)."""
    all_smiles = [smiles_of[ik] for s in sampled_sets for ik in s if ik in smiles_of]
    return {
        "modes_discovered": float(modes_discovered(sampled_sets, annotated_ik)),
        "mean_pairwise_tanimoto": float(mean_pairwise_tanimoto(all_smiles)),
        "n_unique_scaffolds": float(n_unique_scaffolds(all_smiles)),
        "set_size_calibration": float(set_size_calibration(sampled_sets, annotated_ik)),
    }


def evaluate_matrix(
    trainer: SetGFlowNetTrainer,
    generator,
    reranker,
    beam_tree: "MetabolicTree",
    eval_bundle,
    n_eval: int,
    n_samples: int,
    max_size: int,
    top_k: int,
    max_pool: int,
    device,
) -> Dict[str, float]:
    """Dual eval matrix at matched output budget K = max_size. See module docstring."""
    substrates = list(eval_bundle.map.keys())[:n_eval]
    annotated_ik_fn = _make_annotated_ik_fn(eval_bundle)

    gflownet_recall, reranker_recall, beam_recall = [], [], []
    diversity_rows: List[Dict[str, float]] = []
    n_evaluated = 0

    for root in substrates:
        annotated_ik = set(annotated_ik_fn(root))
        if not annotated_ik:
            continue
        root_mol = Chem.MolFromSmiles(root)
        if root_mol is None:
            continue

        # Sample M forests; keep the SMILES for every produced InChIKey (needed for
        # diversity/tanimoto/scaffold metrics and for materializing the gflownet@K set).
        sampled_sets: List[frozenset] = []
        smiles_of: Dict[str, str] = {}
        best_log_r, best_state = None, None
        with torch.no_grad():
            trainer.reranker.eval()
            for _ in range(n_samples):
                state, _sum_log_pf, _post_add = trainer.sample_forest(root)
                terminal = state.terminal_set()
                sampled_sets.append(terminal)
                log_r = set_coverage_logreward(
                    terminal, annotated_ik, trainer.config.beta, getattr(trainer.config, "lam", 0.1)
                )
                if best_log_r is None or log_r > best_log_r:
                    best_log_r, best_state = log_r, state

        # Reconstruct smiles for every InChIKey touched by any sampled forest via the
        # trainer's candidate cache (populated during sample_forest for root + all
        # frontier nodes visited); fall back to the root's own candidate pool.
        smiles_of[_tautomer_inchikey(root)] = root
        for p_smiles, kids in trainer._child_cache.items():
            for c_smiles, _g, _rid in kids:
                ik = _tautomer_inchikey(c_smiles)
                if ik is not None:
                    smiles_of.setdefault(ik, c_smiles)

        best_terminal = best_state.terminal_set() if best_state is not None else frozenset()
        best_smiles = [smiles_of[ik] for ik in best_terminal if ik in smiles_of][:max_size]
        gflownet_recall.append(_recall_at_k(best_smiles, annotated_ik))

        reranker_smiles = _reranker_topk_smiles(
            reranker, generator, root, k=max_size, top_k=top_k, max_pool=max_pool, device=device
        )
        reranker_recall.append(_recall_at_k(reranker_smiles, annotated_ik))

        beam_out = beam_tree.beam_search(root, max_output=max_size)
        beam_smiles = [s for s, _score in beam_out]
        beam_recall.append(_recall_at_k(beam_smiles, annotated_ik))

        diversity_rows.append(_diversity_block(sampled_sets, smiles_of, annotated_ik))
        n_evaluated += 1

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics: Dict[str, float] = {
        "n_substrates": float(n_evaluated),
        f"gflownet_recall@{max_size}": _mean(gflownet_recall),
        f"reranker_recall@{max_size}": _mean(reranker_recall),
        f"beam_recall@{max_size}": _mean(beam_recall),
    }
    for key in ("modes_discovered", "mean_pairwise_tanimoto", "n_unique_scaffolds", "set_size_calibration"):
        metrics[key] = _mean([row[key] for row in diversity_rows])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2b Set-GFlowNet: train + dual-eval matrix")
    parser.add_argument("--train-substrates", type=int, default=300)
    parser.add_argument(
        "--test-substrates", type=int, default=2000,
        help="Test substrates for --eval-split test. Default 2000 exceeds the full clean "
             "test split, so the touch-once eval uses the ENTIRE test set (no subsampling).",
    )
    parser.add_argument(
        "--eval-split", choices=["val", "test"], default="val",
        help="Which split to evaluate on. 'val' for selection; 'test' ONCE for the final report.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=6.0, help="Reward sharpness in set_coverage_logreward.")
    parser.add_argument("--lam", type=float, default=0.1, help="Size penalty in set_coverage_logreward.")
    parser.add_argument("--max-depth", type=int, default=2, help="Forest rollout depth cap.")
    parser.add_argument("--max-size", type=int, default=15, help="Forest size cap == matched output budget K.")
    parser.add_argument("--top-k", type=int, default=200, help="Generator candidates enumerated per frontier node.")
    parser.add_argument("--epochs", type=int, default=10, help="Set-GFlowNet TB training epochs.")
    parser.add_argument("--n-samples", type=int, default=32, help="Forests sampled per eval substrate (M).")
    parser.add_argument(
        "--bootstrap", action="store_true", default=True,
        help="Fine-tune the reranker on the depth-2 bootstrap (build_intermediate_pairs) "
             "after the depth-1 InfoNCE fit.",
    )
    parser.add_argument("--no-bootstrap", dest="bootstrap", action="store_false")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel pool-generation workers for the reranker pool cache (spawn Pool). 1 = serial.",
    )
    parser.add_argument("--out", type=str, default=None, help="Override the results JSON path.")
    # Reranker warm-start knobs, mirroring run_reranker_gate.py's --arch bi defaults.
    parser.add_argument("--rerank-epochs", type=int, default=15, help="Depth-1 reranker InfoNCE epochs.")
    parser.add_argument("--bootstrap-epochs", type=int, default=5, help="Depth-2 bootstrap fine-tune epochs.")
    parser.add_argument("--max-pool", type=int, default=100)
    args = parser.parse_args()

    t_start = time.time()
    seed_everything(args.seed)
    print(
        f"[gflownet] seed={args.seed} beta={args.beta} lam={args.lam} max_depth={args.max_depth} "
        f"max_size={args.max_size} top_k={args.top_k} bootstrap={args.bootstrap}",
        flush=True,
    )

    print("[gflownet] loading trained generator ...", flush=True)
    t0 = time.time()
    generator, rules = _load_generator()
    print(f"[gflownet] generator loaded in {time.time()-t0:.1f}s; num_rules={generator.num_rules}", flush=True)

    print("[gflownet] loading trained filter (beam baseline) ...", flush=True)
    t0 = time.time()
    filt = _load_filter()
    print(f"[gflownet] filter loaded in {time.time()-t0:.1f}s", flush=True)

    eval_is_test = args.eval_split == "test"
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
        max_val_substrates=(1 if eval_is_test else 300),
        max_test_substrates=(args.test_substrates + 60 if eval_is_test else 1),
        sampling_seed=args.seed,
    )
    print("[gflownet] loading dataset bundle (SDF standardization is the slow load) ...", flush=True)
    t0 = time.time()
    bundle = load_dataset_bundle(cfg)
    eval_bundle = bundle.test if eval_is_test else bundle.val
    eval_count = args.test_substrates if eval_is_test else len(eval_bundle.map)
    eval_prefix = "test" if eval_is_test else "val"
    print(
        f"[gflownet] bundle loaded in {time.time()-t0:.1f}s; "
        f"train={len(bundle.train.map)} {eval_prefix}={len(eval_bundle.map)}",
        flush=True,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_cache = CACHE_DIR / f"train_bi_s{args.train_substrates}_seed{args.seed}_k{args.top_k}.pt"

    print("[gflownet] assembling TRAIN pools for reranker warm-start (cached, bi-encoder path) ...", flush=True)
    t0 = time.time()
    train_examples = load_or_build_examples_bi(
        generator, bundle.train, args.train_substrates, train_cache,
        top_k=args.top_k, max_pool=args.max_pool,
        workers=args.workers, gen_ckpt=str(GEN_CKPT),
    )
    print(f"[gflownet] train examples={len(train_examples)} in {time.time()-t0:.1f}s", flush=True)

    print("[gflownet] warm-starting reranker (P_F init) via depth-1 listwise InfoNCE ...", flush=True)
    t0 = time.time()
    reranker = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    rr_trainer = BiRerankerTrainer(reranker, lr=1e-3, seed=args.seed)
    rr_trainer.fit(train_examples, epochs=args.rerank_epochs)
    print(f"[gflownet] reranker warm-start done in {time.time()-t0:.1f}s", flush=True)

    if args.bootstrap:
        print("[gflownet] building depth-2 bootstrap examples (build_intermediate_pairs) ...", flush=True)
        t0 = time.time()
        # NOTE: n_substrates here bounds ROOTS SCANNED, not examples produced (unlike the
        # depth-1 builders above) -- see build_intermediate_pairs's docstring.
        bootstrap_examples = build_intermediate_pairs(
            generator, bundle.train, args.train_substrates, top_k=args.top_k, max_pool=args.max_pool,
        )
        print(
            f"[gflownet] bootstrap examples={len(bootstrap_examples)} "
            f"(scanned up to {args.train_substrates} roots) in {time.time()-t0:.1f}s",
            flush=True,
        )
        if bootstrap_examples:
            combined = train_examples + bootstrap_examples
            print(
                f"[gflownet] fine-tuning reranker on depth-1 + depth-2-bootstrap "
                f"({len(train_examples)} + {len(bootstrap_examples)} = {len(combined)} examples) ...",
                flush=True,
            )
            t0 = time.time()
            rr_trainer.fit(combined, epochs=args.bootstrap_epochs)
            print(f"[gflownet] bootstrap fine-tune done in {time.time()-t0:.1f}s", flush=True)
        else:
            print("[gflownet] no depth-2 bootstrap examples found; skipping fine-tune.", flush=True)

    print("[gflownet] training Set-GFlowNet (TB loss over forest rollouts) ...", flush=True)
    t0 = time.time()
    gfn_config = GFlowNetConfig(
        max_depth=args.max_depth,
        beta=args.beta,
        lam=args.lam,
        max_size=args.max_size,
        top_k=args.top_k,
        epochs=args.epochs,
    )
    train_substrates_list = list(bundle.train.map.keys())[: args.train_substrates]
    trainer = SetGFlowNetTrainer(
        generator, reranker, gfn_config, _make_annotated_ik_fn(bundle.train), device=rr_trainer.device,
    )
    trainer.fit(train_substrates_list, epochs=args.epochs, verbose=True)
    print(f"[gflownet] Set-GFlowNet training done in {time.time()-t0:.1f}s", flush=True)

    print(f"[gflownet] evaluating dual matrix on {eval_prefix.upper()} (touch-once for test) ...", flush=True)
    t0 = time.time()
    multistep_cfg = MultiStepConfig(enabled=True, max_depth=args.max_depth, per_node_top_k=10)
    beam_tree = MetabolicTree(generator, filt, multistep_cfg)
    metrics = evaluate_matrix(
        trainer, generator, reranker, beam_tree, eval_bundle,
        n_eval=eval_count, n_samples=args.n_samples, max_size=args.max_size,
        top_k=args.top_k, max_pool=args.max_pool, device=rr_trainer.device,
    )
    print(f"[gflownet] eval done in {time.time()-t0:.1f}s", flush=True)

    result = {
        "seed": args.seed,
        "config": {
            "train_substrates_requested": args.train_substrates,
            "eval_split": args.eval_split,
            "eval_substrates_requested": eval_count,
            "beta": args.beta,
            "lam": args.lam,
            "max_depth": args.max_depth,
            "max_size": args.max_size,
            "top_k": args.top_k,
            "epochs": args.epochs,
            "n_samples": args.n_samples,
            "bootstrap": args.bootstrap,
            "rerank_epochs": args.rerank_epochs,
            "bootstrap_epochs": args.bootstrap_epochs,
        },
        "counts": {
            "train_examples": len(train_examples),
            "eval_substrates_evaluated": int(metrics["n_substrates"]),
        },
        "metrics": metrics,
        "wall_seconds": time.time() - t_start,
    }

    if args.out:
        results_path = Path(args.out)
    else:
        suffix = ""
        if eval_is_test:
            suffix += "_test"
        if args.seed != 0:
            suffix += f"_seed{args.seed}"
        results_path = RESULTS_PATH if suffix == "" else RESULTS_PATH.with_name(f"gflownet{suffix}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as handle:
        json.dump(result, handle, indent=2)

    print("\n========== STAGE 2b SET-GFLOWNET ==========", flush=True)
    print(f"  eval split: {args.eval_split.upper()}", flush=True)
    print(f"  {eval_prefix} substrates evaluated: {int(metrics['n_substrates'])}", flush=True)
    print(
        f"  recall@{args.max_size}  gflownet={metrics[f'gflownet_recall@{args.max_size}']:.4f}  "
        f"reranker={metrics[f'reranker_recall@{args.max_size}']:.4f}  "
        f"beam={metrics[f'beam_recall@{args.max_size}']:.4f}",
        flush=True,
    )
    print(
        f"  diversity: modes_discovered={metrics['modes_discovered']:.2f}  "
        f"mean_pairwise_tanimoto={metrics['mean_pairwise_tanimoto']:.4f}  "
        f"n_unique_scaffolds={metrics['n_unique_scaffolds']:.2f}  "
        f"set_size_calibration={metrics['set_size_calibration']:.2f}",
        flush=True,
    )
    print(f"  results -> {results_path}", flush=True)
    print(f"  total wall: {result['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

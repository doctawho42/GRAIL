#!/usr/bin/env python3
"""Train a filter (single OR pair mode) on a matched SUBSET, for the single-vs-pair comparison.

A full-data pair filter is impractical: pair mode does one RDKit MCS (`from_pair`) per training
pair, and the deployed run has ~165k pairs -> ~12-15h just to featurize. To compare the two filter
architectures fairly at tractable cost, we train BOTH on the SAME small subset (same substrates,
same capped negatives, same seed, same filter_optim) and change ONLY `mode`. The comparison is
paired on data; the subset caveat is reported honestly.

Reuses the deployed run's cached MolFrame (fast prepare), then subsamples the loaded positives +
negatives in memory. For pair mode it pre-builds the merged MCS graphs with a visible progress bar
(so featurization is never a silent stall), which `filter.fit` then reuses. Output:
`artifacts/<name>/checkpoints/filter.pt`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from grail_metabolism.artifacts import ArtifactStore
from grail_metabolism.config import ExperimentConfig
from grail_metabolism.utils.seed import seed_everything
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter
from grail_metabolism.workflows.training import FilterTrainingWorkflow


def subsample(frame, n_subs: int, max_negs: int) -> tuple[int, int]:
    """Keep the first n_subs substrates; cap negatives per substrate. Mutates frame in place."""
    subs = list(frame.map.keys())[:n_subs]
    frame.map = {s: frame.map[s] for s in subs}
    negs = getattr(frame, "negs", {}) or {}
    frame.negs = {s: list(negs.get(s, []))[:max_negs] for s in subs}
    n_pos = sum(len(v) for v in frame.map.values())
    n_neg = sum(len(v) for v in frame.negs.values())
    return n_pos, n_neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "artifacts" / "full5000_single" / "config.yaml"))
    ap.add_argument("--mode", choices=["single", "pair"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-train", type=int, default=800)
    ap.add_argument("--max-val", type=int, default=120)
    ap.add_argument("--max-negs", type=int, default=15)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    config = ExperimentConfig.from_yaml(args.config)
    config.filter.mode = args.mode
    config.filter.in_channels = 18 if args.mode == "pair" else 16
    config.filter.train_on_candidates = False
    config.name = args.name
    print(f"[cfg] mode={config.filter.mode} in={config.filter.in_channels} hidden={config.filter.hidden_dims} "
          f"| optim lr={config.filter_optim.lr} epochs={config.filter_optim.epochs} "
          f"prior={config.filter_optim.prior} nnpu={config.filter_optim.nnpu} seed={config.seed}", flush=True)
    print(f"[cfg] subset: max_train={args.max_train} max_val={args.max_val} max_negs={args.max_negs}", flush=True)

    seed_everything(config.seed)
    artifacts = ArtifactStore.create(config.output_dir or "artifacts", config.name)
    config.dump_yaml(artifacts.path("config.yaml"))

    t0 = time.perf_counter()
    bundle = load_dataset_bundle(config.dataset)
    bundle.prepare(
        rules=bundle.rules,
        include_val=True,
        include_test=False,
        include_pair_graphs=False,
        include_morgan=False,
        single_substrates_only=False,  # hit the deployed single run's cache (fast prepare)
    )
    tr_pos, tr_neg = subsample(bundle.train, args.max_train, args.max_negs)
    va_pos, va_neg = subsample(bundle.val, args.max_val, args.max_negs)
    print(f"[prepare] {time.perf_counter()-t0:.0f}s  train {len(bundle.train.map)} subs "
          f"({tr_pos} pos, {tr_neg} neg)  val {len(bundle.val.map)} subs ({va_pos} pos, {va_neg} neg)", flush=True)

    if args.mode == "pair":
        t1 = time.perf_counter()
        print("[pairgraphs] building merged MCS graphs (train)...", flush=True)
        bundle.train.pairgraphs()
        print("[pairgraphs] building merged MCS graphs (val)...", flush=True)
        bundle.val.pairgraphs()
        print(f"[pairgraphs] {time.perf_counter()-t1:.0f}s", flush=True)

    filter_model = build_filter(config.filter)
    t2 = time.perf_counter()
    FilterTrainingWorkflow(config, artifacts).run(filter_model, bundle)
    print(f"[filter_train] {time.perf_counter()-t2:.0f}s -> {artifacts.path('checkpoints/filter.pt')}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

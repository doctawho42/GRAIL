#!/usr/bin/env python3
"""Re-evaluate saved subset-trained checkpoints WITHOUT retraining, comparing the current
hard filter-gate against a rank-only policy (rank by filter*gen, take top-k, no gating).

(в) showed the filter gate halves recall (generator 0.355 -> ensemble 0.205 @15). This
isolates whether that loss is the gating POLICY (fixable in code) vs the filter quality.
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

from grail_metabolism.config import DatasetConfig, EvaluationConfig, FilterConfig, GeneratorConfig
from grail_metabolism.metrics import aggregate_prediction_metrics
from grail_metabolism.model.wrapper import ModelWrapper
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter, build_generator

SYGMA_RECALL = {"5": 0.470, "10": 0.531, "12": 0.543, "15": 0.558}
KS = [1, 3, 5, 10, 12, 15]


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    print(f"loaded {Path(path).name}: missing={len(missing)} unexpected={len(unexpected)} "
          f"threshold={model.calibrated_threshold}", flush=True)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, default=str(ROOT / "artifacts" / "subset_train_v" / "checkpoints"))
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--candidate-top-k", type=int, default=30)
    ap.add_argument("--max-output", type=int, default=15)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ckpt = Path(args.ckpt_dir)
    generator = _load(ckpt / "generator.pt", lambda arch, rules: build_generator(GeneratorConfig(**arch), rules))
    generator.gen_normalization = "canonical"  # model was trained with standardize=False
    filter_model = _load(ckpt / "filter.pt", lambda arch, rules: build_filter(FilterConfig(**arch)))
    model = ModelWrapper(filter_model, generator)

    # Same test split as the trained run: canonical normalization, max 250 substrates.
    dataset = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False, max_test_substrates=args.test, sampling_seed=args.sampling_seed,
    )
    print("loading test split...", flush=True)
    bundle = load_dataset_bundle(dataset)
    items = list(bundle.test.map.items())
    print(f"test substrates: {len(items)}", flush=True)

    gen_threshold = getattr(generator, "calibrated_threshold", None)

    def predictions(filter_threshold):
        rows = []
        t = time.perf_counter()
        for i, (sub, prods) in enumerate(items, 1):
            if i == 1 or i % 50 == 0 or i == len(items):
                print(f"  {('rank-only' if filter_threshold == 0.0 else 'gated')} {i}/{len(items)} "
                      f"({time.perf_counter()-t:.0f}s)", flush=True)
            ranked = model.generate(sub, top_k=args.candidate_top_k, threshold=gen_threshold,
                                    filter_threshold=filter_threshold, max_output=args.max_output)
            rows.append({"substrate": sub, "predicted": ranked, "real": sorted(prods)})
        return rows

    print("\n== GATED (current calibrated filter threshold) ==", flush=True)
    gated = aggregate_prediction_metrics(predictions(None), KS, match="inchikey")
    print("\n== RANK-ONLY (no gate, rank by filter*gen, top-k) ==", flush=True)
    rank_only = aggregate_prediction_metrics(predictions(0.0), KS, match="inchikey")

    def comp(m):
        return {f"recall@{k}": round(m.get(f"top_{k}_recall", 0.0), 3) for k in [5, 10, 12, 15]} | {
            "precision": round(m.get("precision", 0.0), 3),
            "mean_output_size": round(m.get("mean_output_size", 0.0), 2),
        }

    report = {
        "gated": comp(gated),
        "rank_only": comp(rank_only),
        "sygma_recall_at": SYGMA_RECALL,
        "filter_calibrated_threshold": filter_model.calibrated_threshold,
    }
    out = ROOT / "results" / "reeval_ranking_report.json"
    out.write_text(json.dumps(report, indent=2))
    print("\n==== gated vs rank-only vs SyGMa (recall@k, InChIKey) ====", flush=True)
    for k in ["5", "10", "12", "15"]:
        g = report["gated"][f"recall@{k}"]
        r = report["rank_only"][f"recall@{k}"]
        print(f"  @{k:>2}: gated={g:.3f}  rank_only={r:.3f}  sygma={SYGMA_RECALL[k]:.3f}", flush=True)
    print(f"\ngated precision={report['gated']['precision']} out={report['gated']['mean_output_size']} | "
          f"rank_only precision={report['rank_only']['precision']} out={report['rank_only']['mean_output_size']}", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

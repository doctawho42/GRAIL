#!/usr/bin/env python3
"""(в) Subsampled end-to-end training of the GRAIL ensemble on the real data, then
report trained recall@k vs the SyGMa baseline (InChIKey matching).

CPU-only here, and reaction labeling applies the full 7581-rule bank per substrate
(~4.5s each), so this is a small feasibility/baseline run, not a headline number.
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

from grail_metabolism.experiments.presets import get_experiment_preset
from grail_metabolism.experiments.runner import ExperimentRunner

# SyGMa baseline on the same test split (InChIKey matching), from scripts/run_benchmark.py.
SYGMA_RECALL = {"5": 0.470, "10": 0.531, "12": 0.543, "15": 0.558}
SYGMA_PRECISION = {"5": 0.175, "10": 0.105, "12": 0.090, "15": 0.074}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=80)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--gen-epochs", type=int, default=5)
    ap.add_argument("--filter-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    config = get_experiment_preset("paper_full_ensemble").with_overrides(
        name="subset_train_v",
        seed=args.seed,
        dataset={
            "max_train_substrates": args.train,
            "max_val_substrates": args.val,
            "max_test_substrates": args.test,
            "cache_preprocessed": True,
            # Canonical normalization (NOT tautomer standardization) for ~5x faster
            # reaction labeling; tautomer/charge robustness is recovered at eval via
            # InChIKey matching (evaluation.match="inchikey" below).
            "standardize": False,
        },
        # Supervised, no pretraining (USPTO load is slow and off-task for this run).
        pretrain={"enabled": False},
        generator={"use_pretraining": False, "use_maccs_pretraining": False, "top_k": 30},
        generator_optim={"epochs": args.gen_epochs, "batch_size": 32},
        filter_optim={"epochs": args.filter_epochs, "batch_size": 64, "nnpu": False},
        evaluation={
            "generator_top_k": [1, 3, 5, 10, 12, 15],
            "candidate_top_k": 30,
            "max_output": 15,
            "match": "inchikey",
            "export_predictions": True,
        },
    )

    started = time.perf_counter()
    result = ExperimentRunner(output_dir="artifacts").run_config(config)
    runtime = time.perf_counter() - started

    ens = result.metrics.get("ensemble", {})
    ens_val = result.metrics.get("ensemble_val", {})
    gen = result.metrics.get("generator", {})
    flt = result.metrics.get("filter", {})

    def row(k):
        rk = ens.get(f"top_{k}_recall")
        return None if rk is None else round(rk, 3)

    summary = {
        "config": {"train": args.train, "val": args.val, "test": args.test,
                   "gen_epochs": args.gen_epochs, "filter_epochs": args.filter_epochs, "seed": args.seed},
        "runtime_sec": round(runtime, 1),
        "ensemble_test": {k: round(v, 4) for k, v in ens.items()},
        "ensemble_val": {k: round(v, 4) for k, v in ens_val.items()},
        "generator_test": {k: round(v, 4) for k, v in gen.items()},
        "filter_test": {k: round(v, 4) for k, v in flt.items()},
        "sygma_baseline_recall_at": SYGMA_RECALL,
        "comparison_recall_at": {
            k: {"grail": row(int(k)), "sygma": SYGMA_RECALL[k],
                "delta": (None if row(int(k)) is None else round(row(int(k)) - SYGMA_RECALL[k], 3))}
            for k in ["5", "10", "12", "15"]
        },
    }
    out = ROOT / "results" / "subset_train_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\n==== (в) subsampled trained GRAIL vs SyGMa (InChIKey, top-k) ====", flush=True)
    print(json.dumps(summary["comparison_recall_at"], indent=2), flush=True)
    print(f"\nmean_output_size={ens.get('mean_output_size')}  precision={ens.get('precision')}  "
          f"recall={ens.get('recall')}  f1={ens.get('f1')}", flush=True)
    print(f"filter MCC/ROC-AUC: {flt}", flush=True)
    print(f"runtime={runtime:.0f}s  artifacts={result.artifact_dir}", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

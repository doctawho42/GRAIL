#!/usr/bin/env python3
"""Run one experiment preset over several seeds and report mean +/- std.

A single-run point estimate is not a defensible headline number. This runs the same
configuration over N seeds (model init + training + shuffling all reseeded) and reports
mean +/- std for each ensemble metric, plus the per-seed values, so headline numbers
can be stated with a spread.

Usage:
  python scripts/run_multiseed.py --preset paper_minimal_baseline --seeds 0 1 2 \
      --max-train 40 --max-val 20 --max-test 20 --gen-epochs 2 --filter-epochs 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.artifacts import ArtifactStore
from grail_metabolism.experiments.presets import get_experiment_preset
from grail_metabolism.workflows.ensemble import EnsembleWorkflow


def _mean_std(values: List[float]):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="paper_minimal_baseline")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--gen-epochs", type=int, default=None)
    ap.add_argument("--filter-epochs", type=int, default=None)
    ap.add_argument("--section", default="ensemble", help="metrics section to summarize")
    ap.add_argument("--out", default=str(ROOT / "results" / "multiseed_report.json"))
    args = ap.parse_args()

    base = get_experiment_preset(args.preset)
    dataset_overrides: Dict[str, object] = {"cache_preprocessed": False}
    if args.max_train is not None:
        dataset_overrides["max_train_substrates"] = args.max_train
    if args.max_val is not None:
        dataset_overrides["max_val_substrates"] = args.max_val
    if args.max_test is not None:
        dataset_overrides["max_test_substrates"] = args.max_test

    per_seed: List[Dict[str, float]] = []
    for seed in args.seeds:
        overrides: Dict[str, object] = {"name": f"{args.preset}_seed{seed}", "seed": seed, "dataset": dataset_overrides}
        if args.gen_epochs is not None:
            overrides["generator_optim"] = {"epochs": args.gen_epochs}
        if args.filter_epochs is not None:
            overrides["filter_optim"] = {"epochs": args.filter_epochs}
        cfg = base.with_overrides(**overrides)
        print(f"\n=== seed {seed} ===", flush=True)
        artifacts = ArtifactStore.create(cfg.output_dir, cfg.name)
        metrics = EnsembleWorkflow(cfg, artifacts).run()
        section = metrics.get(args.section, {})
        print(f"seed {seed} {args.section}: " + json.dumps({k: round(v, 4) for k, v in section.items()}), flush=True)
        per_seed.append({k: float(v) for k, v in section.items()})

    keys = sorted({k for row in per_seed for k in row})
    summary = {}
    for key in keys:
        values = [row.get(key, 0.0) for row in per_seed]
        mean, std = _mean_std(values)
        summary[key] = {"mean": mean, "std": std, "values": values}

    report = {
        "preset": args.preset,
        "seeds": args.seeds,
        "section": args.section,
        "summary": summary,
        "per_seed": per_seed,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== mean +/- std ===", flush=True)
    for key in keys:
        s = summary[key]
        print(f"  {key}: {s['mean']:.4f} +/- {s['std']:.4f}", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

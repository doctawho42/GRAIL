"""Multi-seed reproduction of the deployed GRAIL headline (macro recall@15, tautomer-InChIKey).

Loads the EXACT deployed config (`artifacts/full5000_single/config.yaml`) that produced the
0.330 headline, retrains the deployed generator+filter under N seeds (model init + training +
shuffling all reseeded via ExperimentConfig.seed), evaluates each on the FULL clean test split
(not the 291-substrate cap the deployed config carries), and reports mean +/- std for every
ensemble metric. This certifies the single-checkpoint headline is not a training-seed artifact
(the `[PENDING: multi-seed]` marker in the manuscript).

Why a separate driver (not scripts/run_multiseed.py): run_multiseed loads a *named preset*; the
deployed headline is a saved yaml, so we source ExperimentConfig.from_yaml to guarantee zero
preset-drift from the checkpoint that produced 0.330.

The dataset preprocessing (MolFrame graphs) is seed-independent (DatasetConfig.sampling_seed=42
is fixed and distinct from ExperimentConfig.seed), so cache_preprocessed stays ON and only the
first seed pays the data-prep cost.

Usage:
  python scripts/multiseed_headline.py --seeds 0 1 2
  python scripts/multiseed_headline.py --smoke            # fast harness validation
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.artifacts import ArtifactStore
from grail_metabolism.config import ExperimentConfig
from grail_metabolism.workflows.ensemble import EnsembleWorkflow

DEPLOYED_CONFIG = ROOT / "artifacts" / "full5000_single" / "config.yaml"


def _mean_std(values: List[float]):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEPLOYED_CONFIG),
                    help="deployed ExperimentConfig yaml (default: the full5000_single headline config)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--section", default="ensemble", help="metrics section to summarize")
    ap.add_argument("--out", default=str(ROOT / "results" / "multiseed_headline.json"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny/fast harness validation: max_train=40, max_test=20, gen/filter epochs=2, seeds 0 1")
    args = ap.parse_args()

    base = ExperimentConfig.from_yaml(args.config)

    # Full clean test split (the deployed config caps test at 300 -> the 291-subset; the headline
    # 0.330 is full-1170). Keep preprocessing cache ON (data is seed-independent).
    dataset_overrides: Dict[str, object] = {"max_test_substrates": None, "cache_preprocessed": True}
    seeds = args.seeds
    optim_overrides: Dict[str, object] = {}
    if args.smoke:
        dataset_overrides.update({"max_train_substrates": 40, "max_val_substrates": 20, "max_test_substrates": 20})
        optim_overrides = {"generator_optim": {"epochs": 2}, "filter_optim": {"epochs": 2}}
        seeds = [0, 1]

    per_seed: List[Dict[str, float]] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        overrides: Dict[str, object] = {
            "name": f"multiseed_full5000_seed{seed}",
            "seed": seed,
            "dataset": dataset_overrides,
        }
        overrides.update(optim_overrides)
        cfg = base.with_overrides(**overrides)
        t0 = time.time()
        print(f"\n=== seed {seed} (train+eval; match={cfg.evaluation.match}, "
              f"max_output={cfg.evaluation.max_output}) ===", flush=True)
        artifacts = ArtifactStore.create(cfg.output_dir, cfg.name)
        metrics = EnsembleWorkflow(cfg, artifacts).run()
        section = metrics.get(args.section, {})
        dt = time.time() - t0
        row = {k: float(v) for k, v in section.items()}
        per_seed.append(row)
        print(f"seed {seed} {args.section} ({dt/60:.1f} min): "
              + json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)
        # write partial results after each seed so a mid-run death still leaves progress
        out.write_text(json.dumps({"config": args.config, "seeds_done": seeds[: len(per_seed)],
                                    "per_seed": per_seed}, indent=2))

    keys = sorted({k for row in per_seed for k in row})
    summary = {}
    for key in keys:
        values = [row.get(key, 0.0) for row in per_seed]
        mean, std = _mean_std(values)
        summary[key] = {"mean": mean, "std": std, "values": values}

    report = {
        "config": args.config,
        "seeds": seeds,
        "section": args.section,
        "note": "deployed generator+filter retrained per seed; evaluated on the full clean test "
                "split under inchikey_tautomer match, max_output=15 (the headline protocol).",
        "summary": summary,
        "per_seed": per_seed,
    }
    out.write_text(json.dumps(report, indent=2))

    print("\n=== mean +/- std ===", flush=True)
    for key in keys:
        s = summary[key]
        print(f"  {key}: {s['mean']:.4f} +/- {s['std']:.4f}", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

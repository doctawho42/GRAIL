#!/usr/bin/env python3
"""Dump VAL-split predictions in the same format as `test_predictions.csv`.

Why: exploratory probes in this repo have been scored on the TEST split (rule-granularity probes,
budget-matched frontier, cross-method decomposition) -- a soft breach of touch-test-once that is
now recorded in STATUS §0a. Test is frozen for the single final GRAIL-vs-MetaTox row, so every
further exploratory run needs a val equivalent, and none was cached. This produces it.

Uses the SAME code path as the test dump (`workflows.evaluation._ensemble_predictions` on the
bundle's val frame, then the `substrate,predicted,real` pipe-joined CSV written by
`EnsembleWorkflow`), so val and test numbers are directly comparable and every downstream script
(`rank_flip_ci`, `ablate_id_embedding`, prune-and-re-rank, ...) can point at it unchanged.

Output: artifacts/full5000_single/predictions/val_predictions.csv

IMPORTANT -- CHECKPOINT PAIRING (do not compare this file naively to test_predictions.csv):
this dump uses the `full5000_priors` GENERATOR + `full5000_single` filter, the pairing the probe
scripts use (budget_matched_frontier, ablate_id_embedding) because the priors generator has a
non-degenerate rule prior. `test_predictions.csv` was written by the `full5000_single` run with its
OWN generator. So the two files are different model configurations, and a val-vs-test difference
computed across them mixes split effects with configuration effects. Measured, config-matched:
  priors-config  val 0.3882 (n=994, this file)  vs  test 0.3665 (n=291, ablate_id_embedding baseline)
      -> val is only ~+0.022 easier
  single-config  val 0.3439            vs  test 0.3342  (the run's own reports/metrics.json)
      -> +0.010, the val/test agreement the manuscript cites
A naive read of this file against test_predictions.csv gives a spurious +0.054 gap.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
DEPLOYED_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
OUT = ROOT / "artifacts" / "full5000_single" / "predictions" / "val_predictions.csv"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-substrates", type=int, default=None,
                    help="cap the val substrates (default: all)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from grail_metabolism.config import EvaluationConfig, FilterConfig, GeneratorConfig
    from grail_metabolism.experiments.presets import get_experiment_preset
    from grail_metabolism.model.wrapper import ModelWrapper
    from grail_metabolism.workflows.data import _load_split, _resolve_triples_path
    from grail_metabolism.workflows.evaluation import _ensemble_predictions
    from grail_metabolism.workflows.factory import build_filter, build_generator

    gs = torch.load(DEPLOYED_GEN, map_location="cpu", weights_only=False)
    gen = build_generator(GeneratorConfig(**gs["arch"]), gs.get("rules"))
    gen.load_state_dict(gs["state_dict"], strict=False)
    gen.calibrated_threshold = gs.get("calibrated_threshold")
    gen.eval()
    gen.gen_normalization = "canonical"
    fss = torch.load(DEPLOYED_FILTER, map_location="cpu", weights_only=False)
    filt = build_filter(FilterConfig(**fss["arch"]))
    filt.load_state_dict(fss["state_dict"], strict=False)
    filt.calibrated_threshold = fss.get("calibrated_threshold")
    filt.eval()
    print(f"loaded deployed checkpoints ({len(gs.get('rules') or [])} rules)", flush=True)

    # Load ONLY the val split (the deployed preset's dataset paths + clean-split resolution).
    # load_dataset_bundle would also parse the 1.2GB train SDF, which this does not need.
    dcfg = get_experiment_preset("paper_full_ensemble").dataset
    val = _load_split(
        dcfg.val_sdf,
        _resolve_triples_path(dcfg.val_triples, dcfg.use_clean_splits),
        standardize=dcfg.standardize,
        max_substrates=args.max_substrates or dcfg.max_val_substrates,
        seed=dcfg.sampling_seed,
    )
    print(f"val substrates with annotations: {len(val.map)}", flush=True)

    model = ModelWrapper(filter=filt, generator=gen, rules=gs.get("rules"))
    # same evaluation config as the deployed test dump (paper_full_ensemble): rank-only policy,
    # candidate_top_k=128, tautomer matching, max_output as configured there.
    cfg = EvaluationConfig(candidate_top_k=128, max_output=15, match="inchikey_tautomer")

    t0 = time.perf_counter()
    rows = _ensemble_predictions(model, val, cfg)
    print(f"predicted {len(rows)} substrates in {time.perf_counter()-t0:.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["substrate", "predicted", "real"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "substrate": r["substrate"],
                "predicted": "|".join(r["predicted"]),
                "real": "|".join(r["real"]),
            })
    mean_pred = sum(len(r["predicted"]) for r in rows) / max(len(rows), 1)
    print(f"wrote {out}  ({len(rows)} rows, mean {mean_pred:.2f} predictions/substrate)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

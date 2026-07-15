#!/usr/bin/env python3
"""Task 6 decision-gate eval: END-TO-END recall@15 (tautomer, macro) + the precision
frontier for the factorized (type -> site -> filter) generator on the clean test.

This is the actual deployed-shaped pipeline -- FactorizedGenerator.type_logits/site_logits
feeding RDKit rule application feeding the frozen filter -- NOT the val-only type/site-head
comparison `scripts/train_factorized.py` runs (that measures the heads in isolation against
a live-reapplied ground truth; this measures what a user actually gets out).

Committed test-set baselines to compare against (see docs/superpowers/plans/
2026-07-14-grail-factorized-generator.md and .superpowers/sdd/task-6-brief.md):
  deployed generator x filter        recall@15 = 0.330  (results/recall_factorization.json,
                                                          full 1170-substrate clean test)
  broad rule selection (top_k=300)
    + the same learned filter         recall@15 = 0.413  (results/selection_ablation.json,
                                                          a 245-substrate subset)
  SyGMa (phase1+phase2)               recall@15 = 0.572  (results/recall_factorization.json,
                                                          full 1170-substrate clean test)

Usage:
  python scripts/eval_factorized.py --max-substrates 250   # fast subset signal
  python scripts/eval_factorized.py                        # full 1170-substrate clean test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from grail_metabolism.config import FilterConfig
from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.metrics import recall as _set_recall
from grail_metabolism.model.factorized import FactorizedGenerator
from grail_metabolism.model.factorized_infer import build_rule_by_type, generate
from grail_metabolism.model.grail import _load_checkpoint_payload, _read_checkpoint
from grail_metabolism.stats import paired_diff_bootstrap_ci
from grail_metabolism.workflows.factory import build_filter

from run_benchmark import load_test_map  # noqa: E402  (scripts/ sibling module)

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "factorized_v1" / "checkpoints" / "factorized.pt"
DEFAULT_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
DEFAULT_VOCAB = ROOT / "grail_metabolism" / "resources" / "coarse_type_vocab.json"

BASELINE_DEPLOYED = 0.330
BASELINE_BROAD_FILTER = 0.413
BASELINE_SYGMA = 0.572

FRONTIER_KS = (1, 3, 5, 10, 15)


def _load_filter(path: Path):
    state = _read_checkpoint(path)
    if state is None:
        raise FileNotFoundError(f"could not read filter checkpoint {path}")
    arch = state.get("arch") if isinstance(state, dict) else None
    if not arch:
        raise ValueError(f"filter checkpoint {path} has no saved 'arch' -- cannot reconstruct")
    filter_model = build_filter(FilterConfig(**arch))
    if not _load_checkpoint_payload(filter_model, path, strict=False):
        raise RuntimeError(f"failed to load filter checkpoint {path} (see warnings above)")
    filter_model.eval()
    return filter_model


def _per_substrate_recall(predicted: List[str], real: List[str]) -> float:
    real_keys = {_tautomer_inchikey(s) for s in real}
    if not real_keys:
        return 0.0
    pred_keys = {_tautomer_inchikey(s) for s in predicted}
    return _set_recall(pred_keys, real_keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-substrates", type=int, default=0, help="0 = full clean test (1170)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-types", type=int, default=10)
    ap.add_argument("--max-output", type=int, default=15)
    ap.add_argument("--rule-timeout", type=float, default=1.0, help="per-rule RDKit RunReactants timeout (s)")
    ap.add_argument("--rule-max-products", type=int, default=20, help="per-rule RDKit maxProducts cap")
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--filter-checkpoint", type=str, default=str(DEFAULT_FILTER))
    ap.add_argument("--vocab", type=str, default=str(DEFAULT_VOCAB))
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "factorized_eval.json"))
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    print(f"loading factorized checkpoint {args.checkpoint} ...", flush=True)
    model = FactorizedGenerator.load(args.checkpoint)
    model.eval()

    print(f"loading filter checkpoint {args.filter_checkpoint} ...", flush=True)
    filter_model = _load_filter(Path(args.filter_checkpoint))

    print(f"loading vocab {args.vocab} ...", flush=True)
    rule_to_type: Dict[str, int] = json.loads(Path(args.vocab).read_text())["rule_to_type"]
    rule_by_type = build_rule_by_type(rule_to_type)
    print(
        f"  num_types(model)={model.arch['num_types']}  "
        f"typed_rules={sum(len(v) for v in rule_by_type.values())}/{len(rule_to_type)}",
        flush=True,
    )

    test_map = load_test_map(args.max_substrates or None, args.seed)
    n = len(test_map)

    predictions: List[Dict[str, object]] = []
    per_substrate_recall: List[float] = []
    n_empty = 0
    t0 = time.perf_counter()
    for i, (sub, mets) in enumerate(test_map.items(), 1):
        if i == 1 or i % args.log_every == 0 or i == n:
            elapsed = time.perf_counter() - t0
            rate = elapsed / i
            eta = rate * (n - i)
            print(
                f"  [{i}/{n}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining "
                f"({rate:.2f}s/substrate, empty={n_empty})",
                flush=True,
            )
        real = list(mets)
        try:
            candidates = generate(
                model,
                filter_model,
                rule_by_type,
                sub,
                top_types=args.top_types,
                max_output=args.max_output,
                rule_timeout=args.rule_timeout,
                rule_max_products=args.rule_max_products,
            )
        except Exception as exc:  # a single pathological substrate must not kill the whole run
            print(f"  WARNING: generate() failed on substrate #{i} ({exc!r}); scoring as empty", flush=True)
            candidates = []
        if not candidates:
            n_empty += 1
        predictions.append({"predicted": candidates, "real": real})
        per_substrate_recall.append(_per_substrate_recall(candidates, real))

    elapsed_total = time.perf_counter() - t0
    print(f"generation done: {n}/{n} substrates, {n_empty} empty ({elapsed_total:.0f}s total)", flush=True)

    # ---- headline recall@max_output (macro, tautomer-InChIKey) ----------------------------
    headline = aggregate_prediction_metrics(predictions, ks=[args.max_output], match="inchikey_tautomer")
    recall_at_max = headline["recall"]
    mean_output = headline["mean_output_size"]

    point, lo, hi = paired_diff_bootstrap_ci(per_substrate_recall, seed=args.seed)

    # ---- precision/recall frontier: slice the SAME ranked candidate lists per k -----------
    frontier: Dict[str, Dict[str, float]] = {}
    for k in FRONTIER_KS:
        if k > args.max_output:
            continue
        sliced = [{"predicted": row["predicted"][:k], "real": row["real"]} for row in predictions]
        m = aggregate_prediction_metrics(sliced, ks=[k], match="inchikey_tautomer")
        frontier[str(k)] = {
            "recall": round(m["recall"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "mean_output_size": round(m["mean_output_size"], 4),
        }

    report = {
        "config": {
            "n_substrates": n,
            "seed": args.seed,
            "top_types": args.top_types,
            "max_output": args.max_output,
            "rule_timeout": args.rule_timeout,
            "rule_max_products": args.rule_max_products,
            "checkpoint": str(args.checkpoint),
            "filter_checkpoint": str(args.filter_checkpoint),
            "vocab": str(args.vocab),
        },
        "n_empty_predictions": n_empty,
        "mean_output": round(mean_output, 4),
        f"recall@{args.max_output}": round(recall_at_max, 4),
        f"recall@{args.max_output}_bootstrap_ci": {
            "point": round(point, 4),
            "lo": round(lo, 4),
            "hi": round(hi, 4),
            "n_boot": 10000,
            "note": (
                "Unpaired percentile bootstrap (substrate resampling) on THIS run's own "
                "per-substrate recall -- NOT a paired comparison against the 0.413 broad+filter "
                "baseline. That baseline (results/selection_ablation.json) was measured on a "
                "different 245-substrate subset with no per-substrate breakdown persisted, so a "
                "true paired-diff bootstrap (stats.paired_diff_bootstrap_ci over per-substrate "
                "deltas) would require rerunning that baseline on this exact substrate set -- out "
                "of scope for this eval. Read the CI here as this run's own sampling uncertainty."
            ),
        },
        "precision_frontier": frontier,
        "baselines": {
            "deployed_generator_x_filter": BASELINE_DEPLOYED,
            "broad_top_k300_plus_filter": BASELINE_BROAD_FILTER,
            "sygma_phase1_phase2": BASELINE_SYGMA,
        },
        "elapsed_seconds": round(elapsed_total, 1),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

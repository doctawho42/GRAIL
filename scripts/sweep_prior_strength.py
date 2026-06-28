#!/usr/bin/env python3
"""Inference-only sweep of the generator's empirical-rule-prior weight (prior_strength).

Data scaling saturated at ~0.33@15 (60% of SyGMa). SyGMa ranks by empirical per-rule
transformation probabilities; GRAIL already has rule_prior_logits (empirical) added with
prior_strength (trained=0.4). This tests whether weighting those priors MORE (toward
SyGMa's behavior) improves recall@k -- NO retraining, just re-rank the rule logits.

prior_strength changes which rules clear top_k, so each value needs a fresh generation
pass (unlike the SoM beta post-hoc reweight). Rank-only, cap=32, tautomer InChIKey match.
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

from grail_metabolism.config import DatasetConfig, FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter, build_generator

SYGMA = {"5": 0.470, "10": 0.531, "12": 0.543, "15": 0.558}
KS = [5, 10, 12, 15]


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    return model, state["arch"]


def _dedup_cap(smiles_list, mo):
    out, seen = [], set()
    for s in smiles_list:
        try:
            key = _tautomer_inchikey(s)
        except Exception:
            key = s
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= mo:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, default=str(ROOT / "artifacts" / "full5000_single" / "checkpoints"))
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-substrates", type=int, default=250)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--candidate-top-k", type=int, default=30)
    ap.add_argument("--filter-cap", type=int, default=32)
    ap.add_argument("--max-output", type=int, default=15)
    ap.add_argument("--prior-strength", type=str, default="0.4,0.7,1.0,1.5,2.0,3.0")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ck = Path(args.ckpt_dir)
    generator, _ = _load(ck / "generator.pt", lambda a, r: build_generator(GeneratorConfig(**a), r))
    generator.gen_normalization = "canonical"
    filter_model, _ = _load(ck / "filter.pt", lambda a, r: build_filter(FilterConfig(**a)))
    strengths = [float(s) for s in str(args.prior_strength).split(",") if s.strip()]
    trained_ps = float(getattr(generator, "prior_strength", 0.4))
    print(f"trained prior_strength={trained_ps}; sweeping {strengths}", flush=True)

    dataset = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        # We only score the eval split; keep the other splits tiny so the bundle loads fast
        # (otherwise load_dataset_bundle parses the full train SDF).
        max_train_substrates=8,
        max_val_substrates=(args.max_substrates if args.split == "val" else 8),
        max_test_substrates=(args.max_substrates if args.split == "test" else 8),
        sampling_seed=args.sampling_seed,
    )
    print(f"loading {args.split} split...", flush=True)
    bundle = load_dataset_bundle(dataset)
    items = list((bundle.val if args.split == "val" else bundle.test).map.items())
    gen_threshold = getattr(generator, "calibrated_threshold", None)
    filt_threshold = float(getattr(filter_model, "calibrated_threshold", 0.5) or 0.5)
    mo, cap = args.max_output, args.filter_cap
    print(f"{args.split} substrates: {len(items)}", flush=True)

    report = {"split": args.split, "n": len(items), "trained_prior_strength": trained_ps, "sygma": SYGMA, "by_ps": {}}
    for ps in strengths:
        generator.prior_strength = ps  # re-weights empirical rule priors in forward
        rows = []
        t = time.perf_counter()
        for i, (sub, prods) in enumerate(items, 1):
            if i == 1 or i % 50 == 0 or i == len(items):
                print(f"  ps={ps}: {i}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)
            scored = generator.generate_scored(sub, top_k=args.candidate_top_k, threshold=gen_threshold)[:cap]
            cands = [s for s, _ in scored]
            fscores = filter_model.score_batch(sub, cands) if cands else []
            combined = [(s, float(fs) * float(gs)) for (s, gs), fs in zip(scored, fscores)]
            ranked = [s for s, _ in sorted(combined, key=lambda x: -x[1])]
            rows.append({"predicted": _dedup_cap(ranked, mo), "real": sorted(prods)})
        m = aggregate_prediction_metrics(rows, KS, match="inchikey_tautomer")
        report["by_ps"][str(ps)] = {f"recall@{k}": round(m.get(f"top_{k}_recall", 0.0), 3) for k in KS} | {
            "precision": round(m.get("precision", 0.0), 3), "mean_output": round(m.get("mean_output_size", 0.0), 2)}

    out = ROOT / "results" / "prior_strength_sweep.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n==== recall@k by prior_strength ({args.split}, rank-only, tautomer) ====", flush=True)
    header = f"{'k':>3} | " + " | ".join(f"ps={ps:<4}" for ps in strengths) + f" | {'SyGMa':>6}"
    print(header, flush=True)
    for k in KS:
        cells = " | ".join(f"{report['by_ps'][str(ps)][f'recall@{k}']:>6.3f}" for ps in strengths)
        print(f"{k:>3} | {cells} | {SYGMA[str(k)]:>6.3f}", flush=True)
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

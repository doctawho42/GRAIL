#!/usr/bin/env python3
"""Stage-2 spike: is GRAIL's weak ranking fixable, or is the rule-scoring paradigm capped?

Stage 1 showed the gap to SyGMa is RANKING, not coverage (ceiling 0.718 >> 0.33), the learned
score barely beats the empirical prior (pure-learned 0.31 -> prior-dominated 0.40), and GRAIL's
prior over 7581 mined rules is noisier than SyGMa's ~150 curated rules. This spike, inference
only (no retrain), tests two untested levers on the clean test split:

  (1) PURE-PRIOR reference: crank prior_strength so the empirical per-rule hit-rate dominates
      the learned score (approximates ranking by prior alone).
  (2) RULE-BANK PRUNING: hold the best ranker (prior_strength fixed) and keep only the top-p
      fraction of rules by empirical prior (mask the rest via bias=-1e9 so they never enter
      top_k), sweeping p. Removing low-precision rules trades coverage for a cleaner candidate
      pool. If recall@15 RISES at some p, the bank is the lever (-> a curated/pruned bank +
      Set-GFlowNet on a competitive base). If recall only FALLS (coverage loss dominates), the
      ranking is genuinely paradigm-capped (-> honest method/diversity framing or transformer).

Rank-only (filter x gen), cap=32, max_output=15, tautomer-InChIKey match -- same protocol as
the headline.
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
    return model


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


def _eval_pass(generator, filter_model, items, gen_threshold, cap, mo):
    rows = []
    t = time.perf_counter()
    for i, (sub, prods) in enumerate(items, 1):
        if i == 1 or i % 50 == 0 or i == len(items):
            print(f"    {i}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)
        scored = generator.generate_scored(sub, top_k=30, threshold=gen_threshold)[:cap]
        cands = [s for s, _ in scored]
        fscores = filter_model.score_batch(sub, cands) if cands else []
        combined = [(s, float(fs) * float(gs)) for (s, gs), fs in zip(scored, fscores)]
        ranked = [s for s, _ in sorted(combined, key=lambda x: -x[1])]
        rows.append({"predicted": _dedup_cap(ranked, mo), "real": sorted(prods)})
    m = aggregate_prediction_metrics(rows, KS, match="inchikey_tautomer")
    return {f"recall@{k}": round(m.get(f"top_{k}_recall", 0.0), 3) for k in KS} | {
        "precision": round(m.get("precision", 0.0), 3), "mean_output": round(m.get("mean_output_size", 0.0), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, default=str(ROOT / "artifacts" / "full5000_priors" / "checkpoints"))
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-substrates", type=int, default=120)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--prior-strength", type=float, default=8.0, help="fixed ranker for the prune sweep")
    ap.add_argument("--pure-prior-ps", type=float, default=20.0, help="prior-dominated reference")
    ap.add_argument("--prune-fracs", type=str, default="1.0,0.5,0.25,0.1,0.05")
    ap.add_argument("--filter-cap", type=int, default=32)
    ap.add_argument("--max-output", type=int, default=15)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ck = Path(args.ckpt_dir)
    generator = _load(ck / "generator.pt", lambda a, r: build_generator(GeneratorConfig(**a), r))
    generator.gen_normalization = "canonical"
    filter_model = _load(ck / "filter.pt", lambda a, r: build_filter(FilterConfig(**a)))

    base_bias = generator.bias.data.clone()
    prior = generator.rule_prior_logits.data.clone()
    n_rules = prior.numel()
    print(f"bank: {n_rules} rules | prior range [{prior.min():.2f}, {prior.max():.2f}] "
          f"| nonzero priors {(prior != 0).sum().item()}", flush=True)

    dataset = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=8,
        max_val_substrates=(args.max_substrates if args.split == "val" else 8),
        max_test_substrates=(args.max_substrates if args.split == "test" else 8),
        sampling_seed=args.sampling_seed,
    )
    print(f"loading {args.split} split...", flush=True)
    bundle = load_dataset_bundle(dataset)
    items = list((bundle.val if args.split == "val" else bundle.test).map.items())
    gen_threshold = getattr(generator, "calibrated_threshold", None)
    print(f"{args.split} substrates: {len(items)}", flush=True)

    fracs = [float(x) for x in str(args.prune_fracs).split(",") if x.strip()]
    report = {"split": args.split, "n": len(items), "n_rules": n_rules,
              "ps_ranker": args.prior_strength, "sygma": SYGMA, "rows": {}}

    def set_prune(frac):
        """Keep the top-`frac` rules by empirical prior; mask the rest out of top_k via bias."""
        generator.bias.data.copy_(base_bias)
        if frac >= 1.0:
            return n_rules
        k_keep = max(1, int(round(frac * n_rules)))
        # rules with the highest empirical prior (precision) are kept
        keep_idx = torch.topk(prior, k_keep).indices
        mask = torch.ones(n_rules, dtype=torch.bool)
        mask[keep_idx] = False  # mask == True => prune
        generator.bias.data[mask] = -1e9
        return k_keep

    # (1) pure-prior reference (no prune)
    generator.prior_strength = args.pure_prior_ps
    generator.bias.data.copy_(base_bias)
    print(f"[pure-prior ps={args.pure_prior_ps}, full bank]", flush=True)
    report["rows"][f"pure_prior_ps{args.pure_prior_ps:g}"] = _eval_pass(
        generator, filter_model, items, gen_threshold, args.filter_cap, args.max_output)

    # (2) prune sweep at the fixed ranker prior_strength
    generator.prior_strength = args.prior_strength
    for frac in fracs:
        k_keep = set_prune(frac)
        label = f"prune{frac:g}_keep{k_keep}_ps{args.prior_strength:g}"
        print(f"[{label}]", flush=True)
        report["rows"][label] = _eval_pass(
            generator, filter_model, items, gen_threshold, args.filter_cap, args.max_output)

    out = ROOT / "results" / "ranker_diagnosis.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\n==== ranker diagnosis ({args.split}, n={len(items)}, rank-only, tautomer) ====", flush=True)
    print(f"{'config':<34} | " + " | ".join(f"r@{k}" for k in KS) + " | prec | out", flush=True)
    for label, r in report["rows"].items():
        print(f"{label:<34} | " + " | ".join(f"{r[f'recall@{k}']:.3f}" for k in KS) +
              f" | {r['precision']:.3f} | {r['mean_output']:.1f}", flush=True)
    print(f"SyGMa reference @15 = {SYGMA['15']}", flush=True)
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

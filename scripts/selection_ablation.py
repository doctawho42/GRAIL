#!/usr/bin/env python3
"""Selection-breadth ablation: is GRAIL's recall bottlenecked by rule SELECTION?

The recall decomposition attributes the dominant loss to ``selection_retention`` (0.489) --
the generator applies only its top_k rules, discarding coverage the bank actually has. This
experiment tests that directly, inference-only on the deployed checkpoints: sweep the number
of rules applied (``top_k``) and, at each breadth, measure

  (a) POOL COVERAGE (oracle) -- fraction of substrates whose annotated metabolite appears
      ANYWHERE in the generated candidate pool (tautomer-InChIKey). This is
      coverage_bank x selection_retention(top_k): it isolates selection from ranking. As
      top_k grows it should climb from the deployed ~0.489-of-ceiling toward the 0.735 ceiling.
  (b) DEPLOYED recall@15 -- the full pipeline (filter x generator, top-15). The realized
      payoff of removing selection, still bounded by the ranking factor (0.726).

If (a) and (b) climb with top_k, selection -- not coverage -- is the bottleneck, and applying
rules more broadly (SyGMa's strategy) recovers recall. Reference lines: deployed GRAIL 0.330,
SyGMa 0.572, rule-bank ceiling 0.735 (all macro/oracle tautomer, full clean test).

The trained empirical rule prior lives in artifacts/full5000_priors (the shipped
full5000_single generator.pt has it zeroed); we load the generator from there so prior_strength
is meaningful and the top_k=30 baseline reproduces the 0.330 headline.
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

# SyGMa + ceiling references (macro/oracle tautomer, full clean test), for context in the table.
SYGMA_15 = 0.572
CEILING = 0.735
DEPLOYED_15 = 0.330
KS = [5, 10, 15]


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
        if key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) >= mo:
            break
    return out


def _taut_set(smiles_iter):
    keys = set()
    for s in smiles_iter:
        try:
            keys.add(_tautomer_inchikey(s))
        except Exception:
            keys.add(s)
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"))
    ap.add_argument("--filter-ckpt", default=str(ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"))
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-substrates", type=int, default=200)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--top-ks", type=str, default="30,100,300")
    ap.add_argument("--prior-strength", type=float, default=0.4, help="deployed=0.4; raise to rank rules by the frequency prior")
    ap.add_argument("--max-output", type=int, default=15)
    ap.add_argument("--filter-cap", type=int, default=300, help="candidates fed to the filter; large so a broad pool actually reaches it")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "selection_ablation.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    prior_std = float(generator.rule_prior_logits.std())
    assert prior_std > 0.1, f"rule prior is degenerate (std={prior_std}); load the full5000_priors generator"
    generator.gen_normalization = "canonical"
    generator.prior_strength = args.prior_strength
    filter_model = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))
    gen_threshold = getattr(generator, "calibrated_threshold", None)
    top_ks = [int(t) for t in str(args.top_ks).split(",") if t.strip()]
    print(f"prior std={prior_std:.3f}  prior_strength={args.prior_strength}  top_ks={top_ks}", flush=True)

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
    print(f"{args.split} substrates: {len(items)}", flush=True)

    report = {
        "split": args.split, "n": len(items), "prior_strength": args.prior_strength,
        "reference": {"deployed_recall@15": DEPLOYED_15, "sygma_recall@15": SYGMA_15, "ceiling": CEILING},
        "by_top_k": {},
    }
    for T in top_ks:
        rows, pool_hits, pool_sizes = [], 0, []
        t = time.perf_counter()
        for i, (sub, prods) in enumerate(items, 1):
            if i == 1 or i % 50 == 0 or i == len(items):
                print(f"  top_k={T}: {i}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)
            scored = generator.generate_scored(sub, top_k=T, threshold=gen_threshold)
            pool_smiles = [s for s, _ in scored]
            pool_sizes.append(len(pool_smiles))
            # (a) oracle pool coverage: is any true metabolite reachable in the broad pool?
            true_keys = _taut_set(prods)
            if true_keys & _taut_set(pool_smiles):
                pool_hits += 1
            # (b) deployed recall@15: filter-rank the (capped) pool, top-15
            capped = scored[: args.filter_cap]
            cands = [s for s, _ in capped]
            fscores = filter_model.score_batch(sub, cands) if cands else []
            combined = [(s, float(fs) * float(gs)) for (s, gs), fs in zip(capped, fscores)]
            ranked = [s for s, _ in sorted(combined, key=lambda x: -x[1])]
            rows.append({"predicted": _dedup_cap(ranked, args.max_output), "real": sorted(prods)})
        m = aggregate_prediction_metrics(rows, KS, match="inchikey_tautomer")
        report["by_top_k"][str(T)] = {
            "pool_coverage": round(pool_hits / len(items), 3),
            **{f"recall@{k}": round(m.get(f"top_{k}_recall", 0.0), 3) for k in KS},
            "precision": round(m.get("precision", 0.0), 3),
            "mean_pool_size": round(sum(pool_sizes) / len(pool_sizes), 1),
            "mean_output": round(m.get("mean_output_size", 0.0), 2),
        }
        r = report["by_top_k"][str(T)]
        print(f"  => top_k={T}: pool_cov={r['pool_coverage']} recall@15={r['recall@15']} "
              f"mean_pool={r['mean_pool_size']} out={r['mean_output']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n==== selection-breadth ablation ({args.split}, n={len(items)}, tautomer) ====", flush=True)
    print(f"{'top_k':>6} | {'pool_cov':>8} | {'recall@15':>9} | {'mean_pool':>9} | {'mean_out':>8}", flush=True)
    for T in top_ks:
        r = report["by_top_k"][str(T)]
        print(f"{T:>6} | {r['pool_coverage']:>8.3f} | {r['recall@15']:>9.3f} | {r['mean_pool_size']:>9.1f} | {r['mean_output']:>8.2f}", flush=True)
    print(f"  refs: deployed@15={DEPLOYED_15}  SyGMa@15={SYGMA_15}  ceiling={CEILING}", flush=True)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

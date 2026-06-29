#!/usr/bin/env python3
"""Stage-2a decision probe: does the BCE-trained filter have cross-rule ranking power?

Four orderings on the SAME IK-deduped candidate pool (top_k=100 from generator):
  - generator-alone:  rank by generator score (baseline)
  - filter-alone:     rank by filter.score_batch scores alone
  - filter x gen:     rank by filter_score * gen_score (current ensemble)
  - oracle:           true metabolites first (corrected ceiling on IK-deduped pool)

Decision gate:
  - filter-alone >> 0.376 (toward oracle): cross-rule ranking power EXISTS -> GO build reranker
  - filter-alone ~  0.376 (neutral):       product signal as-trained is weak -> HIGHER RISK

Dedup: tautomer-InChIKey BEFORE scoring so pool is IK-distinct (fixes earlier dedup-timing issue).
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

KS = [5, 10, 15]
SYGMA = {"5": 0.470, "10": 0.531, "15": 0.558}


def _load_model(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    return model


def _ikset(smiles_list):
    out = set()
    for s in smiles_list:
        try:
            out.add(_tautomer_inchikey(s))
        except Exception:
            out.add(s)
    return out


def _ik(s):
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return s


def _dedup_by_ik(scored_list):
    """Remove duplicates by tautomer-InChIKey, keeping generator order. Returns list of (smiles, score, ik)."""
    seen = set()
    out = []
    for smiles, score in scored_list:
        k = _ik(smiles)
        if k not in seen:
            seen.add(k)
            out.append((smiles, score, k))
    return out


def _rows_to_metrics(rows):
    m = aggregate_prediction_metrics(rows, KS, match="inchikey_tautomer")
    return {f"recall@{k}": round(m.get(f"top_{k}_recall", 0.0), 3) for k in KS} | {
        "precision": round(m.get("precision", 0.0), 3),
        "mean_output": round(m.get("mean_output_size", 0.0), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage-2a decision probe: filter ranking power on IK-deduped pool"
    )
    ap.add_argument("--ckpt-dir", type=str,
                    default=str(ROOT / "artifacts" / "full5000_priors" / "checkpoints"))
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-substrates", type=int, default=120)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--prior-strength", type=float, default=8.0)
    ap.add_argument("--top-k", type=int, default=100,
                    help="pool size (top_k candidates from generator before dedup)")
    ap.add_argument("--max-output", type=int, default=15,
                    help="max items in predicted set for recall@k")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ckpt_dir = Path(args.ckpt_dir)
    print(f"Loading generator from {ckpt_dir / 'generator.pt'} ...", flush=True)
    generator = _load_model(
        ckpt_dir / "generator.pt",
        lambda a, r: build_generator(GeneratorConfig(**a), r),
    )
    generator.gen_normalization = "canonical"
    generator.prior_strength = args.prior_strength

    print(f"Loading filter from {ckpt_dir / 'filter.pt'} ...", flush=True)
    filter_model = _load_model(
        ckpt_dir / "filter.pt",
        lambda a, r: build_filter(FilterConfig(**a)),
    )

    gen_threshold = getattr(generator, "calibrated_threshold", None)
    print(f"Generator calibrated_threshold: {gen_threshold}", flush=True)

    dataset = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf",
        train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf",
        val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf",
        test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True,
        standardize=False,
        max_train_substrates=8,
        max_val_substrates=(args.max_substrates if args.split == "val" else 8),
        max_test_substrates=(args.max_substrates if args.split == "test" else 8),
        sampling_seed=args.sampling_seed,
    )
    print(f"Loading {args.split} split ...", flush=True)
    bundle = load_dataset_bundle(dataset)
    items = list((bundle.val if args.split == "val" else bundle.test).map.items())
    print(f"{args.split} substrates: {len(items)}", flush=True)

    # Accumulate rows for each ordering
    rows_gen = []   # generator-alone
    rows_filt = []  # filter-alone
    rows_fxg = []   # filter x generator
    rows_oracle = []  # oracle (hits first)

    t_start = time.perf_counter()
    for i, (sub, true_prods) in enumerate(items, 1):
        if i == 1 or i % 20 == 0 or i == len(items):
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / i) * (len(items) - i)
            print(f"  {i}/{len(items)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

        # 1. Generate candidate pool and dedup by tautomer-IK BEFORE scoring
        scored_raw = generator.generate_scored(sub, top_k=args.top_k, threshold=gen_threshold)
        pool = _dedup_by_ik(scored_raw)  # list of (smiles, gen_score, ik)

        # 2. Score pool with filter in batch
        pool_smiles = [s for s, _, _ in pool]
        pool_gen_scores = [gs for _, gs, _ in pool]
        pool_iks = [k for _, _, k in pool]

        fscores = filter_model.score_batch(sub, pool_smiles) if pool_smiles else []

        # True metabolite IK set
        true_ik_set = _ikset(true_prods)

        # 3. Build orderings (take top max_output for the predicted set)
        mo = args.max_output

        # generator-alone: already in generator order
        gen_ranked = pool_iks[:mo]

        # filter-alone: sort by filter score desc
        filt_order = sorted(range(len(fscores)), key=lambda j: -fscores[j])
        filt_ranked = [pool_iks[j] for j in filt_order][:mo]

        # filter x generator: sort by product
        fxg_scores = [float(fscores[j]) * float(pool_gen_scores[j]) for j in range(len(fscores))]
        fxg_order = sorted(range(len(fxg_scores)), key=lambda j: -fxg_scores[j])
        fxg_ranked = [pool_iks[j] for j in fxg_order][:mo]

        # oracle: hits first, then misses, capped at mo
        hits = [k for k in pool_iks if k in true_ik_set]
        misses = [k for k in pool_iks if k not in true_ik_set]
        oracle_ranked = (hits + misses)[:mo]

        true_sorted = sorted(true_ik_set)

        # aggregate_prediction_metrics expects dicts with 'predicted' and 'real'
        # but our keys are already IKs; we use a thin wrapper that bypasses re-conversion
        # by relying on the "inchikey_tautomer" match path which calls _tautomer_inchikey
        # on each item. Since our items ARE already IKs (strings), that is identity here.
        # We use plain SMILES lists for 'predicted' but IK sets for 'real' won't work —
        # instead pass all as SMILES and let the metric convert. We hold them as IK strings:
        # aggregate_prediction_metrics with match="inchikey_tautomer" will call
        # _tautomer_inchikey on each predicted item. If the item is already an InChIKey
        # (not a SMILES), the RDKit parse will fail and fall back to the raw string.
        # So we pass the true metabolite SMILES for 'real', and pool IK strings for 'predicted'
        # — but this mix causes a mismatch. Instead, let's pass IK strings for both sides.
        # _tautomer_inchikey on an already-valid InChIKey string: RDKit returns None for
        # non-SMILES strings, and the function raises; but looking at metrics.py it catches.
        # Safest: pass real SMILES for 'real' and pool SMILES (in ordering) for 'predicted'.

        # Map IK -> smiles for pool
        ik_to_smiles = {k: s for s, _, k in pool}
        real_list = list(true_prods)

        rows_gen.append({
            "predicted": [ik_to_smiles[k] for k in gen_ranked if k in ik_to_smiles],
            "real": real_list,
        })
        rows_filt.append({
            "predicted": [ik_to_smiles[k] for k in filt_ranked if k in ik_to_smiles],
            "real": real_list,
        })
        rows_fxg.append({
            "predicted": [ik_to_smiles[k] for k in fxg_ranked if k in ik_to_smiles],
            "real": real_list,
        })
        rows_oracle.append({
            "predicted": [ik_to_smiles[k] for k in oracle_ranked if k in ik_to_smiles],
            "real": real_list,
        })

    print(f"\nComputing metrics ...", flush=True)
    m_gen = _rows_to_metrics(rows_gen)
    m_filt = _rows_to_metrics(rows_filt)
    m_fxg = _rows_to_metrics(rows_fxg)
    m_oracle = _rows_to_metrics(rows_oracle)

    total_time = time.perf_counter() - t_start

    report = {
        "split": args.split,
        "n_substrates": len(items),
        "pool_top_k": args.top_k,
        "max_output": args.max_output,
        "prior_strength": args.prior_strength,
        "runtime_seconds": round(total_time, 1),
        "sygma_at15": SYGMA["15"],
        "orderings": {
            "generator_alone": m_gen,
            "filter_alone": m_filt,
            "filter_x_generator": m_fxg,
            "oracle": m_oracle,
        },
    }

    out_path = ROOT / "results" / "probe_filter_ranking.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    # Print summary table
    print(f"\n{'='*70}", flush=True)
    print(f"FILTER RANKING PROBE  ({args.split}, n={len(items)}, pool={args.top_k}, mo={args.max_output})", flush=True)
    print(f"{'='*70}", flush=True)
    header = f"{'ordering':<22} | " + " | ".join(f"r@{k}" for k in KS) + " | prec  | out"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for label, m in report["orderings"].items():
        row = f"{label:<22} | " + " | ".join(f"{m[f'recall@{k}']:.3f}" for k in KS)
        row += f" | {m['precision']:.3f} | {m['mean_output']:.1f}"
        print(row, flush=True)

    print(f"\nSyGMa reference @15 = {SYGMA['15']}", flush=True)
    print(f"Oracle (corrected, IK-dedup pool={args.top_k}) @15 = {m_oracle['recall@15']:.3f}", flush=True)
    print(f"\nFilter-alone recall@15 = {m_filt['recall@15']:.3f}  vs  generator-alone = {m_gen['recall@15']:.3f}", flush=True)
    delta = m_filt["recall@15"] - m_gen["recall@15"]
    if delta > 0.03:
        print("VERDICT: filter-alone >> generator-alone => cross-rule ranking power EXISTS => GO", flush=True)
    else:
        print("VERDICT: filter-alone ~ generator-alone => product signal is weak as-trained => HIGHER RISK", flush=True)

    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

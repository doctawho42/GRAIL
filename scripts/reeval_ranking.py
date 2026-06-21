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
from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.model.som import product_som_score
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter, build_generator

SYGMA_RECALL = {"5": 0.470, "10": 0.531, "12": 0.543, "15": 0.558}
KS = [1, 3, 5, 10, 12, 15]


def _dedup_cap(smiles_list, mo):
    """Mirror ModelWrapper.generate: dedup by tautomer-InChIKey (the metric's match key),
    then cap at mo, so the output budget holds structure-DISTINCT molecules instead of
    tautomer/charge variants of the same one."""
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


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    print(f"loaded {Path(path).name}: missing={len(missing)} unexpected={len(unexpected)} "
          f"threshold={model.calibrated_threshold}", flush=True)
    return model


def _load_som(path):
    from grail_metabolism.model.som import SoMPredictor

    state = torch.load(path, map_location="cpu", weights_only=False)
    model = SoMPredictor(**state["arch"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    print(f"loaded som {Path(path).name}: best_val_auc={state.get('best_val_auc')}", flush=True)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, default=str(ROOT / "artifacts" / "subset_train_v" / "checkpoints"))
    ap.add_argument("--split", choices=["test", "val"], default="test", help="tune beta on val; report on test")
    ap.add_argument("--max-substrates", type=int, default=250)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--candidate-top-k", type=int, default=30)
    ap.add_argument("--max-output", type=int, default=15)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--som-ckpt", type=str, default=None, help="trained som.pt; enables the SoM reweight")
    ap.add_argument("--som-beta", type=str, default="0", help="comma list of betas, swept in ONE generation pass")
    ap.add_argument("--som-agg", choices=["max", "mean"], default="max")
    ap.add_argument("--filter-cap", type=str, default="0", help="comma list of filter candidate caps swept in one pass; 0=all")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ckpt = Path(args.ckpt_dir)
    generator = _load(ckpt / "generator.pt", lambda arch, rules: build_generator(GeneratorConfig(**arch), rules))
    generator.gen_normalization = "canonical"  # model was trained with standardize=False
    filter_model = _load(ckpt / "filter.pt", lambda arch, rules: build_filter(FilterConfig(**arch)))
    som_model = _load_som(args.som_ckpt) if args.som_ckpt else None
    betas = [float(b) for b in str(args.som_beta).split(",") if b.strip() != ""] or [0.0]
    caps = [int(c) for c in str(args.filter_cap).split(",") if c.strip() != ""] or [0]
    caps = [None if c <= 0 else c for c in caps]          # 0 -> no cap (score all)
    finite_caps = [c for c in caps if c is not None]
    max_cap = max(finite_caps) if (finite_caps and None not in caps) else None  # bound the slow pass

    # Same normalization as the trained run (canonical). Cap whichever split we score.
    n = args.max_substrates
    dataset = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_val_substrates=(n if args.split == "val" else None),
        max_test_substrates=(n if args.split == "test" else None),
        sampling_seed=args.sampling_seed,
    )
    print(f"loading {args.split} split...", flush=True)
    bundle = load_dataset_bundle(dataset)
    split_frame = bundle.val if args.split == "val" else bundle.test
    items = list(split_frame.map.items())
    print(f"{args.split} substrates: {len(items)}  betas={betas}  som={'on' if som_model else 'off'}", flush=True)

    gen_threshold = getattr(generator, "calibrated_threshold", None)
    filt_threshold = float(getattr(filter_model, "calibrated_threshold", 0.5) or 0.5)
    mo = args.max_output

    # Generate ONCE per substrate (the expensive part: rule re-encoding). Store raw
    # per-candidate (generator, filter, SoM) scores so every beta sweeps cheaply in memory.
    # som_raw is the beta-independent product SoM score in [0,1]; combined = filt*gen*som_raw**beta.
    from rdkit import Chem

    per_sub = []   # (real, [(smi, gen, filt, som_raw)])
    gen_rows = []  # generator-only baseline (beta-independent)
    t = time.perf_counter()
    for i, (sub, prods) in enumerate(items, 1):
        if i == 1 or i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)
        real = sorted(prods)
        scored = generator.generate_scored(sub, top_k=args.candidate_top_k, threshold=gen_threshold)  # [(smi, gen)], desc
        gen_rows.append({"predicted": _dedup_cap([s for s, _ in scored], mo), "real": real})
        # Only filter-score the generator's top max_cap candidates; the cap sweep simulates
        # any cap <= max_cap from these, bounding the expensive filter pass.
        scored_f = scored if max_cap is None else scored[:max_cap]
        cands = [s for s, _ in scored_f]
        fscores = filter_model.score_batch(sub, cands) if cands else []
        if som_model is not None and cands:
            sub_mol = Chem.MolFromSmiles(sub)
            som_atoms = som_model.score_atoms(sub)
            som_raw = [product_som_score(som_atoms, sub_mol, s, args.som_agg) if sub_mol is not None else 1.0 for s in cands]
        else:
            som_raw = [1.0] * len(cands)
        per_sub.append((real, [(s, float(gs), float(fs), float(sr)) for (s, gs), fs, sr in zip(scored_f, fscores, som_raw)]))

    def build_rank_gated(beta, cap):
        rank_rows, gated_rows = [], []
        for real, rows in per_sub:
            r = rows if cap is None else rows[:cap]   # top-cap by generator score (rows are gen-sorted)
            combined = [(s, fs * gs * (sr ** beta), fs) for (s, gs, fs, sr) in r]
            rank_sorted = [s for s, _, _ in sorted(combined, key=lambda x: -x[1])]
            gated_sorted = [s for s, _, _ in sorted((x for x in combined if x[2] >= filt_threshold), key=lambda x: -x[1])]
            rank_rows.append({"predicted": _dedup_cap(rank_sorted, mo), "real": real})
            gated_rows.append({"predicted": _dedup_cap(gated_sorted, mo), "real": real})
        return rank_rows, gated_rows

    def comp(rows, match):
        m = aggregate_prediction_metrics(rows, KS, match=match)
        return {f"recall@{k}": round(m.get(f"top_{k}_recall", 0.0), 3) for k in [5, 10, 12, 15]} | {
            "precision": round(m.get("precision", 0.0), 3),
            "recall": round(m.get("recall", 0.0), 3),
            "mean_output_size": round(m.get("mean_output_size", 0.0), 2),
        }

    matches = ("inchikey", "inchikey_tautomer")
    cap_label = lambda c: "all" if c is None else str(c)
    report = {
        "split": args.split, "n": len(items), "filter_calibrated_threshold": filt_threshold,
        "sygma_recall_at": SYGMA_RECALL, "som_ckpt": args.som_ckpt, "betas": betas,
        "caps": [cap_label(c) for c in caps], "max_cap_scored": max_cap,
        "generator_only": {m: comp(gen_rows, m) for m in matches},
        "by_beta": {},
    }
    for beta in betas:
        report["by_beta"][str(beta)] = {}
        for cap in caps:
            rank_rows, gated_rows = build_rank_gated(beta, cap)
            report["by_beta"][str(beta)][cap_label(cap)] = {
                m: {"ensemble_gated": comp(gated_rows, m), "ensemble_rank_only": comp(rank_rows, m)} for m in matches
            }

    out = ROOT / "results" / "reeval_ranking_report.json"
    out.write_text(json.dumps(report, indent=2))
    for match in matches:
        for beta in betas:
            print(f"\n==== recall@k ({match}) — {args.split} β={beta}  rank-only by filter-cap ====", flush=True)
            header = f"{'k':>3} | {'gen':>6}"
            for cap in caps:
                header += f" | cap{cap_label(cap):>4}"
            header += f" | {'SyGMa':>6}"
            print(header, flush=True)
            for k in ["5", "10", "12", "15"]:
                row = f"{k:>3} | {report['generator_only'][match][f'recall@{k}']:>6.3f}"
                for cap in caps:
                    row += f" | {report['by_beta'][str(beta)][cap_label(cap)][match]['ensemble_rank_only'][f'recall@{k}']:>7.3f}"
                row += f" | {SYGMA_RECALL[k]:>6.3f}"
                print(row, flush=True)
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Clean prior-vs-learned diagnostic: does GRAIL's learned rule-selection beat a frequency prior?

The old prior_strength SWEEP (sweep_prior_strength.py) was scale-invariant -- it varied the
weight of the empirical prior ADDED to the learned logits over [0.4, 3.0] and found recall flat,
which is nearly tautological (scaling one addend of a rank score rarely changes the top-k, and it
never tested the endpoints). This tests the endpoints properly, holding the product-generation
loop, top_k, filter and match protocol fixed and varying ONLY how rules are scored:

  - learned-only : generator with prior_strength = 0  (GNN logits alone)
  - blend        : generator with prior_strength = 0.4 (trained GRAIL)
  - prior-only   : rules scored by sigmoid(rule_prior_logits) alone -- the empirical per-rule
                   "does this rule yield a true metabolite" frequency, GNN discarded (masked by
                   the same applicability mask, so only fireable rules are applied -- SyGMa-style).

and, orthogonally, isolating the filter confound (the final GRAIL rank is filter x gen, so the
filter can mask the generator entirely) by scoring each mode both ways:

  - gen-only   : rank products by the generator/prior score alone
  - gen x filter : rank by filter_score x gen_score (how GRAIL actually ranks)

The decision-relevant comparison is learned-only vs prior-only: if the learned generator's own
ranking (gen-only) is no better than the frequency prior's, the learned rule-selection adds little
over a trivial baseline -- i.e. the coverage->recall conversion gap lives in ranking the learner
has not closed. recall@k, tautomer-InChIKey match, select-nothing-on-test discipline (report only).
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

import numpy as np
import torch

from grail_metabolism.config import DatasetConfig, FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter, build_generator

SYGMA = {"5": 0.470, "10": 0.531, "12": 0.543, "15": 0.558}  # same split, from run_benchmark
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


def _install_prior_only(generator):
    """Monkeypatch score_rules so rule scores are the empirical prior alone (GNN discarded),
    keeping the real applicability mask. Returns a restore() callable."""
    original = generator.score_rules
    prior_prob = torch.sigmoid(generator.rule_prior_logits.detach()).cpu().numpy().astype(np.float32)

    def prior_score_rules(sub, return_mask=False):
        out = original(sub, return_mask=True)  # (scores, mask); we keep only the mask
        mask = out[1] if isinstance(out, tuple) else np.ones_like(prior_prob)
        scores = prior_prob.copy()
        return (scores, mask) if return_mask else scores

    generator.score_rules = prior_score_rules
    return lambda: setattr(generator, "score_rules", original)


def _recall_row(generator, filter_model, sub, prods, top_k, cap, mo):
    """Return (gen_only_row, gen_x_filter_row) for one substrate, given the current generator
    scoring mode. Rows are {'predicted': [...], 'real': [...]} for aggregate_prediction_metrics."""
    scored = generator.generate_scored(sub, top_k=top_k, threshold=None)[:cap]  # [(smi, gen_score)]
    real = sorted(prods)
    if not scored:
        empty = {"predicted": [], "real": real}
        return empty, dict(empty)
    gen_ranked = [s for s, _ in scored]  # already sorted by -gen_score
    gen_only = {"predicted": _dedup_cap(gen_ranked, mo), "real": real}
    cands = [s for s, _ in scored]
    fscores = filter_model.score_batch(sub, cands) if cands else []
    combined = sorted(zip(cands, (float(f) * float(g) for (_, g), f in zip(scored, fscores))),
                      key=lambda x: -x[1])
    genfilt = {"predicted": _dedup_cap([s for s, _ in combined], mo), "real": real}
    return gen_only, genfilt


def _eval_mode(generator, filter_model, items, top_k, cap, mo, label):
    gen_rows, gf_rows = [], []
    t = time.perf_counter()
    for i, (sub, prods) in enumerate(items, 1):
        if i == 1 or i % 50 == 0 or i == len(items):
            print(f"  [{label}] {i}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)
        g, gf = _recall_row(generator, filter_model, sub, prods, top_k, cap, mo)
        gen_rows.append(g)
        gf_rows.append(gf)
    gm = aggregate_prediction_metrics(gen_rows, KS, match="inchikey_tautomer")
    fm = aggregate_prediction_metrics(gf_rows, KS, match="inchikey_tautomer")
    pack = lambda m: {f"recall@{k}": round(m.get(f"top_{k}_recall", 0.0), 3) for k in KS} | {
        "mean_output": round(m.get("mean_output_size", 0.0), 2)}
    return {"gen_only": pack(gm), "gen_x_filter": pack(fm)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, default=str(ROOT / "artifacts" / "full5000_single" / "checkpoints"))
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--max-substrates", type=int, default=250)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--candidate-top-k", type=int, default=30)
    ap.add_argument("--filter-cap", type=int, default=32)
    ap.add_argument("--max-output", type=int, default=15)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "prior_vs_learned.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ck = Path(args.ckpt_dir)
    generator = _load(ck / "generator.pt", lambda a, r: build_generator(GeneratorConfig(**a), r))
    generator.gen_normalization = "canonical"
    filter_model = _load(ck / "filter.pt", lambda a, r: build_filter(FilterConfig(**a)))

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
    top_k, cap, mo = args.candidate_top_k, args.filter_cap, args.max_output

    report = {"split": args.split, "n": len(items), "candidate_top_k": top_k, "filter_cap": cap,
              "max_output": mo, "match": "inchikey_tautomer", "sygma_recall@": SYGMA, "modes": {}}

    # learned-only (prior_strength = 0) and blend (0.4)
    generator.prior_strength = 0.0
    report["modes"]["learned_only"] = _eval_mode(generator, filter_model, items, top_k, cap, mo, "learned_only")
    generator.prior_strength = 0.4
    report["modes"]["blend_ps0.4"] = _eval_mode(generator, filter_model, items, top_k, cap, mo, "blend_ps0.4")
    # prior-only (frequency prior, GNN discarded)
    restore = _install_prior_only(generator)
    try:
        report["modes"]["prior_only"] = _eval_mode(generator, filter_model, items, top_k, cap, mo, "prior_only")
    finally:
        restore()

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n==== recall@15 ({args.split}, tautomer) -- does learned rule-selection beat the prior? ====", flush=True)
    print(f"{'mode':<16} | {'gen-only':>9} | {'gen x filter':>12}", flush=True)
    for name, d in report["modes"].items():
        print(f"{name:<16} | {d['gen_only']['recall@15']:>9.3f} | {d['gen_x_filter']['recall@15']:>12.3f}", flush=True)
    print(f"{'SyGMa (ref)':<16} | {'-':>9} | {SYGMA['15']:>12.3f}", flush=True)
    print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from grail_metabolism.metrics import (
    _match_keys,
    _tautomer_inchikey,
    aggregate_prediction_metrics,
    top_k_recall,
)
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter, build_generator

SYGMA = {"5": 0.470, "10": 0.531, "12": 0.543, "15": 0.558}  # same split, from run_benchmark
KS = [5, 10, 12, 15]
CI_K = 15  # the headline recall@k the paired-bootstrap CI is computed on


def _row_recall(row, k: int = CI_K) -> float:
    """Per-substrate recall@k under tautomer-InChIKey, consistent with aggregate_prediction_metrics
    (map each predicted SMILES to its tautomer key preserving rank, intersect the top-k with the
    tautomer-keyed reference set)."""
    ranked = [next(iter(_match_keys([s], "inchikey_tautomer"))) for s in row["predicted"]]
    real = _match_keys(row["real"], "inchikey_tautomer")
    return top_k_recall(ranked, real, k)


def _boot_ci(delta, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap over substrates (the correct unit): resample the per-substrate delta vector."""
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        means[b] = delta[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(delta.mean()), float(lo), float(hi)


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
    assert prior_prob.std() > 0, ("prior-only would be a uniform (degenerate) baseline: "
                                  "rule_prior_logits is constant -- populate the trained prior first")

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
    gen_vec = np.array([_row_recall(r) for r in gen_rows], dtype=float)
    filt_vec = np.array([_row_recall(r) for r in gf_rows], dtype=float)
    return {"gen_only": pack(gm), "gen_x_filter": pack(fm)}, {"gen": gen_vec, "filter": filt_vec}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, default=str(ROOT / "artifacts" / "full5000_single" / "checkpoints"))
    ap.add_argument("--priors-generator", type=str,
                    default=str(ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"),
                    help="checkpoint carrying the TRAINED rule_prior_logits (the full5000_single "
                         "generator.pt predates the persistent-buffer fix and lacks it). The empirical "
                         "prior is copied from here into the (byte-identical-weights) headline generator "
                         "so prior-only is the real frequency prior and blend actually differs from learned.")
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

    # The headline full5000_single generator.pt predates the persistent rule_prior_logits fix, so
    # loading it strict=False leaves that buffer at its zero init -> prior-only would be a degenerate
    # uniform baseline and blend would trivially equal learned. Copy the TRAINED prior from the
    # byte-identical-weights full5000_priors checkpoint, then hard-guard against a degenerate prior.
    priors_state = torch.load(args.priors_generator, map_location="cpu", weights_only=False)["state_dict"]
    assert "rule_prior_logits" in priors_state, f"no rule_prior_logits in {args.priors_generator}"
    trained_prior = priors_state["rule_prior_logits"].to(generator.rule_prior_logits.dtype)
    assert trained_prior.shape == generator.rule_prior_logits.shape, "rule count / bank mismatch"
    with torch.no_grad():
        generator.rule_prior_logits.copy_(trained_prior)
    rp = generator.rule_prior_logits
    assert float(rp.abs().max()) > 0 and float(rp.std()) > 0, \
        "rule_prior_logits is degenerate (zero/constant) -- prior comparison would be invalid"
    print(f"loaded trained prior: nonzero={int((rp != 0).sum())}/{rp.numel()} std={float(rp.std()):.3f} "
          f"range[{float(rp.min()):.2f},{float(rp.max()):.2f}]", flush=True)

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

    vecs = {}  # mode -> {"gen": per-substrate recall@CI_K vector, "filter": ...}
    # learned-only (prior_strength = 0) and blend (0.4)
    generator.prior_strength = 0.0
    report["modes"]["learned_only"], vecs["learned_only"] = _eval_mode(generator, filter_model, items, top_k, cap, mo, "learned_only")
    generator.prior_strength = 0.4
    report["modes"]["blend_ps0.4"], vecs["blend_ps0.4"] = _eval_mode(generator, filter_model, items, top_k, cap, mo, "blend_ps0.4")
    # prior-only (frequency prior, GNN discarded)
    restore = _install_prior_only(generator)
    try:
        report["modes"]["prior_only"], vecs["prior_only"] = _eval_mode(generator, filter_model, items, top_k, cap, mo, "prior_only")
    finally:
        restore()

    # Paired-bootstrap CIs (over substrates) on the decision-relevant recall@CI_K gaps: is the
    # learned rule-selection significantly better than the frequency prior, and is the prior
    # non-redundant on top of the learned scorer (blend - learned should be ~0)?
    gap_specs = [("learned_only", "prior_only", "gen"), ("learned_only", "prior_only", "filter"),
                 ("blend_ps0.4", "learned_only", "gen"), ("blend_ps0.4", "learned_only", "filter")]
    report["bootstrap_ci"] = {"k": CI_K, "n_boot": 10000, "seed": 0, "gaps": {}}
    print(f"\n==== paired-bootstrap CI on recall@{CI_K} gaps (n={len(items)}, tautomer) ====", flush=True)
    for a, b, axis in gap_specs:
        mean, lo, hi = _boot_ci(vecs[a][axis] - vecs[b][axis])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (spans 0)"
        report["bootstrap_ci"]["gaps"][f"{a}__minus__{b}__{axis}"] = {
            "delta": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "verdict": sig}
        print(f"  {a} - {b} [{axis:>6}]: {mean:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}]  {sig}", flush=True)
    # within-mode filter lift (gen_x_filter - gen_only): does the pair-filter add ranking value on
    # top of a given generator ordering? (These back the "filter does real ranking work" claim, which
    # was previously an un-bootstrapped point estimate.)
    for mode in ("learned_only", "prior_only"):
        mean, lo, hi = _boot_ci(vecs[mode]["filter"] - vecs[mode]["gen"])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (spans 0)"
        report["bootstrap_ci"]["gaps"][f"{mode}__filter_lift"] = {
            "delta": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "verdict": sig}
        print(f"  {mode} filter_lift : {mean:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}]  {sig}", flush=True)

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

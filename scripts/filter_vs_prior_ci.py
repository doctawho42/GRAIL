#!/usr/bin/env python3
"""Is the learned filter's advantage over the frequency prior, at equal selection breadth, real?

The paper states that widening selection to 300 rules and ranking the resulting pool by the learned
filter reaches recall@15 0.413 against the rule frequency prior's 0.374, and reads the 0.039 as
evidence that the learned scorer beats the prior at the ranking stage. Both numbers come from
selection_ablation.py runs that persisted no per-substrate breakdown, so the difference has never
carried an interval -- factorized_eval.json says so in its own note, and the pre-submission audit
(paper/SELF_CLAIMS.md, row 4) carries it as an open item. This closes it.

The two arms differ in exactly one thing: how the pool is ordered. The pool itself --
generate_scored_with_details at top_k=300, generator-score sorted, capped at 300 -- is identical, so
generating it once and ranking it both ways makes the pairing exact by construction rather than by
assumption, and costs one pass instead of two. Everything else follows selection_ablation.py
verbatim: the same two checkpoints, canonical generator normalisation, prior_strength 0.4, the
calibrated generator threshold, tautomer-aware dedup capped at 15, and the same 245-substrate
sample (max_test_substrates=245, sampling_seed=42).

Gates, fixed before running. Each arm must reproduce its published marginal to within 0.003 -- the
published figures are rounded to three places and the pool is not resampled, so anything larger
means the configuration drifted rather than that the estimate moved. And the per-substrate vector
this script bootstraps must average to what aggregate_prediction_metrics reports for the same rows,
which is what makes the interval an interval on the published quantity rather than on a lookalike.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import pathlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rel(p) -> str:
    """Repo-relative path. Absolute paths in a committed artifact name the author's home
    directory, which is an anonymity leak in a double-blind submission and portable to nobody."""
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)

import torch

from grail_metabolism.config import DatasetConfig, FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter, build_generator

PUBLISHED = {"filter": 0.413, "prior": 0.374}
PUBLISHED_POOL = 107.6
TOL = 0.003
N_BOOT, SEED = 10000, 0


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    return model


def _key(s):
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return s


def _dedup_cap(smiles_list, mo):
    out, seen = [], set()
    for s in smiles_list:
        k = _key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
        if len(out) >= mo:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"))
    ap.add_argument("--filter-ckpt", default=str(ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"))
    ap.add_argument("--max-substrates", type=int, default=245)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--prior-strength", type=float, default=0.4)
    ap.add_argument("--max-output", type=int, default=15)
    ap.add_argument("--filter-cap", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N substrates; skips the gates")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "filter_vs_prior_ci.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    prior_std = float(generator.rule_prior_logits.std())
    assert prior_std > 0.1, f"rule prior is degenerate (std={prior_std}); load the full5000_priors generator"
    generator.gen_normalization = "canonical"
    generator.prior_strength = args.prior_strength
    filter_model = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))
    gen_threshold = getattr(generator, "calibrated_threshold", None)
    prior_vec = generator.rule_prior_logits.detach().cpu()
    print(f"prior std={prior_std:.3f}  top_k={args.top_k}  prior_strength={args.prior_strength}", flush=True)

    dataset = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=8, max_val_substrates=8, max_test_substrates=args.max_substrates,
        sampling_seed=args.sampling_seed,
    )
    print("loading test split...", flush=True)
    items = list(load_dataset_bundle(dataset).test.map.items())
    if args.limit:
        items = items[: args.limit]
    print(f"substrates: {len(items)}", flush=True)

    rows = {"filter": [], "prior": []}
    per_sub = {"filter": [], "prior": []}
    subs, pool_sizes, t = [], [], time.perf_counter()
    for i, (sub, prods) in enumerate(items, 1):
        if i == 1 or i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)
        detailed = generator.generate_scored_with_details(sub, top_k=args.top_k,
                                                          threshold=gen_threshold, compute_sites=False)
        detailed.sort(key=lambda d: (-d[1], d[0]))
        pool_sizes.append(len(detailed))
        capped = detailed[: args.filter_cap]
        cands = [d[0] for d in capped]
        fscores = filter_model.score_batch(sub, cands) if cands else []
        ranking = {
            "filter": [(d[0], float(fs) * float(d[1])) for d, fs in zip(capped, fscores)],
            "prior": [(d[0], float(prior_vec[d[2]])) for d in capped],
        }
        true_keys = {_key(p) for p in prods}
        subs.append(sub)
        for arm, keyed in ranking.items():
            ranked = [s for s, _ in sorted(keyed, key=lambda x: -x[1])]
            pred = _dedup_cap(ranked, args.max_output)
            rows[arm].append({"predicted": pred, "real": sorted(prods)})
            hit = len({_key(p) for p in pred} & true_keys)
            per_sub[arm].append(hit / len(true_keys) if true_keys else 0.0)

    mean_pool = float(np.mean(pool_sizes))
    # Record the configuration, not only the result. The artifact this script exists to certify --
    # selection_ablation_ranksignal.json -- stores n=245 and neither the cap nor the seed that
    # produced those 245, so its substrate set cannot be recovered and its numbers cannot be
    # reproduced from it. Committing a result file is necessary for regenerability and not
    # sufficient.
    rep = {"config": {"max_substrates": args.max_substrates, "sampling_seed": args.sampling_seed,
                      "top_k": args.top_k, "prior_strength": args.prior_strength,
                      "filter_cap": args.filter_cap, "max_output": args.max_output,
                      "gen_ckpt": _rel(args.gen_ckpt), "filter_ckpt": _rel(args.filter_ckpt),
                      "use_clean_splits": True, "standardize": False},
           "n": len(items), "top_k": args.top_k, "match": "inchikey_tautomer",
           "max_output": args.max_output, "mean_pool_size": round(mean_pool, 1),
           "n_boot": N_BOOT, "seed": SEED, "arms": {}}
    # Gate 0: the two arms can only differ where the pool exceeds the output budget, so a pool that
    # does not reproduce the published breadth makes the comparison vacuous rather than merely off.
    print(f"mean pool size {mean_pool:.1f} against a published 107.6")
    if not args.limit and abs(mean_pool - PUBLISHED_POOL) > 1.0:
        raise SystemExit(f"mean pool {mean_pool:.1f} != published {PUBLISHED_POOL} -- selection breadth drifted")
    print(f"\n{'arm':10}{'this run':>10}{'aggregator':>12}{'published':>11}")
    for arm in ("filter", "prior"):
        v = np.array(per_sub[arm])
        agg = aggregate_prediction_metrics(rows[arm], [args.max_output], match="inchikey_tautomer")
        agg_r = float(agg.get(f"top_{args.max_output}_recall", 0.0))
        print(f"{arm:10}{v.mean():10.4f}{agg_r:12.4f}{PUBLISHED[arm]:11.3f}")
        # Gate 1: the vector being bootstrapped must be the published quantity, not a lookalike.
        if abs(v.mean() - agg_r) > 1e-6:
            raise SystemExit(f"{arm}: per-substrate mean {v.mean():.6f} != aggregator {agg_r:.6f} "
                             f"-- the vector is not the reported metric")
        # Gate 2: the configuration must reproduce the published marginal.
        if not args.limit and abs(agg_r - PUBLISHED[arm]) > TOL:
            raise SystemExit(f"{arm}: recomputes to {agg_r:.4f} against a published {PUBLISHED[arm]} "
                             f"-- refusing to report a gap between numbers that are not the paper's")
        rep["arms"][arm] = {"recall@15": round(float(v.mean()), 4), "published": PUBLISHED[arm]}

    f, p = np.array(per_sub["filter"]), np.array(per_sub["prior"])
    d = f - p
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    rep["gap"] = {"estimand": "recall@15(learned filter) - recall@15(frequency prior), paired on substrate",
                  "point": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                  "excludes_zero": bool(lo > 0 or hi < 0),
                  "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
                  "n_tied": int((d == 0).sum())}
    print(f"\ngap = {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"{'SIG' if rep['gap']['excludes_zero'] else 'n.s.'}")
    print(f"filter better on {rep['gap']['n_better']}, worse on {rep['gap']['n_worse']}, "
          f"tied on {rep['gap']['n_tied']} substrates")
    rep["per_substrate"] = [{"sub": s, "filter": round(a, 4), "prior": round(b, 4)}
                            for s, a, b in zip(subs, f, p)]
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

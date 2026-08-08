#!/usr/bin/env python3
"""Does GRAIL's wider bank beat SyGMa once selection is removed and the budget is matched?

The paper's central claim is that the gap to SyGMa is the price of having a selection stage rather
than a worse rule base: GRAIL's bank reaches 0.735 where SyGMa's reaches 0.542. The direct test of
that claim has never been run. Its form is forced by the claim itself -- apply the bank without a
selector, rank the whole pool with the trained scorer, and truncate at the number of candidates
SyGMa actually emits -- and the existing budget sweep does not answer it, because that sweep ranks
the pool the deployed 30-rule selector produces, which contains only eight to twelve candidates and
so has nothing to give at k=64.

Two outcomes and both are informative. If the wide-bank arm passes SyGMa at matched output, the
decomposition's reading is demonstrated rather than inferred and the bank is genuinely not the
bottleneck. If it does not, the claim needs weakening: the loss is then distributed across selection
and ranking rather than located in selection, and the title overstates what the measurement shows.

Selection breadth is swept rather than fixed at 300. Table 5 reports pool coverage of 0.608 at 300
rules and calls that essentially every applicable rule, but the ceiling on those same 245 substrates
is 0.728, so 300 rules leave an eighth of the reachable references unreached and the sweep has to
run past it to separate the cutoff from the bank.

SyGMa is scored from its frozen full-split predictions restricted to the same substrates, truncated
at the same k, under the same matcher, so the only thing differing between arms is which system
produced the ranked list.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from grail_metabolism.config import DatasetConfig, FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter, build_generator

N_BOOT, SEED = 10000, 0
def ceiling_subset(subs) -> dict:
    """The committed ceiling restricted to exactly these substrates, both aggregations.

    Read from the artifact rather than frozen here. A literal is a snapshot of the answer at the
    moment it was written; this one sat at 0.7284 through a correction that took the same quantity
    to 0.8007, and the value it reported went on looking committed.
    """
    rows = {r["sub"]: r for r in
            json.loads((ROOT / "results/recall_factorization.json").read_text())["per_substrate"]}
    hit = [rows[s] for s in subs if s in rows]
    micro = sum(r["Cfull"] for r in hit) / max(sum(r["U"] for r in hit), 1)
    macro = sum(r["Cfull"] / r["U"] for r in hit if r["U"]) / max(len(hit), 1)
    return {"n_matched": len(hit), "micro": round(micro, 4), "macro": round(macro, 4),
            "source": "results/recall_factorization.json, restricted to these substrates"}


def _rel(p) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _code_version() -> dict:
    import subprocess
    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    return model


# Tautomer canonicalisation is a search, not a lookup, and it dominates this script: keying SyGMa's
# 19,487 structures for these substrates took 47 minutes on the first attempt, before a single
# substrate had been scored. results/key_tables/inchikey_tautomer.json already holds 108,967 of them,
# built and verified by build_key_tables.py, and covers 100% of what SyGMa needs here. Look up first,
# compute only on a miss, and memoise -- GRAIL's wide-bank pools repeat structures across substrates.
_TABLE = json.loads((ROOT / "results" / "key_tables" / "inchikey_tautomer.json").read_text())
_MISS: dict = {}


def _key(s):
    k = _TABLE.get(s)
    if k is not None:
        return k
    k = _MISS.get(s)
    if k is not None:
        return k
    try:
        k = _tautomer_inchikey(s)
    except Exception:
        k = s
    _MISS[s] = k
    return k


def _keys_parallel(smiles, pool):
    """Tautomer keys for a candidate list, table first and the misses farmed out.

    Canonicalisation runs at roughly seven structures a second and GRAIL's wide-bank pools are
    novel, so this is the whole cost of the script: serial keying of one 300-rule pool took longer
    than the rest of the arm put together."""
    out = [_TABLE.get(x) or _MISS.get(x) for x in smiles]
    todo = [(i, x) for i, (k, x) in enumerate(zip(out, smiles)) if k is None]
    if todo:
        for (i, x), k in zip(todo, pool.map(_tautomer_inchikey_safe, [x for _, x in todo], 64)):
            _MISS[x] = k
            out[i] = k
    return out


def _tautomer_inchikey_safe(x):
    try:
        return _tautomer_inchikey(x)
    except Exception:
        return x


def _dedup(smiles, cap=None, pool=None):
    """Unique keys in rank order, stopping at cap -- beyond the largest budget they are never read."""
    if pool is not None:
        keys = _keys_parallel(smiles, pool)
    else:
        keys = [_key(s) for s in smiles]
    out, seen = [], set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
            if cap and len(out) >= cap:
                break
    return out


def recall_at(pred_keys, ref_keys, k):
    if not ref_keys:
        return float("nan")
    return len(set(pred_keys[:k]) & ref_keys) / len(ref_keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts/full5000_priors/checkpoints/generator.pt"))
    ap.add_argument("--filter-ckpt", default=str(ROOT / "artifacts/full5000_single/checkpoints/filter.pt"))
    ap.add_argument("--max-substrates", type=int, default=250)
    ap.add_argument("--sampling-seed", type=int, default=42)
    ap.add_argument("--top-ks", default="300,1000,7581",
                    help="selection breadth; 7581 is the whole bank, i.e. no selector at all")
    ap.add_argument("--budgets", default="8,15,32,64,82",
                    help="output budgets to score both arms at; 82 is SyGMa's emitted size")
    ap.add_argument("--prior-strength", type=float, default=0.4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "bank_without_selection.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    assert float(generator.rule_prior_logits.std()) > 0.1, "load the full5000_priors generator"
    generator.gen_normalization = "canonical"
    generator.prior_strength = args.prior_strength
    filt = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))
    gen_threshold = getattr(generator, "calibrated_threshold", None)
    top_ks = [int(t) for t in args.top_ks.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]

    dataset = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf", train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf", val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf", test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=8, max_val_substrates=8, max_test_substrates=args.max_substrates,
        sampling_seed=args.sampling_seed)
    print("loading test split...", flush=True)
    items = list(load_dataset_bundle(dataset).test.map.items())
    if args.limit:
        items = items[: args.limit]
    print(f"substrates: {len(items)}", flush=True)

    sygma_preds = json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text())
    missing = [s for s, _ in items if s not in sygma_preds]
    if missing and not args.limit:
        raise SystemExit(f"SyGMa predictions missing for {len(missing)} of the evaluated substrates")

    refs = {s: {k for k in (_key(p) for p in prods) if k} for s, prods in items}
    cap = max(budgets)
    with Pool(args.threads) as pool:
        sy = {s: _dedup(sygma_preds.get(s, []), cap, pool) for s, _ in items}
    print(f"SyGMa mean emitted (deduplicated): {np.mean([len(v) for v in sy.values()]):.1f} "
          f"(key table hits {len(_TABLE)}, computed {len(_MISS)})", flush=True)

    rep = {"config": {**_code_version(), "max_substrates": args.max_substrates,
                      "sampling_seed": args.sampling_seed, "top_ks": top_ks, "budgets": budgets,
                      "prior_strength": args.prior_strength, "gen_ckpt": _rel(args.gen_ckpt),
                      "filter_ckpt": _rel(args.filter_ckpt)},
           "n": len(items), "match": "inchikey_tautomer",
           "ceiling_on_this_subset": ceiling_subset([s for s, _ in items]), "arms": {}}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(items), (N_BOOT, len(items)))
    sy_vec = {b: np.array([recall_at(sy[s], refs[s], b) for s, _ in items]) for b in budgets}

    keypool = Pool(args.threads)
    for T in top_ks:
        ranked, pool_sizes, t = {}, [], time.perf_counter()
        for i, (sub, _) in enumerate(items, 1):
            if i == 1 or i % 25 == 0 or i == len(items):
                print(f"  top_k={T}: {i}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)
            # The whole bank means no selector, and the calibrated threshold IS a selector: leaving
            # it on would prune rules below it however large top_k is. The widest arm therefore
            # drops it, which is what makes it the counterfactual the decomposition's claim needs.
            thr = None if T >= 7581 else gen_threshold
            det = generator.generate_scored_with_details(sub, top_k=T, threshold=thr,
                                                         compute_sites=False)
            det.sort(key=lambda d: (-d[1], d[0]))
            pool_sizes.append(len(det))
            cands = [d[0] for d in det]
            fs = filt.score_batch(sub, cands) if cands else []
            order = sorted(zip(cands, [float(a) * float(d[1]) for a, d in zip(fs, det)]),
                           key=lambda x: -x[1])
            ranked[sub] = _dedup([c for c, _ in order], cap, keypool)
        arm = {"threshold": None if T >= 7581 else gen_threshold,
               "mean_pool": round(float(np.mean(pool_sizes)), 1),
               "mean_unique": round(float(np.mean([len(v) for v in ranked.values()])), 1),
               "pool_coverage": round(float(np.mean(
                   [len(set(ranked[s]) & refs[s]) / len(refs[s]) if refs[s] else np.nan
                    for s, _ in items])), 4),
               "by_budget": {}}
        print(f"  => top_k={T}: pool {arm['mean_pool']}, unique {arm['mean_unique']}, "
              f"pool coverage {arm['pool_coverage']}", flush=True)
        for b in budgets:
            g = np.array([recall_at(ranked[s], refs[s], b) for s, _ in items])
            d = g - sy_vec[b]
            bt = d[idx].mean(axis=1)
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            arm["by_budget"][str(b)] = {
                "grail": round(float(g.mean()), 4), "sygma": round(float(sy_vec[b].mean()), 4),
                "gap": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
            r = arm["by_budget"][str(b)]
            print(f"     k={b:>3}: GRAIL {r['grail']:.4f}  SyGMa {r['sygma']:.4f}  "
                  f"gap {r['gap']:+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"{'SIG' if r['excludes_zero'] else 'n.s.'}", flush=True)
        rep["arms"][str(T)] = arm

    keypool.close()
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

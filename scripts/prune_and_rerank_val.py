#!/usr/bin/env python3
"""Prune-and-re-rank on VAL: does cutting training-dead rules clean the ranking, or cost coverage?

The last live rule-granularity hypothesis (STATUS §0a). Its history matters: the original framing
("dead rules inject noise into the 7,581-way target / via untrained id embeddings") was UNDERCUT by
the zero-id ablation, which showed the id component is inert. What survives is a different,
id-independent, INFERENCE-side mechanism: rules that fire but never produce a true metabolite emit
DISTRACTOR products that compete for the top-k slots, so removing them could raise recall@15 without
any retraining.

Two mechanisms make opposite predictions, and this run separates them by measuring BOTH pool
coverage and realised recall:
  - "distractor cut"    -> pool coverage UNCHANGED, recall@15 UP     (ranking was diluted)
  - "reachability loss" -> pool coverage DOWN,      recall@15 DOWN   (the rules were load-bearing
                            on unseen chemistry -- the caveat flagged in STATUS §0a, since a rule
                            useless on train can be a true label on val)
  - "harmless ballast"  -> both ~unchanged                            (dead rules never win a slot
                            anyway; pruning is deploy hygiene only)
All three are pre-registered; whichever lands is the finding.

Design choices that keep it honest:
  * The prune set is derived from TRAINING data ONLY (rules with zero positive labels across the
    4,787 training substrates, `artifacts/preprocessed/train/.../reaction_labels.pt`). No val or
    test information selects the rules, so evaluating on val is leakage-free.
  * Evaluation is on VAL (`val_predictions.csv` supplies substrates + ground truth), honouring the
    test-freeze recorded in STATUS §0a. Same checkpoint pairing as that dump (priors generator +
    single filter), so numbers are comparable to it.
  * Pruning is applied by zeroing the applicability mask, so pruned rules are invisible to the
    top-k selection rather than silently consuming candidate slots -- i.e. it simulates a smaller
    bank, not a crippled one.
  * Paired bootstrap over substrates for the delta (same machinery as rank_flip_ci).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger

from grail_metabolism.config import FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.utils.preparation import _normalize_smiles_cached
from grail_metabolism.workflows.factory import build_filter, build_generator

RDLogger.DisableLog("rdApp.*")

VAL_CSV = ROOT / "artifacts" / "full5000_single" / "predictions" / "val_predictions.csv"
TRAIN_LABELS = ROOT / "artifacts" / "preprocessed" / "train" / "ea9ee257861324be" / "reaction_labels.expanded.pt"
DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
DEPLOYED_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
OUT = ROOT / "results" / "prune_and_rerank_val.json"
TOP_K, MAX_OUT = 128, 15


def _taut(s):
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return None


def boot_ci(delta: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = delta[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(delta.mean()), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="val substrates to evaluate")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    # ---- val substrates + ground truth (no SDF parse needed) ----
    rows = list(csv.DictReader(open(VAL_CSV)))
    subs, truth = [], {}
    for r in rows:
        real = [x for x in r.get("real", "").split("|") if x]
        keys = {k for k in (_taut(x) for x in real) if k}
        if keys:
            subs.append(r["substrate"])
            truth[r["substrate"]] = keys
    subs = subs[: args.n]
    print(f"val substrates: {len(subs)}", flush=True)

    # ---- models (same pairing as the val dump) ----
    gs = torch.load(DEPLOYED_GEN, map_location="cpu", weights_only=False)
    gen = build_generator(GeneratorConfig(**gs["arch"]), gs.get("rules"))
    gen.load_state_dict(gs["state_dict"], strict=False)
    gen.calibrated_threshold = gs.get("calibrated_threshold")
    gen.eval()
    gen.gen_normalization = "canonical"
    fss = torch.load(DEPLOYED_FILTER, map_location="cpu", weights_only=False)
    filt = build_filter(FilterConfig(**fss["arch"]))
    filt.load_state_dict(fss["state_dict"], strict=False)
    filt.eval()
    thr = getattr(gen, "calibrated_threshold", None)
    n_rules = len(gs.get("rules") or [])

    # ---- prune set from TRAINING positives only ----
    lab = torch.load(TRAIN_LABELS, map_location="cpu", weights_only=False)
    M = np.array([list(map(int, v)) for v in lab.values()], dtype=np.int32)
    pos = M.sum(0)
    dead = np.where(pos == 0)[0]
    print(f"training substrates {M.shape[0]} | rules {n_rules} | training-dead (0 positives): "
          f"{len(dead)} ({len(dead)/n_rules:.1%}) -> pruned bank {n_rules - len(dead)}", flush=True)
    dead_set = set(int(i) for i in dead)

    # ---- prune by zeroing the applicability mask (invisible to top-k selection) ----
    orig_score_rules = gen.score_rules

    def masked_score_rules(sub, return_mask=False):
        out = orig_score_rules(sub, return_mask=True)
        scores, mask = out
        mask = np.array(mask, copy=True)
        mask[dead] = 0.0
        scores = np.array(scores, copy=True)
        scores[dead] = -1e9
        return (scores, mask) if return_mask else scores

    def run_arm(label: str):
        """-> (recall@15 vector, pool-coverage vector, mean pool size)"""
        r15 = np.zeros(len(subs))
        rinf = np.zeros(len(subs))
        pool_sizes = []
        t0 = time.time()
        for i, sub in enumerate(subs):
            if i and i % 50 == 0:
                print(f"    [{label}] {i}/{len(subs)} ({time.time()-t0:.0f}s)", flush=True)
            keys = truth[sub]
            mol = Chem.MolFromSmiles(sub)
            detailed = gen.generate_scored_with_details(sub, top_k=TOP_K, threshold=thr,
                                                        compute_sites=False) if mol is not None else []
            smis = [_normalize_smiles_cached(d[0], "canonical") for d in detailed]
            pool_sizes.append(len(smis))
            if not smis:
                continue
            fs = filt.score_batch(sub, smis)
            scored = sorted(zip(smis, (float(f) * float(d[1]) for f, d in zip(fs, detailed))),
                            key=lambda x: -x[1])
            ranked, seen = [], set()
            for s, _ in scored:
                k = _taut(s)
                if k and k not in seen:
                    seen.add(k)
                    ranked.append(k)
            r15[i] = len(set(ranked[:MAX_OUT]) & keys) / len(keys)
            rinf[i] = len(set(ranked) & keys) / len(keys)   # pool coverage (reachability realised)
        return r15, rinf, float(np.mean(pool_sizes))

    print("  arm 1/2: FULL bank", flush=True)
    full15, fullinf, full_pool = run_arm("full")
    print("  arm 2/2: PRUNED bank", flush=True)
    gen.score_rules = masked_score_rules
    gen._rule_embedding_cache = None
    prune15, pruneinf, prune_pool = run_arm("pruned")

    d15, lo15, hi15 = boot_ci(prune15 - full15, args.n_boot, args.seed)
    dinf, loinf, hiinf = boot_ci(pruneinf - fullinf, args.n_boot, args.seed)

    def verdict():
        cov_hurt = hiinf < 0
        rec_up = lo15 > 0
        rec_down = hi15 < 0
        if rec_up and not cov_hurt:
            return "DISTRACTOR CUT: recall up with coverage intact -> pruning cleans the ranking"
        if rec_down or cov_hurt:
            return "REACHABILITY LOSS: pruned rules were load-bearing on unseen chemistry"
        return "HARMLESS BALLAST: dead rules neither help nor hurt -> pruning is deploy hygiene only"

    report = {
        "split": "val", "n_substrates": len(subs), "match": "inchikey_tautomer",
        "operating_point": {"candidate_top_k": TOP_K, "max_output": MAX_OUT, "policy": "rank-only"},
        "bank": {"full": n_rules, "pruned": n_rules - len(dead), "removed": len(dead),
                 "prune_rule": "zero positive labels across 4,787 TRAINING substrates (no val/test info)"},
        "recall_at_15": {"full": round(float(full15.mean()), 4), "pruned": round(float(prune15.mean()), 4),
                         "delta": round(d15, 4), "ci95": [round(lo15, 4), round(hi15, 4)]},
        "pool_coverage_recall_inf": {"full": round(float(fullinf.mean()), 4),
                                     "pruned": round(float(pruneinf.mean()), 4),
                                     "delta": round(dinf, 4), "ci95": [round(loinf, 4), round(hiinf, 4)]},
        "mean_pool_size": {"full": round(full_pool, 2), "pruned": round(prune_pool, 2)},
        "verdict": verdict(),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n=== PRUNE-AND-RE-RANK (val, n={len(subs)}, tautomer) ===", flush=True)
    print(f"bank {n_rules} -> {n_rules - len(dead)}  (removed {len(dead)} training-dead)", flush=True)
    print(f"recall@15      full {full15.mean():.4f}  pruned {prune15.mean():.4f}  "
          f"delta {d15:+.4f} 95%CI[{lo15:+.4f},{hi15:+.4f}]", flush=True)
    print(f"pool coverage  full {fullinf.mean():.4f}  pruned {pruneinf.mean():.4f}  "
          f"delta {dinf:+.4f} 95%CI[{loinf:+.4f},{hiinf:+.4f}]", flush=True)
    print(f"mean pool size full {full_pool:.1f}  pruned {prune_pool:.1f}", flush=True)
    print(f"VERDICT: {report['verdict']}", flush=True)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

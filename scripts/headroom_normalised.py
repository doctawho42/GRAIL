#!/usr/bin/env python3
"""Scale-free (headroom-normalised) form of the differential match-sensitivity endpoint.

The primary endpoint is a paired difference of raw gains on a [0,1]-bounded metric, and the arms
enter the relaxation step from different baselines. Externally the arm with more headroom is the
arm that gains more, so the headroom confound runs WITH the reported effect there rather than
against it (internally it runs against it). This computes the obvious scale-free version so the
direction can be checked rather than argued about:

    normalised gain  =  mean(recall_relaxed - recall_strict) / (1 - mean(recall_strict))

i.e. the gain expressed as the fraction of the remaining headroom it consumes. The differential is
the difference of two methods' normalised gains, resampled paired over substrates (substrates are
the independent unit, and the same resample indexes both methods and both protocols, so the ratio's
numerator and denominator move together).

This is a ROBUSTNESS column, not a replacement estimand: the endpoint remains the raw paired
difference of a differential re-scoring of frozen predictions, which is well defined.

Populations, budgets and criteria are the same ones the raw endpoint uses:
  * external: the 37 GLORYx drugs, k=15, inchikey -> inchikey_tautomer (which equals the stereo
    step for every method whose no-stereo and tautomer columns coincide);
  * internal: the 150-substrate shared subset, k=15, canonical -> inchikey_tautomer.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from grail_metabolism.metrics import _match_keys, top_k_recall  # noqa: E402
from scripts.eval_on_gloryx import load_gloryx  # noqa: E402
from scripts.run_match_sensitivity import _dedup_canon, load_grail_csv  # noqa: E402

DATA = ROOT / "docs" / "benchmark" / "data"
OUT = ROOT / "results" / "headroom_normalised.json"
K = 15
N_BOOT = 10000
SEED = 0

EXT_PRED = {
    "BioTransformer": DATA / "biotransformer_gloryx.json",
    "MetaPredictor": DATA / "metapredictor_gloryx.json",
    "GRAIL": DATA / "grail_reranker_gloryx.json",
}


def per_substrate_recall(preds, reals, subs, match, k, dedup_canon: bool):
    v = np.empty(len(subs), dtype=float)
    for i, s in enumerate(subs):
        if dedup_canon:
            raw = _dedup_canon(preds.get(s, []))
            ranked = [next(iter(_match_keys([item], match))) for item in raw]
        else:
            ranked, seen = [], set()
            for item in preds.get(s, []):
                key = next(iter(_match_keys([item], match)), None)
                if key and key not in seen:
                    seen.add(key)
                    ranked.append(key)
        v[i] = top_k_recall(ranked, _match_keys(reals[s], match), k)
    return v


def boot_idx(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def norm_gain(strict, relaxed, idx):
    """Point estimate and paired-bootstrap CI of mean(gain)/(1 - mean(strict))."""
    def stat(sel):
        s = strict[sel].mean()
        g = (relaxed[sel] - strict[sel]).mean()
        return g / (1.0 - s) if s < 1.0 else float("nan")
    pt = stat(slice(None))
    b = np.array([stat(i) for i in idx])
    return float(pt), [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def norm_diff(sA, rA, sB, rB, idx):
    """B's normalised gain minus A's, paired over substrates."""
    def stat(sel):
        gA = (rA[sel] - sA[sel]).mean() / (1.0 - sA[sel].mean())
        gB = (rB[sel] - sB[sel]).mean() / (1.0 - sB[sel].mean())
        return gB - gA
    pt = stat(slice(None))
    b = np.array([stat(i) for i in idx])
    lo, hi = float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
    return {"b_minus_a": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "verdict": "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."}


def block(vec, methods, idx, strict, relaxed, n):
    rep = {"n_substrates": n, "k": K, "strict": strict, "relaxed": relaxed,
           "per_method": {}, "pairwise_normalised_differential": {}}
    for m in methods:
        s, r = vec[(m, strict)], vec[(m, relaxed)]
        pt, ci = norm_gain(s, r, idx)
        rep["per_method"][m] = {
            "recall_strict": round(float(s.mean()), 4),
            "recall_relaxed": round(float(r.mean()), 4),
            "raw_gain": round(float((r - s).mean()), 4),
            "headroom": round(float(1.0 - s.mean()), 4),
            "normalised_gain": round(pt, 4),
            "normalised_gain_ci95": [round(ci[0], 4), round(ci[1], 4)],
        }
    for a, b in combinations(methods, 2):
        rep["pairwise_normalised_differential"][f"{a}_vs_{b}"] = norm_diff(
            vec[(a, strict)], vec[(a, relaxed)], vec[(b, strict)], vec[(b, relaxed)], idx)
    return rep


def main() -> int:
    report = {"n_boot": N_BOOT, "seed": SEED,
              "estimand": "mean(relaxed - strict) / (1 - mean(strict)), paired bootstrap over substrates",
              "note": "robustness column for the raw differential-sensitivity endpoint; not a replacement estimand"}

    # ---- external: 37 GLORYx drugs, inchikey -> inchikey_tautomer -------------------------------
    reals = load_gloryx(DATA / "gloryx_test.json")
    subs = sorted(s for s in reals if reals[s])
    print(f"external: {len(subs)} GLORYx substrates with references", flush=True)
    methods = {}
    for name, path in EXT_PRED.items():
        d = json.loads(path.read_text())
        methods[name] = {s: d.get(s, []) for s in subs}
    from scripts.eval_on_gloryx import sygma_predictions
    print("  SyGMa: recomputing with py-sygma ...", flush=True)
    sp = sygma_predictions(subs)
    methods["SyGMa"] = {s: sp.get(s, [])[:K] for s in subs}

    vec = {}
    for name, preds in methods.items():
        for proto in ("inchikey", "inchikey_tautomer"):
            vec[(name, proto)] = per_substrate_recall(preds, reals, subs, proto, K, dedup_canon=False)
    idx = boot_idx(len(subs), N_BOOT, SEED)
    report["external"] = block(vec, list(methods), idx, "inchikey", "inchikey_tautomer", len(subs))

    # ---- internal: 150-substrate shared subset, canonical -> inchikey_tautomer -------------------
    grail_preds, ireals = load_grail_csv(ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv")
    isubs = [s for s in ireals if ireals[s]][:150]
    ireals = {s: ireals[s] for s in isubs}
    bt = json.loads((ROOT / "artifacts" / "tier2" / "biotransformer_preds.json").read_text())
    imeth = {"GRAIL": {s: grail_preds.get(s, []) for s in isubs},
             "BioTransformer": {s: bt.get(s, []) for s in isubs}}
    # Same guard as rank_flip_ci.load_method: a cache built for a different substrate list would
    # silently score the missing substrates 0. Keyset membership is the test, as there (a method
    # may legitimately return an empty list for a substrate it covers).
    for name, src in (("GRAIL", grail_preds), ("BioTransformer", bt)):
        cov = sum(1 for s in isubs if s in src)
        if cov / len(isubs) < 0.98:
            raise SystemExit(f"ERROR: {name} covers only {cov}/{len(isubs)} of the subset")
    ivec = {}
    for name, preds in imeth.items():
        for proto in ("canonical", "inchikey_tautomer"):
            ivec[(name, proto)] = per_substrate_recall(preds, ireals, isubs, proto, K, dedup_canon=True)
    iidx = boot_idx(len(isubs), N_BOOT, SEED)
    report["internal"] = block(ivec, list(imeth), iidx, "canonical", "inchikey_tautomer", len(isubs))

    for scope in ("external", "internal"):
        r = report[scope]
        print(f"\n=== {scope} (n={r['n_substrates']}, {r['strict']} -> {r['relaxed']}) ===")
        for m, v in r["per_method"].items():
            print(f"  {m:>15}  strict {v['recall_strict']:.4f}  raw {v['raw_gain']:+.4f}  "
                  f"headroom {v['headroom']:.4f}  normalised {v['normalised_gain']:+.4f} "
                  f"[{v['normalised_gain_ci95'][0]:+.4f},{v['normalised_gain_ci95'][1]:+.4f}]")
        for pair, v in r["pairwise_normalised_differential"].items():
            print(f"  {pair:>36}  Δnorm {v['b_minus_a']:+.4f} "
                  f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] {v['verdict']}")

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

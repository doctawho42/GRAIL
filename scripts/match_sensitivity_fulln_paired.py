#!/usr/bin/env python3
"""Match sensitivity for three methods under five criteria on the FULL clean test split, paired.

The published five-method table is scored on 150 shared substrates, which is the first objection a
reviewer raises. GRAIL, SyGMa and MetaPredictor all have predictions for the whole 1,170-substrate
split, so the load-bearing claim can be made at eight times the sample size, paired on identical
substrates, with interactions corrected for multiplicity.

Uses the paper's own matcher (scripts.gloryx_rank_flip_ci.per_substrate_recall) so these numbers
are comparable with the subset table rather than a second convention.
"""
from __future__ import annotations
import json, math, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_key_tables import recall_vector  # torch-free; tables built and checked upstream

MODES = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
STRICT, TOLERANT = "canonical", "inchikey_tautomer"
K, N_BOOT, SEED, ALPHA = 15, 10000, 0, 0.05
OUT = ROOT / "results" / "match_sensitivity_fulln_paired.json"


def load_methods(subs):
    m = {}
    recs = json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]
    m["GRAIL"] = {r["sub"]: r["deployed_top15"] for r in recs}
    m["MetaPredictor"] = json.loads((ROOT / "artifacts" / "tier2_1170" / "metapredictor_preds.json").read_text())
    sy = ROOT / "results" / "sygma_fulltest_predictions.json"
    if sy.exists():
        m["SyGMa"] = json.loads(sy.read_text())
    else:
        print("WARNING: SyGMa predictions absent -- running with two methods", flush=True)
    for name, preds in m.items():
        missing = sum(1 for s in subs if not preds.get(s))
        print(f"  {name:15} covers {len(set(subs) & set(preds))}/{len(subs)}, empty {missing}", flush=True)
        if len(set(subs) & set(preds)) < len(subs):
            raise SystemExit(f"ERROR: {name} does not cover the split; a paired design needs all of it.")
    return m


def p_from_boot(draws):
    """Two-sided bootstrap p: twice the smaller tail mass at zero."""
    lo = float((draws <= 0).mean())
    hi = float((draws >= 0).mean())
    return max(min(2 * min(lo, hi), 1.0), 1.0 / len(draws))


def holm(tests, alpha=ALPHA):
    out, live = [], True
    for i, (name, p, extra) in enumerate(sorted(tests, key=lambda t: t[1])):
        thr = alpha / (len(tests) - i)
        rej = live and p < thr
        live = rej
        out.append({"pair": name, "p": round(p, 6), "threshold": round(thr, 5), "rejected": rej, **extra})
    return out


def main() -> int:
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    subs = sorted(truth)
    print(f"substrates with references: {len(subs)}", flush=True)
    methods = load_methods(subs)

    # Key every distinct structure once, in parallel, then score from the table. Tautomer
    # canonicalisation is the cost and the prediction sets barely repeat, so a per-process cache
    # does not help. Each mode's table is checked against the serial scorer before it is used.
    vec = {}
    for mode in MODES:
        table = json.loads((ROOT / "results" / "key_tables" / f"{mode}.json").read_text())
        print(f"  {mode:20} {len(table):,} keys", flush=True)
        for name, preds in methods.items():
            vec[(name, mode)] = recall_vector(preds, truth, subs, table, K)
            print(f"    {name:15} {vec[(name, mode)].mean():.4f}", flush=True)

    rep = {"n_substrates": len(subs), "k": K, "n_boot": N_BOOT, "seed": SEED, "modes": MODES,
           "strict": STRICT, "tolerant": TOLERANT,
           "recall_at_15": {n: {m: round(float(vec[(n, m)].mean()), 4) for m in MODES} for n in methods},
           "sensitivity": {}, "pairwise": {}}

    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(subs), len(subs)) for _ in range(N_BOOT)]

    gain = {n: vec[(n, TOLERANT)] - vec[(n, STRICT)] for n in methods}
    for n in methods:
        d = np.array([gain[n][i].mean() for i in idx])
        rep["sensitivity"][n] = {"gain": round(float(gain[n].mean()), 4),
                                 "ci95": [round(float(np.quantile(d, .025)), 4),
                                          round(float(np.quantile(d, .975)), 4)]}

    tests = []
    for a, b in combinations(sorted(methods), 2):
        diff = gain[b] - gain[a]
        d = np.array([diff[i].mean() for i in idx])
        p = p_from_boot(d)
        extra = {"interaction_b_minus_a": round(float(diff.mean()), 4),
                 "ci95": [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)],
                 "reorders": bool(np.sign(vec[(a, STRICT)].mean() - vec[(b, STRICT)].mean()) !=
                                  np.sign(vec[(a, TOLERANT)].mean() - vec[(b, TOLERANT)].mean()))}
        tests.append((f"{a}_vs_{b}", p, extra))
    rep["pairwise"] = holm(tests)

    print("\nHolm-Bonferroni over %d pairs:" % len(tests), flush=True)
    for r in rep["pairwise"]:
        print(f"  {r['pair']:30} d={r['interaction_b_minus_a']:+.4f} {r['ci95']}  "
              f"p={r['p']:.2e}  {'rejected' if r['rejected'] else 'not rejected'}"
              f"{'  REORDERS' if r['reorders'] else ''}", flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Partial identification of F1 under positive-unlabelled annotation.

Every precision figure in this evaluation is a lower bound: references are annotated metabolites,
so a predicted structure that is a genuine metabolite nobody recorded is charged as a false
positive. That threatens the metric-reordering result specifically, because the method it demotes
is the one emitting eighty candidates, which is the one with the most room for unrecorded hits.

Model the annotation as a uniform propensity: each true metabolite of a substrate is recorded
independently with probability c. Then for a fixed prediction set of size n covering t true
metabolites out of T,

    E[observed TP] = c*t,  E[|references|] = c*T

so observed recall = c*t/(c*T) = t/T is UNBIASED, observed precision = c*t/n is the truth scaled by
c, and observed F1 = 2ct/(n + cT) is neither -- c enters numerator and denominator differently.
Precision's ORDERING is therefore identified for any c, recall's is identified outright, and F1's
is not. This computes the F1 ordering as a function of c and reports the critical c at which each
adjacent pair swaps, which is the quantity that says whether the reordering finding survives.

c=1 recovers the reported numbers. Smaller c means sparser annotation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODE = "inchikey_tautomer"
GRID = np.round(np.arange(0.05, 1.0001, 0.005), 4)
OUT = ROOT / "results" / "pu_propensity_bounds.json"


def keyset(items, table):
    return {k for k in (table.get(s) for s in items) if k}


def obs_vectors(preds, truth, subs, table):
    """Per-substrate observed precision and recall, plus prediction-set size."""
    P, R, N = (np.zeros(len(subs)) for _ in range(3))
    keep = np.zeros(len(subs), dtype=bool)
    for i, s in enumerate(subs):
        pred, real = keyset(preds.get(s, []), table), keyset(truth[s], table)
        if not real:
            continue
        keep[i] = True
        tp = len(pred & real)
        N[i] = len(pred)
        P[i] = tp / len(pred) if pred else 0.0
        R[i] = tp / len(real)
    return P[keep], R[keep], N[keep]


def macro_f1(P_obs, R_obs, c):
    """Macro F1 after correcting precision for propensity c; recall needs no correction.

    Precision is a probability, so the corrected value is capped at 1. The cap is what makes the
    correction conservative for the wide-output method: it cannot be inflated past certainty.
    """
    P = np.minimum(P_obs / c, 1.0)
    denom = P + R_obs
    f1 = np.zeros_like(P)
    nz = denom > 0
    f1[nz] = 2 * P[nz] * R_obs[nz] / denom[nz]
    return float(f1.mean())


def main() -> int:
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    grail = {r["sub"]: r["deployed_top15"]
             for r in json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]}
    methods = {
        "GRAIL": grail,
        "SyGMa": json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text()),
        "MetaPredictor": json.loads((ROOT / "artifacts" / "tier2_1170" / "metapredictor_preds.json").read_text()),
    }
    table = json.loads((ROOT / "results" / "key_tables" / f"{MODE}.json").read_text())
    subs = sorted(set.intersection(*(set(m) for m in methods.values())) & set(truth))
    subs = [s for s in subs if truth[s]]

    vec = {n: obs_vectors(p, truth, subs, table) for n, p in methods.items()}
    names = sorted(methods)
    curves = {n: [macro_f1(vec[n][0], vec[n][1], c) for c in GRID] for n in names}

    # Critical c per adjacent pair: the largest c at which the c=1 ordering no longer holds.
    crossings = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = np.array(curves[a]) - np.array(curves[b])
            sign_at_1 = np.sign(d[-1])
            flipped = np.where(np.sign(d) != sign_at_1)[0]
            crossings[f"{a}-{b}"] = {
                "diff_at_c1": round(float(d[-1]), 4),
                "flips": bool(len(flipped)),
                "critical_c": round(float(GRID[flipped[-1]]), 4) if len(flipped) else None,
            }

    rep = {"mode": MODE, "n_substrates": len(subs), "grid": GRID.tolist(),
           "macro_f1_by_c": {n: [round(v, 4) for v in curves[n]] for n in names},
           "crossings": crossings,
           "mean_output": {n: round(float(vec[n][2].mean()), 2) for n in names}}
    OUT.write_text(json.dumps(rep, indent=1))

    print(f"n={len(subs)} substrates, criterion={MODE}\n")
    print("macro F1 under annotation propensity c")
    print(f"  {'c':>6}  " + "  ".join(f"{n:>14}" for n in names))
    for c in (1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05):
        j = int(np.argmin(np.abs(GRID - c)))
        print(f"  {GRID[j]:>6.3f}  " + "  ".join(f"{curves[n][j]:>14.4f}" for n in names))
    print("\nordering flips")
    for k, v in crossings.items():
        state = f"flips at c={v['critical_c']}" if v["flips"] else "never flips on the grid"
        print(f"  {k:32} diff at c=1: {v['diff_at_c1']:+.4f}   {state}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

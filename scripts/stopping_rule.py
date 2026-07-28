#!/usr/bin/env python3
"""Does a stopping rule on the model's own scores beat a fixed output budget?

The oracle cut is worth 48-76% relative on macro F1, and `cardinality_crossfit.py` showed a
substrate-conditioned head recovers 3-4% of it, because most of the headroom is knowledge of where
the hits fell in the ranking. The scores carry exactly that knowledge, so this asks whether they can
be turned into a per-substrate cut.

Everything is cross-fitted: the calibration map, the thresholds and the constant baseline are all
fit on training folds and evaluated on held-out substrates, so no arm sees its own answer.

  emitted     the deployed output, every candidate the pipeline returns
  constant    the best single k, fit on the training folds
  threshold   emit while the raw ranking score exceeds tau, tau fit on the training folds
  calibrated  the score is first mapped to a hit probability by isotonic regression fit on the
              training folds, then the rule emits while that probability exceeds tau
  expected-F1 per substrate, the k maximising expected F1 under the calibrated probabilities, with
              the reference count estimated self-consistently as the probability mass of the pool.
              This is the rule the reward analysis in the GFlowNet appendix points at: emit while
              the marginal candidate's hit probability exceeds F1/2
  oracle      the best k per substrate, the unreachable bound

The scores come from `results/scored_predictions.json`, which reproduces the deployed output on all
1,170 substrates, so the `emitted` arm here equals the deployed number by construction.
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
MODE = "inchikey_tautomer"
KMAX, FOLDS, SEED = 32, 5, 0
OUT = ROOT / "results" / "stopping_rule.json"


def f1_at(hits, n_ref, k):
    """hits is the cumulative-hit indicator per rank; k is 1-based."""
    tp = int(hits[:k].sum())
    n = min(k, len(hits))
    return 2 * tp / (n + n_ref) if (n + n_ref) else 0.0


def curve(hits, n_ref):
    return np.array([f1_at(hits, n_ref, k) for k in range(1, KMAX + 1)])


def main() -> int:
    dump = json.loads((ROOT / "results" / "scored_predictions.json").read_text())
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    table = json.loads((ROOT / "results" / "key_tables" / f"{MODE}.json").read_text())

    rows = []
    for r in dump["rows"]:
        if r.get("error") or r["sub"] not in truth or not truth[r["sub"]]:
            continue
        real = {k for k in (table.get(x) for x in truth[r["sub"]]) if k}
        if not real:
            continue
        seen, hits, scores = set(), [], []
        for c in r["candidates"]:
            key = table.get(c["smiles"])
            if not key or key in seen:
                continue
            seen.add(key)
            hits.append(1.0 if key in real else 0.0)
            scores.append(c["combined"])
        if not hits:
            continue
        h = np.array(hits + [0.0] * max(0, KMAX - len(hits)))[:KMAX]
        s = np.array(scores + [0.0] * max(0, KMAX - len(scores)))[:KMAX]
        rows.append({"h": h, "s": s, "n_ref": len(real), "n_pred": len(hits),
                     "curve": curve(np.array(hits), len(real))})

    C = np.array([r["curve"] for r in rows])
    H = np.array([r["h"] for r in rows])
    S = np.array([r["s"] for r in rows])
    NP = np.array([r["n_pred"] for r in rows])
    n = len(rows)
    print(f"{n} substrates, ranking recorded to {KMAX}, criterion {MODE}\n")

    arms = {k: [] for k in ("emitted", "constant", "threshold", "calibrated", "expected_f1")}
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for tr, te in kf.split(C):
        for i in te:
            arms["emitted"].append(C[i, min(NP[i], KMAX) - 1])

        k_const = int(np.argmax(C[tr].mean(axis=0))) + 1
        for i in te:
            arms["constant"].append(C[i, k_const - 1])

        # raw-score threshold: sweep tau on train, apply on test
        cand = np.unique(np.round(S[tr][S[tr] > 0], 4))
        cand = cand[:: max(1, len(cand) // 400)] if len(cand) > 400 else cand
        def apply_tau(idx, tau, sc):
            out = []
            for i in idx:
                keep = int((sc[i][: NP[i]] > tau).sum())
                out.append(C[i, max(1, keep) - 1])
            return out
        best_tau = max(cand, key=lambda t: float(np.mean(apply_tau(tr, t, S)))) if len(cand) else 0.0
        arms["threshold"] += apply_tau(te, best_tau, S)

        # isotonic calibration of score -> P(hit), fit on train ranks only
        flat_s = np.concatenate([S[i][: NP[i]] for i in tr])
        flat_h = np.concatenate([H[i][: NP[i]] for i in tr])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(flat_s, flat_h)
        P = np.zeros_like(S)
        for i in range(n):
            P[i][: NP[i]] = iso.predict(S[i][: NP[i]])
        cand_p = np.unique(np.round(np.concatenate([P[i][: NP[i]] for i in tr]), 4))
        cand_p = cand_p[:: max(1, len(cand_p) // 400)] if len(cand_p) > 400 else cand_p
        best_p = max(cand_p, key=lambda t: float(np.mean(apply_tau(tr, t, P)))) if len(cand_p) else 0.0
        arms["calibrated"] += apply_tau(te, best_p, P)

        # expected-F1 cut: reference mass estimated as the pool's probability mass
        for i in te:
            p = P[i][: NP[i]]
            ref_hat = max(p.sum(), 1e-6)
            exp_f1 = [2 * p[:k].sum() / (k + ref_hat) for k in range(1, len(p) + 1)]
            k_hat = int(np.argmax(exp_f1)) + 1 if len(exp_f1) else 1
            arms["expected_f1"].append(C[i, k_hat - 1])

    # Paired bootstrap over substrates on the arms' per-substrate values. The folds partition the
    # substrates, so each arm has exactly one value per substrate and the pairing is exact.
    order = np.concatenate([te for _, te in KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(C)])
    per_sub = {k: np.zeros(n) for k in arms}
    for k, v in arms.items():
        per_sub[k][order] = np.array(v)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, (10000, n))
    paired = {}
    for k in ("emitted", "threshold", "calibrated", "expected_f1"):
        d = per_sub[k] - per_sub["constant"]
        bt = d[idx].mean(axis=1)
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        paired[k] = {"delta": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "excludes_zero": bool(lo > 0 or hi < 0)}

    oracle = float(C.max(axis=1).mean())
    means = {k: round(float(np.mean(v)), 4) for k, v in arms.items()}
    base = means["constant"]
    rep = {"n": n, "folds": FOLDS, "seed": SEED, "mode": MODE, "macro_f1": means,
           "oracle": round(oracle, 4),
           "gain_over_constant": {k: round(v - base, 4) for k, v in means.items() if k != "constant"},
           "paired_vs_constant": paired,
           "oracle_share": {k: (round((v - base) / (oracle - base), 3) if oracle > base else None)
                            for k, v in means.items() if k != "constant"}}
    OUT.write_text(json.dumps(rep, indent=1))

    print(f"{'arm':14}{'macro F1':>10}{'vs constant':>13}{'share of oracle':>17}")
    for k in ("emitted", "constant", "threshold", "calibrated", "expected_f1"):
        g = "" if k == "constant" else f"{means[k]-base:+.4f}"
        sh = "" if k == "constant" else (f"{rep['oracle_share'][k]:.1%}" if rep["oracle_share"][k] is not None else "n/a")
        print(f"{k:14}{means[k]:10.4f}{g:>13}{sh:>17}")
    print(f"{'oracle':14}{oracle:10.4f}{oracle-base:+13.4f}{'100.0%':>17}")
    print(f"\n{'arm':14}{'delta vs constant':>19}{'95% CI':>24}{'':>6}")
    for k, v in paired.items():
        print(f"  {k:12}{v['delta']:+19.4f}   [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
              f"   {'SIG' if v['excludes_zero'] else 'n.s.'}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

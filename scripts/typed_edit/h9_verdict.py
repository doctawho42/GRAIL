"""The H9 check: the pool cap, on the validation split.

H9 fixes a cap of 100 candidates by generator score before any validation measurement and
predicts it beats the uncapped pool by at least +0.015 of micro recall@15, roughly half what it
showed on the 291 it was read off. The cap is adopted as a cost guard whichever way this falls;
only the claim that it improves the ranking is decided here.

The absent substrate is bounded rather than excluded, on the same terms as the H7 check: its
references are credited first to one arm and then to the other, and the verdict stands only if
both extremes agree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

CAP, THRESHOLD = 100, 0.015
N_BOOT, SEED = 10000, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "results/val_pools.json"))
    ap.add_argument("--out", default=str(ROOT / "results/h9_verdict.json"))
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--declared-n", type=int, default=294)
    ap.add_argument("--missing-refs", type=int, default=2)
    args = ap.parse_args()

    d = json.loads(Path(args.pools).read_text())
    pools, refs = d["pools"], d["references"]
    subs = sorted(s for s in pools if refs.get(s))
    ks = (1, 5, 10, 15, 20, 30, 50)

    h_cap = {k: [] for k in ks}
    h_unc = {k: [] for k in ks}
    n_ref, kept = [], []
    for s in subs:
        real = set(refs[s])
        n_ref.append(len(real))
        unc = [c["key"] for c in rrf_order(pools[s])]
        keep = sorted(pools[s], key=lambda c: -c["generator"])[:CAP]
        kept.append(len(keep))
        cap = [c["key"] for c in rrf_order(keep)]
        for k in ks:
            h_unc[k].append(len(set(unc[:k]) & real))
            h_cap[k].append(len(set(cap[:k]) & real))

    U = np.array(n_ref, dtype=float)
    N = float(U.sum())
    a = {k: np.array(v, dtype=float) for k, v in h_cap.items()}
    b = {k: np.array(v, dtype=float) for k, v in h_unc.items()}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)
    kk = args.k
    diff = a[kk] - b[kk]
    bt = diff[idx].sum(axis=1) / denom
    lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
    point = float(diff.sum() / N)

    gap = args.declared_n - len(subs)
    bound = None
    if gap > 0:
        m, tot = args.missing_refs, N + args.missing_refs
        bound = {"absent_substrates": gap, "absent_references": m,
                 "worst_case": (diff.sum() - m) / tot, "best_case": (diff.sum() + m) / tot}

    verdict = "supported" if (point >= THRESHOLD and (lo > 0 or hi < 0)) else "failed"
    if bound and (bound["worst_case"] >= THRESHOLD) != (bound["best_case"] >= THRESHOLD):
        verdict = "undetermined: the absent substrates can flip it"

    out = {"provenance": stamp(__file__), "hypothesis": "H9",
           "registered_threshold": THRESHOLD, "cap": CAP, "split": "validation",
           "match": d.get("match"), "n_substrates": len(subs), "n_references": N,
           "declared_population": args.declared_n, "k": kk,
           "aggregation": "micro, ratio of sums",
           "mean_pool": {"uncapped": round(float(np.mean([len(pools[s]) for s in subs])), 1),
                         "capped": round(float(np.mean(kept)), 1)},
           "micro": {"uncapped": round(float(b[kk].sum() / N), 4),
                     "capped": round(float(a[kk].sum() / N), 4),
                     "difference": round(point, 4)},
           "bootstrap": {"n": N_BOOT, "seed": SEED, "ci95": [round(lo, 4), round(hi, 4)],
                         "excludes_zero": bool(lo > 0 or hi < 0)},
           "curves_micro": {str(k): {"uncapped": round(float(b[k].sum() / N), 4),
                                     "capped": round(float(a[k].sum() / N), 4)} for k in ks},
           "absence_bound": bound, "verdict": verdict,
           "note": "The cap is adopted as a cost guard whichever way this falls; only the claim "
                   "that it improves the ranking is decided here."}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"H9 on {len(subs)} validation substrates, {N:.0f} references")
    print(f"  mean pool  {out['mean_pool']['uncapped']} -> {out['mean_pool']['capped']}")
    print(f"\n{'k':>4}{'uncapped':>11}{'capped':>9}{'diff':>9}")
    for k in ks:
        c = out["curves_micro"][str(k)]
        print(f"{k:>4}{c['uncapped']:>11.4f}{c['capped']:>9.4f}{c['capped']-c['uncapped']:>+9.4f}")
    print(f"\n  at k={kk}: {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"threshold {THRESHOLD:+.3f}")
    if bound:
        print(f"  {bound['absent_substrates']} absent, {bound['absent_references']} references: "
              f"difference in [{bound['worst_case']:+.4f}, {bound['best_case']:+.4f}]")
    print(f"  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

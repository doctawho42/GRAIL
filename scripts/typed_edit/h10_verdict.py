"""The H10 check: what the whole bank buys over the trained rule budget.

H10 fixes the operating point at the budget the checkpoint records, 30, and predicts the whole
bank of 7,581 templates buys at most +0.05 of micro recall@15 over it. Both arms carry the H9
cap of 100 and the H7 fusion rule, so they differ only in how many templates were applied.

The two pools were built separately and need not cover the same substrates. The comparison runs
on the substrates both hold, and every substrate either lacks is bounded rather than dropped:
its references are credited first to one arm and then to the other, and the verdict stands only
if both extremes agree.
"""
from __future__ import annotations

import argparse
import glob
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

CAP, THRESHOLD = 100, 0.05
N_BOOT, SEED = 10000, 0
KS = (1, 5, 10, 15, 20, 30, 50)


def load(spec):
    pools, refs, tk = {}, {}, set()
    for f in sorted(glob.glob(spec)) or [spec]:
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
        if "top_k" in d:
            tk.add(d["top_k"])
    return pools, refs, (sorted(tk) or [None])[0]


def ordered_keys(pool):
    keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
    return [c["key"] for c in rrf_order(keep)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whole-bank", default=str(ROOT / "results/val_pools.json"))
    ap.add_argument("--trained", default=str(ROOT / "results/valpools_k30/all.json"))
    ap.add_argument("--out", default=str(ROOT / "results/h10_verdict.json"))
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    big, refs_b, tk_b = load(args.whole_bank)
    small, refs_s, tk_s = load(args.trained)
    refs = {**refs_s, **refs_b}

    both = sorted(s for s in set(big) & set(small) if refs.get(s))
    only_big = sorted(s for s in set(big) - set(small) if refs.get(s))
    only_small = sorted(s for s in set(small) - set(big) if refs.get(s))
    print(f"whole bank {len(big)} substrates (top_k={tk_b}), "
          f"trained {len(small)} (top_k={tk_s}), both {len(both)}")

    hb = {k: [] for k in KS}
    hs = {k: [] for k in KS}
    n_ref, pool_b, pool_s, sizes_b, sizes_s = [], [], [], [], []
    for s in both:
        real = set(refs[s])
        n_ref.append(len(real))
        pool_b.append(len(big[s])); pool_s.append(len(small[s]))
        ob, os_ = ordered_keys(big[s]), ordered_keys(small[s])
        sizes_b.append(len(ob)); sizes_s.append(len(os_))
        for k in KS:
            hb[k].append(len(set(ob[:k]) & real))
            hs[k].append(len(set(os_[:k]) & real))

    U = np.array(n_ref, dtype=float)
    N = float(U.sum())
    a = {k: np.array(v, dtype=float) for k, v in hb.items()}   # whole bank
    b = {k: np.array(v, dtype=float) for k, v in hs.items()}   # trained budget

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(both), (N_BOOT, len(both)))
    denom = np.maximum(U[idx].sum(axis=1), 1)
    kk = args.k
    diff = a[kk] - b[kk]                    # what the whole bank buys
    bt = diff[idx].sum(axis=1) / denom
    lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
    point = float(diff.sum() / N)

    missing = sorted(set(only_big) | set(only_small))
    m = sum(len(refs[s]) for s in missing)
    bound = None
    if missing:
        tot = N + m
        bound = {"absent_substrates": len(missing), "absent_references": m,
                 "worst_case": (diff.sum() + m) / tot,   # worst for H10 is the bank buying more
                 "best_case": (diff.sum() - m) / tot}

    verdict = "supported" if point <= THRESHOLD else "failed"
    if bound and (bound["worst_case"] <= THRESHOLD) != (bound["best_case"] <= THRESHOLD):
        verdict = "undetermined: the absent substrates can flip it"

    out = {"provenance": stamp(__file__), "hypothesis": "H10",
           "registered_threshold": THRESHOLD, "direction": "the whole bank must buy no more",
           "split": "validation", "cap": CAP,
           "top_k": {"whole_bank": tk_b, "trained": tk_s},
           "top_k_note": "the whole-bank pool artifact predates the flag that records the rule "
                         "budget, so it carries none; it was built by build_val_pools.py when "
                         "that script passed 7581 unconditionally, which git records and the "
                         "artifact does not",
           "n_substrates": len(both), "n_references": N,
           "aggregation": "micro, ratio of sums",
           "mean_pool": {"whole_bank": round(float(np.mean(pool_b)), 1),
                         "trained": round(float(np.mean(pool_s)), 1)},
           "micro": {"whole_bank": round(float(a[kk].sum() / N), 4),
                     "trained": round(float(b[kk].sum() / N), 4),
                     "bought_by_the_whole_bank": round(point, 4)},
           "bootstrap": {"n": N_BOOT, "seed": SEED, "ci95": [round(lo, 4), round(hi, 4)],
                         "excludes_zero": bool(lo > 0 or hi < 0)},
           "curves_micro": {str(k): {"whole_bank": round(float(a[k].sum() / N), 4),
                                     "trained": round(float(b[k].sum() / N), 4)} for k in KS},
           # A budget the arm cannot fill is not a ranking result. At 30 templates the pool
           # holds 17 candidates on average, so above k=15 the trained arm is not ranking worse,
           # it is running out of things to return, and the two are not the same claim.
           "substrates_whose_pool_is_smaller_than_the_budget": {
               str(k): {"whole_bank": int(sum(1 for n in sizes_b if n < k)),
                        "trained": int(sum(1 for n in sizes_s if n < k))} for k in KS},
           "absence_bound": bound,
           "absent_substrates": {"only_in_the_whole_bank_pool": len(only_big),
                                 "only_in_the_trained_pool": len(only_small)},
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"\n{len(both)} substrates, {N:.0f} references")
    print(f"  mean pool  whole bank {out['mean_pool']['whole_bank']}  "
          f"trained {out['mean_pool']['trained']}")
    print(f"\n{'k':>4}{'whole bank':>13}{'trained':>10}{'bought':>9}")
    for k in KS:
        c = out["curves_micro"][str(k)]
        print(f"{k:>4}{c['whole_bank']:>13.4f}{c['trained']:>10.4f}"
              f"{c['whole_bank']-c['trained']:>+9.4f}")
    print(f"\n  at k={kk}: the whole bank buys {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  registered ceiling {THRESHOLD:+.2f}")
    if bound:
        print(f"  {bound['absent_substrates']} absent, {bound['absent_references']} references: "
              f"bought lies in [{bound['best_case']:+.4f}, {bound['worst_case']:+.4f}]")
    print("\n  substrates whose pool is smaller than the budget:")
    for k in KS:
        r = out["substrates_whose_pool_is_smaller_than_the_budget"][str(k)]
        print(f"    k={k:<3} whole bank {r['whole_bank']:>4}   trained {r['trained']:>4}"
              f"  of {len(both)}")
    print(f"\n  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

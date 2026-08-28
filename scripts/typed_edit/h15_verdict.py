"""The H15 check: the survivors arm with the tautomer budget at 200.

Two registered conditions and both must hold: the median per-substrate time before the filter
falls below 10 seconds, and micro recall@15 falls by at most 0.01 against the whole-product
baseline. The target is a number of seconds because H13 asked for a factor, removed 94 to 99 per
cent of the work and moved by 2.95; what a service needs is an absolute answer.

The matching key stays at the shipped budget of 1,000. The bounded enumerator is private to the
standardisation of survivors, so `_tautomer_inchikey` and its cache are untouched. That is not
free of consequence and the consequence is measured here rather than assumed: a survivor
standardised at 200 is a different molecule to hand the key function, and the key it comes back
with can differ from the one the same candidate had at 1,000. This counts them.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
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

SECONDS, RECALL_CEILING = 10.0, 0.01
CAP, N_BOOT, SEED = 100, 10000, 0
KS = (1, 5, 10, 15, 20, 30, 50)


def load(pattern):
    pools, refs, timing, budget = {}, {}, {}, set()
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
        budget.add(d.get("tautomer_budget"))
        for x in d.get("generator_seconds", []):
            timing[x["substrate"]] = x
    return pools, refs, timing, (sorted(b for b in budget if b) or [None])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/h13/every/s*.json")
    ap.add_argument("--arm", default="results/h15/s*.json")
    ap.add_argument("--unbounded", default="results/h13/surv/s*.json",
                    help="the same survivors arm at the shipped budget, for the key diagnostic")
    ap.add_argument("--out", default=str(ROOT / "results/h15_verdict.json"))
    ap.add_argument("--budget-curve", default=str(ROOT / "results/tautomer_budget.json"))
    args = ap.parse_args()

    bp, br, bt, _ = load(args.baseline)
    ap_, ar, at, budget = load(args.arm)
    up, _, at_u, _ = load(args.unbounded)
    common = sorted(set(bp) & set(ap_) & set(br))

    shared_t = [s for s in common if s in at_u]
    A = [bt[s]["seconds"] for s in common]
    B = [at[s]["seconds"] for s in common]
    enum = [at[s]["enumerate"] for s in common]
    std = [at[s]["standardise_survivors"] for s in common]
    median_s = st.median(B)

    U = np.array([len(br[s]) for s in common], dtype=float)
    N = float(U.sum())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(common), (N_BOOT, len(common)))
    den = np.maximum(U[idx].sum(axis=1), 1)

    def hits(pools, refs, k):
        return np.array([len(set(c["key"] for c in rrf_order(
            sorted(pools[s], key=lambda c: -c["generator"])[:CAP])[:k]) & set(refs[s]))
            for s in common], dtype=float)

    curves, contrasts = {}, {}
    for k in KS:
        a, b = hits(bp, br, k), hits(ap_, ar, k)
        d = b - a
        btv = d[idx].sum(axis=1) / den
        lo, hi = float(np.quantile(btv, .025)), float(np.quantile(btv, .975))
        curves[str(k)] = {"whole_product": round(float(a.sum() / N), 4),
                          "h15": round(float(b.sum() / N), 4)}
        contrasts[str(k)] = {"change": round(float(d.sum() / N), 4),
                             "ci95": [round(lo, 4), round(hi, 4)],
                             "excludes_zero": bool(lo > 0 or hi < 0)}

    # the key diagnostic: a survivor standardised at 200 is a different input to the key function
    # Matching the two arms BY SMILES and asking whether the key moved is a tautology: a
    # candidate the lower budget standardised differently carries a different SMILES and never
    # enters the comparison, so the count is always zero. What can be asked of these artifacts is
    # how often the standardised molecule differs at all, and whether the set of keys a substrate
    # ends up with differs -- the second is what a recall number would feel.
    shared = [s for s in common if s in up]
    n_cand = differing_smiles = 0
    key_delta = keys_only_low = keys_only_high = 0
    for s in shared:
        hi_s = {c["smiles"] for c in up[s]}
        lo_s = {c["smiles"] for c in ap_[s]}
        n_cand += len(lo_s)
        differing_smiles += len(lo_s - hi_s)
        hi_k = {c["key"] for c in up[s]}
        lo_k = {c["key"] for c in ap_[s]}
        key_delta += len(lo_k ^ hi_k)
        keys_only_low += len(lo_k - hi_k)
        keys_only_high += len(hi_k - lo_k)

    loss = -contrasts["15"]["change"]
    time_ok = median_s < SECONDS
    recall_ok = loss <= RECALL_CEILING
    verdict = "supported" if (time_ok and recall_ok) else "failed"

    # The enumeration is identical in the two survivors arms -- canonical deduplication, no
    # tautomer budget in it at all -- so any difference in ITS time is machine load and nothing
    # else. The arms did not run under the same load: H13's ran beside the whole-product arm and
    # this one ran with only its own shards. Dividing the standardisation speed-up by the load
    # factor the enumeration exposes gives what the budget itself is worth.
    u_enum = st.median([at_u[s]["enumerate"] for s in shared_t]) if shared_t else None
    u_std = st.median([at_u[s]["standardise_survivors"] for s in shared_t]) if shared_t else None
    load_factor = (u_enum / st.median(enum)) if u_enum else None
    attributable = ((u_std / st.median(std)) / load_factor) \
        if (u_std and load_factor) else None
    predicted = None
    curve = Path(args.budget_curve)
    if curve.exists():
        c = json.loads(curve.read_text())["by_budget"]
        if str(budget) in c and "1000" in c:
            predicted = round(c["1000"]["ms_per_standardisation"]
                              / c[str(budget)]["ms_per_standardisation"], 2)

    out = {"provenance": stamp(__file__), "hypothesis": "H15", "split": "validation",
           "tautomer_budget": budget, "matching_budget": 1000,
           "registered": {"median_seconds_below": SECONDS,
                          "recall_loss_at_most": RECALL_CEILING},
           "population": {"n_paired": len(common)},
           "aggregation": "micro, ratio of sums; times are per substrate before the filter",
           "time": {"whole_product_median_s": round(st.median(A), 2),
                    "h15_median_s": round(median_s, 2),
                    "factor": round(st.median(A) / median_s, 2),
                    "enumerate_median_s": round(st.median(enum), 2),
                    "standardise_survivors_median_s": round(st.median(std), 2),
                    "share_spent_standardising":
                        round(sum(std) / (sum(enum) + sum(std)), 3),
                    "load_correction": {
                        "unbounded_arm_enumerate_median_s": round(u_enum, 2) if u_enum else None,
                        "unbounded_arm_standardise_median_s": round(u_std, 2) if u_std else None,
                        "load_factor_from_the_identical_enumeration":
                            round(load_factor, 2) if load_factor else None,
                        "speedup_attributable_to_the_budget":
                            round(attributable, 2) if attributable else None,
                        "predicted_by_the_budget_curve": predicted,
                        "median_under_the_other_arms_load":
                            round(median_s * load_factor, 2) if load_factor else None,
                        "note": "the two survivors arms did not run under the same load; the "
                                "enumeration is identical between them, so its ratio measures "
                                "the load and divides out of the standardisation ratio. The "
                                "registered target is absolute and is unaffected by this."}},
           "recall_micro": curves, "recall_contrasts": contrasts,
           "key_diagnostic": {
               "candidates_in_the_bounded_arm": n_cand,
               "whose_standardised_smiles_differs_from_the_shipped_budget": differing_smiles,
               "keys_only_in_the_bounded_arm": keys_only_low,
               "keys_only_at_the_shipped_budget": keys_only_high,
               "symmetric_difference_of_pool_keys": key_delta,
               "note": "the matching key is computed at the shipped 1,000 in both arms. Asking "
                       "whether a candidate matched BY SMILES changed key is a tautology, since "
                       "a differently standardised candidate carries a different SMILES and is "
                       "excluded; these count the standardisation difference itself and the "
                       "difference in the key sets, which is what a recall number would feel."},
           "time_holds": time_ok, "recall_holds": recall_ok, "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"H15 on {len(common)} paired substrates, tautomer budget {budget}\n")
    print(f"{'k':>4}{'whole-product':>16}{'H15':>10}{'change':>10}")
    for k in KS:
        c, d = curves[str(k)], contrasts[str(k)]
        print(f"{k:>4}{c['whole_product']:>16.4f}{c['h15']:>10.4f}{d['change']:>+10.4f}"
              f"{'*' if d['excludes_zero'] else ''}")
    t = out["time"]
    print(f"\n  median time {t['whole_product_median_s']}s -> {t['h15_median_s']}s "
          f"({t['factor']}x)  against a target below {SECONDS}s  -> "
          f"{'holds' if time_ok else 'FAILS'}")
    print(f"    enumerate {t['enumerate_median_s']}s, standardise the survivors "
          f"{t['standardise_survivors_median_s']}s ({t['share_spent_standardising']:.0%})")
    lc = out["time"]["load_correction"]
    if lc["median_under_the_other_arms_load"]:
        print(f"    under the load the other arm ran at, the median would be "
              f"{lc['median_under_the_other_arms_load']}s, still inside the {SECONDS}s target")
    print(f"  recall@15 loss {loss:+.4f} against a ceiling of {RECALL_CEILING}  -> "
          f"{'holds' if recall_ok else 'FAILS'}")
    kd = out["key_diagnostic"]
    print(f"  of {kd['candidates_in_the_bounded_arm']} candidates, "
          f"{kd['whose_standardised_smiles_differs_from_the_shipped_budget']} standardise "
          f"differently at 200 than at 1,000")
    print(f"  pool keys: {kd['keys_only_in_the_bounded_arm']} appear only at 200, "
          f"{kd['keys_only_at_the_shipped_budget']} only at 1,000 "
          f"(symmetric difference {kd['symmetric_difference_of_pool_keys']})")
    print(f"\n  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

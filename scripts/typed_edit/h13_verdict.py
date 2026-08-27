"""The H13 check: standardisation off the enumeration loop.

H13 predicts two things and both must hold. Micro recall@15 on validation falls by at most 0.01,
and the median per-substrate generator time falls by at least ten times.

Both arms are timed over everything before the filter, which is the only comparison that is fair:
the survivors arm does work the whole-product arm never did separately -- standardising the
hundred that survive the cap -- because in the whole-product arm that work was spread over four
thousand products. Timing only the enumeration would answer a question nobody registered.

The peptide at index 83 appears in the survivors arm and not in the whole-product arm, which has
never finished it in any run. Its absence is stated rather than dropped; every figure here is on
the 293 both arms hold.
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

RECALL_CEILING, TIME_FACTOR = 0.01, 10.0
CAP, N_BOOT, SEED = 100, 10000, 0
KS = (1, 5, 10, 15, 20, 30, 50)


def load(pattern):
    pools, refs, timing = {}, {}, {}
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
        for x in d.get("generator_seconds", []):
            timing[x["substrate"]] = x
    return pools, refs, timing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", default="results/h13/every/s*.json")
    ap.add_argument("--survivors", default="results/h13/surv/s*.json")
    ap.add_argument("--out", default=str(ROOT / "results/h13_verdict.json"))
    args = ap.parse_args()

    ep, er, et = load(args.every)
    sp, sr, stt = load(args.survivors)
    common = sorted(set(ep) & set(sp) & set(er))
    absent = sorted(set(sp) - set(ep))

    A = [et[s]["seconds"] for s in common]
    B = [stt[s]["seconds"] for s in common]
    enum = [stt[s]["enumerate"] for s in common]
    surv_std = [stt[s]["standardise_survivors"] for s in common]
    ratio = sorted(a / b for a, b in zip(A, B) if b > 0)
    med_factor = st.median(A) / st.median(B)

    U = np.array([len(er[s]) for s in common], dtype=float)
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
        a, b = hits(ep, er, k), hits(sp, sr, k)
        d = b - a
        bt = d[idx].sum(axis=1) / den
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        curves[str(k)] = {"every_product": round(float(a.sum() / N), 4),
                          "survivors": round(float(b.sum() / N), 4)}
        contrasts[str(k)] = {"change": round(float(d.sum() / N), 4),
                             "ci95": [round(lo, 4), round(hi, 4)],
                             "excludes_zero": bool(lo > 0 or hi < 0)}

    loss = -contrasts["15"]["change"]
    recall_ok = loss <= RECALL_CEILING
    time_ok = med_factor >= TIME_FACTOR
    verdict = "supported" if (recall_ok and time_ok) else "failed"

    out = {"provenance": stamp(__file__), "hypothesis": "H13", "split": "validation",
           "registered": {"recall_loss_at_most": RECALL_CEILING,
                          "median_time_falls_by_at_least": TIME_FACTOR},
           "population": {"n_paired": len(common),
                          "absent_from_every_product": len(absent),
                          "absent_note": "the 291-heavy-atom peptide at index 83, which the "
                                         "whole-product arm has never finished in any run"},
           "aggregation": "micro, ratio of sums; times are per substrate before the filter",
           "cap": CAP,
           "time": {"every_product_median_s": round(st.median(A), 2),
                    "survivors_median_s": round(st.median(B), 2),
                    "median_factor": round(med_factor, 2),
                    "every_product_total_h": round(sum(A) / 3600, 2),
                    "survivors_total_h": round(sum(B) / 3600, 2),
                    "per_substrate_factor": {"median": round(ratio[len(ratio) // 2], 2),
                                             "p90": round(ratio[int(.9 * len(ratio))], 2),
                                             "max": round(ratio[-1], 2)},
                    "inside_survivors": {
                        "enumerate_median_s": round(st.median(enum), 2),
                        "standardise_survivors_median_s": round(st.median(surv_std), 2),
                        "share_spent_standardising":
                            round(sum(surv_std) / (sum(enum) + sum(surv_std)), 3)},
                    "enumeration_alone_against_the_old_total":
                        round(st.median(A) / st.median(enum), 1)},
           "recall_micro": curves, "recall_contrasts": contrasts,
           "recall_holds": recall_ok, "time_holds": time_ok,
           "reading": "the mechanism registered was right and the total was not: enumeration "
                      "collapses by 13.9 times, and standardising the hundred survivors is 72 "
                      "per cent of what remains, so the arm as a whole moves by 2.95",
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"H13 on {len(common)} paired substrates ({len(absent)} absent from the whole-product "
          f"arm)\n")
    print(f"{'k':>4}{'every-product':>16}{'survivors':>12}{'change':>10}")
    for k in KS:
        c, d = curves[str(k)], contrasts[str(k)]
        print(f"{k:>4}{c['every_product']:>16.4f}{c['survivors']:>12.4f}{d['change']:>+10.4f}"
              f"{'*' if d['excludes_zero'] else ''}")
    t = out["time"]
    print(f"\n  recall@15 loss {loss:+.4f} against a ceiling of {RECALL_CEILING}  -> "
          f"{'holds' if recall_ok else 'fails'}")
    print(f"  median time {t['every_product_median_s']}s -> {t['survivors_median_s']}s = "
          f"{t['median_factor']}x against {TIME_FACTOR}x  -> {'holds' if time_ok else 'fails'}")
    print(f"  enumeration alone: {t['enumeration_alone_against_the_old_total']}x; "
          f"standardising the survivors is {t['inside_survivors']['share_spent_standardising']:.0%}"
          f" of the new arm")
    print(f"\n  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

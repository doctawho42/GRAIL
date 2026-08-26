"""Does the registered pool-relative emission rule survive the change of ranking?

Section 0.2a registers "emit every candidate scoring at least alpha times the best", with
alpha = 0.5, defined on the product of the filter and generator scores. The ranking is now rank
fusion. This measures what the same rule emits under each, because a policy that transfers in
name and not in behaviour is worse than one that is replaced.

It also records why. A relative threshold assumes the score carries a scale. Fusion scores are
sums of reciprocal ranks, so a pool's best and worst differ by a few per cent while the product's
differ by orders of magnitude, and the same alpha therefore means two unrelated things. The
artifact carries the spread of both so the claim is checkable rather than asserted.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import RRF_K, competition_ranks  # noqa: E402

ALPHAS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99)


def fusion_scores(pool):
    rf = competition_ranks(pool, lambda c: c["filter"])
    rg = competition_ranks(pool, lambda c: c["generator"])
    return [1.0 / (RRF_K + rf[i]) + 1.0 / (RRF_K + rg[i]) for i in range(len(pool))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "results/widepools_k30/all.json"))
    ap.add_argument("--out", default=str(ROOT / "results/emission_rule_transfer.json"))
    args = ap.parse_args()

    pools = {}
    for f in sorted(glob.glob(args.pools)) or [args.pools]:
        pools.update(json.loads(Path(f).read_text())["pools"])
    subs = sorted(s for s in pools if pools[s])

    spread_p, spread_f = [], []
    for s in subs:
        p = pools[s]
        if len(p) < 2:
            continue
        c = sorted((x["combined"] for x in p), reverse=True)
        r = sorted(fusion_scores(p), reverse=True)
        if c[0] > 0:
            spread_p.append(c[-1] / c[0])
        if r[0] > 0:
            spread_f.append(r[-1] / r[0])

    rows = {}
    for a in ALPHAS:
        n_p, n_f = [], []
        for s in subs:
            p = pools[s]
            top_c = max(x["combined"] for x in p)
            n_p.append(sum(1 for x in p if x["combined"] >= a * top_c))
            r = fusion_scores(p)
            top_r = max(r)
            n_f.append(sum(1 for x in r if x >= a * top_r))
        rows[str(a)] = {"emitted_by_product": round(st.mean(n_p), 2),
                        "emitted_by_fusion": round(st.mean(n_f), 2)}

    rep = {"provenance": stamp(__file__),
           "population": {"n": len(subs), "pool": args.pools},
           "mean_pool": round(st.mean(len(pools[s]) for s in subs), 2),
           "registered_alpha": 0.5,
           "worst_over_best_score_in_a_pool": {
               "product": {"median": round(st.median(spread_p), 8),
                           "max": round(max(spread_p), 8)},
               "fusion": {"median": round(st.median(spread_f), 4),
                          "min": round(min(spread_f), 4)}},
           "by_alpha": rows,
           "reading": "the same rule emits about two candidates under the product and nearly the "
                      "whole pool under fusion, because fusion scores span a few per cent where "
                      "the product spans orders of magnitude; a relative threshold on a rank "
                      "statistic is a rank cutoff with a nonlinear knob"}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"{len(subs)} substrates, mean pool {rep['mean_pool']}")
    print(f"\n{'alpha':>7}{'by product':>13}{'by fusion':>12}")
    for a in ALPHAS:
        r = rows[str(a)]
        print(f"{a:>7}{r['emitted_by_product']:>13.2f}{r['emitted_by_fusion']:>12.2f}")
    sp, sf = rep["worst_over_best_score_in_a_pool"]["product"], \
        rep["worst_over_best_score_in_a_pool"]["fusion"]
    print(f"\nworst over best score in a pool: product median {sp['median']:.2e}, "
          f"fusion median {sf['median']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

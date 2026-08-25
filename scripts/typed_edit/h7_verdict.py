#!/usr/bin/env python3
"""The H7 check: rank fusion against the product, on the validation split.

H7 fixes reciprocal rank fusion with k=60 before any validation measurement and predicts it
beats the deployed product by at least +0.05 of micro recall@15 over the same selector-free
pool. This computes both rankings on the merged validation pools, the paired bootstrap on
their difference, and the verdict against the registered threshold.

Micro recall is the ratio of sums (hits over all substrates divided by references over all
substrates), which is what the registration names; the macro mean is reported alongside it
because the two have been confused in this project before and the pair makes that visible.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

RRF_K = 60          # Cormack, Clarke and Buettcher 2009; not tuned here
THRESHOLD = 0.05    # the registered margin
BOOT, BOOT_SEED = 10000, 0


def _ranks(pool, field):
    """Dense competition ranks, 1-based, descending by `field`; ties share the lower rank."""
    order = sorted(range(len(pool)), key=lambda i: -pool[i][field])
    out, prev, cur = [0] * len(pool), None, 1
    for pos, i in enumerate(order, 1):
        v = pool[i][field]
        if v != prev:
            prev, cur = v, pos
        out[i] = cur
    return out


def rankings(pool):
    """Return the pool ordered by the product and by rank fusion of the two components."""
    by_product = sorted(pool, key=lambda c: -c["combined"])
    rf, rg = _ranks(pool, "filter"), _ranks(pool, "generator")
    fused = sorted(range(len(pool)),
                   key=lambda i: -(1.0 / (RRF_K + rf[i]) + 1.0 / (RRF_K + rg[i])))
    return by_product, [pool[i] for i in fused]


def hits(ordered, refs, k):
    return len({c["key"] for c in ordered[:k]} & set(refs))


def micro(per_sub, key):
    h = sum(per_sub[s][key] for s in per_sub)
    n = sum(per_sub[s]["n_ref"] for s in per_sub)
    return h / n if n else 0.0


def macro(per_sub, key):
    vals = [per_sub[s][key] / per_sub[s]["n_ref"] for s in per_sub if per_sub[s]["n_ref"]]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "results/val_pools.json"))
    ap.add_argument("--out", default=str(ROOT / "results/h7_verdict.json"))
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--declared-n", type=int, default=294,
                    help="substrates the registered population contains")
    ap.add_argument("--missing-refs", type=int, default=0,
                    help="references held by substrates absent from the pools")
    args = ap.parse_args()

    d = json.loads(Path(args.pools).read_text())
    pools, refs = d["pools"], d["references"]

    per_sub, ks = {}, (1, 5, 10, 15, 20, 30, 50)
    for s, pool in pools.items():
        r = set(refs[s])
        prod, fused = rankings(pool)
        rec = {"n_ref": len(r), "n_pool": len(pool),
               "ceiling": len({c["key"] for c in pool} & r)}
        for k in ks:
            rec[f"product@{k}"] = hits(prod, r, k)
            rec[f"fusion@{k}"] = hits(fused, r, k)
        per_sub[s] = rec

    subs = sorted(per_sub)
    kk = args.k
    pa, fa = f"product@{kk}", f"fusion@{kk}"
    point = micro(per_sub, fa) - micro(per_sub, pa)

    rng = random.Random(BOOT_SEED)
    diffs = []
    for _ in range(BOOT):
        draw = [per_sub[subs[rng.randrange(len(subs))]] for _ in range(len(subs))]
        n = sum(x["n_ref"] for x in draw)
        if not n:
            continue
        diffs.append((sum(x[fa] for x in draw) - sum(x[pa] for x in draw)) / n)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    p_clears = sum(1 for x in diffs if x >= THRESHOLD) / len(diffs)

    curves = {str(k): {"product": micro(per_sub, f"product@{k}"),
                       "fusion": micro(per_sub, f"fusion@{k}")} for k in ks}

    # An absent substrate is bounded rather than assumed away: the difference is recomputed
    # with every one of its references credited to the product and then to the fusion. If the
    # verdict is the same at both extremes, the absence cannot have produced it.
    n_ref = sum(per_sub[s]["n_ref"] for s in subs)
    gap = args.declared_n - len(subs)
    bound = None
    if gap > 0:
        h = sum(per_sub[s][fa] for s in subs) - sum(per_sub[s][pa] for s in subs)
        m, tot = args.missing_refs, n_ref + args.missing_refs
        bound = {"absent_substrates": gap, "absent_references": m,
                 "worst_case": (h - m) / tot, "best_case": (h + m) / tot}

    verdict = "supported" if point >= THRESHOLD else "failed"
    if bound and (bound["worst_case"] >= THRESHOLD) != (bound["best_case"] >= THRESHOLD):
        verdict = "undetermined: the absent substrates can flip it"
    out = {
        "provenance": stamp(__file__),
        "hypothesis": "H7", "registered_threshold": THRESHOLD, "rrf_k": RRF_K,
        "split": "validation", "match": d.get("match"), "population": d.get("population"),
        "n_substrates": len(subs),
        "n_references": sum(per_sub[s]["n_ref"] for s in subs),
        "declared_population": args.declared_n,
        "k": kk,
        "ceiling_micro": micro(per_sub, "ceiling"),
        "micro": {"product": micro(per_sub, pa), "fusion": micro(per_sub, fa),
                  "difference": point},
        "macro": {"product": macro(per_sub, pa), "fusion": macro(per_sub, fa),
                  "difference": macro(per_sub, fa) - macro(per_sub, pa)},
        "bootstrap": {"n": len(diffs), "seed": BOOT_SEED, "ci95": [lo, hi],
                      "p_difference_at_least_threshold": p_clears},
        "curves_micro": curves,
        "absence_bound": bound,
        "verdict": verdict,
        "caveats": [
            "The fusion rule was chosen after looking at the 291 MetaTox substrates; this "
            "validation check is the out-of-argmax measurement H7 registers, not a second "
            "selection.",
            "The checkpoints were selected on validation, so validation is not fully clean for "
            "them. The bias runs against the fusion rule, which the product-shaped selection "
            "favours, so the margin measured here is conservative.",
        ],
    }
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"H7 on {len(subs)} validation substrates, {out['n_references']} references")
    print(f"  pool ceiling (micro)   {out['ceiling_micro']:.4f}")
    print(f"  product   recall@{kk}   {out['micro']['product']:.4f}")
    print(f"  fusion    recall@{kk}   {out['micro']['fusion']:.4f}")
    print(f"  difference             {point:+.4f}   "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]   threshold {THRESHOLD:+.2f}")
    print(f"  P(difference >= threshold) = {p_clears:.3f}")
    print(f"  macro difference       {out['macro']['difference']:+.4f}")
    if bound:
        print(f"  {bound['absent_substrates']} substrate(s) absent, "
              f"{bound['absent_references']} reference(s): difference lies in "
              f"[{bound['worst_case']:+.4f}, {bound['best_case']:+.4f}]")
    print(f"  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

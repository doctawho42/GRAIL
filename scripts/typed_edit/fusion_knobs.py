#!/usr/bin/env python3
"""Two parameters of the ranking that were left at a default, swept on the pools already built.

A paper arguing that silent parameters decide verdicts carries two of its own. The fusion constant
$K$ is the value reciprocal rank fusion was published with and was never varied. And the
candidate score aggregates over the rules that produce a candidate by noisy-or, which assumes the
rules are independent evidence; in this bank they are demonstrably not, since 152 of SyGMa's
templates and 611 of BioTransformer's sit in it verbatim and the mined half is full of
near-duplicates. Noisy-or over redundant templates inflates whatever many templates can reach.

Only one of the two can be answered from files already on disk. The stored pools carry both
component scores for every candidate, so the fusion can be recomputed at any $K$ without a model
run, and that sweep is here. They carry one aggregated generator score per candidate and not the
per-rule scores a maximum would need, so the aggregation question is named, bounded and left
unmeasured rather than answered with a different quantity.

    python scripts/typed_edit/fusion_knobs.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

CAP = 100
KS_BUDGET = (5, 15, 30, 50)
K_VALUES = (1, 10, 30, 60, 120, 300)
DEPLOYED_K = 60
N_BOOT, SEED = 10000, 0


def fuse(pool, k_const):
    """Reciprocal rank fusion of the two component orderings at a given constant."""
    by_gen = sorted(range(len(pool)), key=lambda i: -pool[i]["generator"])
    by_fil = sorted(range(len(pool)), key=lambda i: -pool[i]["filter"])
    rank = defaultdict(float)
    for r, i in enumerate(by_gen, 1):
        rank[i] += 1.0 / (k_const + r)
    for r, i in enumerate(by_fil, 1):
        rank[i] += 1.0 / (k_const + r)
    return sorted(range(len(pool)), key=lambda i: -rank[i])


def keys_in_order(pool, order, parent):
    out, seen = [], set()
    for i in order:
        key = pool[i].get("key")
        if key and key != parent and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="results/widepools_implicit/w*.json")
    ap.add_argument("--out", default=str(ROOT / "results" / "fusion_knobs.json"))
    args = ap.parse_args()

    from bank_without_selection import _key as tautkey

    pools, refs = {}, {}
    for f in sorted(glob.glob(str(ROOT / args.pools))) or [str(ROOT / args.pools)]:
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"]); refs.update(blob["references"])
    subs = sorted(s for s in pools if refs.get(s))
    real = {s: set(refs[s]) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)
    parent = {s: tautkey(s) for s in subs}
    capped = {s: sorted(pools[s], key=lambda c: -c["generator"])[:CAP] for s in subs}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def hits(orders, k):
        return np.array([len(set(orders[s][:k]) & real[s]) for s in subs], dtype=float)

    def contrast(a, b):
        d = a - b
        bt = d[idx].sum(axis=1) / denom
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(d.sum() / U.sum()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    # The fusion constant.
    orders_by_k = {k: {s: keys_in_order(capped[s], fuse(capped[s], k), parent[s]) for s in subs}
                   for k in K_VALUES}
    base = orders_by_k[DEPLOYED_K]
    k_rows = {}
    for k in K_VALUES:
        row = {"recall": {str(b): round(float(hits(orders_by_k[k], b).sum() / U.sum()), 4)
                          for b in KS_BUDGET}}
        if k != DEPLOYED_K:
            row["against_the_deployed_constant_at_15"] = contrast(
                hits(orders_by_k[k], 15), hits(base, 15))
        k_rows[str(k)] = row

    # The other question cannot be answered from these files, and saying which is part of the
    # answer. Noisy-or over the rules that reach a candidate assumes those rules are independent
    # evidence, and in a bank holding 611 of BioTransformer's templates verbatim they are not. The
    # alternative aggregation is a maximum, which ignores duplication. But the pools store one row
    # per candidate carrying a generator score that has already been aggregated: the per-rule
    # scores the maximum would need are not in them, and recovering them means re-running the
    # generator over the whole comparison set. The comparison is therefore named and not made.
    agg = {"status": "not measurable from the released pools",
           "why": ("the pools carry one aggregated generator score per candidate, not the "
                   "per-rule scores a maximum would need; recomputing them requires re-running "
                   "the generator over the comparison set"),
           "what_it_would_test": ("whether noisy-or inflates candidates reachable by many "
                                  "near-duplicate templates, which this bank contains by "
                                  "construction")}

    report = {
        "provenance": stamp(__file__),
        "population": {"n_substrates": len(subs), "n_references": int(U.sum())},
        "cap": CAP,
        "deployed_constant": DEPLOYED_K,
        "constants_swept": list(K_VALUES),
        "by_constant": k_rows,
        "aggregation": agg,
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "reading": (
            "The constant was left at the value the method was published with. If recall is flat "
            "across two orders of magnitude of it then leaving it there costs nothing and the "
            "paper can say so; if it is not, the deployed value is a choice that had to be "
            "declared like the others."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"{len(subs)} substrates, {int(U.sum())} references\n")
    print(f"{'K':>6s} " + " ".join(f"r@{b:<5d}" for b in KS_BUDGET) + "   vs deployed at 15")
    for k in K_VALUES:
        row = k_rows[str(k)]
        cells = " ".join(f"{row['recall'][str(b)]:.4f}" for b in KS_BUDGET)
        c = row.get("against_the_deployed_constant_at_15")
        tail = "" if c is None else (f"   {c['gap']:+.4f} [{c['ci95'][0]:+.4f}, "
                                     f"{c['ci95'][1]:+.4f}]"
                                     f"{'  separates' if c['excludes_zero'] else ''}")
        mark = "  <- deployed" if k == DEPLOYED_K else ""
        print(f"{k:6d} {cells}{tail}{mark}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

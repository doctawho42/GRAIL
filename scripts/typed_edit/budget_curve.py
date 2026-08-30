#!/usr/bin/env python3
"""What the rule budget buys, measured on validation, so the deployed value is a chosen point.

The interactive mode applies the thirty templates the released checkpoint records, and the paper
has never said why thirty. It is the value the checkpoint was trained at, which is a fact about
how the run was configured and not an argument. This measures the curve: validation pools built at
several rule budgets with everything downstream identical, and recall and cost read off each.

Validation, not the comparison set, because the budget is a design choice and choosing it on the
population the comparison is reported on would be selecting on the test.

Whatever budgets have been built are used, so the curve can be extended by building another pool
without touching this file. The deployed budget is marked in the output rather than assumed to be
the best.

    python scripts/typed_edit/budget_curve.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

CAP = 100
KS = (1, 5, 10, 15, 30, 50)
DEPLOYED = 30
N_BOOT, SEED = 10000, 0


def pools_on_disk() -> dict:
    """{rule budget: path} for every validation pool built, read from the directory names."""
    found = {}
    for path in sorted((ROOT / "results").glob("valpools_k*/all.json")):
        match = re.search(r"valpools_k(\d+)", str(path))
        if match:
            found[int(match.group(1))] = path
    return found


def main() -> int:
    import numpy as np

    from _rrf import rrf_order

    built = pools_on_disk()
    if not built:
        raise SystemExit("no validation pools found; run scripts/typed_edit/run_budget_sweep.sh")

    # One population for every budget, so the curve is paired: the substrates every built pool
    # holds. A budget measured on its own substrates would be a different experiment per point.
    per_budget, refs, sizes = {}, {}, {}
    for budget, path in built.items():
        blob = json.loads(path.read_text())
        per_budget[budget] = blob["pools"]
        sizes[budget] = len(blob["pools"])
        refs.update(blob["references"])

    # A pool still being built holds a prefix of the population, and intersecting it with the
    # others silently shrinks every other budget's population to that prefix. That is how a
    # four-point curve came to be measured on 83 substrates instead of 294. A pool short of the
    # largest is refused and named rather than quietly narrowing the experiment.
    full = max(sizes.values())
    partial = {b: n for b, n in sizes.items() if n < full}
    for budget in partial:
        del per_budget[budget]
    if not per_budget:
        raise SystemExit("every pool is partial; let the sweep finish")
    common = sorted(set.intersection(*(set(p) for p in per_budget.values())))
    subs = [s for s in common if refs.get(s)]
    real = {s: set(refs[s]) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def ranked(pool):
        keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
        out, seen = [], set()
        for c in rrf_order(keep):
            key = c.get("key")
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    orders = {b: {s: ranked(per_budget[b][s]) for s in subs} for b in per_budget}

    def hits(budget, k):
        return np.array([len(set(orders[budget][s][:k]) & real[s]) for s in subs], dtype=float)

    rows = {}
    for budget in sorted(orders):
        lists = orders[budget]
        rows[budget] = {
            "recall_micro": {str(k): round(float(hits(budget, k).sum() / U.sum()), 4)
                             for k in KS},
            "mean_candidates": round(float(np.mean([min(len(lists[s]), CAP) for s in subs])), 2),
            "median_candidates": round(float(np.median([min(len(lists[s]), CAP) for s in subs])), 1),
            "substrates_with_an_empty_pool": sum(1 for s in subs if not lists[s]),
        }

    # Every budget against the deployed one at the budget the paper reports its headline at, paired
    # on the same substrates, so the question "would a different budget have been better" is
    # answered with an interval rather than by comparing two point estimates.
    contrasts = {}
    if DEPLOYED in orders:
        base = hits(DEPLOYED, 15)
        for budget in sorted(orders):
            if budget == DEPLOYED:
                continue
            d = hits(budget, 15) - base
            bt = d[idx].sum(axis=1) / denom
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            contrasts[str(budget)] = {"gap_at_15": round(float(d.sum() / U.sum()), 4),
                                      "ci95": [round(lo, 4), round(hi, 4)],
                                      "excludes_zero": bool(lo > 0 or hi < 0)}

    better = [b for b, c in contrasts.items() if c["gap_at_15"] > 0 and c["excludes_zero"]]
    report = {
        "provenance": stamp(__file__),
        "split": "validation",
        "population": {"n_substrates": len(subs), "n_references": int(U.sum()),
                       "note": "the substrates every built pool holds, so the curve is paired"},
        "budgets_built": sorted(orders),
        "budgets_skipped_as_partial": {str(b): n for b, n in sorted(partial.items())},
        "substrates_each_complete_pool_holds": full,
        "deployed_budget": DEPLOYED,
        "pool_cap": CAP,
        "aggregation": "micro, ratio of sums",
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "by_budget": rows,
        "against_the_deployed_budget_at_k15": contrasts,
        "budgets_that_beat_the_deployed_one": better,
        "reading": (
            "The curve is what the budget buys and the cost column is what it costs. A budget that "
            "beats the deployed one with the interval excluding zero would say the deployed value "
            "is not on the frontier; the list above is that answer, computed on validation so that "
            "choosing from it is not selection on the reported population."),
    }
    (ROOT / "results/budget_curve.json").write_text(json.dumps(report, indent=1))

    print(f"{len(subs)} validation substrates, {int(U.sum())} references, "
          f"budgets {sorted(orders)}")
    for budget, n in sorted(partial.items()):
        print(f"  skipped budget {budget}: {n} of {full} substrates built, still incomplete")
    print()
    print("budget  cand mean   r@5    r@15   r@30   vs deployed at 15")
    for budget in sorted(rows):
        row = rows[budget]
        c = contrasts.get(str(budget))
        tail = "" if c is None else (f"  {c['gap_at_15']:+.4f} "
                                     f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
                                     f"{'  separates' if c['excludes_zero'] else ''}")
        mark = "  <- deployed" if budget == DEPLOYED else ""
        print(f"{budget:6d}  {row['mean_candidates']:9.1f}  "
              f"{row['recall_micro']['5']:.4f} {row['recall_micro']['15']:.4f} "
              f"{row['recall_micro']['30']:.4f}{tail}{mark}")
    print(f"\nbudgets beating the deployed one at 15: {better or 'none'}")
    print("wrote results/budget_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

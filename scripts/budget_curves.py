#!/usr/bin/env python3
"""Does the output budget reorder the leaderboard, without any budget being selected?

An earlier version of this measurement chose the best constant k per method and reported F1 there.
That invites the obvious objection: a k chosen against the same numbers it is reported on is
optimistically biased, and a reordering could be an artifact of the choice. Cross-fitting the choice
answers part of it, but the whole procedure still lives on the test split, which is not the
discipline this paper argues for.

So no k is selected here. The F1-versus-k curve is reported for each method, the region where the
ordering changes is read off the curves, and the certified comparison is made at a single round k
fixed in advance rather than chosen from the data. If the ordering depends on the budget, that is a
property of the curves, and no selection enters.

The mechanism is worth reading off the same table: a method's curve goes flat once k passes the
number of candidates it emits, so a large budget compares a method that has stopped against one that
has not.

Two families are recorded at every k rather than only at the fixed one, because the appendix reads
numbers off both and an earlier version of this script produced neither. `paired_by_k` is the same
paired bootstrap as `paired_at_k_fixed`, over the same resample indices, evaluated at each k;
`truncated_fraction_by_k` is the fraction of substrates on which the cut actually binds, which is
what "budget-matched" and "cut on almost every substrate" refer to. Reporting an interval at one k
while the prose quotes intervals at several was a provenance hole, not a difference of opinion: no
committed artifact held them, and a reader could not have checked them. `paired_at_k_fixed` is kept
and still reproduces bit for bit, since selection remains confined to the one k fixed in advance --
the per-k intervals are descriptive, and the prose says which is which.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODE = "inchikey_tautomer"
KMAX = 32
K_FIXED = 5          # fixed in advance, not chosen from these curves
N_BOOT, SEED = 10000, 0
OUT = ROOT / "results" / "budget_curves.json"


def keyed(items, table):
    seen, out = set(), []
    for s in items:
        k = table.get(s)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def main() -> int:
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    table = json.loads((ROOT / "results" / "key_tables" / f"{MODE}.json").read_text())
    grail = {r["sub"]: r["deployed_top15"]
             for r in json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]}
    methods = {
        "GRAIL": grail,
        "SyGMa": json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text()),
        "MetaPredictor": json.loads((ROOT / "artifacts" / "tier2_1170" / "metapredictor_preds.json").read_text()),
    }
    subs = [s for s in sorted(set.intersection(*(set(m) for m in methods.values())) & set(truth)) if truth[s]]

    curves, emitted, sizes_by = {}, {}, {}
    for name, preds in methods.items():
        rows, sizes = [], []
        for s in subs:
            pk = keyed(preds.get(s, []), table)
            real = {k for k in (table.get(x) for x in truth[s]) if k}
            sizes.append(len(pk))
            rows.append([2 * len(set(pk[:k]) & real) / (min(k, len(pk)) + len(real))
                         if (min(k, len(pk)) + len(real)) else 0.0 for k in range(1, KMAX + 1)])
        curves[name] = np.array(rows)
        emitted[name] = float(np.mean(sizes))
        sizes_by[name] = sizes

    names = list(methods)
    means = {n: curves[n].mean(axis=0) for n in names}
    order_at = {k: sorted(names, key=lambda n: -means[n][k - 1]) for k in range(1, KMAX + 1)}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))

    def paired_at(a: str, b: str, k: int) -> dict:
        """Paired bootstrap of the mean F1 difference at budget k, over the shared resample."""
        d = curves[a][:, k - 1] - curves[b][:, k - 1]
        bt = d[idx].mean(axis=1)
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"delta": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    paired = {f"{a}-{b}": paired_at(a, b, K_FIXED) for a, b in pairs}
    paired_by_k = {str(k): {f"{a}-{b}": paired_at(a, b, k) for a, b in pairs}
                   for k in range(1, KMAX + 1)}
    # The cut binds on a substrate when the method offered more candidates than the budget admits.
    truncated = {n: [round(float(np.mean(np.array(sizes_by[n]) > k)), 4) for k in range(1, KMAX + 1)]
                 for n in names}

    distinct = sorted({tuple(v) for v in order_at.values()})
    rep = {
        "mode": MODE, "n_substrates": len(subs), "k_max": KMAX,
        "k_fixed_in_advance": K_FIXED, "n_boot": N_BOOT, "seed": SEED,
        "mean_emitted": {n: round(emitted[n], 2) for n in names},
        "macro_f1_by_k": {n: [round(float(x), 4) for x in means[n]] for n in names},
        "ordering_by_k": {str(k): order_at[k] for k in range(1, KMAX + 1)},
        "distinct_orderings": [list(o) for o in distinct],
        "paired_at_k_fixed": paired,
        "paired_by_k": paired_by_k,
        "truncated_fraction_by_k": truncated,
    }
    OUT.write_text(json.dumps(rep, indent=1))

    print(f"{len(subs)} substrates, criterion {MODE}\n")
    print(f"{'k':>3}  " + "  ".join(f"{n:>13}" for n in names) + "   ordering")
    for k in (1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 32):
        print(f"{k:>3}  " + "  ".join(f"{means[n][k-1]:13.4f}" for n in names)
              + "   " + " > ".join(order_at[k]) + ("   <- the field's budget" if k == 15 else ""))
    print(f"\nmean emitted: " + ", ".join(f"{n} {emitted[n]:.1f}" for n in names))
    print(f"distinct orderings across k: {len(distinct)}")
    print(f"\npaired at k={K_FIXED}, fixed in advance:")
    for k, v in paired.items():
        print(f"  {k:24} {v['delta']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
              f"  {'SIG' if v['excludes_zero'] else 'n.s.'}")
    print(f"\ntruncated fraction (the cut binds), selected k:")
    for k in (1, 2, 5, 8, 15):
        print(f"  k={k:<3} " + ", ".join(f"{n} {100*truncated[n][k-1]:.1f}%" for n in names))
    print(f"\npaired GRAIL-SyGMa across k (descriptive; selection stays at k={K_FIXED}):")
    for k in (1, 2, 3, 5, 8, 12, 15):
        v = paired_by_k[str(k)]["GRAIL-SyGMa"]
        print(f"  k={k:<3} {v['delta']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
              f"  {'SIG' if v['excludes_zero'] else 'n.s.'}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

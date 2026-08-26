"""What a cap on the candidate pool costs in recall.

The filter scores every candidate the bank produces, so the work a substrate causes grows with
its pool. Capping the pool bounds that work; the question this answers is what the cap costs,
and it is answerable on pools already built rather than by running the pipeline again.

The cap is applied the way it would ship: keep the highest-scoring candidates by the generator,
which is the score available before any pair graph is built and therefore before the expensive
part, then rank the survivors by the registered fusion rule. Capping after the filter has run
would bound nothing.
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

from bank_without_selection import _dedup  # noqa: E402

METATOX = ROOT / "results/metatox_smirks_preds.json"
FOUR = ROOT / "results/four_method_291.json"
N_BOOT, SEED = 10000, 0

CAPS = (50, 100, 250, 500, 1000, 2000, 4000, 0)   # 0 means uncapped
BUDGETS = (5, 10, 15, 20, 30, 50)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="results/widepools_implicit/w*.json")
    ap.add_argument("--out", default=str(ROOT / "results/pool_cap_cost.json"))
    args = ap.parse_args()

    pools, refs = {}, {}
    for p in sorted(glob.glob(args.pools)):
        d = json.loads(Path(p).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    subs = sorted(s for s in pools if refs.get(s))
    N = sum(len(refs[s]) for s in subs)

    sizes = sorted(len(pools[s]) for s in subs)
    real = {s: set(refs[s]) for s in subs}
    mtx = json.loads(METATOX.read_text())["predictions"]
    mt = {s: _dedup(mtx.get(s, []), max(BUDGETS)) for s in subs}

    per = {}          # cap -> budget -> per-substrate hit counts, which the bootstrap needs
    table = {}
    for cap in CAPS:
        kept_total, h = 0, {b: [] for b in BUDGETS}
        for s in subs:
            pool = pools[s]
            if cap:
                pool = sorted(pool, key=lambda c: -c["generator"])[:cap]
            kept_total += len(pool)
            ordered = [c["key"] for c in rrf_order(pool)]
            for b in BUDGETS:
                h[b].append(len(set(ordered[:b]) & real[s]))
        per[cap] = {b: np.array(v, dtype=float) for b, v in h.items()}
        table[str(cap)] = {"recall_micro": {str(b): round(float(per[cap][b].sum() / N), 4)
                                            for b in BUDGETS},
                           "mean_pool_after_cap": round(kept_total / len(subs), 1),
                           "substrates_touched": sum(1 for s in subs
                                                     if cap and len(pools[s]) > cap)}

    h_mt = {b: np.array([len(set(mt[s][:b]) & real[s]) for s in subs], dtype=float)
            for b in BUDGETS}
    U = np.array([len(real[s]) for s in subs], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def contrast(a, b_arr):
        d = a - b_arr
        bt = d[idx].sum(axis=1) / denom
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(d.sum() / U.sum()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    for cap in CAPS:
        if cap == 0:
            continue
        table[str(cap)]["vs_uncapped"] = {
            str(b): contrast(per[cap][b], per[0][b]) for b in BUDGETS}
        table[str(cap)]["vs_metatox"] = {
            str(b): contrast(per[cap][b], h_mt[b]) for b in BUDGETS}
    table["0"]["vs_metatox"] = {str(b): contrast(per[0][b], h_mt[b]) for b in BUDGETS}

    four = json.loads(FOUR.read_text())["per_method"]["MetaTox"]["recall"]
    # compare at the precision the committed artifact records, not below it: a gate that
    # fires on the fifth decimal of a number stored to four decimals reports nothing.
    mism = [f"k={b}: {round(float(h_mt[b].sum() / N), 4)} vs committed {four[str(b)]}"
            for b in BUDGETS if str(b) in four
            and abs(round(float(h_mt[b].sum() / N), 4) - four[str(b)]) > 1e-9]

    base = table["0"]["recall_micro"]
    for cap in CAPS:
        table[str(cap)]["delta_recall@15_vs_uncapped"] = round(
            table[str(cap)]["recall_micro"]["15"] - base["15"], 4)

    rep = {"provenance": stamp(__file__),
           "population": {"n": len(subs), "n_references": N,
                          "source": "the 291 of results/four_method_291.json"},
           "aggregation": "micro, ratio of sums",
           "cap_rule": "keep the top-cap candidates by generator score, then rank by the "
                       "registered fusion rule; 0 is uncapped",
           "pool_sizes": {"mean": round(sum(sizes) / len(sizes), 1),
                          "median": sizes[len(sizes) // 2],
                          "p90": sizes[int(0.90 * len(sizes))],
                          "p99": sizes[int(0.99 * len(sizes))], "max": sizes[-1]},
           "gate": {"reproduces_four_method_291_metatox": not mism, "mismatches": mism},
           "status": "UPPER BOUND. Eight caps were computed on the population this is reported "
                     "on and the best was read off, which is an argmax over the reported set. "
                     "The cap must be fixed in advance and checked where it was not chosen "
                     "before any of these numbers is quoted, on the same terms as H7.",
           "n_boot": N_BOOT, "seed": SEED,
           "by_cap": table}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"pool sizes: mean {rep['pool_sizes']['mean']}  median {rep['pool_sizes']['median']}  "
          f"p90 {rep['pool_sizes']['p90']}  p99 {rep['pool_sizes']['p99']}  "
          f"max {rep['pool_sizes']['max']}")
    print(f"\n{'cap':>7}{'mean pool':>11}{'substrates cut':>16}"
          + "".join(f"{'r@'+str(b):>9}" for b in BUDGETS) + f"{'d@15':>9}")
    for cap in CAPS:
        t = table[str(cap)]
        name = "none" if cap == 0 else str(cap)
        print(f"{name:>7}{t['mean_pool_after_cap']:>11.1f}{t['substrates_touched']:>16}"
              + "".join(f"{t['recall_micro'][str(b)]:>9.4f}" for b in BUDGETS)
              + f"{t['delta_recall@15_vs_uncapped']:>+9.4f}")
    print(f"\ngate reproduces the committed MetaTox column: {not mism}")
    print("\nk=15, paired bootstrap:")
    for cap in CAPS:
        t = table[str(cap)]
        u = t.get("vs_uncapped", {}).get("15")
        m = t["vs_metatox"]["15"]
        name = "none" if cap == 0 else str(cap)
        us = (f"vs uncapped {u['gap']:+.4f} [{u['ci95'][0]:+.4f},{u['ci95'][1]:+.4f}]"
              f"{'*' if u['excludes_zero'] else ' '}" if u else " " * 38)
        print(f"  cap {name:>5}  {us}   vs metatox {m['gap']:+.4f} "
              f"[{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]{'*' if m['excludes_zero'] else ' '}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

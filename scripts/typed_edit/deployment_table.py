"""The comparison as it would ship: both rule budgets, the H9 cap, the H7 fusion, MetaTox.

Every arm here is a configuration someone could deploy, which the whole-bank selector-free pool
was not: it exists to measure a ceiling. The two GRAIL arms differ only in how many of the 7,581
templates were applied, both carry the cap of 100 candidates H9 registers and the rank fusion H7
registers, and MetaTox is read from its own predictions with no cap of ours imposed on it.

Budgets a pool cannot fill are counted rather than averaged over. Thirty templates yield about
seventeen candidates, so above a budget of fifteen that arm is returning a short list, and a
recall gap against it there is an empty list on one side rather than a difference in ordering.
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

CAP = 100
KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
N_BOOT, SEED = 10000, 0
METATOX = ROOT / "results/metatox_smirks_preds.json"
FOUR = ROOT / "results/four_method_291.json"


def load(spec):
    pools, refs, tk = {}, {}, set()
    for f in sorted(glob.glob(spec)) or [spec]:
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
        if d.get("top_k"):
            tk.add(d["top_k"])
    return pools, refs, (sorted(tk) or [None])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whole-bank", default="results/widepools_implicit/w*.json")
    ap.add_argument("--trained", default=str(ROOT / "results/widepools_k30/all.json"))
    ap.add_argument("--out", default=str(ROOT / "results/deployment_table.json"))
    args = ap.parse_args()

    from bank_without_selection import _dedup

    big, refs_b, tk_b = load(args.whole_bank)
    small, refs_s, tk_s = load(args.trained)
    refs = {**refs_b, **refs_s}
    subs = sorted(s for s in set(big) & set(small) if refs.get(s))
    real = {s: set(refs[s]) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)
    N = float(U.sum())
    print(f"{len(subs)} substrates, {N:.0f} references "
          f"(whole bank top_k={tk_b}, trained top_k={tk_s})", file=sys.stderr)

    def ranked(pool):
        keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
        return [c["key"] for c in rrf_order(keep)]

    mtx = json.loads(METATOX.read_text())["predictions"]
    arms = {"whole bank": {s: ranked(big[s]) for s in subs},
            "trained budget": {s: ranked(small[s]) for s in subs},
            "metatox": {s: _dedup(mtx.get(s, []), max(KS)) for s in subs}}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def hits(arm, k):
        return np.array([len(set(arms[arm][s][:k]) & real[s]) for s in subs], dtype=float)

    def contrast(a, b):
        d = a - b
        bt = d[idx].sum(axis=1) / denom
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(d.sum() / U.sum()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    table, contrasts, exhausted = {}, {}, {}
    for k in KS:
        h = {a: hits(a, k) for a in arms}
        table[str(k)] = {a: round(float(v.sum() / N), 4) for a, v in h.items()}
        contrasts[str(k)] = {
            "whole bank - metatox": contrast(h["whole bank"], h["metatox"]),
            "trained budget - metatox": contrast(h["trained budget"], h["metatox"]),
            "trained budget - whole bank": contrast(h["trained budget"], h["whole bank"])}
        exhausted[str(k)] = {a: int(sum(1 for s in subs if len(arms[a][s]) < k)) for a in arms}

    four = json.loads(FOUR.read_text())["per_method"]["MetaTox"]["recall"]
    mism = [f"k={k}: {table[str(k)]['metatox']} vs committed {four[str(k)]}"
            for k in KS if str(k) in four
            and abs(table[str(k)]["metatox"] - four[str(k)]) > 1e-9]

    mean_pool = {a: round(float(np.mean([len(arms[a][s]) for s in subs])), 1) for a in arms}
    rep = {"provenance": stamp(__file__),
           "population": {"n": len(subs), "n_references": N,
                          "source": "the 291 of results/four_method_291.json"},
           "aggregation": "micro, ratio of sums",
           "configuration": {"cap": CAP, "fusion": "H7 reciprocal rank fusion, k=60",
                             "top_k": {"whole bank": tk_b, "trained budget": tk_s}},
           "gate": {"reproduces_four_method_291_metatox": not mism, "mismatches": mism},
           "mean_output_length": mean_pool,
           "recall_micro": table, "contrasts": contrasts,
           "substrates_whose_list_is_shorter_than_the_budget": exhausted,
           "status": "H10 was registered and checked on validation; this population is the "
                     "deployment report, not a second test of it.",
           "n_boot": N_BOOT, "seed": SEED}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    order = ["whole bank", "trained budget", "metatox"]
    print(f"\nmean list length: " + "  ".join(f"{a} {mean_pool[a]}" for a in order))
    print(f"\n{'k':>4}" + "".join(f"{a:>16}" for a in order)
          + f"{'trained-mtx':>14}{'bank-mtx':>12}")
    for k in KS:
        t, c = table[str(k)], contrasts[str(k)]
        tm, bm = c["trained budget - metatox"], c["whole bank - metatox"]
        print(f"{k:>4}" + "".join(f"{t[a]:>16.4f}" for a in order)
              + f"{tm['gap']:>+13.4f}{'*' if tm['excludes_zero'] else ' '}"
              + f"{bm['gap']:>+11.4f}{'*' if bm['excludes_zero'] else ' '}")
    print(f"\ngate reproduces the committed MetaTox column: {not mism}")
    print("\nsubstrates whose list is shorter than the budget:")
    for k in KS:
        e = exhausted[str(k)]
        print(f"  k={k:<3}" + "  ".join(f"{a} {e[a]:>3}" for a in order) + f"  of {len(subs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

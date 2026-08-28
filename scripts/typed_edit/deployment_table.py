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

# results/four_method_291.json is the artifact that DEFINES this population, and it carries four
# methods. Reporting only MetaTox from it was a defect: SyGMa and MetaPredictor have predictions
# on the identical substrates and beat both GRAIL arms at the tight budgets this comparison is
# built on. Every comparator that artifact names is read here.
COMPARATORS = {
    "metatox": ("results/metatox_smirks_preds.json", "predictions"),
    "sygma": ("results/sygma_fulltest_predictions.json", None),
    "metapredictor": ("artifacts/tier2_1170/metapredictor_preds.json", None),
}


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

    # The artifact that defines this population drops a prediction equal to the substrate before
    # the budget bites: returning the parent is not a prediction, though it does consume a slot.
    # That is a declared convention and it has to be the same for every arm, ours included, or
    # the arms are not measured on one axis. Without it MetaPredictor's recall here missed the
    # committed column by two references at three budgets, which the gate caught.
    from bank_without_selection import _key as _tautkey
    parent = {s: _tautkey(s) for s in subs}

    def drop_parent(keys, s):
        return [k for k in keys if k and k != parent[s]]

    def ranked(pool, s):
        keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
        return drop_parent([c["key"] for c in rrf_order(keep)], s)

    arms = {"whole bank": {s: ranked(big[s], s) for s in subs},
            "trained budget": {s: ranked(small[s], s) for s in subs}}
    absent = []
    for name, (rel, key) in COMPARATORS.items():
        path = ROOT / rel
        if not path.exists():
            absent.append(name)
            continue
        blob = json.loads(path.read_text())
        preds = blob[key] if key else blob
        arms[name] = {s: drop_parent(_dedup(preds.get(s, []), max(KS) + 5), s)[:max(KS)]
                      for s in subs}
    ours = ("whole bank", "trained budget")
    others = [a for a in arms if a not in ours]

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
        row = {"trained budget - whole bank": contrast(h["trained budget"], h["whole bank"])}
        for a in ours:
            for b in others:
                cov = sum(1 for s in subs if arms[b][s])
                row[f"{a} - {b}"] = (contrast(h[a], h[b]) if cov else
                                     {"unavailable": f"{b} covers no substrate here"})
        contrasts[str(k)] = row
        exhausted[str(k)] = {a: int(sum(1 for s in subs if len(arms[a][s]) < k)) for a in arms}

    # the gate now covers every comparator the defining artifact records, not just one
    committed = json.loads(FOUR.read_text())["per_method"]
    names = {"metatox": "MetaTox", "sygma": "SyGMa", "metapredictor": "MetaPredictor"}
    mism = []
    for arm, label in names.items():
        if arm not in table["1"] or label not in committed:
            continue
        r = committed[label]["recall"]
        mism += [f"{label} k={k}: {table[str(k)][arm]} vs committed {r[str(k)]}"
                 for k in KS if str(k) in r and abs(table[str(k)][arm] - r[str(k)]) > 1e-9]

    mean_pool = {a: round(float(np.mean([len(arms[a][s]) for s in subs])), 1) for a in arms}
    rep = {"provenance": stamp(__file__),
           "population": {"n": len(subs), "n_references": N,
                          "source": "the 291 of results/four_method_291.json"},
           "aggregation": "micro, ratio of sums",
           "convention": "a prediction whose key equals the substrate's is dropped before the "
                         "budget, for every arm alike; results/four_method_291.json, which "
                         "defines this population, does the same",
           "configuration": {"cap": CAP, "fusion": "H7 reciprocal rank fusion, k=60",
                             "top_k": {"whole bank": tk_b, "trained budget": tk_s}},
           "gate": {"reproduces_four_method_291_metatox": not mism, "mismatches": mism},
           "mean_output_length": mean_pool,
           "comparators_absent": absent,
           "recall_micro": table, "contrasts": contrasts,
           "substrates_whose_list_is_shorter_than_the_budget": exhausted,
           "status": "H10 was registered and checked on validation; this population is the "
                     "deployment report, not a second test of it.",
           "n_boot": N_BOOT, "seed": SEED}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    order = list(ours) + others
    print(f"\nmean list length: " + "  ".join(f"{a} {mean_pool[a]}" for a in order))
    print(f"\n{'k':>4}" + "".join(f"{a:>16}" for a in order))
    for k in KS:
        t, c = table[str(k)], contrasts[str(k)]
        print(f"{k:>4}" + "".join(f"{t[a]:>16.4f}" for a in order))
    print(f"\ngate reproduces the committed MetaTox column: {not mism}")
    print("\nsubstrates whose list is shorter than the budget:")
    for k in KS:
        e = exhausted[str(k)]
        print(f"  k={k:<3}" + "  ".join(f"{a} {e[a]:>3}" for a in order) + f"  of {len(subs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

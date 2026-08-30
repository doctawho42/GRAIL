#!/usr/bin/env python3
"""The comparison again with every arm cut to the same list length, substrate by substrate.

The wide-budget leads in this work are read at a nominal budget, and a nominal budget is not a
length: at fifty, most comparator lists have already ended while the exhaustive arm is still at its
cap. Declaring the budget therefore does not control what the budget was introduced to control,
and the paper says as much in one sentence and leaves it there.

This closes it. For each substrate and each comparator, both arms are truncated to the number of
candidates that comparator actually returned for that substrate, so the two are read over the same
number of slots on every substrate rather than on average. A lead that survives is a lead in
ordering; one that does not was a lead in length.

The control is deliberately hostile to this work: where a comparator returns two candidates, the
exhaustive arm is allowed two. It is also the only comparison here in which the budget is a
property of the substrate rather than of the experiment, so the budget axis does not appear in the
output and cannot.

    python scripts/typed_edit/matched_length.py
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

CAP = 100
N_BOOT, SEED = 10000, 0
COMPARATORS = {
    "metatox": ("results/metatox_smirks_preds.json", "predictions"),
    "sygma": ("results/sygma_fulltest_predictions.json", None),
    "metapredictor": ("artifacts/tier2_1170/metapredictor_preds.json", None),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whole-bank", default="results/widepools_implicit/w*.json")
    ap.add_argument("--trained", default="results/widepools_k30/all.json")
    ap.add_argument("--out", default=str(ROOT / "results" / "matched_length.json"))
    args = ap.parse_args()

    from _rrf import rrf_order
    from bank_without_selection import _dedup, _key as tautkey

    def load(spec):
        pools, refs = {}, {}
        for f in sorted(glob.glob(str(ROOT / spec))) or [str(ROOT / spec)]:
            blob = json.loads(Path(f).read_text())
            pools.update(blob["pools"]); refs.update(blob["references"])
        return pools, refs

    big, refs_b = load(args.whole_bank)
    small, refs_s = load(args.trained)
    refs = {**refs_b, **refs_s}
    subs = sorted(s for s in set(big) & set(small) if refs.get(s))
    real = {s: set(refs[s]) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)
    parent = {s: tautkey(s) for s in subs}

    def drop_parent(keys, s):
        return [k for k in keys if k and k != parent[s]]

    def ranked(pool, s):
        keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
        return drop_parent([c["key"] for c in rrf_order(keep)], s)

    ours = {"whole bank": {s: ranked(big[s], s) for s in subs},
            "trained budget": {s: ranked(small[s], s) for s in subs}}
    theirs = {}
    for name, (rel, key) in COMPARATORS.items():
        path = ROOT / rel
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        preds = blob[key] if key else blob
        theirs[name] = {s: drop_parent(_dedup(preds.get(s, []), CAP + 5), s) for s in subs}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    rows = {}
    for cname, clists in theirs.items():
        # The length the comparator itself chose on each substrate. A substrate where it returns
        # nothing gives both arms nothing and contributes only to the denominator, which is the
        # honest treatment: neither arm was allowed a slot there.
        lengths = np.array([len(clists[s]) for s in subs], dtype=int)
        their_hits = np.array([len(set(clists[s]) & real[s]) for s in subs], dtype=float)
        for oname, olists in ours.items():
            our_hits = np.array([len(set(olists[s][:len(clists[s])]) & real[s]) for s in subs],
                                dtype=float)
            d = our_hits - their_hits
            bt = d[idx].sum(axis=1) / denom
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            rows[f"{oname} - {cname}"] = {
                "recall_ours": round(float(our_hits.sum() / U.sum()), 4),
                "recall_theirs": round(float(their_hits.sum() / U.sum()), 4),
                "gap": round(float(d.sum() / U.sum()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "mean_slots": round(float(lengths.mean()), 2),
                "median_slots": int(np.median(lengths)),
                "substrates_where_the_comparator_returned_nothing": int((lengths == 0).sum()),
            }

    report = {
        "provenance": stamp(__file__),
        "population": {"n_substrates": len(subs), "n_references": int(U.sum()),
                       "note": "the comparison set, as everywhere else"},
        "design": ("for each substrate, both arms are truncated to the number of candidates the "
                   "comparator returned on that substrate, so the two are read over the same "
                   "number of slots on every substrate and not on average"),
        "aggregation": "micro, ratio of sums",
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "contrasts": rows,
        "reading": (
            "A nominal budget is not a length. This asks the question the budget axis was "
            "introduced to ask, with the length taken from the comparator rather than from the "
            "experiment: what remains is ordering, and what disappears was list length."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"{len(subs)} substrates, {int(U.sum())} references\n")
    print(f"{'contrast':38s} {'slots':>7s} {'ours':>7s} {'theirs':>7s} {'gap':>9s}")
    for name, row in rows.items():
        star = " separates" if row["excludes_zero"] else ""
        print(f"{name:38s} {row['mean_slots']:7.1f} {row['recall_ours']:7.4f} "
              f"{row['recall_theirs']:7.4f} {row['gap']:+9.4f}"
              f" [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]{star}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

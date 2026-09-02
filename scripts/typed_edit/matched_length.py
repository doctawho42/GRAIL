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

from _provenance import record_inputs, stamp  # noqa: E402

CAP = 100
N_BOOT, SEED = 10000, 0
COMPARATORS = {
    "metatox": ("results/metatox_smirks_preds.json", "predictions"),
    "sygma": ("results/sygma_fulltest_predictions.json", None),
    "metapredictor": ("artifacts/tier2_1170/metapredictor_preds.json", None),
    # The fourth arrived after this control was written, and its matched-length figure lived in
    # its own section while the table captioned as every arm against every comparator carried
    # three. It is the arm the control matters most for: what separates it at a wide budget is
    # how much each side emits, which is the one thing matching lengths removes.
    "biotransformer": ("results/biotransformer_allhuman_one_step_preds.json", None),
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
    theirs, uncapped = {}, {}
    for name, (rel, key) in COMPARATORS.items():
        path = ROOT / rel
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        preds = blob[key] if key else blob
        # Two readings of "the length the comparator chose". The capped one truncates the
        # comparator's list at CAP + 5 before the slot count is taken, which matters because a
        # comparator that emits more candidates than this work's pool cap allows would otherwise
        # be given slots this work could never fill: SyGMa's uncapped mean is a fifth longer than
        # its capped one. The capped reading is the one the control is run at, because beyond the
        # cap the comparison measures the pool cap and not the ordering; the uncapped reading is
        # run beside it so that choice is priced instead of being a number nobody stated.
        theirs[name] = {s: drop_parent(_dedup(preds.get(s, []), CAP + 5), s) for s in subs}
        uncapped[name] = {s: drop_parent(_dedup(preds.get(s, []), 10 ** 6), s) for s in subs}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def contrast(clists, oname, olists):
        """One arm against one comparator over the comparator's own per-substrate length."""
        lengths = np.array([len(clists[s]) for s in subs], dtype=int)
        their_hits = np.array([len(set(clists[s]) & real[s]) for s in subs], dtype=float)
        our_hits = np.array([len(set(olists[s][:len(clists[s])]) & real[s]) for s in subs],
                            dtype=float)
        d = our_hits - their_hits
        bt = d[idx].sum(axis=1) / denom
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(d.sum() / U.sum()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "mean_slots": round(float(lengths.mean()), 2)}

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
            # The same contrast over the comparator's list with no cap on it, so the cap the
            # control applies is a measured choice rather than an undeclared one.
            if cname in uncapped:
                ulen = np.array([len(uncapped[cname][s]) for s in subs], dtype=int)
                cell = contrast(uncapped[cname], oname, olists)
                # What the cut actually costs the comparator: annotated metabolites it placed
                # past the cut and that the capped reading therefore never counted. The claim
                # that the cut is free rests on this being zero, so it is stored rather than
                # inferred from two gaps that happen to agree.
                past = sum(len(set(uncapped[cname][s][len(clists[s]):]) & real[s]) for s in subs)
                rows[f"{oname} - {cname}"]["without_the_cap_on_the_comparator"] = {
                    **cell,
                    "mean_slots_uncapped": round(float(ulen.mean()), 2),
                    "substrates_the_cap_binds_on": int((ulen > CAP + 5).sum()),
                    "annotated_metabolites_the_comparator_placed_past_the_cut": int(past),
                    "verdict_unchanged": bool(
                        cell["excludes_zero"]
                        == rows[f"{oname} - {cname}"]["excludes_zero"]
                        and cell["gap"] * rows[f"{oname} - {cname}"]["gap"] > 0)}

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([ROOT / rel for rel, _ in COMPARATORS.values()]),
        "population": {"n_substrates": len(subs), "n_references": int(U.sum()),
                       "note": "the comparison set, as everywhere else"},
        "design": ("for each substrate, both arms are truncated to the number of candidates the "
                   "comparator returned on that substrate, after deduplication, after dropping a "
                   "prediction equal to the substrate, and after truncating the comparator's own "
                   "list at " + str(CAP + 5) + "; so the two are read over the same number of "
                   "slots on every substrate and not on average"),
        "cap_on_the_comparator_list": CAP + 5,
        "why_the_cap": ("this work's pool is capped at " + str(CAP) + ", so slots beyond that "
                        "could not be filled by either of its arms and the comparison there "
                        "would measure the pool cap rather than the ordering; the contrast "
                        "without the cap is reported for every pair so the choice is priced"),
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

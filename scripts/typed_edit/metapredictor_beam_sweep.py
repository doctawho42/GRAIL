#!/usr/bin/env python3
"""MetaPredictor's beam swept upward, which is the direction that bears on the leads over it.

This work asks other papers to declare the setting that decides how many candidates a system may
emit, and sweeps SyGMa's scenario because that knob is one the authors can turn. MetaPredictor's
beam is the same kind of knob and equally ours: it is a local checkout with its own environment,
run here at the setting its own repository ships. Sweeping one comparator's emission and not the
other's is an asymmetry a reader is entitled to name, and the direction that matters is upward,
because the leads this work claims over MetaPredictor are at budgets where its list has already
ended on every substrate.

The wide decode is the same models, the same pipeline and the same seed; only the beam and the
number of hypotheses kept differ, so the two candidate sets nest and the comparison is of one
system at two settings rather than of two systems.

    python scripts/typed_edit/metapredictor_beam_sweep.py
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

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
CAP = 100
N_BOOT, SEED = 10000, 0
DEPLOYED = ROOT / "artifacts/tier2_1170/metapredictor_preds.json"
WIDE = ROOT / "results/metapredictor_wide_beam_preds.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "metapredictor_beam_sweep.json"))
    args = ap.parse_args()

    if not WIDE.exists():
        print(f"the wide decode is not on disk yet: {WIDE}", file=sys.stderr)
        return 1

    from _rrf import rrf_order
    from bank_without_selection import _dedup, _key as tautkey

    pools, refs = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"]); refs.update(blob["references"])
    subs = sorted(s for s in pools if refs.get(s))
    real = {s: set(refs[s]) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)
    parent = {s: tautkey(s) for s in subs}

    def drop_parent(keys, s):
        return [k for k in keys if k and k != parent[s]]

    ours = {s: drop_parent([c["key"] for c in rrf_order(
        sorted(pools[s], key=lambda c: -c["generator"])[:CAP])], s) for s in subs}

    deployed_raw = json.loads(DEPLOYED.read_text())
    wide_raw = json.loads(WIDE.read_text())
    # A substrate the wide decode did not produce keeps the deployed list, so the wide arm is
    # never worse for want of a prediction and the count is reported rather than hidden.
    missing = [s for s in subs if s not in wide_raw]
    arms = {
        "deployed beam": {s: drop_parent(_dedup(deployed_raw.get(s, []), 10 ** 6), s)
                          for s in subs},
        "wide beam": {s: drop_parent(_dedup(wide_raw.get(s, deployed_raw.get(s, [])), 10 ** 6), s)
                      for s in subs},
    }

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def hits(order, k):
        return np.array([len(set(order[s][:k]) & real[s]) for s in subs], dtype=float)

    rows = {}
    for name, arm in arms.items():
        row = {"mean_emitted": round(float(np.mean([len(arm[s]) for s in subs])), 1),
               "recall": {str(k): round(float(hits(arm, k).sum() / U.sum()), 4) for k in KS}}
        for k in (15, 30, 50):
            d = hits(ours, k) - hits(arm, k)
            bt = d[idx].sum(axis=1) / denom
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            row[f"exhaustive_minus_metapredictor_at_{k}"] = {
                "gap": round(float(d.sum() / U.sum()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
        rows[name] = row

    survives = [name for name, r in rows.items()
                if r["exhaustive_minus_metapredictor_at_30"]["excludes_zero"]
                and r["exhaustive_minus_metapredictor_at_30"]["gap"] > 0]

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([DEPLOYED, WIDE]),
        "question": ("whether the lead over MetaPredictor at wide budgets is a property of the "
                     "two systems or of the beam the comparator was run at"),
        "population": {"n_substrates": len(subs), "n_references": int(U.sum())},
        "settings": {"deployed beam": "n_best 8 / beam 8, then 2 / 8, sixteen candidates",
                     "wide beam": "n_best 16 / beam 16, then 4 / 16, sixty-four candidates"},
        "substrates_the_wide_decode_did_not_produce": len(missing),
        "criterion": "tautomer-aware InChIKey, as everywhere else",
        "convention": "a prediction equal to the substrate is dropped, for both arms",
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "by_setting": rows,
        "settings_at_which_the_lead_at_thirty_still_separates": survives,
        "reading": (
            "The beam is MetaPredictor's emission knob and this work asks other papers to declare "
            "such a knob. Turning it upward is the test of whether a lead read at a budget its "
            "deployed list never reaches is a statement about the systems or about the setting."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\n{'setting':16s} {'emitted':>8s} {'r@15':>7s} {'r@30':>7s} {'r@50':>7s}"
          f"   exhaustive minus MetaPredictor at 30")
    for name, row in rows.items():
        c = row["exhaustive_minus_metapredictor_at_30"]
        print(f"{name:16s} {row['mean_emitted']:8.1f} {row['recall']['15']:7.4f} "
              f"{row['recall']['30']:7.4f} {row['recall']['50']:7.4f}"
              f"   {c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"{'  separates' if c['excludes_zero'] else ''}")
    if missing:
        print(f"\n{len(missing)} substrates kept the deployed decode's list")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

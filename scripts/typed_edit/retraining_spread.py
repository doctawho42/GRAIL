#!/usr/bin/env python3
"""What retraining moves, at the configuration this paper reports and on its own population.

Every number in the manuscript comes from one trained checkpoint, and the paired intervals
quantify the substrate population rather than the training. The spread the manuscript offers for
scale was measured on a run at an earlier operating point, so its level is not comparable with
the tables and the margins are being read against a ruler from a different system.

Three seeds exist at the deployed label presentation and the deployed bank. This scores their
pools exactly as the comparison scores the released one: the same ranking, the same pool cap, the
same parent-drop rule, the same tautomer-aware key and the same population. What comes out is a
spread the reported margins can be read against, and the released checkpoint's own position
inside it.

It is a spread over training seeds at one operating point. It is not the spread of a full-split
run, which no checkpoint here provides, and the report says so rather than letting the number
stand for more than it is.

    python scripts/typed_edit/retraining_spread.py
    python scripts/typed_edit/retraining_spread.py --arm exhaustive
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
CAP = 100
# The arm each seed pool belongs to, and the released column it is read against.
ARMS = {"interactive": ("results/seedpools/interactive_seed*.json", "trained budget"),
        "exhaustive": ("results/seedpools/exhaustive_seed*.json", "whole bank")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=tuple(ARMS) + ("both",), default="both")
    ap.add_argument("--out", default=str(ROOT / "results" / "retraining_spread.json"))
    args = ap.parse_args()

    from _rrf import rrf_order
    from bank_without_selection import _key as tautkey

    deployed = json.loads((ROOT / "results/deployment_table.json").read_text())
    released = deployed["recall_micro"]

    rows, inputs = {}, []
    for arm in [a for a in ARMS if args.arm in (a, "both")]:
        pattern, column = ARMS[arm]
        files = sorted(glob.glob(str(ROOT / pattern)))
        if not files:
            print(f"{arm}: no seed pools under {pattern}", file=sys.stderr)
            continue
        inputs.extend(Path(f) for f in files)
        per_seed = {}
        for path in files:
            blob = json.loads(Path(path).read_text())
            pools, refs = blob["pools"], blob["references"]
            subs = sorted(s for s in pools if refs.get(s))
            real = {s: set(refs[s]) for s in subs}
            parent = {s: tautkey(s) for s in subs}
            U = float(sum(len(real[s]) for s in subs))
            order = {s: [k for k in (c["key"] for c in rrf_order(
                sorted(pools[s], key=lambda c: -c["generator"])[:CAP]))
                if k and k != parent[s]] for s in subs}
            per_seed[Path(path).stem] = {
                "n_substrates": len(subs), "n_references": int(U),
                "recall": {str(k): round(sum(len(set(order[s][:k]) & real[s])
                                             for s in subs) / max(U, 1), 4) for k in KS}}
        by_budget = {}
        for k in KS:
            values = [row["recall"][str(k)] for row in per_seed.values()]
            released_here = released.get(str(k), {}).get(column)
            by_budget[str(k)] = {
                "seeds": values,
                "mean": round(st.mean(values), 4),
                "sd": round(st.stdev(values), 4) if len(values) > 1 else None,
                "released": released_here,
                "released_inside_the_seed_range": (
                    None if released_here is None
                    else bool(min(values) <= released_here <= max(values))),
            }
        # The released checkpoint is one of these seeds: artifacts/full5000_implicit and
        # artifacts/multiseed_full5000_implicit_seed0 are byte-identical. Its pool must therefore
        # reproduce the released column exactly, and if it does not, this build differs from the
        # one the comparison was scored on in some way other than the seed, which would make the
        # spread a spread of two things at once.
        agree = None
        seed0 = next((v for k, v in per_seed.items() if k.endswith("seed0")), None)
        if seed0 is not None:
            off = {str(k): round(seed0["recall"][str(k)] - released[str(k)][column], 4)
                   for k in KS if released.get(str(k), {}).get(column) is not None}
            agree = {"budgets_that_differ": {k: v for k, v in off.items() if abs(v) > 5e-5}}
            agree["reproduces_the_released_column"] = not agree["budgets_that_differ"]
        rows[arm] = {"seeds": per_seed, "by_budget": by_budget,
                     "released_column": column,
                     "the_released_checkpoint_is_seed_0": agree}

    # What the spread is for: a margin smaller than it is a margin the training noise could have
    # produced, and the manuscript reads several margins at k = 30.
    against = {}
    if "exhaustive" in rows:
        sd30 = rows["exhaustive"]["by_budget"]["30"]["sd"]
        for name, cell in deployed.get("contrasts", {}).get("30", {}).items():
            if not name.startswith("whole bank") or sd30 in (None, 0):
                continue
            against[name] = {"gap": cell["gap"], "in_units_of_the_seed_sd": round(
                abs(cell["gap"]) / sd30, 1)}

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs(inputs),
        "question": ("how much of a reported margin retraining alone could move, measured at the "
                     "configuration the manuscript reports rather than at an earlier one"),
        "seeds": sorted({Path(f).stem.rsplit("_", 1)[-1] for f in
                         (str(p) for p in inputs)}),
        "population": "the comparison set, as everywhere else",
        "ranking": "reciprocal rank fusion over the pool capped at 100, parent dropped",
        "criterion": "tautomer-aware InChIKey",
        "by_arm": rows,
        "margins_at_thirty_in_units_of_the_seed_spread": against,
        "what_this_is_not": (
            "a spread over training-set size or over the full split: every seed here sees the "
            "same 5,000 training substrates the released run saw, so this measures the seed and "
            "not the data"),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    for arm, row in rows.items():
        print(f"\n{arm}: {len(row['seeds'])} seeds against the released "
              f"{row['released_column']} column")
        print(f"  {'k':>4s} {'mean':>8s} {'sd':>8s} {'released':>9s}  inside the seed range")
        for k in KS:
            c = row["by_budget"][str(k)]
            sd = "n/a" if c["sd"] is None else f"{c['sd']:.4f}"
            rel = "n/a" if c["released"] is None else f"{c['released']:.4f}"
            print(f"  {k:>4d} {c['mean']:8.4f} {sd:>8s} {rel:>9s}  "
                  f"{c['released_inside_the_seed_range']}")
    for name, cell in against.items():
        print(f"  {name} at 30: {cell['gap']:+.4f}, "
              f"{cell['in_units_of_the_seed_sd']} seed standard deviations")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

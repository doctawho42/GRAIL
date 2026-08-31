#!/usr/bin/env python3
"""The drawing's effect on the substrates the drawing actually changes, not diluted by the rest.

The sweep over how the substrate is drawn reports its effect pooled over the whole comparison set,
and finds it small. That is the right number for reading the comparison, and it is the wrong number
for a user: most substrates in that set carry no amide, so their two drawings are the same molecule
and contribute a difference of exactly zero to the pool. Averaging the affected substrates with
them understates what the drawing does where it does anything.

This conditions on the substrates whose stored and standardised forms actually differ. The pools
are the ones the sweep already built, so nothing is re-run; what changes is the population the
paired difference is computed over, and the count of substrates it is computed on is reported
beside it because a conditional effect on a small set is a wide interval.

    python scripts/typed_edit/dialect_conditional.py
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
from dialect_sweep import ARMS  # noqa: E402

KS = (5, 15, 30, 50)
CAP = 100
N_BOOT, SEED = 10000, 0


def load(pattern):
    pools, refs = {}, {}
    for spec in (pattern if isinstance(pattern, list) else [pattern]):
        for f in sorted(glob.glob(str(ROOT / spec))):
            blob = json.loads(Path(f).read_text())
            pools.update(blob["pools"]); refs.update(blob["references"])
    return pools, refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "dialect_conditional.json"))
    args = ap.parse_args()

    from _rrf import rrf_order
    from bank_without_selection import _key as tautkey
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from grail_metabolism.utils.preparation import standardize_mol

    rows = {}
    for name, (stored_pattern, drawn_pattern) in ARMS.items():
        stored, refs = load(stored_pattern)
        drawn, _ = load(drawn_pattern)
        subs = sorted(s for s in set(stored) & set(drawn) if refs.get(s))
        if not subs:
            continue

        # The substrates the drawing changes at all. Where the standardiser returns the stored
        # structure the two arms are the same molecule and can only contribute zero.
        moved = []
        for s in subs:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            try:
                if Chem.MolToSmiles(standardize_mol(mol)) != s:
                    moved.append(s)
            except Exception:
                continue

        parent = {s: tautkey(s) for s in subs}

        def ranked(pool, s):
            keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
            return [k for k in (c["key"] for c in rrf_order(keep))
                    if k and k != parent[s]]

        a = {s: ranked(stored[s], s) for s in subs}
        b = {s: ranked(drawn[s], s) for s in subs}
        real = {s: set(refs[s]) for s in subs}

        entry = {"substrates": len(subs), "substrates_the_drawing_changes": len(moved),
                 "share_changed": round(len(moved) / max(len(subs), 1), 4), "by_budget": {}}
        for population, tag in ((subs, "all"), (moved, "changed_only")):
            if not population:
                continue
            U = np.array([len(real[s]) for s in population], dtype=float)
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, len(population), (N_BOOT, len(population)))
            denom = np.maximum(U[idx].sum(axis=1), 1)
            for k in KS:
                ha = np.array([len(set(b[s][:k]) & real[s]) for s in population], dtype=float)
                hb = np.array([len(set(a[s][:k]) & real[s]) for s in population], dtype=float)
                d = ha - hb
                bt = d[idx].sum(axis=1) / denom
                lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
                entry["by_budget"].setdefault(str(k), {})[tag] = {
                    "difference": round(float(d.sum() / U.sum()), 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "n": len(population)}
        rows[name] = entry

    report = {
        "provenance": stamp(__file__),
        "difference": ("recall with the substrate as the declared standardiser draws it, minus "
                       "recall with it as the corpus stores it"),
        "budgets": list(KS),
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "arms": rows,
        "reading": (
            "The pooled figure is the one that decides how the comparison reads, because the "
            "comparison is over the whole set. The conditional figure is the one a user should "
            "read, because a user submits one substrate and it is either affected or not."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    for name, entry in rows.items():
        print(f"\n{name}: {entry['substrates_the_drawing_changes']} of {entry['substrates']} "
              f"substrates change under the standardiser "
              f"({entry['share_changed']:.1%})")
        print(f"  {'k':>4s}  {'all':>28s}  {'changed only':>28s}")
        for k in KS:
            cells = entry["by_budget"][str(k)]
            def fmt(c):
                return (f"{c['difference']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]"
                        + ("*" if c["excludes_zero"] else " "))
            print(f"  {k:>4d}  {fmt(cells['all']):>28s}  "
                  f"{fmt(cells.get('changed_only', cells['all'])):>28s}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""SyGMa's scenario swept upward, which is the direction that bears on the lead this work claims.

The comparison runs SyGMa at the scenario its own documentation gives, one cycle of each ruleset,
and an earlier measurement swept that downward to the uncomposed case. A referee pointed out that
the direction not swept is the one that matters: a deeper scenario emits more candidates, and the
only wide-budget lead this work still claims after its other corrections is over SyGMa, at budgets
where SyGMa's list has already ended on many substrates.

SyGMa is an installed module, so the sweep costs a run rather than a request to anybody. Each
scenario is applied to the comparison set, scored under the same criterion and the same
parent-drop convention, and compared with this work's exhaustive arm at the budgets the lead is
claimed at.

    python scripts/typed_edit/sygma_scenario_sweep.py
    python scripts/typed_edit/sygma_scenario_sweep.py --substrates 40    # a probe
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
N_BOOT, SEED = 10000, 0
CAP = 100
# (phase-1 cycles, phase-2 cycles). The first is the deployed setting, taken from the version's
# own documentation; the rest emit more.
# Deeper scenarios grow the tree combinatorially, so the sweep is one step up and no
# further, with a per-substrate deadline. What does not finish is reported as such.
SCENARIOS = ((2, 1),)
DEADLINE_S = 90.0
_SC = None
_SPEC: tuple = (1, 1)


def _init(spec):
    global _SPEC
    _SPEC = spec


def _worker(smiles):
    global _SC
    import sygma
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if _SC is None:
        p1, p2 = _SPEC
        _SC = sygma.Scenario([[sygma.ruleset["phase1"], p1], [sygma.ruleset["phase2"], p2]])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, []
    from grail_metabolism.metrics import _tautomer_inchikey as _tk

    try:
        tree = _SC.run(mol)
        tree.calc_scores()
        ranked = [e[0] for e in tree.to_smiles()]
        pk = _tk(smiles) if _tk else None
        keep = [x for x in ranked
                if x != smiles and not (pk is not None and _tk(x) == pk)]
        return smiles, keep
    except Exception:
        return smiles, []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", type=int, default=0, help="0 means the whole comparison set")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "sygma_scenario_sweep.json"))
    args = ap.parse_args()

    from _rrf import rrf_order
    from bank_without_selection import _dedup, _key as tautkey

    pools, refs = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"]); refs.update(blob["references"])
    subs = sorted(s for s in pools if refs.get(s))
    if args.substrates:
        subs = subs[: args.substrates]
    real = {s: set(refs[s]) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)
    parent = {s: tautkey(s) for s in subs}

    def drop_parent(keys, s):
        return [k for k in keys if k and k != parent[s]]

    ours = {s: drop_parent([c["key"] for c in rrf_order(
        sorted(pools[s], key=lambda c: -c["generator"])[:CAP])], s) for s in subs}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    ctx = multiprocessing.get_context("spawn")

    def hits(order, k):
        return np.array([len(set(order[s][:k]) & real[s]) for s in subs], dtype=float)

    # The deployed scenario is already frozen and does not need re-running.
    frozen = json.loads((ROOT / "results/sygma_fulltest_predictions.json").read_text())
    rows = {}
    base_label = f"phase1 x{1}, phase2 x{1}"
    base = {s: drop_parent(_dedup(frozen.get(s, []), 10 ** 6), s) for s in subs}
    rows[base_label] = {
        "mean_emitted": round(float(np.mean([len(base[s]) for s in subs])), 1),
        "recall": {str(k): round(float(hits(base, k).sum() / U.sum()), 4) for k in KS},
        "source": "results/sygma_fulltest_predictions.json, the frozen deployed run",
        "unfinished": 0}
    for k in (30, 50):
        d = hits(ours, k) - hits(base, k)
        bt = d[idx].sum(axis=1) / denom
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        rows[base_label][f"exhaustive_minus_sygma_at_{k}"] = {
            "gap": round(float(d.sum() / U.sum()), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}

    for spec in SCENARIOS:
        label = f"phase1 x{spec[0]}, phase2 x{spec[1]}"
        print(f"\n{label}: {len(subs)} substrates on {workers} workers", flush=True)
        t0, out, unfinished = time.perf_counter(), {}, 0
        # A substrate that exceeds the deadline keeps the deployed scenario's list, so the arm is
        # never worse than the one it is being compared with and the count is reported. The
        # deadline is per substrate: an earlier version used one timeout on the whole iterator, so
        # the first slow substrate abandoned every remaining one and the deeper scenario would
        # have come back looking identical to the deployed one.
        pool = ctx.Pool(workers, initializer=_init, initargs=(spec,))
        try:
            pending = [(s, pool.apply_async(_worker, (s,))) for s in subs]
            for n, (s, handle) in enumerate(pending, 1):
                try:
                    smiles, keep = handle.get(timeout=DEADLINE_S)
                    out[smiles] = drop_parent(_dedup(keep, 10 ** 6), smiles)
                except Exception:
                    unfinished += 1
                if n % 25 == 0 or n == len(subs):
                    print(f"  {n}/{len(subs)} ({time.perf_counter() - t0:.0f}s, "
                          f"{unfinished} over the deadline)", flush=True)
        finally:
            pool.terminate(); pool.join()
        for s in subs:
            out.setdefault(s, base[s])
        emitted = float(np.mean([len(out[s]) for s in subs]))
        row = {"mean_emitted": round(emitted, 1),
               "recall": {str(k): round(float(hits(out, k).sum() / U.sum()), 4) for k in KS},
               "unfinished": unfinished,
               "unfinished_note": ("substrates the deeper scenario did not finish inside the "
                                   "deadline keep the deployed scenario's list, so this arm is "
                                   "never handicapped by the timeout"),
               "seconds": round(time.perf_counter() - t0, 1)}
        for k in (30, 50):
            d = hits(ours, k) - hits(out, k)
            bt = d[idx].sum(axis=1) / denom
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            row[f"exhaustive_minus_sygma_at_{k}"] = {
                "gap": round(float(d.sum() / U.sum()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
        rows[label] = row

    survives = [lab for lab, r in rows.items()
                if r["exhaustive_minus_sygma_at_50"]["excludes_zero"]
                and r["exhaustive_minus_sygma_at_50"]["gap"] > 0]

    report = {
        "provenance": stamp(__file__),
        "population": {"n_substrates": len(subs), "n_references": int(U.sum())},
        "deployed_scenario": base_label,
        "criterion": "tautomer-aware InChIKey, as everywhere else",
        "convention": "a prediction equal to the substrate is dropped, for both arms",
        "by_scenario": rows,
        "scenarios_at_which_the_lead_at_50_still_separates": survives,
        "reading": (
            "The scenario is SyGMa's own emission knob and this work asks other papers to declare "
            "such a knob. Sweeping it upward is the test of whether the remaining wide-budget "
            "lead is a property of the two systems or of the setting one of them was run at."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\n{'scenario':26s} {'emitted':>8s} {'r@15':>7s} {'r@30':>7s} {'r@50':>7s}"
          f"   exhaustive minus SyGMa at 50")
    for label, row in rows.items():
        c = row["exhaustive_minus_sygma_at_50"]
        print(f"{label:26s} {row['mean_emitted']:8.1f} {row['recall']['15']:7.4f} "
              f"{row['recall']['30']:7.4f} {row['recall']['50']:7.4f}"
              f"   {c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"{'  separates' if c['excludes_zero'] else ''}"
              f"{'  <- deployed' if label == report['deployed_scenario'] else ''}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

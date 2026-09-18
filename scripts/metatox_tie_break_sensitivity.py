#!/usr/bin/env python3
"""How much of the MetaTox column's recall is the method, and how much is its supplier's file order.

Writes results/metatox_tie_break_sensitivity.json.

The column is ordered by the method's own Pa for the Metabolite class, descending. That decides
about half of it. PASS writes a spectrum only above its own threshold, so every prediction below it
carries no score at all, and those keep the order the delivery gave them; equal Pa values leave ties
among the scored ones too. Inside a tie the method has expressed no preference, so any order within
it is an equally faithful reading of what MetaTox said, and the order that survives into the column
is an accident of the supplier's export.

So this permutes, many times, ONLY within the classes the method itself ties, and asks what recall@k
does. A permutation that crossed a real difference in Pa would scramble the method's own ordering
and report a sensitivity the method does not have; that is the one thing this must not do.

WHY IT IS WORTH THE COMPUTE. The axis reads each arm at five budgets and reports a contrast against
this work's two arms. Any contrast whose margin is smaller than the movement this produces is a cell
an arbitrary choice could have decided. Those cells are named in the output rather than left for a
reader to work out, because a sensitivity reported as one number, with no list of what it reaches,
discloses the existence of a problem and not its extent.

The result is not symmetric and that is the point. On the wider half the delivered order is worse
than almost every permutation at every budget, so MetaTox is understated there and every cell in
which this work leads it is flattered by exactly that much.

    python scripts/metatox_tie_break_sensitivity.py [--permutations 60] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

COLUMN = ROOT / "results" / "metatox_smirks_preds_evaluated1170.json"
TRUTH = ROOT / "results" / "test_references.json"
AXIS = ROOT / "results" / "population_definition.json"
OUT = ROOT / "results" / "metatox_tie_break_sensitivity.json"

KS = (5, 10, 15, 30, 50)
HALVES = ("comparison_set", "wider")
# The axis reads the two populations under these names; the column names its halves the other way.
POP_OF_HALF = {"comparison_set": "the comparison set", "wider": "the whole evaluated test set"}


def tie_classes(rows):
    """The prediction list split into runs the method does not distinguish.

    One class per distinct Pa, in the order the column already has, and one class for the whole
    unscored block. Rounding to six places rather than comparing floats exactly: PASS reports three
    decimals, so anything finer is representation noise and splitting on it would understate how
    much of the order is arbitrary.
    """
    out, cur, cur_key = [], [], object()
    for r in rows:
        pa = r[1] if isinstance(r[1], (int, float)) and r[1] == r[1] else None
        key = round(pa, 6) if pa is not None else "unscored"
        if key != cur_key:
            if cur:
                out.append(cur)
            cur, cur_key = [], key
        cur.append(r[0])
    if cur:
        out.append(cur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from grail_metabolism.metrics import _tautomer_inchikey

    col = json.loads(COLUMN.read_text())
    truth = json.loads(TRUTH.read_text())
    det, src = col["predictions_with_scores"], col["source_of_each_substrate"]

    cache: dict = {}

    def key(smiles):
        if smiles not in cache:
            cache[smiles] = _tautomer_inchikey(smiles)
        return cache[smiles]

    # The matching convention the scoring uses, so a structure counted here is a structure the
    # evaluation would count. Anything cheaper would measure a different quantity.
    print("keying the references", flush=True)
    refs = {s: {key(x) for x in v} for s, v in truth.items()}
    print("keying the column", flush=True)
    classes = {s: tie_classes(rows) for s, rows in det.items()}

    n_entries = sum(len(r) for r in det.values())
    n_classes = sum(len(c) for c in classes.values())

    def recall(order_of):
        per = {h: {k: [] for k in KS} for h in HALVES}
        for s, seq in order_of.items():
            half = src.get(s)
            if half not in per or not refs.get(s):
                continue
            keys, seen = [], set()
            for smi in seq:
                kk = key(smi)
                if kk in seen:
                    continue
                seen.add(kk)
                keys.append(kk)
            for k in KS:
                per[half][k].append(len(set(keys[:k]) & refs[s]) / len(refs[s]))
        return {h: {k: statistics.mean(v) for k, v in per[h].items() if v} for h in per}

    delivered_order = {s: [x for c in cs for x in c] for s, cs in classes.items()}
    base = recall(delivered_order)

    rng = random.Random(args.seed)
    dist = {h: {k: [] for k in KS} for h in HALVES}
    for i in range(args.permutations):
        if i % 10 == 0:
            print(f"  permutation {i}", flush=True)
        order = {}
        for s, cs in classes.items():
            seq = []
            for c in cs:
                cc = list(c)
                rng.shuffle(cc)
                seq.extend(cc)
            order[s] = seq
        got = recall(order)
        for h in HALVES:
            for k in KS:
                dist[h][k].append(got[h][k])

    halves = {}
    for h in HALVES:
        halves[h] = {}
        for k in KS:
            d = sorted(dist[h][k])
            b = base[h][k]
            halves[h][str(k)] = {
                "delivered": round(b, 6),
                "permuted_mean": round(statistics.mean(d), 6),
                "permuted_lo": round(d[0], 6),
                "permuted_hi": round(d[-1], 6),
                "shift": round(b - statistics.mean(d), 6),
                "percentile": round(100.0 * sum(1 for x in d if x < b) / len(d), 1),
                "n_substrates": len([s for s in det if src.get(s) == h and refs.get(s)]),
            }

    # The permutations are the expensive part and everything below is cheap, so the expensive part
    # is banked first. An earlier version computed the reach list afterwards, tripped over the shape
    # of the axis, and threw away half an hour of permutations to a TypeError.
    Path(args.out).write_text(json.dumps({"halves": halves, "incomplete": True}, indent=1))

    # Which contrast cells this could have decided. Read from the axis rather than typed, so the
    # list stops being true out loud if the axis is rebuilt and the margins move.
    #
    # Only the two contrast arms are walked. The axis puts `recall` and
    # `substrates_with_no_prediction` beside them under the same comparator, so iterating every key
    # here reaches a dict of floats and an int; the guard below says what a cell is rather than
    # assuming everything in this position is one.
    reach = []
    axis = json.loads(AXIS.read_text()) if AXIS.exists() else {"contrasts": {}}
    ARMS = ("deployed_minus_comparator", "exhaustive_minus_comparator")
    for half, pop_key in POP_OF_HALF.items():
        rows = (axis["contrasts"].get(pop_key, {}).get("metatox") or {})
        for arm in ARMS:
            cells = rows.get(arm) or {}
            for k, c in cells.items():
                if not isinstance(c, dict) or "excludes_zero" not in c:
                    continue
                sh = halves[half].get(str(k))
                if not sh or not c["excludes_zero"]:
                    continue
                # WHAT "WITHIN REACH" HAS TO MEAN. An earlier version compared the shift against
                # the cell's DIFFERENCE, and reported nothing in reach. That is the wrong test: a
                # cell separates because zero lies outside its interval, so what decides whether an
                # arbitrary re-ordering could unmake it is the distance from the interval's near
                # edge to zero, not the distance from the point estimate. At fifteen on the wider
                # population the exhaustive arm leads by 0.0219 with a lower bound of 0.0008 while
                # the shift is 0.0111 -- the wrong test called that safe by a factor of two, and the
                # right one shows the interval would cross zero.
                lo, hi = c["ci95"]
                near = lo if c["difference"] > 0 else -hi
                if near <= abs(sh["shift"]):
                    reach.append({
                        "population": pop_key, "arm": arm, "budget": int(k),
                        "difference": c["difference"], "ci95": c["ci95"],
                        "separates": c["excludes_zero"], "shift": sh["shift"],
                        "margin_to_zero": round(near, 6),
                        "direction": ("the shift runs against this work, so a neutral reading would "
                                      "widen this lead" if sh["shift"] > 0 else
                                      "the shift flatters this work, so a neutral reading would "
                                      "narrow this lead and the interval would cross zero"),
                        "note": ("the interval's near edge is closer to zero than the movement the "
                                 "delivery's own record order produces at this budget, so this "
                                 "cell could have been decided by an order nobody chose"),
                    })
    reach.sort(key=lambda r: r["margin_to_zero"])

    art = {
        "what_this_is": ("how much of the MetaTox column's recall at each budget is the method's "
                         "own ordering and how much is the order its supplier's file happened to "
                         "have"),
        "provenance": stamp(__file__),
        "inputs": record_inputs([COLUMN, TRUTH, AXIS]),
        "permutations": args.permutations,
        "seed": args.seed,
        "what_was_permuted": (
            "only within classes the method itself ties: one class per distinct Pa and one for the "
            "whole unscored block, which PASS leaves unranked because it writes a spectrum only "
            "above its own threshold. Every permutation is therefore an equally faithful reading "
            "of what MetaTox returned; none crosses a difference the method expressed"),
        "matching": ("distinct structures under grail_metabolism.metrics._tautomer_inchikey, the "
                     "convention the scoring matches with"),
        "classes": {"entries": n_entries, "tie_classes": n_classes, "substrates": len(det),
                    "mean_entries_per_class": round(n_entries / n_classes, 3)},
        "halves": halves,
        "cells_within_reach": reach,
        "how_to_read_it": (
            "a negative shift means the delivered order gives LESS recall than a neutral reading of "
            "the same predictions, so MetaTox is understated at that budget and any cell in which "
            "this work leads it is flattered by that much; a positive shift means the reverse"),
    }
    Path(args.out).write_text(json.dumps(art, indent=1))
    print(f"\nwrote {Path(args.out).relative_to(ROOT)}")
    for h in HALVES:
        print(f"  {h}")
        for k in KS:
            r = halves[h][str(k)]
            print(f"    k={k:2d}  delivered {r['delivered']:.4f}  shift {r['shift']:+.4f}  "
                  f"percentile {r['percentile']:5.1f}")
    print(f"  cells within reach of that shift: {len(reach)}")
    for r in reach:
        print(f"    {r['population']} / {r['arm']} / k={r['budget']}: "
              f"{r['difference']:+.4f} against a shift of {r['shift']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Does the emission rule's lead survive the grid, or only the cell it was read in?

The pool-relative rule leads on macro F1 at the published cell: the tautomer-aware criterion,
the comparators at the budgets they emit. A lead in one cell is what this project exists to
distrust, so the same comparison is run in every cell of a declared grid -- five matching
criteria crossed with ten budgets the comparators are read at -- and the verdict is dominance
or it is not a verdict.

The rule is GRAIL's budget, so it is not swept: it emits what its scores say, about two
candidates, and the question is whether that beats each comparator whatever budget the
comparator is read at. The rule capped at k is reported beside it, because at k below its own
output the two differ and a reader should see which is being quoted.

Keys are computed once per substrate, arm and criterion and reused across budgets: truncation
at k is a slice of a rank-ordered key list, so nothing is rescored. The published cell is gated
against results/emission_leaderboard.json before any other cell is read.

One caveat the artifact records rather than hides: BioTransformer returns an unranked set, so
truncating it at k picks by file order and not by confidence. That is the same treatment the
committed budget-matched table gives it, and it flatters neither side consistently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from grail_metabolism.metrics import _match_keys, f1 as f1_of  # noqa: E402
from scripts.run_benchmark import load_test_map  # noqa: E402
from scripts.run_match_sensitivity import _dedup_canon  # noqa: E402

SHARED = ROOT / "artifacts" / "tier2" / "substrates.json"
DUMP = ROOT / "results" / "scored_predictions.json"
COMMITTED = ROOT / "results" / "emission_leaderboard.json"
PREDS = {"BioTransformer": ROOT / "artifacts/tier2/biotransformer_preds.json",
         "MetaPredictor": ROOT / "artifacts/tier2/metapredictor_preds.json",
         "MetaTrans": ROOT / "artifacts/tier2/metatrans_preds.json"}
SYGMA = ROOT / "results" / "sygma_fulltest_predictions.json"
CRITERIA = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
BUDGETS = [1, 3, 5, 8, 10, 15, 20, 32, 50, 74]
PUBLISHED = ("inchikey_tautomer", 15)


def key_list(smiles, match):
    """Rank-ordered keys, one per prediction, as aggregate_prediction_metrics builds them."""
    return [next(iter(_match_keys([s], match))) for s in _dedup_canon(list(smiles))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "results" / "emission_grid.json"))
    args = ap.parse_args()

    shared = json.loads(SHARED.read_text())
    tm = load_test_map(None, 42)
    dump = {r["sub"]: r["candidates"] for r in json.loads(DUMP.read_text())["rows"]}
    subs = sorted(s for s in shared if s in tm and tm[s] and s in dump)

    rule = f"GRAIL, pool-relative alpha={args.alpha}"
    arms = {
        rule: {s: [c["smiles"] for c in dump[s]
                   if dump[s] and c["combined"] >= args.alpha * dump[s][0]["combined"]]
               for s in subs},
        "GRAIL, deployed budget": {s: [c["smiles"] for c in dump[s][:15]] for s in subs},
    }
    for name, path in PREDS.items():
        if path.exists():
            arms[name] = json.loads(path.read_text())
    if SYGMA.exists():
        arms["SyGMa"] = json.loads(SYGMA.read_text())

    grid, mean_output = {}, {}
    for crit in CRITERIA:
        print(f"  keys for {crit} ...", file=sys.stderr, flush=True)
        real = {s: _match_keys(tm[s], crit) for s in subs}
        keys = {name: {s: key_list(p.get(s, []), crit) for s in subs}
                for name, p in arms.items()}
        cell = {}
        for name in arms:
            mean_output[name] = round(
                sum(len(_dedup_canon(list(arms[name].get(s, [])))) for s in subs) / len(subs), 2)
            row = {}
            for k in BUDGETS:
                row[str(k)] = round(
                    sum(f1_of(keys[name][s][:k], real[s]) for s in subs) / len(subs), 4)
            row["uncapped"] = round(
                sum(f1_of(keys[name][s], real[s]) for s in subs) / len(subs), 4)
            cell[name] = row
        grid[crit] = cell

    # The gate: the published cell has to be the committed one. Every arm is compared UNCAPPED,
    # because the committed table scores each method at its own emission -- SyGMa's 74 and
    # BioTransformer's 10.8, not a shared budget. Comparing at k=15 read SyGMa at a budget it
    # never emits at and reported a mismatch of 0.08 that was the gate's error, not the grid's.
    want = json.loads(COMMITTED.read_text())["table"]
    got = grid[PUBLISHED[0]]
    mism = []
    for name, row in want.items():
        if name not in got:
            continue
        ours = got[name]["uncapped"]
        if abs(round(ours, 3) - row["f1"]) > 1e-9:
            mism.append(f"{name}: grid {ours:.4f} vs committed {row['f1']}")

    # dominance: the rule, emitting what its scores say, against each comparator in every cell
    others = [n for n in arms if n != rule]
    verdict = {}
    for name in others:
        cells = [(c, k) for c in CRITERIA for k in BUDGETS
                 if grid[c][rule]["uncapped"] <= grid[c][name][str(k)]]
        verdict[name] = {"cells": len(CRITERIA) * len(BUDGETS),
                         "lost_or_tied": len(cells),
                         "dominates": not cells,
                         "worst_cells": [{"criterion": c, "budget": k,
                                          "rule": grid[c][rule]["uncapped"],
                                          "them": grid[c][name][str(k)]} for c, k in cells[:6]]}

    # Where the rule loses is not scattered: it loses at small budgets, where a comparator is
    # read at an output comparable to the rule's own two candidates and its ranking decides.
    # The boundary is the finding, so it is derived here rather than left to a reader's eye.
    by_budget = {}
    for k in BUDGETS:
        lost = {n: sum(1 for c in CRITERIA if grid[c][rule]["uncapped"] <= grid[c][n][str(k)])
                for n in others}
        by_budget[str(k)] = {"lost_or_tied_per_comparator": lost,
                             "beats_every_comparator_everywhere": not any(lost.values())}
    clean = [int(k) for k, v in by_budget.items() if v["beats_every_comparator_everywhere"]]
    dominance_range = {"budgets_where_it_beats_everything": clean,
                       "lowest_such_budget": min(clean) if clean else None,
                       "reading": ("the rule emits about two candidates, so at budgets at or "
                                   "below its own output a comparator is read at a comparable "
                                   "size and its ranking decides. Above that the rule's output "
                                   "policy decides. The lead is a fact about output size, not "
                                   "about ranking.")}

    rep = {"provenance": stamp(__file__), "population": {"n": len(subs)},
           "criteria": CRITERIA, "budgets": BUDGETS, "alpha": args.alpha,
           "mean_output": mean_output,
           "gate": {"published_cell": list(PUBLISHED),
                    "reproduces_emission_leaderboard": not mism, "mismatches": mism},
           "grid": grid, "dominance_of_the_rule": verdict,
           "dominance_by_budget": by_budget, "dominance_range": dominance_range,
           "caveat": "BioTransformer returns an unranked set, so truncating it at k picks by "
                     "file order and not by confidence"}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n{len(subs)} substrates, {len(CRITERIA)} criteria x {len(BUDGETS)} budgets\n")
    print(f"{'comparator':<26}{'cells':>7}{'rule loses or ties':>20}  verdict")
    for name, v in verdict.items():
        print(f"{name:<26}{v['cells']:>7}{v['lost_or_tied']:>20}  "
              f"{'dominates' if v['dominates'] else 'does NOT dominate'}")
    print(f"\nrule mean output {mean_output[rule]}")
    print(f"it beats every comparator under every criterion at budgets "
          f"{dominance_range['budgets_where_it_beats_everything']}")
    if mism:
        print(f"\nGATE FAILED: {mism}")
        return 1
    print("gate: the published cell reproduces the committed emission table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

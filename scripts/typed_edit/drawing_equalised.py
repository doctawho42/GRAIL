#!/usr/bin/env python3
"""The comparison with every arm that can be re-run on the drawing a user would submit.

The comparison table hands five of six arms the substrate exactly as the corpus stores it and
MetaTox the natural tautomer, because the submission script re-tautomerised the file sent to the
service. The asymmetry has been measured arm by arm, each in its own section, and each of those
measurements holds the other five arms fixed. None of them answers the question a reader of the
table has: what does the comparison look like when everything that can move is on one drawing?

It is answerable without a new run for four of the five. The standardised pools exist for both of
this work's arms, and MetaPredictor and BioTransformer were re-run on the natural drawing and
their predictions frozen. SyGMa is an installed module and is re-run here. MetaTox cannot move
and is carried unchanged, marked, and reported beside the rest rather than dropped: it is the one
arm whose column means something different from the others, and hiding it would make the table
look equalised when it is not.

Everything else is the deployment table's: the same 291 substrates, the same references looked up
under the corpus string in every arm, the same tautomer-aware key, the same parent-drop rule and
the same pool cap.

    python scripts/typed_edit/drawing_equalised.py
    python scripts/typed_edit/drawing_equalised.py --substrates 20    # a probe
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _pools import assert_released  # noqa: E402
from _provenance import record_inputs, stamp  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
CAP = 100
N_BOOT, SEED = 10000, 0

# Where each arm's predictions on the standardiser's drawing are held. SyGMa is absent because it
# is re-run in process; MetaTox is absent because it cannot be re-run at all.
DRAWN = {
    "GRAIL exhaustive": ["results/widepools_std/w*.json", "results/widepools_std_fine/p*.json"],
    "GRAIL interactive": ["results/widepools_k30_std/all.json"],
    "MetaPredictor": ["results/metapredictor_natural_drawing_preds.json"],
    "BioTransformer": ["results/biotransformer_allhuman_one_step_natural_drawing_preds.json"],
}
STORED = {
    "GRAIL exhaustive": ["results/widepools_implicit/w*.json"],
    "GRAIL interactive": ["results/widepools_k30/all.json"],
    "MetaPredictor": ["artifacts/tier2_1170/metapredictor_preds.json"],
    "BioTransformer": ["results/biotransformer_allhuman_one_step_preds.json"],
    "SyGMa": ["results/sygma_fulltest_predictions.json"],
    "MetaTox": ["results/metatox_smirks_preds.json"],
    "GLORYx": ["results/gloryx_service_preds.json"],
}
OURS = ("GRAIL exhaustive", "GRAIL interactive")
# Arms whose predictions this repository holds frozen and cannot regenerate on another drawing:
# both are web services operated by their authors. MetaTox was declared here and GLORYx was not,
# so a control that set one aside with a footnote dropped the other without saying so, and the row
# it dropped is one this work does better on. Derived from the two facts that make an arm
# un-re-runnable rather than listed, so a third such arm cannot go quietly.
CANNOT_RERUN = ("MetaTox", "GLORYx")


def load_pools(patterns):
    """Candidate pools keyed by substrate, merged over shards; a repeat is taken once.

    Every shard has to record the released checkpoints, for the reason the drawing sweep gives:
    a merge cannot see that some of its sources were scored by a superseded model.
    """
    pools, sources = {}, []
    for spec in patterns:
        for path in sorted(glob.glob(str(ROOT / spec))):
            sources.append(path)
            for substrate, pool in json.loads(Path(path).read_text())["pools"].items():
                pools.setdefault(substrate, pool)
    assert_released(sources)
    return pools


def load_lists(patterns):
    """A frozen {substrate: [smiles, ...]} delivery, with the MetaTox file's own shape allowed."""
    out = {}
    for spec in patterns:
        for path in sorted(glob.glob(str(ROOT / spec))) or [str(ROOT / spec)]:
            blob = json.loads(Path(path).read_text())
            if isinstance(blob, dict) and "predictions" in blob:
                blob = blob["predictions"]
            for substrate, lst in blob.items():
                out.setdefault(substrate, list(lst))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", type=int, default=0, help="0 means the whole comparison set")
    ap.add_argument("--out", default=str(ROOT / "results" / "drawing_equalised.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    import multiprocessing
    import os

    from _rrf import rrf_order
    from bank_without_selection import _key as tautkey, _keys_parallel
    from grail_metabolism.utils.preparation import standardize_mol
    from sygma_by_dialect import _enumerate
    from vs_metatox import population

    workers = max(1, (os.cpu_count() or 4) - 2)
    keypool = multiprocessing.get_context("spawn").Pool(workers)

    subs, truth, _ = population()
    subs = sorted(s for s in subs if truth.get(s))
    if args.substrates:
        subs = subs[: args.substrates]
    real = {s: {k for k in (tautkey(p) for p in truth[s]) if k} for s in subs}
    subs = [s for s in subs if real[s]]
    parent = {s: tautkey(s) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)
    print(f"{len(subs)} substrates carrying {int(U.sum())} references", flush=True)

    # The drawing a user would submit, which for most substrates is the stored one.
    drawn = {}
    for s in subs:
        try:
            drawn[s] = Chem.MolToSmiles(standardize_mol(Chem.MolFromSmiles(s)))
        except Exception:
            drawn[s] = s
    moved = [s for s in subs if drawn[s] != s]
    print(f"the standardiser moves {len(moved)} of them", flush=True)

    def keyed(lists):
        """One arm's per-substrate key lists, deduplicated, with the parent dropped.

        Keys are computed in one batch across every arm and substrate, because tautomer
        canonicalisation is a search rather than a lookup and dominates this script: keying
        serially took the run past two hours before a single number was computed, and the pool
        this project already shares farms the misses out.
        """
        flat = [s2 for s in subs for s2 in lists.get(s, [])]
        table = dict(zip(flat, _keys_parallel(flat, keypool)))
        out = {}
        for s in subs:
            ks, seen = [], set()
            for smiles in lists.get(s, []):
                k = table.get(smiles)
                if k and k != parent[s] and k not in seen:
                    seen.add(k)
                    ks.append(k)
            out[s] = ks
        return out

    def from_pools(pools):
        return {s: [c["smiles"] for c in
                    rrf_order(sorted(pools.get(s, []), key=lambda c: -c["generator"])[:CAP])]
                for s in subs}

    def collect(spec, fallback=None):
        """One drawing's arms. `fallback` fills a substrate a re-run does not cover.

        MetaPredictor was re-run only on the substrates whose two drawings are different
        molecules, because the rest contribute a paired difference of exactly zero and a
        re-run of them would be a re-run of the same input. Its file therefore holds 79 of the
        291, and reading it without the fallback scored the other 212 as empty: the arm's recall
        collapsed from .48 to .13 and the table would have shown a drawing effect four times the
        size of any measured anywhere, which is what a silent join failure looks like.
        """
        arms, covered = {}, {}
        for name, patterns in spec.items():
            if name.startswith("GRAIL"):
                pools = load_pools(patterns)
                arms[name] = keyed(from_pools(pools))
                covered[name] = set(pools)
            else:
                raw = load_lists(patterns)
                arms[name] = keyed(raw)
                covered[name] = set(raw)
            if fallback is not None:
                # A substrate the re-run does not cover keeps its stored list. The test is
                # membership in the re-run's file and not an empty list, so an arm that
                # genuinely returns nothing on a substrate still reads as nothing.
                base = fallback[name]
                missing = [s for s in subs if s not in covered[name]]
                for s in missing:
                    arms[name][s] = base[s]
                if missing:
                    print(f"  {name}: {len(missing)} substrates keep their stored predictions, "
                          f"the re-run covering the other {len(subs) - len(missing)}", flush=True)
        return arms, covered

    stored, _ = collect(STORED)
    equalised, covered = collect(DRAWN, fallback=stored)

    # A substrate the standardiser does not move is the same molecule in both runs, so the two
    # arms must return the same list on it. Where they do not, the two files are not the same
    # measurement and the table would be comparing something other than the drawing.
    unmoved = [s for s in subs if drawn[s] == s]
    disagreeing = {name: sum(1 for s in unmoved if s in covered[name]
                             and equalised[name][s] != stored[name][s])
                   for name in DRAWN}
    # The agreement count above cannot fire on an arm whose re-run deliberately covers only the
    # substrates the drawing moves, because the two sets are then disjoint. MetaPredictor is such
    # an arm, and it is the one whose join failed. The bound that does bite is on coverage: every
    # substrate a re-run does not cover has to be one the drawing does not move, or the arm is
    # being read on a mixture of the two drawings without saying so.
    moved_set = set(moved)
    uncovered_moved = {name: sorted(moved_set - covered[name]) for name in DRAWN}
    for name, missing in uncovered_moved.items():
        if missing:
            print(f"  {name}: {len(missing)} substrates the drawing moves are absent from its "
                  f"re-run, so its equalised column is part one drawing and part the other",
                  flush=True)
    for name, n_bad in disagreeing.items():
        if n_bad:
            print(f"  {name}: {n_bad} of {len(unmoved)} unmoved substrates differ between the "
                  f"two runs, which the drawing cannot explain", flush=True)

    # SyGMa is installed, so its standardised arm is produced here rather than read.
    t0, per = time.perf_counter(), {}
    for i, s in enumerate(subs, 1):
        per[s] = _enumerate(drawn[s])
        if i % 50 == 0 or i == len(subs):
            print(f"  SyGMa on the drawn form {i}/{len(subs)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    equalised["SyGMa"] = keyed(per)
    # The arms that cannot move. Their columns are the same numbers as in the stored table, and
    # saying so in the artifact is the point: the table is equalised in the columns it can be.
    for _arm in CANNOT_RERUN:
        if _arm in stored:
            equalised[_arm] = stored[_arm]

    # PAIRED, so the population is the substrates every re-run answers for in BOTH drawings.
    # A substrate one drawing crashed on cannot be scored as an empty list here: the delta would
    # then be that arm's whole output on that substrate and would be attributed to the drawing,
    # which is the defect the coverage record above exists to catch. BioTransformer crashes
    # deterministically on a handful in each drawing, five of them substrates the drawing moves.
    # The set that is dropped, and its size, are recorded so the caption can name them.
    # ONLY the substrates the drawing MOVES. An unmoved substrate is the same molecule in both
    # columns, so a re-run is not needed for it and its absence from one is not a gap: an arm may
    # deliberately re-run only the moved set, and MetaPredictor is such an arm. A first version of
    # this rule dropped every substrate any re-run did not cover and cut the population from 291
    # to 76, turning a deliberate design into a reported failure.
    _drop = sorted({s for name in DRAWN for s in moved_set if s not in covered[name]})
    if _drop:
        subs = [s for s in subs if s not in set(_drop)]
        real = {s: real[s] for s in subs}
        U = np.array([len(real[s]) for s in subs], dtype=float)
        print(f"  paired population: {len(subs)} substrates; {len(_drop)} dropped because at "
              f"least one arm has no answer in one of the two drawings", flush=True)
    # The union, as a LIST. The coverage record beside it holds per-arm COUNTS, and a gate that
    # compares a count against a count cannot tell whether the substrates dropped are the same
    # substrates that were uncovered; it can only tell that two totals agree.
    excluded = {"uncovered_moved_union": sorted({s for v in uncovered_moved.values() for s in v}),
                "n_dropped": len(_drop),
                "n_paired": len(subs),
                "dropped_because_a_re_run_has_no_answer": _drop,
                "of_those_the_drawing_moves": sorted(set(_drop) & moved_set)}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def hits(arm, k):
        return np.array([len(set(arm[s][:k]) & real[s]) for s in subs], dtype=float)

    def recall(arms):
        return {a: {str(k): round(float(hits(arms[a], k).sum() / U.sum()), 4) for k in KS}
                for a in arms}

    def verdicts(arms, exclude=()):
        others = [a for a in arms if a not in OURS and a not in exclude]
        rows = {}
        for k in KS:
            best_ours = max(OURS, key=lambda a: hits(arms[a], k).sum())
            best_other = max(others, key=lambda a: hits(arms[a], k).sum())
            d = hits(arms[best_ours], k) - hits(arms[best_other], k)
            bt = d[idx].sum(axis=1) / denom
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            rows[str(k)] = {"ours": best_ours, "theirs": best_other,
                            "gap": round(float(d.sum() / U.sum()), 4),
                            "ci95": [round(lo, 4), round(hi, 4)],
                            "verdict": "leads" if lo > 0 else ("trails" if hi < 0 else "neither")}
        return rows

    stored_v, equal_v = verdicts(stored), verdicts(equalised)
    # The same grid with MetaTox set aside, since it is the one arm still on its own drawing and a
    # cell read against it is not a cell of an equalised table.
    equal_v_wo = verdicts(equalised, exclude=CANNOT_RERUN)
    moved_cells = [k for k in equal_v if equal_v[k]["verdict"] != stored_v[k]["verdict"]]

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([ROOT / p for spec in (STORED, DRAWN)
                                 for pats in spec.values() for p in pats if "*" not in p]),
        "question": ("what the comparison reads when every arm that can be re-run is on the "
                     "drawing the declared standardiser produces"),
        "population": {"n_substrates": len(subs), "n_references": int(U.sum()),
                       "substrates_the_standardiser_moves": len(moved)},
        "paired_exclusion": excluded,
        "criterion": "tautomer-aware InChIKey, as everywhere else",
        "convention": "parent dropped, pool capped at 100, references looked up under the corpus "
                      "string in every arm so the two tables are scored against one annotation",
        "arms_that_could_not_be_re_run": [a for a in CANNOT_RERUN if a in stored],
        "substrates_the_re_run_does_not_cover": {name: len([s for s in subs
                                                            if s not in covered[name]])
                                                 for name in DRAWN},
        "unmoved_substrates_where_the_two_runs_disagree": disagreeing,
        "substrates_the_drawing_moves_that_a_re_run_does_not_cover": {
            name: len(v) for name, v in uncovered_moved.items()},
        "why_that_matters": ("a substrate the standardiser does not move is the same molecule in "
                             "both runs, so a disagreement there is not a drawing effect"),
        "why": ("MetaTox is a web service with no re-run available to us, and it is the one arm "
                "that received the natural drawing in the original submission, so its column is "
                "already on a different input from the other five"),
        "recall_as_stored": recall(stored),
        "recall_equalised": recall(equalised),
        "verdicts_as_stored": stored_v,
        "verdicts_equalised": equal_v,
        "verdicts_equalised_without_the_service_arms": equal_v_wo,
        "the_service_arms": list(CANNOT_RERUN),
        "budgets_whose_verdict_moves": moved_cells,
        "reading": (
            "The stored grid is what the paper reports and the equalised grid is what a user "
            "submitting a drawing would meet. Where the two agree, the verdict is a property of "
            "the systems; where they differ, it is a property of the dialect the comparison was "
            "run in. MetaTox's column is unchanged in both, which is the residue this work "
            "cannot remove."),
    }
    keypool.close()
    keypool.join()
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\n{'k':>4s}  {'stored':>34s}  {'equalised':>34s}")
    for k in KS:
        a, b = stored_v[str(k)], equal_v[str(k)]
        def fmt(c):
            return (f"{c['verdict']:7s} vs {c['theirs']:14s} {c['gap']:+.4f}")
        print(f"{k:>4d}  {fmt(a):>34s}  {fmt(b):>34s}"
              f"{'   <- moves' if a['verdict'] != b['verdict'] else ''}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

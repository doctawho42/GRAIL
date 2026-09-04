#!/usr/bin/env python3
"""The containment claim on the population the claim is about.

The manuscript says the shortfall is a property of the corpus rather than of the bank, and supports
it with 567 of 607: of the test references whose transformation type the bank does not hold, that
many have a type the training annotation does not hold either. But 607 is every reference of absent
type, whether or not the bank reaches it, while the number the Abstract and the Conclusions are
arguing about is the 337 the bank both fails to reach and cannot type. The bank reaches 270 of the
607, so the two populations differ by more than the gap they are used to explain.

Nothing in this work cross-tabulated them, because the census records counts and not which
references it counted. This runs the census and the typing on one population in one pass, so the
containment can be stated over the references it is supposed to be about.

The run is gated on reproducing the census it re-derives: if the four cells of the committed
decomposition do not come back, the artifact is not written, because a containment figure computed
over a population that no longer matches the published one explains nothing.

    python scripts/typed_edit/containment_on_the_uncovered.py
    python scripts/typed_edit/containment_on_the_uncovered.py --shard 0 --shards 8 --out-shard f.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

COMMITTED = {"novel_type": 337, "known_type": 98, "untypeable": 40, "uncovered": 475}


TRAIN_CACHE = ROOT / "results" / "containment_train_types.json"


def train_type_keys(limit=None):
    """Every transformation type the training annotation contains, under the census's definition.

    This is the same typing route missing_types_in_train.py uses, reused rather than reimplemented,
    so the two artifacts cannot drift into disagreeing about what a type is.

    It is computed once and cached, and the reason is not speed. The typing route runs an MCS with
    a wall-clock timeout and treats a cancelled search as an untypeable pair, so under load more
    pairs go untyped, fewer types enter this set, and more test references come back "absent from
    training" -- which is the direction that flatters the claim. Eight shards each typing the
    training split disagreed about how many types it holds. So the set is built in one process and
    then required to reproduce the published typing before anything is measured against it.
    """
    if TRAIN_CACHE.exists():
        blob = json.loads(TRAIN_CACHE.read_text())
        return set(blob["keys"]), blob["pairs"], blob["typed"]

    from missing_types_in_train import annotated_pairs
    from rdkit import Chem
    from coverage_gap_types import pair_to_type

    keys, typed = set(), 0
    pairs = annotated_pairs("train", limit)
    for i, (sub, prod) in enumerate(pairs, 1):
        a, b = Chem.MolFromSmiles(sub), Chem.MolFromSmiles(prod)
        if a is None or b is None:
            continue
        try:
            t = pair_to_type(a, b)
        except Exception:
            t = None
        if t is not None:
            typed += 1
            keys.add(json.dumps(t, sort_keys=True))
        if i % 1000 == 0:
            print(f"  train {i}/{len(pairs)}, {typed} typed, {len(keys)} types", flush=True)

    published = json.loads((ROOT / "results/missing_types_in_train.json").read_text())["train"]
    TRAIN_CACHE.write_text(json.dumps(
        {"what_this_is": "the training annotation's transformation types, typed once in one "
                         "process because the typing route's MCS timeout is load-sensitive",
         "pairs": len(pairs), "typed": typed, "distinct_types": len(keys),
         "the_published_typing": {k: published[k] for k in ("pairs", "typed", "distinct_types")},
         "agrees_with_the_published_typing": all(
             published[k] == mine for k, mine in (("pairs", len(pairs)), ("typed", typed),
                                                  ("distinct_types", len(keys)))),
         "why_it_may_not": ("the published typing ran on a pool of workers and the route cancels "
                            "an MCS that exceeds a wall-clock timeout, so under load it types "
                            "fewer pairs and finds fewer types; fewer training types is the "
                            "direction that inflates a containment share"),
         "keys": sorted(keys)}))
    return keys, len(pairs), typed


TEST_CACHE = ROOT / "results" / "containment_test_types.json"


def test_type_keys(items):
    """The type of every annotated test pair, typed once in one process and cached.

    Same reason as the training side: the route cancels an MCS on a wall-clock timeout, so a pair
    typed inside a parallel shard can come back untypeable purely because the machine was busy.
    Typing here and applying rules there keeps the only load-sensitive step off the parallel path.
    """
    if TEST_CACHE.exists():
        blob = json.loads(TEST_CACHE.read_text())
        return {tuple(k.split("\t", 1)): v for k, v in blob["types"].items()}, blob["typed"]

    from rdkit import Chem
    from coverage_gap_types import pair_to_type

    out, typed, n = {}, 0, 0
    for i, (sub, mets) in enumerate(items, 1):
        sub_mol = Chem.MolFromSmiles(sub)
        for met in mets:
            n += 1
            met_mol = Chem.MolFromSmiles(met) if sub_mol is not None else None
            if met_mol is None:
                out[(sub, met)] = None
                continue
            try:
                t = pair_to_type(sub_mol, met_mol)
            except Exception:
                t = None
            out[(sub, met)] = json.dumps(t, sort_keys=True) if t is not None else None
            typed += t is not None
        if i % 100 == 0 or i == len(items):
            print(f"  test {i}/{len(items)} substrates, {typed}/{n} pairs typed", flush=True)

    TEST_CACHE.write_text(json.dumps(
        {"what_this_is": "the transformation type of every annotated test pair, typed once in "
                         "one process so a parallel shard cannot change it",
         "pairs": n, "typed": typed,
         "types": {f"{a}\t{b}": v for (a, b), v in out.items()}}))
    return out, typed


def sweep(items, rules, bank_types, train_keys, test_types):
    """The census over one shard of test substrates, with the typing already fixed."""
    from rdkit import Chem
    from grail_metabolism.metrics import _tautomer_inchikey
    from engine_knobs import apply_with

    cov, gap, novel = Counter(), Counter(), Counter()
    t0 = time.perf_counter()
    for i, (sub, true_prods) in enumerate(items, 1):
        sub_mol = Chem.MolFromSmiles(sub)
        if sub_mol is None:
            continue
        products = apply_with(sub_mol, rules, False, "canonical", False)
        covered_keys = set()
        for p in products:
            try:
                covered_keys.add(_tautomer_inchikey(p))
            except Exception:
                continue
        for met in true_prods:
            try:
                mk = _tautomer_inchikey(met)
            except Exception:
                continue
            if mk in covered_keys:
                cov["covered"] += 1
                continue
            cov["uncovered"] += 1
            key = test_types.get((sub, met))
            if key is None:
                gap["untypeable"] += 1
                continue
            if key in bank_types:
                gap["known_type"] += 1
            else:
                gap["novel_type"] += 1
                # The cell this script exists for: of the misses the bank cannot type, how many
                # carry chemistry the training annotation does not contain either.
                novel["absent_from_training" if key not in train_keys
                      else "present_in_training"] += 1
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)} ({time.perf_counter()-t0:.0f}s) uncovered={cov['uncovered']}"
                  f" novel={gap['novel_type']} absent={novel['absent_from_training']}", flush=True)
    return cov, gap, novel


def merge(paths, out) -> int:
    """Add the shards up and write the one artifact, with the gate applied to the total."""
    from collections import Counter as C

    cov, gap, novel, subs = C(), C(), C(), 0
    head = None
    for f in paths:
        blob = json.loads(Path(f).read_text())
        cov.update(blob["cov"]); gap.update(blob["gap"]); novel.update(blob["novel"])
        subs += blob["substrates"]
        head = head or blob
    return write({"substrates": subs, "cov": dict(cov), "gap": dict(gap), "novel": dict(novel),
                  "n_train_pairs": head["n_train_pairs"], "n_train_types": head["n_train_types"],
                  "n_train_typed": head["n_train_typed"],
                  "all_test_cells": head.get("all_test_cells", {}),
                  "n_rules": head["n_rules"], "n_bank_types": head["n_bank_types"]}, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", nargs="*", default=None,
                    help="shard files to add up and write as the artifact")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out-shard", default="")
    ap.add_argument("--train-pairs", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "containment_on_the_uncovered.json"))
    args = ap.parse_args()

    if args.merge:
        return merge(args.merge, args.out)

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from grail_metabolism.model.reaction_types import canonical_type
    from grail_metabolism.utils.preparation import load_default_rules
    from run_benchmark import load_test_map

    rules = load_default_rules()
    # As key strings, because everything else here compares key strings and a type that
    # survives a JSON round trip is a list where the fresh one is a tuple.
    bank_types = {json.dumps(t, sort_keys=True)
                  for t in (canonical_type(r) for r in rules) if t is not None}
    print(f"bank: {len(rules)} rules, {len(bank_types)} types", flush=True)

    train_keys, n_train, n_typed = train_type_keys(args.train_pairs or None)
    print(f"train: {n_train} pairs, {n_typed} typed, {len(train_keys)} types",
          flush=True)

    items = sorted(load_test_map(None, 0).items())
    test_types, n_test_typed = test_type_keys(items)
    print(f"test: {n_test_typed} of {len(test_types)} annotated pairs typed", flush=True)

    # The published cross-tabulation, restated under this typing. It is the same four cells
    # missing_types_in_train.json reports, over every annotated test reference rather than over
    # the uncovered ones, so the two populations can be read side by side under one instrument.
    cells = Counter()
    for key in test_types.values():
        if key is None:
            continue
        cells["in the bank and in training" if key in bank_types and key in train_keys else
              "in the bank, not in training" if key in bank_types else
              "not in the bank, in training" if key in train_keys else
              "not in the bank, nor in training"] += 1

    mine = [it for i, it in enumerate(items) if i % args.shards == args.shard]
    print(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(items)} substrates", flush=True)

    cov, gap, novel = sweep(mine, rules, bank_types, train_keys, test_types)

    if args.out_shard:
        Path(args.out_shard).write_text(json.dumps(
            {"shard": args.shard, "shards": args.shards, "substrates": len(mine),
             "cov": dict(cov), "gap": dict(gap), "novel": dict(novel),
             "n_train_pairs": n_train, "n_train_types": len(train_keys),
             "n_train_typed": n_typed, "all_test_cells": dict(cells), "n_rules": len(rules), "n_bank_types": len(bank_types)}))
        print(f"wrote {args.out_shard}", flush=True)
        return 0

    return write(
        {"substrates": len(mine), "cov": dict(cov), "gap": dict(gap), "novel": dict(novel),
         "n_train_pairs": n_train, "n_train_types": len(train_keys),
         "n_train_typed": n_typed, "all_test_cells": dict(cells), "n_rules": len(rules), "n_bank_types": len(bank_types)}, args.out)


def write(merged, out) -> int:
    from _provenance import stamp

    cov, gap, novel = merged["cov"], merged["gap"], merged["novel"]
    computed = {"novel_type": gap.get("novel_type", 0), "known_type": gap.get("known_type", 0),
                "untypeable": gap.get("untypeable", 0), "uncovered": cov.get("uncovered", 0)}
    mismatch = {k: (v, COMMITTED[k]) for k, v in computed.items() if v != COMMITTED[k]}
    if mismatch:
        print("REFUSING: this run does not reproduce the published decomposition: "
              + ", ".join(f"{k}: {mine} against {theirs}" for k, (mine, theirs) in mismatch.items()),
              file=sys.stderr)
        return 1

    absent = novel.get("absent_from_training", 0)
    present = novel.get("present_in_training", 0)
    total = absent + present
    report = {
        "provenance": stamp(__file__),
        "question": ("of the uncovered test references whose type the bank does not hold, how many "
                     "carry a type the training annotation does not hold either"),
        "why_it_is_not_the_published_figure": (
            "the published containment is over every reference of absent type, reached or not; "
            "this is over the ones the bank also fails to reach, which is the population the "
            "Abstract's claim about corpora is about"),
        "typing": "radius-0 reaction type by the mining route, the definition the census uses",
        "gate": {"reproduces_the_published_decomposition": True, "computed": computed,
                 "committed": COMMITTED},
        "population": {"substrates": merged["substrates"], "uncovered": computed["uncovered"],
                       "of_absent_type": total},
        "training_annotation": {"pairs": merged["n_train_pairs"],
                                "typed": merged["n_train_typed"],
                                "distinct_types": merged["n_train_types"]},
        "bank": {"rules": merged["n_rules"], "distinct_types": merged["n_bank_types"]},
        "the_published_cross_tabulation_restated": {
            "what_this_is": ("the four cells missing_types_in_train.json reports, recomputed here "
                             "under a typing that does not depend on machine load"),
            "cells": merged.get("all_test_cells", {}),
        },
        "absent_from_training": absent,
        "present_in_training": present,
        "share_absent_from_training": round(absent / total, 4) if total else None,
        "reading": ("a share near the published one means the two populations agree and the "
                    "claim survives its own restatement; a share below it means part of what the "
                    "bank misses is chemistry the corpus did contain"),
    }
    Path(out).write_text(json.dumps(report, indent=1))
    print(f"\n{absent} of {total} uncovered references of absent type carry a type training "
          f"lacks too ({report['share_absent_from_training']})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

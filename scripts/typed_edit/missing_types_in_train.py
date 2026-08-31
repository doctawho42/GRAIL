#!/usr/bin/env python3
"""Is the chemistry the bank misses absent from the corpus, or only from the bank?

The manuscript measures that most uncovered references need a transformation type the bank does not
hold, and then states, in the abstract and twice more, that this is a property of the corpus rather
than of any one bank. Those are different propositions and only the first was measured. The bank is
mined from the training split and a template survives only if it regenerates its own product and
passes a selectivity filter, so a type can sit in the training annotation and never reach the bank.

The two are separated by typing both halves of the annotation under the definition the census uses
and intersecting them. A test reference whose type occurs in the training annotation is chemistry
the corpus contains and the miner did not carry across; one whose type occurs nowhere in training
is chemistry the corpus does not contain, which is the proposition the abstract asserts.

This types every annotated pair of both splits, so it needs no record of which references were
uncovered and answers the question over the whole annotation rather than over a recorded head.

    python scripts/typed_edit/missing_types_in_train.py
    python scripts/typed_edit/missing_types_in_train.py --pairs 2000     # a smaller probe
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402


def _worker(pair):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from coverage_gap_types import pair_to_type

    sub, prod = pair
    sub_mol, prod_mol = Chem.MolFromSmiles(sub), Chem.MolFromSmiles(prod)
    if sub_mol is None or prod_mol is None:
        return None
    try:
        t = pair_to_type(sub_mol, prod_mol)
    except Exception:
        return None
    return json.dumps(t, sort_keys=True) if t is not None else None


def annotated_pairs(split: str, limit: int | None):
    """(substrate, product) for every annotated positive pair of one split."""
    import re

    data = ROOT / "grail_metabolism" / "data"
    index_to_smiles = {}
    cur, key = {}, None
    with open(data / f"{split}.sdf", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                match = re.match(r">\s*<([^>]+)>", line)
                key = match.group(1) if match else None
                continue
            if key is not None:
                value = line.strip()
                if value == "":
                    key = None
                else:
                    cur.setdefault(key, value)
                continue
            if line.startswith("$$$$"):
                try:
                    index_to_smiles[int(cur["Index"])] = cur.get("SMILES")
                except Exception:
                    pass
                cur, key = {}, None

    pairs, seen = [], set()
    with open(data / f"{split}_triples_clean.txt") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 3 or parts[2] != "1":
                continue
            a = index_to_smiles.get(int(parts[0]))
            b = index_to_smiles.get(int(parts[1]))
            if not a or not b:
                continue
            key2 = (a, b)
            if key2 in seen:
                continue
            seen.add(key2)
            pairs.append(key2)
            if limit and len(pairs) >= limit:
                break
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=0, help="0 means every annotated pair")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "missing_types_in_train.json"))
    args = ap.parse_args()

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    ctx = multiprocessing.get_context("spawn")

    def types_of(split):
        pairs = annotated_pairs(split, args.pairs or None)
        print(f"{split}: {len(pairs)} annotated pairs on {workers} workers", flush=True)
        seen, typed, t0 = {}, 0, time.perf_counter()
        with ctx.Pool(workers) as pool:
            for n, key in enumerate(pool.imap_unordered(_worker, pairs, 32), 1):
                if key is not None:
                    typed += 1
                    seen[key] = seen.get(key, 0) + 1
                if n % 2000 == 0 or n == len(pairs):
                    print(f"  {n}/{len(pairs)} ({time.perf_counter() - t0:.0f}s) "
                          f"typed {typed}, distinct types {len(seen)}", flush=True)
        return {"pairs": len(pairs), "typed": typed, "types": seen}

    train = types_of("train")
    test = types_of("test")

    # The bank's own types, from the templates rather than from anything they produce.
    from coverage_gap_types import canonical_type
    from grail_metabolism.utils.preparation import load_default_rules

    bank_types = set()
    for rule in load_default_rules():
        try:
            bt = canonical_type(rule)
        except Exception:
            continue
        if bt is not None:
            bank_types.add(json.dumps(bt, sort_keys=True))
    print(f"bank: {len(bank_types)} distinct types", flush=True)

    train_types = set(train["types"])
    test_types = set(test["types"])
    shared = test_types & train_types
    only_test = test_types - train_types
    # References, not types: a type carrying many test references matters more than one carrying
    # a single reference, and the singleton tail is where most of the missing mass sits.
    refs_shared = sum(test["types"][k] for k in shared)
    refs_only_test = sum(test["types"][k] for k in only_test)

    # The four cells the argument turns on, counted by reference.
    cells = {"in the bank and in training": 0, "in the bank, not in training": 0,
             "not in the bank, in training": 0, "not in the bank, nor in training": 0}
    for key, count in test["types"].items():
        in_bank, in_train = key in bank_types, key in train_types
        if in_bank and in_train:
            cells["in the bank and in training"] += count
        elif in_bank:
            cells["in the bank, not in training"] += count
        elif in_train:
            cells["not in the bank, in training"] += count
        else:
            cells["not in the bank, nor in training"] += count
    outside = cells["not in the bank, in training"] + cells["not in the bank, nor in training"]

    report = {
        "provenance": stamp(__file__),
        "question": ("whether the transformation types of the test annotation occur in the "
                     "training annotation, which separates a shortfall of the corpus from a "
                     "shortfall of the bank mined from it"),
        "typing": ("radius-0 reaction type by the mining route, the definition "
                   "coverage_gap_types.py uses for the census"),
        "train": {"pairs": train["pairs"], "typed": train["typed"],
                  "distinct_types": len(train_types)},
        "test": {"pairs": test["pairs"], "typed": test["typed"],
                 "distinct_types": len(test_types)},
        "test_types_also_in_train": len(shared),
        "test_types_absent_from_train": len(only_test),
        "test_references_whose_type_is_in_train": refs_shared,
        "test_references_whose_type_is_absent_from_train": refs_only_test,
        "share_of_test_references_whose_type_is_absent_from_train": round(
            refs_only_test / max(refs_shared + refs_only_test, 1), 4),
        "bank_types": len(bank_types),
        "references_by_cell": cells,
        "references_whose_type_the_bank_lacks": outside,
        "of_those_the_corpus_lacks_too": cells["not in the bank, nor in training"],
        "share_of_the_bank_s_type_gap_the_corpus_also_lacks": round(
            cells["not in the bank, nor in training"] / max(outside, 1), 4),
        "reading": (
            "A test reference whose type occurs in the training annotation is chemistry the "
            "corpus contains, so a bank that misses it misses something the corpus could have "
            "supplied. One whose type occurs nowhere in training is chemistry the corpus does not "
            "contain on the training side, which is what the manuscript's claim about corpora "
            "requires. The two counts are reported and neither is inferred from the other."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\ntrain: {len(train_types)} distinct types over {train['typed']} typed pairs")
    print(f"test : {len(test_types)} distinct types over {test['typed']} typed pairs")
    print(f"  test types also in train : {len(shared)}")
    print(f"  test types absent        : {len(only_test)}")
    print(f"  test references by type in train : {refs_shared}")
    print(f"  test references by type absent   : {refs_only_test} "
          f"({report['share_of_test_references_whose_type_is_absent_from_train']:.1%})")
    print("\nreferences by cell:")
    for name, count in cells.items():
        print(f"  {name:34s} {count}")
    print(f"  of the {outside} whose type the bank lacks, "
          f"{cells['not in the bank, nor in training']} are absent from training too "
          f"({report['share_of_the_bank_s_type_gap_the_corpus_also_lacks']:.1%})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""What the coverage ceiling loses if every borrowed template is removed from the bank.

`curated_third_party.py` finds that a large part of the curated half is present character for
character in a rule set published by somebody else. Whether those templates can stay is a licensing
question; what they are worth is a measurement, and the licensing question is much easier to answer
once the measurement exists. Removing them is one of the options, and an option whose cost is
unknown cannot be compared with the others.

This reports, on the same substrates and under the same conventions as the ceiling the paper
prints, what the bank reaches whole and what it reaches with every borrowed template dropped: the
references only the whole bank recovers, and the paired interval on the difference. The loss is
also split by source, so the two obligations can be priced separately -- one of them may be cheap
to discharge and the other not.

Removal is exact, not approximate: the borrowed set is defined by string equality against the
published files, the same instrument that found them, so what is dropped is what was identified.

Only the disjoint pieces are applied, and the variants are recovered by union. Applying the whole
bank and then each variant would run the expensive part four times over for numbers the cheap
parts determine.

The expensive part is then run on a subset of the substrates, and the reason it is sound is worth
stating because it is what makes this affordable. A reference that the borrowed templates do not
recover cannot be lost by removing them, whatever the remainder does with it. So the borrowed
subsets are applied everywhere -- they are small -- and the 6,818-template remainder only where the
borrowed set recovered something. Everywhere else the two banks agree by construction, and the
count of substrates skipped on that ground is reported rather than left implicit.

    python scripts/typed_edit/licence_removal_cost.py
    python scripts/typed_edit/licence_removal_cost.py --population subsample245 --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _population import POPULATIONS, population_items, tagged_out  # noqa: E402
from _provenance import stamp  # noqa: E402

N_BOOT, SEED = 10000, 0
_BANKS: dict = {}


def borrowed_sets() -> dict:
    """{source: set of templates} for every published rule set this repository holds a copy of."""
    out = {}

    core = ROOT / "artifacts/tier2/biotransformer/database/metabolicReactions.json"
    if core.exists():
        out["BioTransformer"] = set(
            re.findall(r'"([^"\n]*>>[^"\n]*)"', core.read_text(errors="replace")))

    try:
        import sygma

        base = Path(os.path.dirname(sygma.__file__))
        found = set()
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in (".txt", ".json", ".py", ".dat"):
                found |= set(re.findall(r"([^\s\"']*>>[^\s\"']*)",
                                        path.read_text(errors="replace")))
        out["SyGMa"] = {r for r in found if ">>" in r}
    except Exception:
        pass
    return out


def _init(banks):
    global _BANKS
    _BANKS = banks

    from grail_metabolism.utils.preparation import _standardize_smiles_cached  # noqa: F401


def _worker(item):
    """Keys each bank variant reaches on one substrate, and which references they cover."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from bank_without_selection import _key
    from engine_knobs import apply_with

    substrate, references = item
    mol = Chem.MolFromSmiles(substrate)
    if mol is None:
        return None
    wanted = set()
    for met in references:
        try:
            key = _key(met)
        except Exception:
            continue
        if key:
            wanted.add(key)
    # The recovered reference keys, not a count, so the caller can take unions across pieces.
    row = {"substrate": substrate, "references": sorted(wanted)}
    for name, rules in _BANKS.items():
        keys = set()
        for product in apply_with(mol, rules, False, "canonical", False):
            try:
                key = _key(product)
            except Exception:
                continue
            if key:
                keys.add(key)
        row[name] = sorted(wanted & keys)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="grail_metabolism/resources/extended_smirks.txt")
    ap.add_argument("--population", default="clean_test", choices=POPULATIONS)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "licence_removal_cost.json"))
    args = ap.parse_args()
    args.out = tagged_out(args.out, args.population)

    bank = [line.strip() for line in open(ROOT / args.bank) if line.strip()]
    sources = borrowed_sets()
    if not sources:
        raise SystemExit("no published rule set is available to compare against")

    in_bank = {name: [r for r in bank if r in pool] for name, pool in sources.items()}
    every = set().union(*(set(v) for v in in_bank.values()))
    # The pieces, disjoint by construction. Every variant the report names is a union of these.
    pieces = {"remainder": [r for r in bank if r not in every]}
    for name, hits in in_bank.items():
        pieces[name] = list(hits)

    print(f"bank {len(bank)} templates")
    for name, hits in sorted(in_bank.items()):
        print(f"  {name:16s} {len(hits):4d} present verbatim")
    print(f"  together         {len(every):4d} distinct templates would be removed")
    for name, rules in pieces.items():
        print(f"  piece {name:34s} {len(rules):5d} templates")
    n_skipped_note = None

    items = population_items(args.population)
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    ctx = multiprocessing.get_context("spawn")
    remainder = pieces.pop("remainder")

    print(f"\n{len(items)} substrates on {workers} workers")
    print("  pass 1: the borrowed subsets, everywhere", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ctx.Pool(workers, initializer=_init, initargs=(pieces,)) as pool:
        for n, row in enumerate(pool.imap_unordered(_worker, items, 8), 1):
            if row is not None:
                rows.append(row)
            if n % 100 == 0 or n == len(items):
                print(f"    {n}/{len(items)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    by_substrate = {r["substrate"]: r for r in rows}
    touched = [s for s, r in by_substrate.items()
               if any(r[name] for name in pieces)]
    skipped = len(rows) - len(touched)
    print(f"  pass 2: the remainder, on the {len(touched)} substrates where a borrowed template "
          f"recovered a reference ({skipped} skipped, nothing there to lose)", flush=True)
    todo = [(s, by_substrate[s]["references"]) for s in touched]
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init, initargs=({"remainder": remainder},)) as pool:
        for n, row in enumerate(pool.imap_unordered(_worker, todo, 4), 1):
            if row is not None:
                by_substrate[row["substrate"]]["remainder"] = row["remainder"]
            if n % 25 == 0 or n == len(todo):
                print(f"    {n}/{len(todo)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    # A substrate the remainder was not run on recovers, by the argument above, exactly what the
    # whole bank recovers there; it is filled in as such so the union arithmetic below is uniform.
    for row in rows:
        row.setdefault("remainder", row["references"])
    pieces["remainder"] = remainder

    rows.sort(key=lambda r: r["substrate"])
    U = np.array([len(r["references"]) for r in rows], dtype=float)

    # The variants, assembled from the disjoint pieces rather than measured again.
    variants = {"whole bank": ["remainder"] + list(in_bank),
                "without every borrowed template": ["remainder"]}
    for name in in_bank:
        variants[f"without {name}"] = ["remainder"] + [k for k in in_bank if k != name]
    sizes = {"whole bank": len(bank),
             "without every borrowed template": len(pieces["remainder"])}
    for name in in_bank:
        sizes[f"without {name}"] = len(bank) - len(set(in_bank[name]))

    covered = {}
    for name, parts in variants.items():
        covered[name] = np.array(
            [len(set().union(*(set(r[p]) for p in parts))) for r in rows], dtype=float)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    whole = covered["whole bank"]
    report_variants, contrasts = {}, {}
    for name, hits in covered.items():
        report_variants[name] = {
            "templates": sizes[name],
            "references_recovered": int(hits.sum()),
            "ceiling": round(float(hits.sum() / U.sum()), 4),
        }
        if name == "whole bank":
            continue
        d = hits - whole
        bt = d[idx].sum(axis=1) / denom
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        contrasts[name] = {
            "references_lost": int((whole - hits).sum()),
            "substrates_that_lose_at_least_one": int((whole - hits > 0).sum()),
            "ceiling_change": round(float(d.sum() / U.sum()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0),
        }

    report = {
        "provenance": stamp(__file__),
        "population": {"name": args.population, "n_substrates": len(rows),
                       "n_references": int(U.sum())},
        "substrates_the_remainder_was_run_on": len(touched),
        "substrates_skipped_because_no_borrowed_template_recovered_anything": skipped,
        "instrument": ("one uncapped depth-1 pass of each bank variant, hydrogens implicit, "
                       "tautomer-aware matching; the same conventions as the reported ceiling"),
        "borrowed": {name: len(hits) for name, hits in sorted(in_bank.items())},
        "borrowed_distinct": len(every),
        "variants": report_variants,
        "against_the_whole_bank": contrasts,
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "reading": (
            "This prices one option and does not choose it. A template that is borrowed and whose "
            "chemistry another template also reaches costs nothing to remove; the references lost "
            "are the ones no remaining template recovers, which is the quantity a licensing "
            "decision turns on."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\n{len(rows)} substrates, {int(U.sum())} references")
    for name, row in report_variants.items():
        cell = contrasts.get(name)
        tail = "" if cell is None else (
            f"   {cell['ceiling_change']:+.4f} "
            f"[{cell['ci95'][0]:+.4f}, {cell['ci95'][1]:+.4f}]"
            f"{'  separates' if cell['excludes_zero'] else ''}"
            f"   {cell['references_lost']} references lost")
        print(f"  {name:34s} {row['templates']:5d} rules  ceiling {row['ceiling']:.4f}{tail}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Completing the loop is itself an undeclared choice, and the two ways of doing it disagree.

Section 4 reports that expanding a substrate and never contracting the product costs a bank most of
what it appears to lose to the hydrogen convention, and that one call restores it. That call is not
unique. `AddHs` marks every heavy atom as taking no implicit hydrogens; `RemoveHs` then deletes the
drawn hydrogens without lifting the mark, so a template that consumes a mapped hydrogen and puts
nothing in its place leaves the atom one short and RDKit records an unpaired electron rather than
refilling the valence. Clearing the mark first restores the capacity the expansion suspended.

Both are one-line implementations of the same intention and neither is written down in any paper.
This measures what choosing between them costs, on the deployed bank and on substrates of the clean
test split:

    parseable       products that survive sanitisation under each contraction
    radicals        of those, how many carry an unpaired electron
    disagreements   firings where both parse and the two structures differ

A metabolite corpus contains no radicals, so a product that acquires one cannot match any reference:
the loss is silent rather than loud, which is what makes the choice worth reporting rather than
fixing quietly. The intention is not in doubt -- an engine predicting metabolites should not emit
radicals -- so this is not a symmetric fork. It is an undeclared step whose default is wrong.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem, RDLogger

from _contract import contract, contract_by_removing_only
from grail_metabolism.utils.preparation import _iter_reaction_products, load_default_rules

RDLogger.DisableLog("rdApp.*")


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "contraction_choice.json"))
    args = ap.parse_args()

    rules = load_default_rules()
    subs = list(json.loads((ROOT / "results/test_references.json").read_text()))
    random.Random(args.seed).shuffle(subs)
    subs = subs[: args.substrates]

    n = {"one_call": 0, "restored": 0}
    rad = {"one_call": 0, "restored": 0}
    firings = both = differ = 0
    for i, s in enumerate(subs, 1):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        expanded = Chem.AddHs(mol)
        for rule in rules:
            for product in _iter_reaction_products(Chem.Mol(expanded), rule):
                firings += 1
                out = {}
                for tag, fn in (("one_call", contract_by_removing_only), ("restored", contract)):
                    try:
                        m = fn(product)
                    except Exception:
                        continue
                    out[tag] = Chem.MolToSmiles(m)
                    n[tag] += 1
                    rad[tag] += sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()) > 0
                if len(out) == 2:
                    both += 1
                    differ += out["one_call"] != out["restored"]
        print(f"  {i}/{len(subs)}  firings {firings}", flush=True)

    rep = {"config": {**_code_version(), "n_substrates": len(subs), "seed": args.seed,
                      "n_rules": len(rules), "population": "clean_test",
                      "note": "one_call is RemoveHs then sanitise; restored clears the "
                              "no-implicit mark and zeroes explicit counts first"},
           "firings": firings,
           "parseable": n, "carrying_an_unpaired_electron": rad,
           "radical_share": {k: round(rad[k] / max(n[k], 1), 4) for k in n},
           "products_only_the_restored_contraction_yields": n["restored"] - n["one_call"],
           "both_parsed": both, "both_parsed_and_differ": differ,
           "share_of_shared_products_that_differ": round(differ / max(both, 1), 4)}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\n  one call : {n['one_call']} parse, {rad['one_call']} carry an unpaired electron "
          f"({rep['radical_share']['one_call']})")
    print(f"  restored : {n['restored']} parse, {rad['restored']} carry an unpaired electron "
          f"({rep['radical_share']['restored']})")
    print(f"  the restored contraction yields "
          f"{rep['products_only_the_restored_contraction_yields']} more parseable products, and of "
          f"the {both} both produce, {differ} differ "
          f"({rep['share_of_shared_products_that_differ']})")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

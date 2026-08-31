#!/usr/bin/env python3
"""How much of the annotation a protonation-aware matching criterion could ever separate.

The Limitations bound what a configuration-aware criterion could cost by counting the references
that carry a stereogenic element their substrate does not. Protonation is the other layer the
declared criteria differ in and it had no such bound: the first-block criterion drops the
protonation layer along with stereochemistry and isotopes, so a thiol and its thiolate are one
structure under it, and nothing said how much of the annotation that could touch.

This counts it, on the same population and in the same form as the stereochemistry bound: how many
annotated metabolites carry a non-zero formal charge on any atom, split into net-neutral
zwitterions and net-charged species. It is a ceiling on what such a criterion could separate and
not an estimate of what it would cost, which is the same thing the stereochemistry number is.

    python scripts/typed_edit/reference_charge.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "reference_charge.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    source = ROOT / "results/test_references.json"
    refs = json.loads(source.read_text())
    pairs = [(s, m) for s, mets in refs.items() for m in mets]

    typed = charged = zwitter = net = 0
    substrates_with_one = set()
    by_element = Counter()
    for substrate, metabolite in pairs:
        mol = Chem.MolFromSmiles(metabolite)
        if mol is None:
            continue
        typed += 1
        charges = [(a.GetSymbol(), a.GetFormalCharge()) for a in mol.GetAtoms()]
        nonzero = [(s, q) for s, q in charges if q]
        if not nonzero:
            continue
        charged += 1
        substrates_with_one.add(substrate)
        for symbol, q in nonzero:
            by_element[f"{symbol}{'+' if q > 0 else '-'}"] += 1
        if sum(q for _, q in charges) == 0:
            zwitter += 1
        else:
            net += 1

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([source]),
        "question": ("how much of the annotation a criterion sensitive to the protonation layer "
                     "could separate, which is the protonation counterpart of the stereochemistry "
                     "bound the Limitations already give"),
        "population": {"annotated_pairs": len(pairs), "typed": typed,
                       "substrates": len(refs),
                       "source": "results/test_references.json, the evaluated test set"},
        "references_carrying_a_formal_charge": charged,
        "share": round(charged / max(typed, 1), 4),
        "of_those_net_neutral_zwitterions": zwitter,
        "of_those_net_charged": net,
        "substrates_with_at_least_one": len(substrates_with_one),
        "charged_atoms_by_element": dict(by_element.most_common()),
        "reading": (
            "A ceiling and not an estimate. These are the references whose identity a criterion "
            "that kept the protonation layer could in principle resolve differently from one that "
            "drops it; whether any verdict would move is a separate question the criterion sweep "
            "answers, and it answers no."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"typed {typed} of {len(pairs)} annotated metabolites over {len(refs)} substrates")
    print(f"  carrying a formal charge on any atom : {charged} ({report['share']:.2%})")
    print(f"    net-neutral zwitterions            : {zwitter}")
    print(f"    net-charged                        : {net}")
    print(f"  substrates with at least one         : {len(substrates_with_one)}")
    print("  commonest charged atoms: "
          + ", ".join(f"{k} {v}" for k, v in list(by_element.most_common(5))))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

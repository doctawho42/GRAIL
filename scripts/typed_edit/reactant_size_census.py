#!/usr/bin/env python3
"""How constrained the bank's templates are, which the bank's other censuses do not say.

The bank is characterised by count, by reaction type, by provenance, by support and by how much
of it is more than one enzymatic step. It is nowhere characterised by how specific its templates
are, and a template whose reactant side is two atoms will fire almost anywhere: for those, the
site discrimination is supplied by the learned stages rather than by the template, which is worth
saying in a paper whose selling point is that a prediction is rule-attributable.

The instrument is the reactant side's atom count, summed over components, read through the
reaction parser so multi-component templates are counted whole.

    python scripts/typed_edit/reactant_size_census.py
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

from _provenance import stamp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "reactant_size_census.json"))
    args = ap.parse_args()

    from rdkit import RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from grail_metabolism.utils.preparation import resolve_default_rule_bank

    bank = Path(resolve_default_rule_bank())
    lines = [line.strip() for line in bank.read_text().splitlines() if line.strip()]
    mined = {line.strip() for line in
             (ROOT / "grail_metabolism/resources/mined_only_v2.txt").read_text().splitlines()
             if line.strip()}

    sizes, per, by_half = Counter(), {}, {"mined": Counter(), "curated": Counter()}
    unparsed = 0
    for smirks in lines:
        try:
            reaction = AllChem.ReactionFromSmarts(smirks)
        except Exception:
            reaction = None
        if reaction is None or reaction.GetNumReactantTemplates() == 0:
            unparsed += 1
            continue
        n = sum(reaction.GetReactantTemplate(i).GetNumAtoms()
                for i in range(reaction.GetNumReactantTemplates()))
        sizes[n] += 1
        per[smirks] = n
        by_half["mined" if smirks in mined else "curated"][n] += 1

    total = sum(sizes.values())
    le2 = sum(v for k, v in sizes.items() if k <= 2)
    le3 = sum(v for k, v in sizes.items() if k <= 3)

    # how much of the worked example's own head comes from that tail
    case = json.loads((ROOT / "results" / "case_study_exhaustive.json").read_text())
    top = [c.get("rule") for c in case["candidates"][:20] if c.get("rule")]
    from_tail = sum(1 for r in top if per.get(r, 99) <= 3)

    report = {
        "provenance": stamp(__file__),
        "bank": str(bank.relative_to(ROOT)),
        "templates_parsed": total, "unparsed": unparsed,
        "reactant_atoms_at_most_two": le2,
        "reactant_atoms_at_most_two_share": round(le2 / max(total, 1), 4),
        "reactant_atoms_at_most_three": le3,
        "reactant_atoms_at_most_three_share": round(le3 / max(total, 1), 4),
        "histogram": {str(k): sizes.get(k, 0) for k in range(1, 16)},
        "by_half": {half: {str(k): c.get(k, 0) for k in range(1, 16)}
                    for half, c in by_half.items()},
        "worked_example_top20_from_three_or_fewer": from_tail,
        "instrument": ("atoms on the reactant side, summed over components, through the reaction "
                       "parser so multi-component templates are counted whole"),
        "reading": ("a template with two or three reactant atoms matches almost anywhere, so for "
                    "those the site discrimination comes from the learned stages and not from "
                    "the template; the selectivity bar in the mining procedure rejects templates "
                    "producing more than 200 products on average, which bounds this but does not "
                    "remove it"),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"{total} templates parsed, {unparsed} not")
    print(f"  reactant side of two atoms or fewer:   {le2:>5}  {le2 / total:.2%}")
    print(f"  reactant side of three atoms or fewer: {le3:>5}  {le3 / total:.2%}")
    print(f"  of the worked example's top 20, {from_tail} come from that tail")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

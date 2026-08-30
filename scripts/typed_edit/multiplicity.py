#!/usr/bin/env python3
"""Which verdicts survive a family-wise correction over the whole sweep, computed rather than said.

The paper reports per-comparison intervals and says so, and it also says which of its claimed
leads a family-wise correction would remove. That second sentence was written from inspection of
the intervals. It is a countable statement about a declared family and it should be counted.

The family is every GRAIL-arm-against-comparator contrast in the budget sweep: two arms, three
comparators, every budget reported. Holm's step-down procedure is applied to the two-sided
bootstrap p-values the same resamples produce, at the level the paper reads its intervals at, and
each cell is marked with whether it separates before the correction, after it, or neither. The
count of cells whose verdict the correction changes is the number the manuscript quotes.

    python scripts/typed_edit/multiplicity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

ALPHA = 0.05
ARMS = ("whole bank", "trained budget")
COMPARATORS = ("metatox", "sygma", "metapredictor")


def holm(pairs, alpha=ALPHA):
    """Holm's step-down: sort ascending, reject while p_(i) <= alpha / (m - i)."""
    ordered = sorted(pairs, key=lambda kv: kv[1])
    m = len(ordered)
    survives, still_rejecting = {}, True
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        if still_rejecting and p <= threshold:
            survives[name] = True
        else:
            still_rejecting = False
            survives[name] = False
    return survives, {name: alpha / (m - i) for i, (name, _) in enumerate(ordered)}


def main() -> int:
    dep = json.loads((ROOT / "results/deployment_table.json").read_text())
    contrasts = dep["contrasts"]
    budgets = sorted(contrasts, key=int)

    cells, missing_p = {}, []
    for k in budgets:
        for arm in ARMS:
            for comp in COMPARATORS:
                cell = contrasts[k].get(f"{arm} - {comp}")
                if cell is None:
                    continue
                if "p_bootstrap" not in cell:
                    missing_p.append(f"{arm} - {comp} at {k}")
                    continue
                cells[f"{arm} - {comp} @ {k}"] = cell
    if missing_p:
        raise SystemExit("re-run deployment_table.py: no bootstrap p-value for "
                         + ", ".join(missing_p[:3]))

    survives, thresholds = holm([(n, c["p_bootstrap"]) for n, c in cells.items()])

    rows, changed, claimed_and_lost = {}, [], []
    for name, cell in cells.items():
        before = cell["excludes_zero"]
        after = survives[name]
        rows[name] = {"gap": cell["gap"], "p": cell["p_bootstrap"],
                      "holm_threshold": round(thresholds[name], 6),
                      "separates_per_comparison": before, "separates_after_holm": after}
        if before != after:
            changed.append(name)
            if cell["gap"] > 0:
                claimed_and_lost.append(name)

    report = {
        "provenance": stamp(__file__),
        "family": (f"every GRAIL-arm-against-comparator contrast in the sweep: {len(ARMS)} arms "
                   f"by {len(COMPARATORS)} comparators by {len(budgets)} budgets"),
        "n_tests": len(cells),
        "alpha": ALPHA,
        "procedure": "Holm step-down on two-sided bootstrap p-values, B and seed as the intervals",
        "n_separating_per_comparison": sum(1 for r in rows.values()
                                           if r["separates_per_comparison"]),
        "n_separating_after_holm": sum(1 for r in rows.values() if r["separates_after_holm"]),
        "cells_whose_verdict_the_correction_changes": sorted(changed),
        "leads_the_correction_removes": sorted(claimed_and_lost),
        "cells": rows,
        "reading": (
            "The paper reads its verdicts from per-comparison intervals and says so. This says "
            "what the same data give under a correction over the whole family, so a reader who "
            "prefers that reading has it without recomputing anything. A verdict that changes is "
            "named; the rest stand under both."),
    }
    (ROOT / "results/multiplicity.json").write_text(json.dumps(report, indent=1))
    print(f"family of {report['n_tests']} tests at alpha {ALPHA}, Holm")
    print(f"  separate per comparison : {report['n_separating_per_comparison']}")
    print(f"  separate after Holm     : {report['n_separating_after_holm']}")
    for name in sorted(changed):
        print(f"  changes verdict: {name}  gap {rows[name]['gap']:+.4f}  p {rows[name]['p']}")
    print("wrote results/multiplicity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

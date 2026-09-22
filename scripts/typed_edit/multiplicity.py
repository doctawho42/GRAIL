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

ALPHA = 0.05
ARMS = ("whole bank", "trained budget")
# The family as declared, before the BioTransformer arm existed. A family fixed in advance is
# not widened after the fact, which is the whole of what declaring one buys.
COMPARATORS = ("metatox", "sygma", "metapredictor")
# Every comparator the sweep now reports a contrast against. The paper prints more tests than the
# declared family covers, and a correction over part of what is printed controls less than it
# appears to, so the wider family is computed beside the declared one and neither is hidden behind
# the other. The declared family is the one the paper's verdicts are read from.
COMPARATORS_REPORTED = ("metatox", "sygma", "metapredictor", "biotransformer", "gloryx",
                        "gloryxr_default", "gloryxr_strict")
# GLORYxR joined the same way, in both of its site-of-metabolism settings. This tuple is read
# against the sweep below and a comparator the sweep carries that this tuple lacks stops the run:
# a "wider family" that is narrower than what the paper prints reports a correction covering more
# than it does, which is the opposite of what computing it beside the declared one is for.
# GLORYx joined after the family was declared, as BioTransformer did. It is in the wider
# family computed here and not in the declared one, which names its three comparators and
# does not move; the verdicts the paper reports as corrected are still corrected over what
# was declared, and every cell outside it is read without a correction and said to be.


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
    # The deployment table was reachable only at its working path, so the correction this file
    # reports could not be recomputed against a corrected arm without overwriting the artifact.
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", default=str(ROOT / "results/deployment_table.json"))
    ap.add_argument("--out", default=str(ROOT / "results/multiplicity.json"))
    args = ap.parse_args()
    dep = json.loads(Path(args.deployment).read_text())
    contrasts = dep["contrasts"]
    budgets = sorted(contrasts, key=int)

    def collect(comparators):
        out, missing = {}, []
        for k in budgets:
            for arm in ARMS:
                for comp in comparators:
                    cell = contrasts[k].get(f"{arm} - {comp}")
                    if cell is None:
                        continue
                    if "p_bootstrap" not in cell:
                        missing.append(f"{arm} - {comp} at {k}")
                        continue
                    out[f"{arm} - {comp} @ {k}"] = cell
        if missing:
            raise SystemExit("re-run deployment_table.py: no bootstrap p-value for "
                             + ", ".join(missing[:3]))
        return out

    cells = collect(COMPARATORS)
    survives, thresholds = holm([(n, c["p_bootstrap"]) for n, c in cells.items()])

    # The same procedure over every contrast the paper prints. Only the count of surviving cells
    # and the ones the wider family costs are kept: this is a sensitivity on the declared family,
    # not a second set of verdicts.
    # Read against the sweep rather than trusted: a comparator the deployment table carries and
    # this tuple lacks would make the wider family narrower than what the paper prints.
    _carried = {p.split(" - ")[1] for row in contrasts.values() for p in row
                if p.startswith("whole bank - ")} - {"trained budget"}
    _missing = sorted(_carried - set(COMPARATORS_REPORTED))
    if _missing:
        sys.exit(f"REFUSING: deployment_table.json carries contrasts against "
                 f"{', '.join(_missing)} and COMPARATORS_REPORTED does not name them, so the "
                 f"wider family would be narrower than the set the paper prints.")
    wide = collect(COMPARATORS_REPORTED)
    wide_survives, _ = holm([(n, c["p_bootstrap"]) for n, c in wide.items()])
    lost_to_the_wider_family = sorted(
        n for n in cells if survives.get(n) and not wide_survives.get(n))

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
        "over_every_contrast_the_paper_prints": {
            "comparators": list(COMPARATORS_REPORTED),
            "n_tests": len(wide),
            "n_separating_after_holm": sum(1 for v in wide_survives.values() if v),
            "declared_family_cells_it_would_remove": lost_to_the_wider_family,
            "note": ("a sensitivity on the declared family and not a second set of verdicts: the "
                     "family was fixed before the BioTransformer arm existed and is not widened "
                     "after the fact, but the paper prints more tests than it covers and what "
                     "that costs is counted rather than left to the reader")},
        "cells": rows,
        "reading": (
            "The paper reads its verdicts from per-comparison intervals and says so. This says "
            "what the same data give under a correction over the whole family, so a reader who "
            "prefers that reading has it without recomputing anything. A verdict that changes is "
            "named; the rest stand under both."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"family of {report['n_tests']} tests at alpha {ALPHA}, Holm")
    w = report["over_every_contrast_the_paper_prints"]
    print(f"  over every printed contrast ({w['n_tests']} tests): "
          f"{w['n_separating_after_holm']} separate; it would remove "
          f"{len(w['declared_family_cells_it_would_remove'])} of the declared family's cells")
    print(f"  separate per comparison : {report['n_separating_per_comparison']}")
    print(f"  separate after Holm     : {report['n_separating_after_holm']}")
    for name in sorted(changed):
        print(f"  changes verdict: {name}  gap {rows[name]['gap']:+.4f}  p {rows[name]['p']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

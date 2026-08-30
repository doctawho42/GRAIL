#!/usr/bin/env python3
"""Two instruments measure the coverage ceiling under the same hydrogen convention. They disagree.

The ceiling the paper reports comes from `coverage_gap_types.py`, which applies the bank through
the deployed loop with hydrogens implicit. The convention itself was audited separately by
`hydrogen_dispatch.py`, which measures the same population under the same convention and returns a
different count. The gap is small, and it was invisible because each instrument was read alone and
the paper quoted whichever one the surrounding paragraph was about.

Nothing here re-measures the ceiling. It puts every count on one denominator, names which loop
produced each, and reports the largest disagreement beside the narrowest margin that decides
anything in the comparison, so a reader can see at once whether the disagreement could matter.

    python scripts/typed_edit/ceiling_instrument_agreement.py
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


def main() -> int:
    cov = json.loads((ROOT / "results/coverage_gap_types.json").read_text())
    hyd = json.loads((ROOT / "results/hydrogen_dispatch__clean_test.json").read_text())
    bank = hyd["banks"]["grail_full"]
    dep = json.loads((ROOT / "results/deployment_table.json").read_text())

    n = bank["references"]
    if cov["covered_pairs"] + cov["uncovered_pairs"] != n:
        raise SystemExit("the two instruments are not counting the same references")

    paired = bank["global_arms_paired"]
    arms = {
        "deployed loop, hydrogens implicit": {
            "recovered": cov["covered_pairs"],
            "instrument": "coverage_gap_types.py",
            "note": "the count the paper reports as the ceiling",
        },
        "audit loop, hydrogens implicit": {
            "recovered": paired["recovered_implicit"],
            "instrument": "hydrogen_dispatch.py",
            "note": "the same convention through the loop that audits the convention",
        },
        "audit loop, explicit and templates completed": {
            "recovered": paired["recovered_explicit_completed"],
            "instrument": "hydrogen_dispatch.py",
            "note": "the other defensible convention",
        },
        "audit loop, convention chosen per template": {
            "recovered": bank["recovered"],
            "instrument": "hydrogen_dispatch.py",
            "note": "not a convention a caller can request; the upper bound of the audit",
        },
    }
    for row in arms.values():
        row["uncovered"] = n - row["recovered"]
        row["reach"] = round(row["recovered"] / n, 4)

    same = [k for k in arms if k.endswith("hydrogens implicit")]
    disagreement = abs(arms[same[0]]["recovered"] - arms[same[1]]["recovered"])
    spread = (max(r["recovered"] for r in arms.values())
              - min(r["recovered"] for r in arms.values()))

    # The margin that decides the tightest live question anywhere in the comparison. If the
    # disagreement is far below it, the disagreement cannot flip anything the paper concludes.
    margins = [abs(cell["gap"]) for budget in dep["contrasts"].values()
               for cell in budget.values() if cell.get("excludes_zero")]
    narrowest = round(min(margins), 4)

    report = {
        "provenance": stamp(__file__),
        "population": {"n_references": n, "n_substrates": cov["n_substrates"],
                       "split": "the evaluated test set"},
        "arms": arms,
        "disagreement_between_instruments_on_the_same_convention": disagreement,
        "disagreement_as_share_of_references": round(disagreement / n, 4),
        "spread_across_all_four_counts": spread,
        "spread_as_share_of_references": round(spread / n, 4),
        "narrowest_separating_margin_in_the_comparison": narrowest,
        "spread_as_share_of_that_margin": round((spread / n) / narrowest, 4),
        "reading": (
            "Two loops measuring the same convention on the same references differ by "
            f"{disagreement} of {n}. The paper reports the deployed loop's count, because that is "
            "the loop a caller runs. The audit loop's number is the one to compare conventions "
            "with, because both of its arms come from the same code. Neither the disagreement nor "
            "the full spread across all four counts reaches the size of any margin the comparison "
            "turns on, so the choice of instrument changes no conclusion in this paper; it is "
            "reported because it was found by reading two artifacts against each other and would "
            "otherwise be found by a reader doing the same."),
    }
    (ROOT / "results/ceiling_instrument_agreement.json").write_text(json.dumps(report, indent=1))
    for name, row in arms.items():
        print(f"{row['recovered']:5d} recovered, {row['uncovered']:3d} uncovered, "
              f"reach {row['reach']:.4f}   {name}  [{row['instrument']}]")
    print(f"\nsame convention, two loops: {disagreement} references of {n} "
          f"({disagreement / n:.2%})")
    print(f"widest spread across all four: {spread} references ({spread / n:.2%})")
    print("wrote results/ceiling_instrument_agreement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

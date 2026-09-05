#!/usr/bin/env python3
"""This paper's thesis, tested on somebody else's published benchmark rather than on its own.

The argument here is that a recall figure whose output budget is undeclared orders methods by how
much they are allowed to emit. That is easy to demonstrate on one's own comparison and easy to
dismiss as a property of it. Gao et al. (J. Chem. Inf. Model. 2026, 66, 2918-2928, DOI
10.1021/acs.jcim.5c03045) published an independent benchmark of five metabolite predictors against
human radiolabelled ADME data for eleven drugs, reporting recall and precision per drug per tool
and concluding that there is a "trade-off between coverage and balanced accuracy". Their own tables
are enough to ask whether that trade-off is a property of the tools or of how many candidates each
was allowed to return.

Nothing here is re-run. Their Tables S12 and S13 are transcribed verbatim from the supporting
information, which ACS deposits openly under CC BY-NC 4.0, and everything else follows by
arithmetic that the reader can repeat:

  * the size of each drug's reference set is recovered as the smallest denominator consistent with
    every tool's recall on that drug, and checked against the one the supporting information states
    outright (Table S2 lists Pamiparib's eight metabolites by name);
  * the number of candidates a tool returned for a drug is then true positives over precision,
    since precision is those true positives over what was emitted;
  * and the question is whether recall across the resulting cells is a function of that count.

If it is, their ordering is a budget ordering, established independently of this work and on data
this work never touched.

    python scripts/typed_edit/external_budget_confound.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

SOURCE = {
    "citation": ("Gao, L.; Yan, S.; Feng, K.; Liu, H.; Zhang, Z.; Diao, X. Benchmarking In Silico "
                 "Metabolite Prediction Tools against Human Radiolabeled ADME Data for "
                 "Small-Molecule Drugs. J. Chem. Inf. Model. 2026, 66, 2918-2928"),
    "doi": "10.1021/acs.jcim.5c03045",
    "tables": "Supporting Information Tables S12 (recall) and S13 (precision), pages S28-S29",
    "supporting_information": ("openly deposited by ACS under CC BY-NC 4.0, DOI "
                              "10.1021/acs.jcim.5c03045.s001; sha256 of the PDF transcribed from "
                              "is recorded below"),
    "si_sha256_16": "907aef9adad03660",
}

DRUGS = ["Sotorasib", "Pamiparib", "Encorafenib", "Fovinaciclib", "Rezafungin", "Isavuconazole",
         "LY3202626", "Iberdomide", "Opicapone", "Tinengotinib", "Nemiralisib"]

TOOL = {"SM": "SyGMa", "GX": "GLORYx", "BT": "BioTransformer", "MP": "MetaPredictor",
        "MT": "MetaTrans", "CM": "consensus of the five"}

# Table S12, recall. Transcribed verbatim; None is the paper's "NC, not computable".
RECALL = {
    "SM": [0.400, 0.500, 0.333, 0.400, 1.000, 0.429, 0.833, 0.417, 0.250, 0.455, 0.571],
    "GX": [0.600, 0.375, 0.583, 0.400, 0.667, 0.571, 0.500, 0.417, 0.250, 0.364, 0.429],
    "BT": [0.200, 0.250, 0.417, 0.300, 0.667, 0.429, 0.167, 0.000, 0.063, 0.273, 0.429],
    "MP": [0.000, 0.250, 0.250, 0.100, 0.000, 0.286, 0.333, 0.167, 0.063, 0.182, 0.000],
    "MT": [0.000, 0.250, 0.083, 0.300, 0.000, 0.143, 0.167, 0.167, 0.125, 0.273, 0.143],
    "CM": [0.200, 0.375, 0.417, 0.300, 0.667, 0.429, 0.333, 0.333, 0.063, 0.364, 0.429],
}
# Table S13, precision.
PRECISION = {
    "SM": [0.009, 0.095, 0.041, 0.024, 0.004, 0.015, 0.044, 0.064, 0.037, 0.046, 0.035],
    "GX": [0.057, 0.136, 0.149, 0.070, 0.012, 0.061, 0.081, 0.109, 0.148, 0.121, 0.077],
    "BT": [0.050, 0.222, 0.278, 0.115, 0.043, 0.143, 0.053, 0.000, 0.125, 0.300, 0.143],
    "MP": [0.000, 0.286, 0.375, 0.143, None, 0.200, 0.286, 0.286, 0.143, 0.154, None],
    "MT": [0.000, 0.100, 0.048, 0.091, 0.000, 0.030, 0.038, 0.056, 0.053, 0.115, 0.048],
    "CM": [0.063, 0.333, 0.625, 0.188, 0.111, 0.214, 0.182, 0.571, 0.125, 0.400, 0.200],
}
# The one reference-set size the supporting information states outright: Table S2 lists
# Pamiparib's metabolites M1, M2, M3, M4, M5, M8, M10 and M25. The recovery below must reproduce
# it, or the recovery is not to be trusted for the other ten.
STATED = {"Pamiparib": 8}


def recover_reference_sizes():
    """The smallest N under which every reported recall for a drug is a whole number over N.

    Recall is hits over reference-set size, and the paper prints it to three decimals, so N is
    identified by requiring every tool's value on that drug to be k/N for an integer k. Rounding
    at three decimals leaves a tolerance, and a multiple of a valid N is also valid, so the
    smallest is taken and the result is checked against the size the paper states for one drug.
    """
    sizes = {}
    for j, drug in enumerate(DRUGS):
        values = [RECALL[t][j] for t in RECALL if RECALL[t][j] is not None]
        for n in range(1, 61):
            if all(abs(v * n - round(v * n)) <= n * 5e-4 + 1e-9 for v in values):
                sizes[drug] = n
                break
        else:
            sizes[drug] = None
    return sizes


def spearman(xs, ys):
    """Rank correlation, written out rather than imported, with ties averaged."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> int:
    sizes = recover_reference_sizes()

    bad = {d: (sizes[d], STATED[d]) for d in STATED if sizes.get(d) != STATED[d]}
    if bad:
        print("REFUSING: the reference-set recovery does not reproduce the size the supporting "
              "information states: " + ", ".join(f"{d}: {got} against {want}"
                                                 for d, (got, want) in bad.items()), file=sys.stderr)
        return 1
    if any(v is None for v in sizes.values()):
        print("REFUSING: no consistent reference-set size for "
              + ", ".join(d for d, v in sizes.items() if v is None), file=sys.stderr)
        return 1

    # What each tool returned, from its own precision: precision is hits over emitted.
    cells = []
    for t in RECALL:
        for j, drug in enumerate(DRUGS):
            r, p = RECALL[t][j], PRECISION[t][j]
            if p is None or p == 0 or r == 0:
                continue                      # no hits, so the emission is not identified
            n = sizes[drug]
            hits = r * n
            cells.append({"tool": TOOL[t], "code": t, "drug": drug, "reference_set": n,
                          "recall": r, "precision": p, "hits": round(hits),
                          "emitted": round(hits / p, 1)})

    emitted, recall = [c["emitted"] for c in cells], [c["recall"] for c in cells]
    rho_all = spearman([math.log(e) for e in emitted], recall)

    # Per tool, the mean of what it emits: this is the axis the benchmark never declares.
    by_tool = {}
    for t, name in TOOL.items():
        mine = [c for c in cells if c["code"] == t]
        if not mine:
            continue
        by_tool[name] = {
            "cells": len(mine),
            "mean_emitted": round(sum(c["emitted"] for c in mine) / len(mine), 1),
            "mean_recall": round(sum(c["recall"] for c in mine) / len(mine), 4),
            "mean_precision": round(sum(c["precision"] for c in mine) / len(mine), 4),
        }
    order_by_emission = sorted(by_tool, key=lambda k: -by_tool[k]["mean_emitted"])
    order_by_recall = sorted(by_tool, key=lambda k: -by_tool[k]["mean_recall"])

    # Within a drug the reference set is fixed, so the comparison across tools there is clean.
    within = []
    for drug in DRUGS:
        here = [c for c in cells if c["drug"] == drug]
        if len(here) >= 4:
            within.append(spearman([math.log(c["emitted"]) for c in here],
                                   [c["recall"] for c in here]))
    positive = sum(1 for v in within if v > 0)

    report = {
        "provenance": stamp(__file__),
        "question": ("whether the ordering an independent benchmark reports is a property of the "
                     "tools or of how many candidates each was allowed to emit"),
        "source": SOURCE,
        "nothing_was_re_run": ("recall and precision are the published values; the reference-set "
                               "sizes and the emitted counts follow from them by arithmetic"),
        "reference_set_sizes": sizes,
        "reference_set_recovery_checked_against": STATED,
        "cells": cells,
        "by_tool": by_tool,
        "ordering_by_mean_emission": order_by_emission,
        "ordering_by_mean_recall": order_by_recall,
        "orderings_agree": order_by_emission == order_by_recall,
        "spearman_recall_against_log_emitted": round(rho_all, 4),
        "n_cells": len(cells),
        "within_drug_spearman": [round(v, 4) for v in within],
        "within_drug_positive": positive,
        "within_drug_n": len(within),
        "reading": ("A benchmark that does not hold output size fixed orders methods partly by "
                    "output size. This does not correct their measurement and does not need their "
                    "reference structures, which are published only as drawings; it says what "
                    "their published numbers already contain."),
    }
    out = ROOT / "results" / "external_budget_confound.json"
    out.write_text(json.dumps(report, indent=1))

    print(f"{len(cells)} cells over {len(DRUGS)} drugs and {len(by_tool)} arms")
    print(f"{'arm':22s} {'emitted':>9s} {'recall':>8s} {'precision':>10s}")
    for name in order_by_emission:
        v = by_tool[name]
        print(f"  {name:20s} {v['mean_emitted']:9.1f} {v['mean_recall']:8.4f} "
              f"{v['mean_precision']:10.4f}")
    print(f"\nSpearman(recall, log emitted) = {rho_all:+.4f} over {len(cells)} cells")
    print(f"within a drug it is positive in {positive} of {len(within)}")
    print(f"orderings by emission and by recall agree: {report['orderings_agree']}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

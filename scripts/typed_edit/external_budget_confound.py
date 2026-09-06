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
information, which ACS deposits openly under CC BY-NC 4.0, and what each tool emitted is counted
from the prediction files the same authors deposited on Zenodo under CC BY 4.0.

The counting matters, and an earlier version of this analysis got it wrong in a way worth recording
so it is not reintroduced. Output size can also be recovered arithmetically, as true positives over
precision, since precision is those positives over what was emitted. But true positives are recall
times the reference-set size, so

    log(emitted) = log(recall) + log(R) - log(precision),

and recall enters the reconstructed axis explicitly and positively. Correlating recall against that
axis correlates recall partly with itself: on these numbers a permutation of precision within a
drug returns the observed rank correlation with p = 0.28. The reconstruction is kept here, but only
as a check on the reference-set recovery, and the association is measured against emission counted
from the deposited files, where recall does not enter the axis at all.

What the permutation preserves is worth stating exactly, because an earlier version of this
docstring and of the manuscript said "both marginals" and that is false. Shuffling emission
independently within each drug preserves every drug's multiset of emissions and every recall in the
table. It does not preserve a tool's emission profile: on this grid the observed per-tool means run
from 15.8 to 187.3 and one permutation returns a set of means with no relation to them. The null is
therefore "within a drug, which arm emitted what is arbitrary", which is the association under
test, and nothing more.

That null cannot separate a within-tool effect from a between-tool one, and with six arms most of
the emission variance is between them. So a second statistic is reported beside it: the rank
correlation of recall against log emission after the additive tool and drug effects are removed
from both. If it is near zero the association lives between the arms rather than inside them, and
that is the level a benchmark orders at, so the claim does not need the separation; if it is not
near zero the association survives inside the arms as well. Either answer is reportable, and the
number is printed rather than the conclusion assumed.

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

# The same authors' prediction files, deposited separately. Reading them turns output size from a
# quantity reconstructed through recall into one that is counted.
DEPOSIT = {"doi": "10.5281/zenodo.17878495", "licence": "CC BY 4.0",
           "what": "the input scripts and per-tool output files of that benchmark"}
RAW = ROOT / "artifacts" / "external" / "gao2026"
# their file naming, which is not the paper's naming for one drug
STEM = {"Sotorasib": "Sotorasib", "Pamiparib": "Pamiparib", "Encorafenib": "Encorafenib",
        "Fovinaciclib": "FCN-437c", "Rezafungin": "Rezafungin",
        "Isavuconazole": "Isavuconazole", "LY3202626": "LY3202626", "Iberdomide": "Iberdomide",
        "Opicapone": "Opicapone", "Tinengotinib": "Tinengotinib", "Nemiralisib": "Nemiralisib"}


def _sygma(stem):
    """SyGMa's output, a literal list of [SMILES, score] pairs."""
    import ast
    return [r[0] for r in ast.literal_eval((RAW / f"{stem}.txt").read_text(errors="ignore").strip())]


def _biotransformer(stem):
    import csv
    with open(RAW / f"{stem}.csv", newline="", errors="ignore") as fh:
        return [r["SMILES"] for r in csv.DictReader(fh) if r.get("SMILES")]


def _gloryx(stem):
    import csv
    import glob as _g
    hit = _g.glob(str(RAW / f"{stem}-gloryx-*.csv"))
    if not hit:
        return None
    with open(hit[0], newline="", errors="ignore") as fh:
        return [r["metabolite_smiles"] for r in csv.DictReader(fh) if r.get("metabolite_smiles")]


def _metapredictor(stem):
    p = RAW / f"{stem}_MP.txt"
    if not p.exists():
        return None
    return [l.strip() for l in p.read_text(errors="ignore").splitlines() if l.strip()]


COUNTERS = {"SyGMa": _sygma, "BioTransformer": _biotransformer, "GLORYx": _gloryx,
            "MetaPredictor": _metapredictor}


def counted_emission():
    """What each tool actually returned per drug, deduplicated, from the deposited files.

    Deduplicated because a recall or precision figure is over distinct structures; the raw count is
    kept beside it so the difference is visible rather than assumed away.
    """
    import hashlib

    if not RAW.is_dir():
        return None, {}
    out, digests = {}, {}
    for drug, stem in STEM.items():
        for tool, fn in COUNTERS.items():
            try:
                got = fn(stem)
            except Exception:
                got = None
            if got is None:
                continue
            out[(drug, tool)] = {"returned": len(got), "distinct": len(set(got))}
    for f in sorted(RAW.iterdir()):
        if f.is_file():
            digests[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out, digests


def permutation_p_reconstructed(cells, sizes, n_perm=10000, seed=0):
    """The same test applied to the reconstructed axis, which is how its emptiness is shown.

    Precision is shuffled within a drug and hits stay where they are, so the association between
    what a tool found and what it emitted is broken while both marginals survive. Under that null
    the reconstructed correlation is still positive, because recall is inside the axis.
    """
    import random

    by_drug = {}
    for i, c in enumerate(cells):
        by_drug.setdefault(c["drug"], []).append(i)
    rec = [c["recall"] for c in cells]
    hits = [c["recall"] * sizes[c["drug"]] for c in cells]
    prec = [c["precision"] for c in cells]
    obs = spearman(rec, [math.log(h / p) for h, p in zip(hits, prec)])
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        shuffled = list(prec)
        for idx in by_drug.values():
            vals = [prec[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                shuffled[i] = v
        if spearman(rec, [math.log(h / p) for h, p in zip(hits, shuffled)]) >= obs:
            ge += 1
    return ge / n_perm


def permutation_p(rows, n_perm=10000, seed=0):
    """Shuffle emission within a drug, holding recall where it is.

    Within a drug rather than across: drugs differ in how hard they are and in how large their
    reference set is, and a shuffle across drugs would break that too and answer a question nobody
    asked.

    What survives the shuffle is each drug's multiset of emissions and every recall in the table.
    What does NOT survive is a tool's emission profile, since the permutations are independent
    across drugs: a tool that emits the most everywhere lands anywhere afterwards. Calling this
    "preserving both marginals" was wrong and `permutation_destroys_the_tool_profile` below
    measures how wrong. The null is that within a drug, which arm emitted what is arbitrary.
    """
    import random

    by_drug = {}
    for i, (drug, _tool, _r, _e) in enumerate(rows):
        by_drug.setdefault(drug, []).append(i)
    rec = [r for _d, _t, r, _e in rows]
    emi = [math.log(e) for _d, _t, _r, e in rows]
    obs = spearman(rec, emi)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        shuffled = list(emi)
        for idx in by_drug.values():
            vals = [emi[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                shuffled[i] = v
        if spearman(rec, shuffled) >= obs:
            hits += 1
    return obs, hits / n_perm


def permutation_destroys_the_tool_profile(rows, n_perm=200, seed=0):
    """How far the within-drug shuffle moves each tool's mean emission, since it does move it.

    The observed per-tool means are the thing a reader would assume a null "preserving both
    marginals" holds fixed. It does not: the permutations are independent across drugs, so under
    the null every tool's expected mean emission is the same grand mean. This reports the observed
    profile, the profile after permutation, and the rank correlation between them, so the corrected
    sentence in the manuscript rests on a number rather than on an argument.
    """
    import random

    tools = sorted({t for _d, t, _r, _e in rows})
    by_drug = {}
    for i, (drug, _t, _r, _e) in enumerate(rows):
        by_drug.setdefault(drug, []).append(i)
    emi = [e for _d, _t, _r, e in rows]

    def profile(vec):
        return [sum(vec[i] for i, (_d, t, _r, _e) in enumerate(rows) if t == tool)
                / max(1, sum(1 for _d, t, _r, _e in rows if t == tool)) for tool in tools]

    observed = profile(emi)
    rng = random.Random(seed)
    agreements, permuted_example = [], None
    for n in range(n_perm):
        shuffled = list(emi)
        for idx in by_drug.values():
            vals = [emi[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                shuffled[i] = v
        if permuted_example is None:
            permuted_example = profile(shuffled)
        agreements.append(spearman(observed, profile(shuffled)))
    return {
        "tools": tools,
        "observed_mean_emission": [round(v, 1) for v in observed],
        "one_permutation_mean_emission": [round(v, 1) for v in permuted_example],
        "mean_rank_correlation_with_the_observed_profile": round(
            sum(agreements) / len(agreements), 4),
        "permutations": n_perm,
        "reading": ("a null that preserved the tool marginal would return the observed profile "
                    "every time and a rank correlation of 1; this does not, which is why the "
                    "manuscript must not say the permutation preserves both marginals"),
    }


def _two_way_residuals(values, keys_a, keys_b, iterations=200, tol=1e-12):
    """Remove additive row and column effects from an unbalanced grid, by alternating demeaning.

    An unbalanced two-way additive fit has no closed form in one pass, but subtracting group means
    alternately converges to the least-squares one, and on a grid this size it converges in a few
    dozen sweeps. Written without numpy so this script keeps the dependency profile it had.
    """
    res = list(values)
    for _ in range(iterations):
        before = list(res)
        for keys in (keys_a, keys_b):
            groups = {}
            for i, k in enumerate(keys):
                groups.setdefault(k, []).append(i)
            for idx in groups.values():
                m = sum(res[i] for i in idx) / len(idx)
                for i in idx:
                    res[i] -= m
        if max(abs(a - b) for a, b in zip(before, res)) < tol:
            break
    return res


def residual_association(rows):
    """The association that is left once the arms and the drugs differ only in what they are.

    A rank correlation over a grid of six arms and eleven drugs is dominated by whichever axis
    varies most, and here that is the arm: the arms differ in mean emission by more than a factor
    of ten. Removing the additive arm and drug effects from both recall and log emission leaves the
    part of the association that lives INSIDE an arm. Near zero means the association is between
    the arms, which is the level at which a benchmark orders and therefore still the level the
    claim is about; away from zero means it survives inside them as well.
    """
    rec = [r for _d, _t, r, _e in rows]
    emi = [math.log(e) for _d, _t, _r, e in rows]
    tools = [t for _d, t, _r, _e in rows]
    drugs = [d for d, _t, _r, _e in rows]
    r_res = _two_way_residuals(rec, tools, drugs)
    e_res = _two_way_residuals(emi, tools, drugs)
    return {
        "spearman_raw": round(spearman(rec, emi), 4),
        "spearman_after_removing_the_arm_effect_and_the_drug_effect": round(
            spearman(r_res, e_res), 4),
        "n_cells": len(rows),
        "arms": len(set(tools)),
        "drugs": len(set(drugs)),
        "how": ("additive arm and drug effects removed from recall and from log emission by "
                "alternating demeaning, then Spearman of the residuals"),
    }


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
    # Reported, and reported as what it is: recall enters this axis by construction, so its
    # permutation null sits high and the figure is not evidence of anything on its own.
    rho_reconstructed = spearman([math.log(e) for e in emitted], recall)

    # The axis that is measured rather than reconstructed.
    counted, digests = counted_emission()
    if not counted:
        print(f"REFUSING: {RAW.relative_to(ROOT)} holds none of the deposited prediction files, "
              f"so emission can only be reconstructed through recall, which is the thing this "
              f"analysis exists to avoid. The deposit is {DEPOSIT['doi']}.", file=sys.stderr)
        return 1
    rec_by = {(c["drug"], c["tool"]): c["recall"] for c in cells}
    measured = [(d, t, rec_by[(d, t)], v["distinct"])
                for (d, t), v in sorted(counted.items()) if (d, t) in rec_by]
    rho_counted, p_counted = permutation_p(measured)
    p_reconstructed = permutation_p_reconstructed(cells, sizes)
    profile_effect = permutation_destroys_the_tool_profile(measured)
    residual = residual_association(measured)

    # Which of the full grid's cells carry a counted emission, and why the others do not. Two
    # reasons, and they are different: an arm whose predictions were never deposited cannot be
    # counted at all, while an arm that was deposited can still be missing one drug's file.
    arms_all = sorted({c["code"] for c in cells})
    have = {(d, t) for d, t, _r, _e in measured}
    NO_ARM = "this arm's predictions are not in the deposit, so its emission cannot be counted"
    NO_FILE = "the arm is in the deposit but has no file for this drug"
    NO_RECALL = ("the deposit holds this arm's file for this drug, so the emission IS counted, but "
                 "the published tables report no recall for the pair, so it cannot enter a "
                 "correlation")
    census = {"grid": len(arms_all) * len(DRUGS), "counted_and_correlatable": len(measured),
              "absent": []}
    for code in arms_all:
        name = TOOL[code]
        for drug in DRUGS:
            if (drug, name) in have:
                continue
            if name not in COUNTERS:
                reason = NO_ARM
            elif (drug, name) in counted:
                reason = NO_RECALL
            else:
                reason = NO_FILE
            census["absent"].append({"drug": drug, "arm": name, "reason": reason})
    for tag, reason in (("absent_because_the_arm_was_never_deposited", NO_ARM),
                        ("absent_because_the_file_is_missing", NO_FILE),
                        ("absent_because_no_recall_is_published", NO_RECALL)):
        census[tag] = sum(1 for a in census["absent"] if a["reason"] == reason)
    census["arms_never_deposited"] = sorted({a["arm"] for a in census["absent"]
                                             if a["reason"] == NO_ARM})
    census["counted_cells_in_total"] = len(counted)

    # The spread and the correlation are over different cell sets, and quoting them side by side
    # without saying so is the defect this paper is about. The spread is a count and can use every
    # counted cell; the correlation needs a published recall too. Both grains are reported.
    per_tool_correlatable = {}
    for tool in COUNTERS:
        vals = [e for _d, t, _r, e in measured if t == tool]
        if vals:
            per_tool_correlatable[tool] = round(sum(vals) / len(vals), 1)


    # And the plain observation, which needs no correlation at all.
    per_tool_counted = {}
    for tool in COUNTERS:
        vals = [v["distinct"] for (_d, t), v in counted.items() if t == tool]
        if vals:
            per_tool_counted[tool] = round(sum(vals) / len(vals), 1)
    widest = max(per_tool_counted.values())
    narrowest = min(per_tool_counted.values())

    within_counted = []
    for drug in DRUGS:
        here = [(r, e) for d, _t, r, e in measured if d == drug]
        if len(here) >= 3 and len({x[0] for x in here}) > 1 and len({x[1] for x in here}) > 1:
            within_counted.append(spearman([x[0] for x in here],
                                           [math.log(x[1]) for x in here]))

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

    # Does the reconstruction reproduce the count, per arm, with numbers rather than a verdict --
    # and does the arm it fails on carry the association?
    agreement = {}
    for tool, counted_mean in per_tool_counted.items():
        recon = by_tool.get(tool, {}).get("mean_emitted")
        if recon is None:
            continue
        agreement[tool] = {"counted_mean": counted_mean, "reconstructed_mean": recon,
                           "ratio": round(recon / counted_mean, 2) if counted_mean else None}
    worst = (max(agreement, key=lambda k: abs(math.log(agreement[k]["ratio"] or 1)))
             if agreement else None)
    without_worst = [row for row in measured if row[1] != worst]
    rho_without, p_without = (permutation_p(without_worst) if len(without_worst) >= 6
                              else (None, None))

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
        "nothing_was_re_run": ("recall and precision are the published values and the emission is "
                               "counted from the authors' own deposited prediction files"),
        "deposit": DEPOSIT,
        "deposited_file_digests": digests,
        "reference_set_sizes": sizes,
        "reference_set_recovery_checked_against": STATED,
        "cells": cells,
        "by_tool": by_tool,
        "ordering_by_mean_emission": order_by_emission,
        "ordering_by_mean_recall": order_by_recall,
        "orderings_agree": order_by_emission == order_by_recall,

        # The plain observation, which is a count and needs no model. The spread is over ARM MEANS,
        # not over cells: an earlier version of this analysis quoted a cell-level spread and the two
        # are different quantities, so the grain is named here rather than left to the reader.
        "counted_emission_per_tool": per_tool_counted,
        "counted_spread": round(widest / narrowest, 1),
        "counted_spread_grain": ("ratio of the largest to the smallest per-arm MEAN counted "
                                 "emission, over the arms whose predictions were deposited; an "
                                 "earlier version of this analysis quoted a cell-level spread, "
                                 "which is a different quantity"),
        "counted_widest": widest,
        "counted_narrowest": narrowest,
        "n_counted_cells": len(measured),
        # The same means over exactly the cells the correlation uses, since one counted cell has no
        # published recall and so sits in the spread but not in the association.
        "counted_emission_per_tool_over_the_correlated_cells": per_tool_correlatable,
        "counted_spread_over_the_correlated_cells": round(
            max(per_tool_correlatable.values()) / min(per_tool_correlatable.values()), 1),

        # Which cells of the full grid carry a counted emission, and why the rest do not. A
        # correlation over 40 of 66 cells is only readable if the 26 are accounted for.
        "counted_cell_census": census,
        # And the cells themselves, so the correlation below is recomputable from this file rather
        # than only by re-running the counter against the deposit.
        "counted_cells": [{"drug": d, "tool": t, "recall": r, "counted_emitted": e}
                          for d, t, r, e in measured],

        # The association, measured on the counted axis and tested against a null that shuffles
        # emission within a drug.
        "spearman_recall_against_counted_emission": round(rho_counted, 4),
        "permutation_p": round(p_counted, 4),
        "permutations": 10000,
        # What that null does and does not hold fixed, measured rather than asserted.
        "permutation_null_and_the_arm_profile": profile_effect,
        # The association that survives once the arms and the drugs differ only in what they are.
        "association_within_the_arms": residual,
        "within_drug_spearman_counted": [round(v, 4) for v in within_counted],
        "within_drug_counted_positive": sum(1 for v in within_counted if v > 0),
        "within_drug_counted_n": len(within_counted),

        # The reconstructed axis, kept as a check on the reference-set recovery and not used as
        # evidence: recall enters it by construction, so its own null sits high.
        "reconstructed_axis": {
            "spearman_recall_against_log_emitted": round(rho_reconstructed, 4),
            "why_it_is_not_evidence": ("emitted is hits over precision and hits is recall times "
                                       "the reference set, so recall enters the axis explicitly "
                                       "and positively; a permutation of precision within a drug "
                                       "returns this correlation with p = 0.28"),
            "permutation_p": round(p_reconstructed, 4),
            "n_cells": len(cells),
            "within_drug_spearman": [round(v, 4) for v in within],
            "within_drug_positive": positive,
            "within_drug_n": len(within),
            "agreement_with_the_count": agreement,
            "agreement_worst_arm": worst,
            "counted_association_without_the_worst_arm": {
                "spearman": None if rho_without is None else round(rho_without, 4),
                "permutation_p": None if p_without is None else round(p_without, 4),
                "n_cells": len(without_worst),
                "why": ("the reconstruction and the count disagree most on this arm, so the "
                        "association is recomputed without it: if the counted figure depends on "
                        "the arm the reconstruction cannot reproduce, that dependence is the "
                        "result rather than the association"),
            },
        },
        "reading": ("A benchmark that does not hold output size fixed orders methods partly by "
                    "output size. The spread is a count and stands on its own; the association is "
                    "measured against emission counted from the authors' deposited predictions, "
                    "so recall does not enter the axis, and it is tested against a permutation "
                    "null rather than read off a positive sign."),
    }
    out = ROOT / "results" / "external_budget_confound.json"
    out.write_text(json.dumps(report, indent=1))

    print(f"{len(cells)} cells over {len(DRUGS)} drugs and {len(by_tool)} arms")
    print(f"{'arm':22s} {'emitted':>9s} {'recall':>8s} {'precision':>10s}")
    for name in order_by_emission:
        v = by_tool[name]
        print(f"  {name:20s} {v['mean_emitted']:9.1f} {v['mean_recall']:8.4f} "
              f"{v['mean_precision']:10.4f}")
    print(f"\ncounted emission per tool: {per_tool_counted}")
    print(f"  spread {widest / narrowest:.1f}x, from {narrowest} to {widest}")
    print(f"\nmeasured axis: Spearman = {rho_counted:+.4f} over {len(measured)} counted cells, "
          f"permutation p = {p_counted:.4f}")
    print(f"  within a drug positive in {sum(1 for v in within_counted if v > 0)} of "
          f"{len(within_counted)}")
    print(f"\nreconstructed axis (kept as a check, not as evidence): "
          f"Spearman = {rho_reconstructed:+.4f} over {len(cells)} cells")
    print(f"orderings by emission and by recall agree: {report['orderings_agree']}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

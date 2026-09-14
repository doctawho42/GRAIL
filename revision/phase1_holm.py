#!/usr/bin/env python3
"""Phase 1 item 4: the family-wise correction, recomputed over the cells T_main actually holds.

Writes revision/T_holm.json. Three families are reported and none is hidden behind another:

    declared            what the manuscript quotes: 2 GRAIL arms x 3 comparators x 9 budgets,
                        on the comparison set, under the tautomer criterion. 54 tests.
    with_evaluated1170  the same criterion and comparators, plus every cell of those contrasts
                        that the whole evaluated set can support.
    union_of_t_main     every contrast the re-tabulation admits: both populations, all six
                        matching criteria, every comparator present on the population.

The declared family is reproduced first. A recomputation that cannot return the artifact's own
counts is not evidence about a wider family, so `recompute("declared")` is gated against
results/multiplicity.json in revision/tests.

Why the contrasts are computed here rather than read. scripts/typed_edit/multiplicity.py does not
compute anything: it reads the stored `p_bootstrap` of each cell from results/deployment_table.json
and runs Holm over those. That table exists only on the comparison set, so there is no stored
p-value for any whole-test-set cell and the widening cannot be done by reading. The contrasts are
therefore rebuilt from per-substrate hits, through the arm assembly the reproduction gate
certifies, with the p-value defined exactly as the producer defines it:

    below = #(bt <= 0), above = #(bt >= 0), p = min(1, 2 * (min(below, above) + 1) / (B + 1))

over the same bootstrap replicates the interval is read from, B = 10,000 at seed 0, one shared
resample matrix per population so the cells of a family are comparable.

    python revision/phase1_holm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "revision"), str(ROOT / "scripts"),
           str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import phase1_tmain as T  # noqa: E402

ALPHA = 0.05
ARMS = ("whole bank", "trained budget")
# The family as the artifact declares it, fixed before the BioTransformer and GLORYx arms existed.
DECLARED_COMPARATORS = ("metatox", "sygma", "metapredictor")
# Every comparator the manuscript prints a contrast against on the comparison set. This is the
# family results/multiplicity.json reports under "over_every_contrast_the_paper_prints": five
# comparators by two arms by nine budgets, ninety tests. It is recomputed here because it is the
# widening the paper actually quotes, and because its size coincides with the two-population
# family below -- ninety either way, over different cells. Coincident counts are why the tests
# compare membership rather than totals.
PAPER_COMPARATORS = ("metatox", "sygma", "metapredictor", "biotransformer", "gloryx")
DECLARED_CRITERION = "inchikey_tautomer"
DECLARED_POPULATION = "comparison291"
OUT = ROOT / "revision" / "T_holm.json"


def holm_reject(pvalues, alpha=ALPHA):
    """Holm's step-down, returned in the order the p-values were given.

    Sort ascending, compare the i-th against alpha / (m - i), and stop at the first failure: every
    test after it is retained however small its p-value. This is the procedure
    scripts/typed_edit/multiplicity.py implements, and the correction the manuscript quotes is that
    one, so the arithmetic here has to be identical rather than merely similar.
    """
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(order)
    out = [False] * m
    still = True
    for rank, i in enumerate(order):
        if still and pvalues[i] <= alpha / (m - rank):
            out[i] = True
        else:
            still = False
    return out


def decidability(n_tests, n_boot=None, alpha=ALPHA):
    """Whether a family of this size can reject anything at all, before any data is looked at.

    A pure function of the family size, the number of resamples and alpha, so it is checkable
    without computing a contrast. The p-value is 2*(min(below, above)+1)/(B+1), which cannot fall
    below 2/(B+1); Holm's strictest threshold is alpha/m. When that floor exceeds the threshold, no
    cell can be rejected however large its effect, and a bare "0 survive" would read as evidence
    evaporating rather than as a test with no resolution to decide with.
    """
    n_boot = T.N_BOOT if n_boot is None else n_boot
    floor = 2.0 / (n_boot + 1.0)
    strictest = alpha / max(n_tests, 1)
    return {
        "smallest_attainable_p": round(floor, 8),
        "holm_strictest_threshold": round(strictest, 8),
        "family_can_reject_at_all": bool(floor <= strictest),
        "n_boot_needed_for_one_rejection": (None if floor <= strictest
                                            else int(2.0 * n_tests / alpha - 1.0)),
    }


def _population_cells(population, criterion, comparators):
    """Every GRAIL-arm-against-comparator contrast on one population under one criterion.

    Returns {cell_name: {gap, ci95, p, separates_per_comparison}}. A comparator with no prediction
    file on this population yields no cells at all, rather than a contrast against an empty list:
    an absent comparator is not a beaten comparator, which is the defect scripts/_contrast.py was
    written to refuse.
    """
    from bank_without_selection import _key as tautkey

    subs, ref_smiles, (big, small), _, pool_refs = T._population(population)
    parent_tauto = {s: tautkey(s) for s in subs}

    if criterion == DECLARED_CRITERION and pool_refs is not None:
        real = {s: set(pool_refs[s]) for s in subs}
    else:
        flat, index = [], {}
        for s in subs:
            index[s] = (len(flat), len(flat) + len(ref_smiles[s]))
            flat.extend(ref_smiles[s])
        keys, _ = T.match_keys(flat, criterion)
        real = {s: set(keys[a:b]) for s, (a, b) in index.items()}
    parent = (parent_tauto if criterion == DECLARED_CRITERION
              else {s: k for s, k in zip(subs, T.match_keys(subs, criterion)[0])})

    ordered = {"whole bank": {s: T._ordered_candidates(big[s]) for s in subs},
               "trained budget": {s: T._ordered_candidates(small[s]) for s in subs}}

    lists = {}
    for name, spec in T.ARMS[population].items():
        built = T.arm_key_lists(spec, subs, ordered.get(name), parent, criterion)
        if built is not None:
            lists[name] = built[0]

    U = np.array([len(real[s]) for s in subs], dtype=float)
    total = U.sum()
    rng = np.random.default_rng(T.SEED)
    idx = rng.integers(0, len(subs), (T.N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def hits(name, k):
        return np.array([len(set(lists[name][s][:k]) & real[s]) for s in subs], dtype=float)

    cells = {}
    for k in T.KS:
        h = {name: hits(name, k) for name in lists}
        for arm in ARMS:
            if arm not in h:
                continue
            for comp in comparators:
                if comp not in h:
                    continue          # absent on this population: no cell, not a zero
                d = h[arm] - h[comp]
                bt = d[idx].sum(axis=1) / denom
                lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
                below, above = float((bt <= 0).sum()), float((bt >= 0).sum())
                p = min(1.0, 2.0 * (min(below, above) + 1.0) / (len(bt) + 1.0))
                cells[f"{arm} - {comp} @ {k}"] = {
                    "gap": round(float(d.sum() / total), 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "p": round(p, 5),
                    "separates_per_comparison": bool(lo > 0 or hi < 0),
                    "population": population, "criterion": criterion}
    return cells


def _families():
    return {
        "declared": [(DECLARED_POPULATION, DECLARED_CRITERION, DECLARED_COMPARATORS)],
        # Renamed from "with_evaluated1170", which read as though it were the widening the paper
        # quotes. It is not: the paper's is five comparators on one population, this is three
        # comparators across two. Both happen to hold ninety tests.
        "declared_across_both_populations": [
            (DECLARED_POPULATION, DECLARED_CRITERION, DECLARED_COMPARATORS),
            ("evaluated1170", DECLARED_CRITERION, DECLARED_COMPARATORS)],
        "every_contrast_the_paper_prints": [
            (DECLARED_POPULATION, DECLARED_CRITERION, PAPER_COMPARATORS)],
        "union_of_t_main": [(pop, crit, tuple(c for c in T.ARMS[pop] if c not in ARMS))
                            for pop in ("comparison291", "evaluated1170")
                            for crit in T.CRITERIA],
    }


_CACHE: dict = {}


def recompute(family="declared", alpha=ALPHA):
    """Holm over one family, with every cell's population recorded beside it."""
    cells = {}
    for population, criterion, comparators in _families()[family]:
        key = (population, criterion, comparators)
        if key not in _CACHE:
            _CACHE[key] = _population_cells(population, criterion, comparators)
        for name, cell in _CACHE[key].items():
            # Always qualified by criterion and population. Qualifying only the union family let
            # the whole-test-set cells overwrite the comparison-set cells of the same name in
            # `with_evaluated1170`: the family collapsed from 90 tests to 54, and worse than the
            # count, half of the declared family was silently replaced by numbers from another
            # population. The same bug made main()'s "what the widening removes" list compare
            # cells across populations while believing they were the same cell.
            cells[f"{name} [{criterion}/{population}]"] = cell

    names = sorted(cells)
    rejected = holm_reject([cells[n]["p"] for n in names], alpha=alpha)
    after = dict(zip(names, rejected))
    changing = sorted(n for n in names
                      if cells[n]["separates_per_comparison"] != after[n])

    return {
        "family": family,
        "alpha": alpha,
        "n_tests": len(names),
        # Recorded on every family so a survivor count is never quoted without the resolution that
        # bounds it.
        **decidability(len(names), alpha=alpha),
        "n_separating_per_comparison": sum(1 for n in names
                                           if cells[n]["separates_per_comparison"]),
        "n_separating_after_holm": sum(1 for n in names if after[n]),
        "cells_changing_status": changing,
        "population_of_each_cell": {n: cells[n]["population"] for n in names},
        "criterion_of_each_cell": {n: cells[n]["criterion"] for n in names},
        "cells": {n: {**cells[n], "separates_after_holm": after[n]} for n in names},
    }


def main() -> int:
    from _provenance import stamp

    report = {"provenance": stamp(__file__),
              "question": ("what the family-wise correction does over the cells the re-tabulation "
                           "holds, and what including the whole-test-set cells changes"),
              "p_value": ("two-sided bootstrap, min(1, 2*(min(#bt<=0, #bt>=0)+1)/(B+1)), over the "
                          "same replicates as the interval; B and seed as "
                          "scripts/typed_edit/deployment_table.py uses them"),
              "why_recomputed": ("multiplicity.py reads stored p-values from deployment_table.json, "
                                 "which exists only on the comparison set, so no whole-test-set "
                                 "cell has one to read"),
              "families": {}}

    declared = recompute("declared")
    published = json.loads((ROOT / "results" / "multiplicity.json").read_text())
    report["reproduces_the_declared_family"] = {
        "n_tests": [declared["n_tests"], published["n_tests"]],
        "n_separating_per_comparison": [declared["n_separating_per_comparison"],
                                        published["n_separating_per_comparison"]],
        "n_separating_after_holm": [declared["n_separating_after_holm"],
                                    published["n_separating_after_holm"]],
        "matches": (declared["n_tests"] == published["n_tests"]
                    and declared["n_separating_per_comparison"]
                    == published["n_separating_per_comparison"]
                    and declared["n_separating_after_holm"]
                    == published["n_separating_after_holm"]),
    }

    for name in ("declared", "declared_across_both_populations",
                 "every_contrast_the_paper_prints", "union_of_t_main"):
        got = recompute(name)
        report["families"][name] = got

    d = report["families"]["declared"]
    w = report["families"]["declared_across_both_populations"]
    lost = sorted(n for n in d["cells"]
                  if d["cells"][n]["separates_after_holm"]
                  and n in w["cells"] and not w["cells"][n]["separates_after_holm"])
    report["what_including_the_whole_test_set_costs"] = {
        "declared_family_size": d["n_tests"],
        "widened_family_size": w["n_tests"],
        "surviving_declared": d["n_separating_after_holm"],
        "surviving_widened": w["n_separating_after_holm"],
        "declared_cells_the_widening_removes": lost,
    }
    OUT.write_text(json.dumps(report, indent=1))

    print(f"reproduces the declared family: {report['reproduces_the_declared_family']['matches']}")
    for k, v in report["reproduces_the_declared_family"].items():
        if k != "matches":
            print(f"  {k:30} recomputed {v[0]}  published {v[1]}")
    print()
    for name, got in report["families"].items():
        undecidable = "" if got["family_can_reject_at_all"] else \
            f"  [CANNOT REJECT: needs B>={got['n_boot_needed_for_one_rejection']}]"
        print(f"  {name:32} tests {got['n_tests']:>4}"
              f"  separating {got['n_separating_per_comparison']:>4}"
              f"  after Holm {got['n_separating_after_holm']:>4}"
              f"  status changes {len(got['cells_changing_status']):>4}{undecidable}")
    print(f"\n  declared cells the widening removes: "
          f"{report['what_including_the_whole_test_set_costs']['declared_cells_the_widening_removes'] or 'none'}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

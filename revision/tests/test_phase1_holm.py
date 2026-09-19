"""Tests for the Phase 1 family-wise recomputation, written before the analysis functions exist.

Run: python -m pytest revision/tests/test_phase1_holm.py -q

The recomputation cannot be built from T_main.csv's per-arm intervals. The correction the
manuscript quotes is Holm over two-sided BOOTSTRAP p-values on paired contrasts
(`results/multiplicity.json`: "Holm step-down on two-sided bootstrap p-values, B and seed as the
intervals"), and a paired contrast needs the per-substrate hit vectors of both arms, not two
marginal intervals. So the module rebuilds the arms through phase1_tmain and contrasts them itself,
with the producer's own p-value definition, gated on reproducing the published family.

(An earlier version of this docstring said the contrasts come from the repository's own
`scripts/_contrast.py`. They do not: the module never imports it and computes the paired contrast
inline. A false statement about where a number comes from is the defect this revision is about, so
it is corrected rather than left.)

The declared family is 54 cells: 2 GRAIL arms by 3 comparators by 9 budgets, on the comparison set
only, under one criterion. Widening it is the point of the exercise, so the tests fix the
arithmetic of the step-down and the accounting of what enters the family, and leave the verdicts to
the run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "typed_edit"))

# Hard import: until the module exists these tests must fail, not skip.
import phase1_holm as holm  # noqa: E402


def test_holm_step_down_rejects_the_prefix_and_stops_at_the_first_failure():
    """Holm is step-down: it sorts ascending, compares p_i against alpha/(m-i), and the moment one
    fails nothing after it is rejected however small it is."""
    p = [0.001, 0.004, 0.03, 0.04]
    rejected = holm.holm_reject(p, alpha=0.05)
    # m=4: thresholds 0.0125, 0.01667, 0.025, 0.05. 0.001 and 0.004 pass, 0.03 fails, so 0.04 is
    # not rejected even though 0.04 < 0.05.
    assert rejected == [True, True, False, False]


def test_holm_is_not_bonferroni_and_is_reported_as_step_down():
    """A p-value that Bonferroni would reject at alpha/m must still be rejected by Holm, and one
    that only survives because a larger p-value blocked it must not be."""
    assert holm.holm_reject([0.01, 0.2], alpha=0.05) == [True, False]
    assert holm.holm_reject([0.03, 0.03], alpha=0.05) == [False, False]


def test_a_single_test_family_reduces_to_the_uncorrected_comparison():
    assert holm.holm_reject([0.04], alpha=0.05) == [True]
    assert holm.holm_reject([0.06], alpha=0.05) == [False]


def test_the_declared_family_is_reproduced_before_it_is_widened():
    """The gate. Recomputing the family the artifact declares must return the artifact's counts:
    54 tests, 33 separating per comparison, 22 surviving Holm. A recomputation that cannot
    reproduce the declared family is not evidence about a wider one."""
    published = json.loads((ROOT / "results" / "multiplicity.json").read_text())
    got = holm.recompute(family="declared")
    assert got["n_tests"] == published["n_tests"]
    assert got["n_separating_per_comparison"] == published["n_separating_per_comparison"]
    assert got["n_separating_after_holm"] == published["n_separating_after_holm"]


def test_widening_the_family_is_accounted_rather_than_asserted():
    """Adding the whole-test-set cells must report the family size before and after, and name
    every cell whose status changes. A count that moves with no named cell is not a finding."""
    got = holm.recompute(family="declared_across_both_populations")
    declared = holm.recompute(family="declared")
    assert got["n_tests"] > declared["n_tests"]
    assert set(got) >= {"n_tests", "n_separating_after_holm", "cells_changing_status",
                        "population_of_each_cell"}
    for cell in got["cells_changing_status"]:
        assert cell in got["population_of_each_cell"], (
            "a cell reported as changing status must be locatable in the family")


def test_a_family_reports_whether_it_can_reject_anything_at_all():
    """Decidability is a property of B and the family size, not of the data.

    Asserted against the pure function, with the family sizes passed in. Written first as three
    calls to `recompute`, one of them on the 864-cell union family, this test recomputed most of
    what `main()` computes -- a twenty-minute suite duplicating the run it guards. The arithmetic
    needs no contrast, so it is checked without one.
    """
    B = 10000
    # The two families the manuscript's numbers come from are decidable at this B.
    for m in (54, 90):
        got = holm.decidability(m, n_boot=B)
        assert got["family_can_reject_at_all"] is True, m
        assert got["smallest_attainable_p"] <= got["holm_strictest_threshold"]
        assert got["n_boot_needed_for_one_rejection"] is None

    # The union of everything the re-tabulation holds is not, and by how much is reported.
    union = holm.decidability(864, n_boot=B)
    assert union["family_can_reject_at_all"] is False
    assert union["smallest_attainable_p"] > union["holm_strictest_threshold"]
    assert union["n_boot_needed_for_one_rejection"] == 34559

    # The boundary: with enough resamples the same family becomes decidable, so the flag tracks B
    # and not the family alone.
    assert holm.decidability(864, n_boot=34559)["family_can_reject_at_all"] is True
    assert holm.decidability(864, n_boot=34558)["family_can_reject_at_all"] is False


def test_the_recorded_decidability_travels_with_every_family():
    """The flag has to be on the family's own record, or a survivor count can be read alone."""
    got = holm.recompute(family="declared")
    assert set(got) >= {"smallest_attainable_p", "holm_strictest_threshold",
                        "family_can_reject_at_all", "n_boot_needed_for_one_rejection"}
    assert got["family_can_reject_at_all"] is True


def test_the_paper_s_widening_is_a_different_family_from_the_two_population_one():
    """Two families under one correction, told apart by MEMBERSHIP and never by size.

    They once held ninety tests each, over different cells, and this asserted the shared total. The
    coincidence ended twice over: GLORYxR added two comparators to the family the paper prints, and
    MetaTox's whole-population file, which did not exist when the other family was written, added
    its cells to the two-population one. A test that pins a coincidence fails when the coincidence
    does, and reports it as if something had broken. The sizes are read and printed here so a
    reader of a failure can see them; what is asserted is what the docstring always claimed.
    """
    paper = holm.recompute(family="every_contrast_the_paper_prints")
    both = holm.recompute(family="declared_across_both_populations")
    assert paper["n_tests"] > 0 and both["n_tests"] > 0, (
        f"a family is empty: paper={paper['n_tests']}, both={both['n_tests']}")
    assert set(paper["cells"]) != set(both["cells"]), (
        f"the two families hold the same cells ({paper['n_tests']} and {both['n_tests']}), so one "
        f"of the two definitions is not the family it is named for")
    assert set(paper["population_of_each_cell"].values()) == {"comparison291"}
    assert set(both["population_of_each_cell"].values()) == {"comparison291", "evaluated1170"}


def test_every_cell_records_which_population_it_came_from():
    """The declared family is silent about its population; the recomputation must not be."""
    got = holm.recompute(family="declared_across_both_populations")
    pops = set(got["population_of_each_cell"].values())
    assert pops == {"comparison291", "evaluated1170"}, pops

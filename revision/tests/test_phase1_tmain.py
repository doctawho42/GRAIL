"""Tests for the Phase 1 re-tabulation, written before the analysis functions exist.

Run: python -m pytest revision/tests -q

The last test is the one that makes the table trustworthy: the re-tabulation must reproduce the
published comparison-set cells of results/deployment_table.json exactly. A re-tabulation that
computes its own slightly different recall is not a re-tabulation, and every downstream cell on the
1,170 would inherit the difference invisibly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "typed_edit"))

# Imported hard, not with importorskip: until the analysis module exists these tests must FAIL,
# and a skip is not a failure. The discipline the revision asks for is that the test fails first.
import phase1_tmain as tmain  # noqa: E402


def test_micro_recall_is_the_ratio_of_sums_not_the_mean_of_ratios():
    """Micro weights a substrate by its reference count; the mean of per-substrate recall does not.

    One substrate with a single reference found and one with ten references all missed is 1/11
    under micro and 0.5 under macro. The manuscript reports micro throughout, so a table that
    quietly computed macro would disagree with every published cell while looking plausible.
    """
    hits = {"a": 1, "b": 0}
    universe = {"a": 1, "b": 10}
    assert tmain.micro_recall(hits, universe) == pytest.approx(1 / 11)


def test_drop_parent_removes_the_substrates_own_key_before_the_budget():
    """Returning the parent is not a prediction, and it must not consume a slot either."""
    ranked = ["PARENT", "A", "B"]
    assert tmain.drop_parent(ranked, "PARENT") == ["A", "B"]
    assert tmain.drop_parent(ranked, "PARENT")[:2] == ["A", "B"]


def test_dedup_keeps_first_occurrence_in_rank_order_and_stops_at_the_cap():
    assert tmain.dedup_in_rank_order(["A", "B", "A", "C"], cap=None) == ["A", "B", "C"]
    assert tmain.dedup_in_rank_order(["A", "B", "A", "C"], cap=2) == ["A", "B"]


def test_keying_counts_its_fallbacks_instead_of_hiding_them():
    """An unkeyable structure must be counted, and it cannot be counted by catching an exception.

    This repository's keying never raises: `_tautomer_inchikey` falls back to `_inchikey` per
    input and `_inchikey` returns the raw SMILES when RDKit cannot parse it. So the failure
    arrives as a "key" that is the SMILES itself, which matches no reference and biases recall
    down with nothing in the table to show it. The keyer must therefore detect the signature, not
    the exception.
    """
    keys, report = tmain.match_keys(["CCO", "not a molecule at all"], "inchikey_tautomer")
    assert len(keys) == 2
    assert report["fallbacks"] == 1, "the unparseable input must be counted exactly once"
    assert report["fallback_inputs"] == ["not a molecule at all"]
    assert keys[0] != "CCO", "a parseable molecule must come back as a key, not as its SMILES"


def test_the_two_criteria_where_a_key_equal_to_the_input_is_correct_are_not_miscounted():
    """`exact` is the raw string by definition and `canonical` returns an already-canonical input
    unchanged, so for those two the signature above is not a failure and must not be counted."""
    _, exact = tmain.match_keys(["CCO"], "exact")
    _, canon = tmain.match_keys(["CCO"], "canonical")
    assert exact["fallbacks"] == 0
    assert canon["fallbacks"] == 0


def test_bootstrap_interval_is_deterministic_and_brackets_the_point():
    per_sub_hits = {f"s{i}": (i % 3) for i in range(40)}
    universe = {f"s{i}": 3 for i in range(40)}
    point = tmain.micro_recall(per_sub_hits, universe)
    lo1, hi1 = tmain.bootstrap_ci(per_sub_hits, universe, n_boot=2000, seed=0)
    lo2, hi2 = tmain.bootstrap_ci(per_sub_hits, universe, n_boot=2000, seed=0)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= point <= hi1


def test_the_comparison_set_cells_reproduce_the_published_table_exactly():
    """The gate. Every (arm, budget) cell on the 291 under the tautomer criterion must equal
    results/deployment_table.json's recall_micro, which is what the manuscript prints."""
    published = json.loads((ROOT / "results" / "deployment_table.json").read_text())["recall_micro"]
    rows = tmain.build_rows(populations=("comparison291",), criteria=("inchikey_tautomer",))
    got = {(r["system"], str(r["k"])): r["recall"] for r in rows}
    checked = 0
    for k, cells in published.items():
        for arm, value in cells.items():
            assert (arm, k) in got, f"the re-tabulation has no cell for {arm} at k={k}"
            assert got[(arm, k)] == pytest.approx(value, abs=5e-5), (
                f"{arm} at k={k}: re-tabulated {got[(arm, k)]} against published {value}")
            checked += 1
    # Nine arms now, not seven: GLORYxR contributes two columns. Read from the artifact rather
    # than typed, so the next arm cannot pass this by leaving the count alone.
    expected_cells = sum(len(c) for c in published.values())
    assert checked == expected_cells, (
        f"the published table holds {expected_cells} cells and this compared {checked}")
    assert checked >= 63, f"the table shrank to {checked} cells"

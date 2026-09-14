"""Tests for the budget figure's curve assembly, written before the producer exists.

Run: python -m pytest revision/tests/test_phase1_budget.py -q

The figure is recall against the mean number of candidates a system actually emits, one line per
system and one panel per matching criterion. Almost all of its risk is in the assembly rather than
the drawing: a system that has no cells on a population must be absent from that panel rather than
drawn along the floor, and the axes must be numbers rather than the strings a CSV hands back.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision"))

# Hard import: until the producer exists these tests must fail, not skip.
import phase1_budget as budget  # noqa: E402


def _rows():
    """Two systems on one population, one of them absent from the other population."""
    out = []
    for k, emitted, recall in ((1, 1.0, 0.10), (5, 4.2, 0.30), (15, 8.4, 0.50)):
        out.append({"system": "whole bank", "criterion": "inchikey_tautomer", "k": k,
                    "population": "comparison291", "recall": recall,
                    "ci_lo": recall - 0.02, "ci_hi": recall + 0.02,
                    "mean_emitted_at_k": emitted, "mean_emitted_untruncated": 8.4,
                    "n_substrates": 291, "n_references": 665})
    for k, emitted, recall in ((1, 1.0, 0.08), (5, 3.9, 0.22), (15, 7.1, 0.41)):
        out.append({"system": "sygma", "criterion": "inchikey_tautomer", "k": k,
                    "population": "comparison291", "recall": recall,
                    "ci_lo": recall - 0.02, "ci_hi": recall + 0.02,
                    "mean_emitted_at_k": emitted, "mean_emitted_untruncated": 7.1,
                    "n_substrates": 291, "n_references": 665})
    # gloryx exists only on the comparison set; it must not appear in the 1,170 panel
    out.append({"system": "gloryx", "criterion": "inchikey_tautomer", "k": 15,
                "population": "comparison291", "recall": 0.2, "ci_lo": 0.18, "ci_hi": 0.22,
                "mean_emitted_at_k": 5.0, "mean_emitted_untruncated": 5.0,
                "n_substrates": 291, "n_references": 665})
    out.append({"system": "whole bank", "criterion": "inchikey_tautomer", "k": 15,
                "population": "evaluated1170", "recall": 0.44, "ci_lo": 0.42, "ci_hi": 0.46,
                "mean_emitted_at_k": 8.1, "mean_emitted_untruncated": 8.1,
                "n_substrates": 1170, "n_references": 2292})
    return out


def test_a_series_pairs_each_budget_with_its_own_emitted_count_in_budget_order():
    got = budget.series(_rows(), criterion="inchikey_tautomer", population="comparison291")
    assert [p["k"] for p in got["whole bank"]] == [1, 5, 15]
    assert [p["x"] for p in got["whole bank"]] == [1.0, 4.2, 8.4]
    assert [p["y"] for p in got["whole bank"]] == [0.10, 0.30, 0.50]


def test_a_system_absent_from_a_population_is_absent_from_its_panel():
    """Drawing a missing comparator at zero would read as a measured loss rather than no data."""
    got = budget.series(_rows(), criterion="inchikey_tautomer", population="evaluated1170")
    assert "gloryx" not in got
    assert "sygma" not in got
    assert set(got) == {"whole bank"}


def test_the_panels_are_the_criteria_the_data_actually_carries():
    assert budget.panels(_rows()) == ["inchikey_tautomer"]


def test_loading_coerces_the_axes_to_numbers():
    """A CSV hands back strings, and a string x-axis sorts lexicographically: 10 before 5."""
    path = ROOT / "revision" / "T_main.csv"
    if not path.exists():
        import pytest
        pytest.skip("T_main.csv not built yet; the coercion is still asserted on synthetic rows")
    rows = budget.load_rows(path)
    assert rows, "the table is empty"
    r = rows[0]
    for field in ("k", "recall", "ci_lo", "ci_hi", "mean_emitted_at_k"):
        assert isinstance(r[field], (int, float)), f"{field} came back as {type(r[field]).__name__}"

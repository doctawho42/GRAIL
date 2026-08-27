"""The contrast helper must refuse the two shapes that produced a confident nothing here."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _contrast import EmptyComparator, paired_contrast  # noqa: E402


def test_an_absent_comparator_raises_instead_of_returning_the_arm():
    a = np.array([2.0, 1.0, 3.0])
    b = np.zeros(3)
    with pytest.raises(EmptyComparator):
        paired_contrast(a, b, [3.0, 2.0, 4.0], comparator_covers=0, name="MetaTox")


def test_coverage_must_be_stated_at_all():
    with pytest.raises(ValueError):
        paired_contrast([1.0], [0.0], [2.0])


def test_a_gap_that_is_the_whole_arm_is_flagged_even_when_coverage_is_nonzero():
    a = np.array([2.0, 1.0, 3.0])
    b = np.array([0.0, 0.0, 0.0])
    out = paired_contrast(a, b, [3.0, 2.0, 4.0], comparator_covers=3)
    assert "suspect" in out and out["gap"] > 0


def test_an_ordinary_contrast_is_not_flagged():
    a = np.array([2.0, 1.0, 3.0])
    b = np.array([1.0, 1.0, 2.0])
    out = paired_contrast(a, b, [3.0, 2.0, 4.0], comparator_covers=3)
    assert "suspect" not in out
    assert out["comparator_covers"] == 3 and out["n_items"] == 3


def test_the_gap_is_the_ratio_of_sums_and_not_a_mean_of_ratios():
    a, b, w = np.array([2.0, 0.0]), np.array([0.0, 0.0]), np.array([2.0, 8.0])
    out = paired_contrast(a, b, w, comparator_covers=2, n_boot=200)
    assert out["gap"] == pytest.approx(2.0 / 10.0)     # micro, not (1.0 + 0.0) / 2

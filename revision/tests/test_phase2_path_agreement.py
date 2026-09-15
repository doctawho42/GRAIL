"""Three modules must name one file for the GLORYx arm on the evaluated population.

Run: python -m pytest revision/tests/test_phase2_path_agreement.py -q

The column is assembled by one module, declared as an arm by a second, and reported by a third:

    revision/phase2_gloryx_merge.py   writes the merged file
    revision/phase1_tmain.py          declares it as the arm the table reads
    revision/phase2_comparators.py    reports its coverage and delivery

Each holds the path as its own constant, and the unmerged run output remains a real file on disk
that any of them could plausibly name -- it holds 879 of the 1,170 substrates, so naming it would
score the other 291 as misses. A path that three modules agree on today and one of them changes
tomorrow is the defect this pins, and checking it by hand once is not the same as checking it.

test_the_table_declares_the_merged_file fails until the arm is added to phase1_tmain.ARMS, which
is the edit it exists to guard.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "revision"), str(ROOT / "scripts"),
           str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import phase1_tmain as T          # noqa: E402
import phase2_comparators as P    # noqa: E402
import phase2_gloryx_merge as M   # noqa: E402


def _rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def test_the_merge_writes_what_the_comparator_record_reads():
    assert _rel(M.OUT) == P.GLORYX_1170


def test_the_merge_reads_what_the_service_run_writes():
    """The wider run's own output, which is the merge's input and nobody else's arm."""
    assert _rel(M.WIDER) == P.GLORYX_1170_RAW


def test_the_table_declares_the_merged_file_as_the_gloryx_arm():
    """Fails until the arm is added; this is the guard for that edit.

    The accessor matters as much as the path: the merged artifact keeps its substrate map under
    `predictions`, and a spec with no accessor would read the envelope's top-level keys as
    substrates -- the envelope-for-substrate-map mistake that has cost this session twice.
    """
    arms = T.ARMS["evaluated1170"]
    assert "gloryx" in arms, (
        "the evaluated population has no gloryx arm; add it once the merge exists")
    spec = arms["gloryx"]
    assert spec[0] == "list"
    assert spec[1] == P.GLORYX_1170, f"the arm names {spec[1]!r}"
    assert spec[2] == "predictions", "the merged artifact keeps its map under 'predictions'"


def test_the_unmerged_run_output_is_declared_as_nobody_s_arm():
    """It holds 879 of the 1,170. Declaring it anywhere scores 291 substrates as misses."""
    for population, arms in T.ARMS.items():
        for arm, spec in arms.items():
            assert spec[1] != P.GLORYX_1170_RAW, (
                f"{population}/{arm} names the unmerged run output, which covers 879 of 1,170")


def test_the_two_paths_are_actually_different_files():
    """A guard on the guard: if the raw and merged constants ever collapse to one value, every
    assertion above passes while the distinction they protect is gone."""
    assert P.GLORYX_1170 != P.GLORYX_1170_RAW

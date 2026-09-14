"""Tests for the Phase 3 changelog, written before the producer exists.

Run: python -m pytest revision/tests/test_phase3_changelog.py -q

Phase 3 asks for every quoted number's old and new value, flagging any that moved beyond its own
interval half-width. The manuscript reaches its numbers only through macros generated from
`results/paper2_numbers.json`, so the changelog is a key-by-key diff of that file across the
re-runs rather than a scan of prose: 2,459 dotted keys, of which 261 carry `.lo` and `.hi` siblings
and 2,198 carry no interval at all.

That asymmetry is what most of these tests are about. A value with no interval cannot be judged
against one, and reporting it as "not flagged" would read as *checked and within tolerance*. It has
to say it could not be checked. The same holds for the keys whose values are not numbers: this file
carries "yes"/"no" verdicts and ISO dates, and an arithmetic comparison of those is meaningless.

The tests use synthetic number-maps, so they pin the arithmetic and the refusals without depending
on which re-runs have landed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "revision"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Hard import: until the producer exists these tests must fail, not skip.
import phase3_changelog as C  # noqa: E402


def _old():
    return {
        "holm.tests": 54,
        "holm.surviving": 22,
        "agg.cmp.hybrid.max": 0.0526,
        "agg.cmp.hybrid.max.lo": 0.0271,
        "agg.cmp.hybrid.max.hi": 0.0787,
        "emitted.metatox": 33.6,
        "holm.trainedmetatoxfifty": "yes",
        "comparators.metatox.date": "2026-07-25",
        "gone.away": 7,
    }


def _new(**over):
    d = dict(_old())
    d.pop("gone.away")
    d["arrived.new"] = 3
    d.update(over)
    return d


# --------------------------------------------------------------------------- loading

def test_loading_reads_through_the_envelope_rather_than_the_top_level(tmp_path):
    """The file is {n_numbers, numbers, provenance}; the values live under `numbers`.

    Reading the top level would yield three "numbers" -- a count, a dict and a provenance block --
    which is the envelope-for-substrate-map mistake in a different costume.
    """
    p = tmp_path / "paper2_numbers.json"
    p.write_text(json.dumps({"n_numbers": 2, "numbers": {"a.b": 1, "c.d": 2},
                             "provenance": {"script": "x"}}))
    got = C.load_numbers(p)
    assert got == {"a.b": 1, "c.d": 2}


# --------------------------------------------------------------------------- the interval rule

def test_a_move_beyond_the_half_width_is_flagged():
    """Half-width of [0.0271, 0.0787] is 0.0258; a move of 0.03 exceeds it."""
    entries = {e["key"]: e for e in C.compare(_old(), _new(**{"agg.cmp.hybrid.max": 0.0826}))}
    e = entries["agg.cmp.hybrid.max"]
    assert e["kind"] == "changed"
    assert e["comparable"] is True
    assert abs(e["half_width"] - 0.0258) < 1e-9
    assert e["moved_beyond_interval"] is True


def test_a_move_within_the_half_width_is_not_flagged():
    entries = {e["key"]: e for e in C.compare(_old(), _new(**{"agg.cmp.hybrid.max": 0.0600}))}
    e = entries["agg.cmp.hybrid.max"]
    assert e["moved_beyond_interval"] is False
    assert e["comparable"] is True


def test_a_changed_value_with_no_interval_says_it_cannot_be_judged():
    """Not False. False reads as "checked and within tolerance", which would be a false negative
    on 2,198 of the 2,459 keys."""
    entries = {e["key"]: e for e in C.compare(_old(), _new(**{"emitted.metatox": 41.2}))}
    e = entries["emitted.metatox"]
    assert e["kind"] == "changed"
    assert e["half_width"] is None
    assert e["moved_beyond_interval"] is None
    assert e["comparable"] is True, "it is a number; it simply has no interval to judge against"


def test_a_changed_non_numeric_value_is_reported_but_not_compared():
    entries = {e["key"]: e for e in C.compare(_old(), _new(**{
        "holm.trainedmetatoxfifty": "no", "comparators.metatox.date": "2026-09-15"}))}
    for key in ("holm.trainedmetatoxfifty", "comparators.metatox.date"):
        e = entries[key]
        assert e["kind"] == "changed"
        assert e["comparable"] is False
        assert e["moved_beyond_interval"] is None
        assert e["half_width"] is None


# --------------------------------------------------------------------------- the accounting

def test_added_and_removed_keys_are_reported_as_such():
    entries = {e["key"]: e for e in C.compare(_old(), _new())}
    assert entries["gone.away"]["kind"] == "removed"
    assert entries["gone.away"]["new"] is None
    assert entries["arrived.new"]["kind"] == "added"
    assert entries["arrived.new"]["old"] is None


def test_unchanged_keys_do_not_appear():
    keys = {e["key"] for e in C.compare(_old(), _new())}
    assert "holm.tests" not in keys
    assert "agg.cmp.hybrid.max" not in keys


def test_the_interval_siblings_themselves_are_reported_when_they_move():
    """An interval that moved is a change in its own right, and hiding it inside its point
    estimate's row would drop it."""
    entries = {e["key"]: e for e in C.compare(_old(), _new(**{"agg.cmp.hybrid.max.hi": 0.0900}))}
    assert "agg.cmp.hybrid.max.hi" in entries
    assert entries["agg.cmp.hybrid.max.hi"]["kind"] == "changed"


# --------------------------------------------------------------------------- rendering

def test_the_rendering_drops_nothing_and_counts_what_it_cannot_judge():
    """Every entry reaches the page, and the ones that could not be judged are counted rather than
    omitted: a changelog that silently shows only the checkable rows understates what moved."""
    entries = C.compare(_old(), _new(**{"agg.cmp.hybrid.max": 0.0826, "emitted.metatox": 41.2,
                                        "holm.trainedmetatoxfifty": "no"}))
    md = C.render(entries)
    for e in entries:
        assert e["key"] in md, f"{e['key']} was dropped from the rendering"
    assert "cannot be judged" in md.lower() or "no interval" in md.lower()
    flagged = [e for e in entries if e["moved_beyond_interval"] is True]
    assert str(len(flagged)) in md, "the count of flagged moves must appear"


def test_the_generator_s_inputs_are_enumerated_from_its_own_source():
    """The quoted numbers come from one generator, and its inputs are what could move them.

    Read from the `art("...")` calls in scripts/paper2_numbers.py rather than from a list kept
    beside it: a hardcoded list next to data that has its own keys goes false the first time the
    generator gains an input, and this repository has paid for that already.
    """
    got = C.generator_inputs()
    assert len(got) > 50, f"only {len(got)} inputs found; the extraction is probably wrong"
    assert any(g.endswith("multiplicity.json") for g in got)
    assert any(g.endswith("deployment_table.json") for g in got)


def test_the_enumeration_catches_a_loop_tuple_name_and_excludes_the_output():
    """Two properties, not a total.

    `case_study_drawn.json` is passed to art() through a loop variable and is named nowhere else,
    so a pattern matching only `art("name")` drops it -- the enumeration must be wider than that.
    Widening it also picks up `paper2_numbers.json`, which is the generator's OUTPUT: counting
    that would have the changelog report it checked the file it is diffing.

    The count itself is deliberately not asserted. A hardcoded total goes false the first time the
    generator gains an input, which is the defect this function exists to avoid.
    """
    got = C.generator_inputs()
    assert "case_study_drawn.json" in got, (
        "an input reached only through a loop variable is still an input")
    assert not any(g.endswith("paper2_numbers.json") for g in got), (
        "the file being diffed is not an input to itself")


def test_an_empty_changelog_distinguishes_nothing_moved_from_nothing_looked_at():
    """An empty diff is a result only if it says what it checked.

    "No quoted number changed" alone cannot be told apart from a diff that never read anything,
    and the honest statement is the stronger one: every input to the generator was checked and
    none of them moved, so no quoted value could have. The counts are rendered, not narrated.
    """
    md = C.render([], inputs_checked=98, inputs_changed=[])
    assert "98" in md, "the number of inputs checked must appear"

    moved = C.render([], inputs_checked=98, inputs_changed=["results/multiplicity.json"])
    assert "results/multiplicity.json" in moved, (
        "an input that moved while no number did is a different statement and must be named")
    assert md != moved


def test_rendering_without_the_input_accounting_still_works():
    """The older call site passes entries only, and must keep working."""
    md = C.render([])
    assert "changed: 0" in md


def test_comparing_a_file_against_itself_yields_an_empty_changelog(tmp_path):
    """The no-op case, which is what a re-run that changed nothing must produce."""
    p = tmp_path / "n.json"
    p.write_text(json.dumps({"numbers": _old()}))
    assert C.compare(C.load_numbers(p), C.load_numbers(p)) == []

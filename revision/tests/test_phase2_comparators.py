"""Tests for the Phase 2 comparator record, written before the producer exists.

Run: python -m pytest revision/tests/test_phase2_comparators.py -q

Phase 2 was asked for two comparator columns on all 1,170 substrates, in a default and a strict-SOM
mode. Some of that is obtainable and some is not, and the point of this artefact is that the two are
distinguishable afterwards: every claim it makes is either a count computed from a file in this
repository or a blocker carrying the evidence that establishes it.

So the tests here are mostly about refusals. A blocker with no evidence, a comparator reported as
runnable without a live check, or a coverage number quoted for a file nobody read are all ways this
record could look complete while asserting nothing, and each has a test.

The archive of GLORYxR models is deliberately not required: it arrived in a session scratchpad and
a record that cannot be rebuilt from a clean checkout without it is not rerunnable, so its absence
must be reported as absence rather than crash.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "revision"), str(ROOT / "scripts"),
           str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Hard import: until the producer exists these tests must fail, not skip.
import phase2_comparators as P  # noqa: E402


# --------------------------------------------------------------------------- coverage

def test_coverage_is_read_through_each_arm_s_own_declared_accessor():
    """The counts must come from the file the table actually reads, not from a similar file.

    This repository holds two MetaTox prediction files: the SMIRKS run every published number is
    computed from, and a 248-substrate side analysis. Reading the wrong one understates MetaTox on
    43 substrates, so the accessor is part of the measurement.
    """
    cov = P.coverage()
    by = {(c["population"], c["arm"]): c for c in cov}
    assert by[("comparison291", "metatox")]["substrates"] == 291
    assert by[("comparison291", "metatox")]["file"].endswith("metatox_smirks_preds.json")
    assert by[("comparison291", "gloryx")]["substrates"] == 291
    assert by[("evaluated1170", "sygma")]["substrates"] == 1170
    for cell in cov:
        assert cell["file"], "every count must name the file it was read from"
        assert cell["substrates"] == cell["in_population"] or cell["in_population"] <= cell["substrates"]


def test_an_arm_with_no_file_on_a_population_is_absent_rather_than_zero():
    """Absent is not beaten: a comparator that was never run must not read as a measured loss.

    The mechanism is checked against a spec naming a file that cannot exist, not against GLORYx on
    the wider population. GLORYx is absent there only until this phase's own run lands, so pinning
    it would make this test fail exactly when the work succeeds: a gate whose green depends on the
    task not being done. MetaTox on that population stays absent through this phase, because only
    its submission files are produced here, so that one is asserted directly.
    """
    synthetic = {"evaluated1170": {"nowhere": ("list", "results/no_such_comparator.json", None)}}
    cell = P.coverage(arms=synthetic)[0]
    assert cell["absent"] is True
    assert cell["substrates"] is None, "an absent arm must carry no count at all"
    assert cell["file"].endswith("no_such_comparator.json")

    absent = {(c["population"], c["arm"]) for c in P.coverage() if c.get("absent")}
    assert ("evaluated1170", "metatox") in absent


# --------------------------------------------------------------------------- submission sets

def test_the_submission_sets_are_the_population_minus_the_published_baseline():
    """Counted against a named baseline, not against whatever happens to be on disk.

    What Phase 2 had to submit is a fact about the published comparison-set runs, so the baseline
    is those artefacts by name. Measured against "whatever exists now" these counts would fall to
    zero as this phase's own results arrive, which would read as though nothing had been needed.
    """
    subs = P.submission_sets()
    assert subs["gloryx"]["baseline"].endswith("gloryx_service_preds.json")
    assert subs["metatox"]["baseline"].endswith("metatox_smirks_preds.json")
    assert subs["gloryx"]["to_submit"] == 879
    assert subs["gloryx"]["held"] == 291
    assert subs["metatox"]["to_submit"] == 879
    assert subs["metatox"]["held"] == 291
    for name, s in subs.items():
        assert s["held"] + s["to_submit"] == 1170, f"{name} does not account for the population"


def test_the_submission_set_is_not_taken_from_a_predictions_csv():
    """The prediction CSVs hold 1169 of the 1,170; sourcing from one drops a substrate silently.

    The missing structure is a CoA thioester. A submission set short by one molecule would produce
    a column whose denominator quietly disagrees with the table it joins, so the record has to name
    where the set came from and it must not be a CSV.
    """
    subs = P.submission_sets()
    for name, s in subs.items():
        assert "test_references.json" in s["source"], f"{name} sourced from {s['source']!r}"
        assert not s["source"].endswith(".csv")


# --------------------------------------------------------------------------- blockers

def test_every_blocker_carries_evidence_that_can_be_checked():
    """A blocker is a claim about why something did not run; unevidenced, it is an opinion."""
    for b in P.blockers():
        assert b.get("finding"), "a blocker must state what it found"
        ev = b.get("evidence") or []
        assert ev, f"blocker {b.get('id')!r} carries no evidence"
        for item in ev:
            assert item.get("kind") in {"file", "digest", "measurement", "service", "document"}, \
                f"evidence of unknown kind in {b.get('id')!r}: {item}"
            assert item.get("detail"), f"evidence with no detail in {b.get('id')!r}"


def test_strict_som_is_reported_unavailable_with_the_service_s_own_parameters():
    """Marked, not invented: the mode was asked for and no user for it exists here."""
    b = {x["id"]: x for x in P.blockers()}
    assert "strict_som_unavailable" in b
    ev = b["strict_som_unavailable"]["evidence"]
    kinds = {e["kind"] for e in ev}
    assert "service" in kinds, "the service's parameter list is what establishes the absence"
    assert b["strict_som_unavailable"].get("mode_requested") == "strict-SOM"


def test_gloryxr_is_blocked_on_code_and_says_so_rather_than_on_the_interpreter():
    """The version mismatch is real but not the binding constraint, and the record must not imply
    that a newer interpreter would unblock the arm: the featuriser and the code are absent."""
    b = {x["id"]: x for x in P.blockers()}
    assert "gloryxr_no_code" in b
    assert b["gloryxr_no_code"]["binding"] is True
    assert b.get("gloryxr_sklearn_version", {}).get("binding") is False


def test_the_model_archive_is_optional_and_its_absence_is_recorded_not_fatal():
    """It arrived in a session scratchpad, so a clean checkout must still produce this record."""
    rec = P.model_archive(Path("/nonexistent/path/models"))
    assert rec["present"] is False
    assert rec.get("checked_path") == "/nonexistent/path/models"
    assert rec.get("models") == []


# --------------------------------------------------------------------------- the report

def test_the_report_states_which_of_the_two_requested_columns_it_delivers():
    rep = P.build()
    asked = rep["requested"]
    assert asked["populations"] == ["evaluated1170"]
    assert set(asked["modes"]) == {"default", "strict-SOM"}
    delivered = rep["delivered"]
    assert "default" in delivered and "strict-SOM" in delivered
    assert delivered["strict-SOM"]["delivered"] is False
    for mode, d in delivered.items():
        if not d["delivered"]:
            assert d.get("blocker_id"), f"{mode} undelivered with no blocker named"


def test_nothing_is_reported_as_run_without_a_live_check():
    rep = P.build()
    for arm, d in rep["arms"].items():
        if d.get("ran"):
            assert d.get("evidence_of_running"), f"{arm} claims a run with no evidence"

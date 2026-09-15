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

import json
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


def test_strict_som_is_reported_as_a_gloryxr_mode_that_was_run():
    """It exists, it ran, and the record says where it exists -- not that it exists nowhere.

    The earlier test pinned the opposite and passed, because the service genuinely has no such
    parameter. Absence on one route was read as absence everywhere, which is the same
    wider-than-the-evidence error the GLORYxR blockers made.
    """
    b = {x["id"]: x for x in P.blockers()}
    entry = b["strict_som_is_a_gloryxr_mode_not_a_service_one"]
    assert entry.get("mode_requested") == "strict-SOM"
    assert entry["binding"] is False
    kinds = {e["kind"] for e in entry["evidence"]}
    assert "service" in kinds, "the service's parameter list still belongs here, as a contrast"
    assert "file" in kinds, "the code that exposes the mode must be cited"
    assert entry.get("what_the_mode_does_not_change"), (
        "coverage is identical between the modes and the record must say so")


def test_the_strict_column_is_delivered_by_gloryxr_not_by_the_service():
    rep = P.build()
    strict = rep["delivered"]["strict-SOM"]
    assert strict["delivered"] is True
    assert strict["file"].endswith("gloryxr_local_preds_strict.json")
    assert "GLORYxR" in strict["system"]
    arms = rep["arms"]
    assert arms["gloryxr strict"]["ran"] is True
    assert arms["gloryxr default"]["ran"] is True
    assert arms["gloryxr strict"]["evidence_of_running"]


def test_the_record_states_where_the_local_columns_are_comparable():
    """Without this the reader would put a GLORYxR row beside gloryx under every criterion."""
    rep = P.build()
    text = rep["comparability"]
    assert "inchikey_tautomer" in text and "inchi_no_stereo" in text
    assert "stereo" in text.lower()
    assert "T_gloryxr.csv" in text


def test_the_retracted_gloryxr_blockers_are_gone_and_name_what_they_retract():
    """Both earlier GLORYxR blockers were false and are replaced, not quietly edited.

    `gloryxr_no_code` claimed the arm could not be run for want of code and a featuriser, with
    binding=True; the code is public, the featuriser arrives as a dependency, and both columns have
    since been produced. `gloryxr_sklearn_version` claimed the predictions could not be trusted
    because newer-to-older is unsupported, comparing against an interpreter the run never used.
    A record that simply dropped them would leave no trace of having been wrong, so each
    replacement names the id it retracts.
    """
    b = {x["id"]: x for x in P.blockers()}
    assert "gloryxr_no_code" not in b
    assert "gloryxr_sklearn_version" not in b
    assert "strict_som_unavailable" not in b

    retracted = {x.get("retracts") for x in P.blockers() if x.get("retracts")}
    assert {"gloryxr_no_code", "gloryxr_sklearn_version"} <= retracted

    env = b["gloryxr_environment_not_guaranteed"]
    assert env["binding"] is False
    assert "measured" in env["finding"] or "moves no number" in env["finding"]
    for x in (env, b["gloryxr_dumps_unpublished"]):
        assert x.get("what_was_wrong_before"), f"{x['id']} does not say what it corrects"


def test_the_remaining_gloryxr_constraint_binds_only_a_third_party():
    """The arm exists here; what a third party cannot do is reproduce it from public sources."""
    b = {x["id"]: x for x in P.blockers()}
    dumps = b["gloryxr_dumps_unpublished"]
    assert dumps["binding"] is False
    assert dumps["binding_for_a_third_party"] is True


def test_the_duplicate_model_blocker_carries_how_much_of_the_rule_table_it_covers():
    """Two keys on one forest is only interpretable with the share of rules affected."""
    b = {x["id"]: x for x in P.blockers()}
    text = json.dumps(b["gloryxr_duplicate_model"])
    assert "224" in text and "260" in text


def test_the_model_archive_is_optional_and_its_absence_is_recorded_not_fatal():
    """It arrived in a session scratchpad, so a clean checkout must still produce this record."""
    rec = P.model_archive(Path("/nonexistent/path/models"))
    assert rec["present"] is False
    assert rec.get("checked_path") == "/nonexistent/path/models"
    assert rec.get("models") == []


# --------------------------------------------------------------------------- the report

def test_the_report_states_which_of_the_two_requested_columns_it_delivers():
    """Both requested modes are accounted for, each either delivered with its file or blocked
    with a blocker that exists.

    This test used to pin `delivered["strict-SOM"] is False`, written while the record wrongly
    held that no strict-SOM mode existed anywhere. The mode does exist -- it is GLORYxR's, it has
    been run over all 1,170 substrates -- so that assertion became a gate defending a false
    claim. It is replaced by the invariant rather than by the opposite constant, so it does not
    need rewriting again the next time the state changes.
    """
    rep = P.build()
    asked = rep["requested"]
    assert asked["populations"] == ["evaluated1170"]
    assert set(asked["modes"]) == {"default", "strict-SOM"}

    delivered = rep["delivered"]
    assert set(delivered) == {"default", "strict-SOM"}, (
        "every requested mode must be accounted for, delivered or not")
    ids = {b["id"] for b in rep["blockers"]}
    for mode, d in delivered.items():
        assert isinstance(d["delivered"], bool)
        if d["delivered"]:
            assert d.get("file"), f"{mode} delivered without naming its file"
            assert d.get("system"), f"{mode} delivered without naming the system that produced it"
            assert d.get("substrates_held") == d.get("population"), (
                f"{mode} reported delivered while holding "
                f"{d.get('substrates_held')} of {d.get('population')} substrates")
        else:
            assert d.get("blocker_id") in ids, (
                f"{mode} undelivered and its blocker_id {d.get('blocker_id')!r} names nothing")


def test_every_named_blocker_id_resolves_to_a_blocker_in_every_state(monkeypatch):
    """A blocker_id naming nothing makes the record read as explained when it is not.

    The undelivered default column blocks for two different reasons -- the merged file is absent,
    or it exists and covers only part of the population -- and `build()` used to choose the second
    id through a fallback that named an id `blockers()` never emitted. Asserting only that the id
    is truthy could not catch that, so all three states are forced and the id is resolved against
    the report's own blocker list.
    """
    states = {"absent": ({}, {"x"}), "incomplete": ({"y": []}, {"x", "y"}),
              "complete": ({"x": []}, {"x"})}
    for state in states:
        monkeypatch.setattr(P, "_default_column_state", lambda s=state: (s, states[s][0],
                                                                         states[s][1]))
        rep = P.build()
        ids = {b["id"] for b in rep["blockers"]}
        for mode, d in rep["delivered"].items():
            if not d["delivered"]:
                assert d.get("blocker_id") in ids, (
                    f"state {state!r}, mode {mode!r}: blocker_id "
                    f"{d.get('blocker_id')!r} names no blocker")
        assert rep["delivered"]["default"]["delivered"] is (state == "complete"), state


def test_the_evaluated_population_arm_is_the_merge_not_the_service_run():
    """The service run's own file holds only what the published run lacked."""
    assert P.GLORYX_1170.endswith("gloryx_service_preds_evaluated1170.json")
    assert P.GLORYX_1170_RAW.endswith("gloryx_service_preds_1170.json")
    assert P.GLORYX_1170 != P.GLORYX_1170_RAW
    by_pop = {(c["population"], c["arm"]): c for c in P.coverage()}
    assert by_pop[("evaluated1170", "gloryx")]["file"] == P.GLORYX_1170


def test_nothing_is_reported_as_run_without_a_live_check():
    rep = P.build()
    for arm, d in rep["arms"].items():
        if d.get("ran"):
            assert d.get("evidence_of_running"), f"{arm} claims a run with no evidence"

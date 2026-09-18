"""The second MetaTox submission exists, and everything asserted about it has to be measured.

Run: python -m pytest revision/tests/test_the_second_metatox_submission_is_recorded_with_its_coverage.py -q

Why this exists. paper2/si.tex stated that MetaTox has no quantity on the wider population "because
a second submission to that web service is not the authors' to make". That was true when written and
is now false: the submission was made and came back for all 879 substrates outside the comparison
set, which with the 291 already held is the whole evaluated 1,170 -- above the 0.99 coverage floor
scripts/typed_edit/population_definition.py applies.

Why it exists in THIS shape is a second story, and a worse one. When coverage stopped being the
reason, an output-budget objection was put in its place and reached the manuscript's prose: the
delivery returns 260.64 records per substrate against the comparison set's 36.43, so the halves were
said not to be one configuration. Every part of that was wrong. It divided a de-duplicated count by
a raw one -- on structures the rates are 39.19 and 36.43, and on the tautomer InChIKey the scorer
matches with, 36.45 and 33.65. It measured a quantity the axis never reads, since every arm is
truncated to k. And it was computed for one arm and asked of no other: an ADMITTED arm sits further
from parity than MetaTox does, and GLORYx -- whose whole-population column is built in exactly the
shape MetaTox's would be, one web-service run over the 291 and a separate run over the other 879 --
was admitted with no such check at all.

So the tests below are shaped by that failure as much as by the delivery:

  the submission is recorded with the digest of each file it came from;
  coverage is computed, and compared against the floor read from the axis rather than typed here;
  a blocker may not rest on a statistic computed for one arm only;
  the withdrawn blocker is recorded rather than reverted;
  both readings of which file is the arm are computed, and the fork is reported honestly;
  being eligible and being in the axis are kept apart.

The third is the one that matters. A rule that is only ever computed where it excludes is the
asymmetry this manuscript exists to object to, and it was committed here before it was caught.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARTIFACT = ROOT / "results" / "metatox_outside_submission.json"


def _blob() -> dict:
    assert ARTIFACT.exists(), (
        f"{ARTIFACT.relative_to(ROOT)} does not exist. The second MetaTox submission was delivered "
        f"and has to be recorded, together with the coverage it reaches and the coverage the "
        f"population axis requires.")
    return json.loads(ARTIFACT.read_text())


def test_the_submission_is_recorded_with_the_archive_it_came_from():
    """A delivery nobody can identify is not evidence.

    The archives are too large to track, so the record has to carry their digests: a reader who is
    sent the same files can check they are the same files, and one who is not can at least see that
    the record names something definite.
    """
    d = _blob()
    src = d.get("sources")
    assert src, "the record names no source archive"
    for s in src:
        assert s.get("name") and s.get("sha256"), f"a source is recorded without a digest: {s}"
        assert re.fullmatch(r"[0-9a-f]{64}", s["sha256"]), f"not a sha256: {s['sha256']!r}"


def test_the_coverage_is_computed_and_the_floor_is_read_from_the_axis():
    """Both halves of the comparison have to be measured, not typed.

    The coverage is a count over the evaluated population, and the floor belongs to
    population_definition.py. A test that hardcoded either would keep passing after the other moved.
    """
    d = _blob()
    cov = d.get("coverage") or {}
    for k in ("evaluated_substrates", "metatox_on_comparison_set",
              "recovered_from_this_submission", "union", "share"):
        assert k in cov, f"coverage does not record {k}"
    assert cov["union"] == cov["metatox_on_comparison_set"] + cov["recovered_from_this_submission"],\
        (f"the union {cov['union']} is not the sum of its parts "
         f"{cov['metatox_on_comparison_set']} + {cov['recovered_from_this_submission']}")
    assert abs(cov["share"] - cov["union"] / cov["evaluated_substrates"]) < 1e-9, \
        "the recorded share is not the union over the evaluated population"

    import population_definition as P
    floor = getattr(P, "COVERAGE_FLOOR", None)
    if floor is None:
        src = (ROOT / "scripts" / "typed_edit" / "population_definition.py").read_text()
        m = re.search(r"0\.99", src)
        assert m, "the axis no longer states a 0.99 coverage floor; this test has to re-read it"
        floor = 0.99
    assert abs(d.get("coverage_floor", -1) - floor) < 1e-9, (
        f"the record says the floor is {d.get('coverage_floor')} while the axis applies {floor}; "
        f"a floor quoted from memory is how a rule gets applied in whichever direction suits")


def test_admission_follows_the_floor_and_then_the_blockers():
    """The asymmetry this paper is about, turned into a check.

    Two rules, in order. Below the floor an arm is never admitted, whatever else is true of it.
    Above the floor it is admitted only if nothing else stops it, and anything else that stops it
    has to be written down as a blocker with the numbers behind it -- because "we left it out" with
    no measurement attached is the move this paper spends its length objecting to.

    Both directions bite. Dropping a blocker to let MetaTox in fails here, and so does keeping the
    arm out while recording no reason for it.
    """
    d = _blob()
    cov = d["coverage"]
    clears = cov["share"] >= d["coverage_floor"]
    assert d.get("coverage_clears_the_floor") is clears, (
        f"coverage is {cov['share']:.4f} against a floor of {d['coverage_floor']}, so clears "
        f"should be {clears}, and the record says {d.get('coverage_clears_the_floor')!r}")

    blockers = d.get("blockers")
    assert isinstance(blockers, list), "the record carries no blockers list, not even an empty one"
    eligible = d.get("eligible_for_the_whole_population")
    assert eligible is (clears and not blockers), (
        f"coverage clears the floor: {clears}; blockers recorded: {len(blockers or [])}; so "
        f"eligible should be {clears and not blockers} and the record says {eligible!r}")

    for b in blockers:
        assert b.get("name") and b.get("what_it_is"), f"a blocker is recorded unnamed: {b}"
        assert b.get("measured"), (
            f"blocker {b.get('name')!r} carries no measurement; an arm kept out on an unmeasured "
            f"objection is the same defect as one let in on an unmeasured one")
    if not clears:
        why = " ".join(b.get("name", "") for b in blockers).lower()
        assert "coverage" in why, "below the floor, the shortfall itself has to be a named blocker"


def test_the_output_budget_of_the_two_halves_is_compared_on_the_same_basis():
    """The comparison the submission README asked for in advance, done on the right quantity.

    This test is the one that was wrong. It used to divide a record count by a de-duplicated count
    and call the result an output budget. It now checks what the record has to compare: distinct
    predicted STRUCTURES per answering substrate, on both sides, de-duplicated the same way.

    Why structures and nothing else. The first batch's artefact is built from lists that
    scripts/metatox_smirks_ingest.py de-duplicates by canonical SMILES before it counts; the scorer
    de-duplicates again on the tautomer InChIKey; and grail_metabolism/metrics.py is set-based, so a
    structure written twice changes no number anywhere downstream. A supplier who writes one record
    per route is describing its export format, not its output size.

    The raw rate is still required in the record, because the mistake has to stay visible: a reader
    who has heard "seven times" needs to find both numbers and the reason one of them was wrong.
    """
    d = _blob()
    b = d.get("budget") or {}
    for k in ("comparison_set_substrates", "comparison_set_structures", "comparison_set_mean_output",
              "this_delivery_substrates", "this_delivery_structures", "this_delivery_mean_output",
              "ratio", "basis", "raw_record_rate", "why_the_raw_rate_is_not_used"):
        assert k in b, f"budget does not record {k}"
    for half in ("comparison_set", "this_delivery"):
        got, want = b[f"{half}_mean_output"], b[f"{half}_structures"] / b[f"{half}_substrates"]
        assert abs(got - want) < 0.005, (
            f"{half} mean output {got} is not {b[f'{half}_structures']}/{b[f'{half}_substrates']} "
            f"= {want:.4f}")
    # Against the counts, not the stored rates: the counts are exact and the rates are rounded, so
    # dividing the rounded pair would test the rounding rather than the ratio.
    exact = ((b["this_delivery_structures"] / b["this_delivery_substrates"])
             / (b["comparison_set_structures"] / b["comparison_set_substrates"]))
    assert abs(b["ratio"] - exact) < 1e-5, (
        f"the recorded ratio {b['ratio']} is not the two recorded structure counts divided, "
        f"which give {exact:.6f}")
    assert "de-duplicat" in b["basis"], (
        "the basis does not say the two sides are de-duplicated the same way, which is the whole "
        "content of this comparison")
    assert b["raw_record_rate"] > b["this_delivery_mean_output"], (
        "the raw record rate is not above the structure rate, so either the delivery has no "
        "duplicate records or one of the two is being measured wrongly; on this delivery the gap "
        "is what made the withdrawn blocker look convincing")


def test_a_blocker_may_not_rest_on_a_statistic_computed_for_one_arm_only():
    """The guard that exists because this record already failed it once.

    An output-budget blocker was asserted here, it became the sole ground for keeping MetaTox off
    the whole-population axis, and it was wrong in the most embarrassing available way: the same
    statistic had never been computed for any other arm. Computed, it turns out an ADMITTED arm sits
    further from parity than MetaTox does, and the arm whose whole-population column is built in
    exactly the same shape -- a web service run once over the 291 and again over the other 879 --
    was admitted with no such check at all. A rule available only where it excludes is the defect
    this manuscript is about.

    So: any blocker that talks about output budget has to carry the same measurement for the arms
    the axis already reads, and has to be outside what they do. This is the check that would have
    caught it, written after the fact and kept so it cannot recur.
    """
    d = _blob()
    peers = d.get("peer_two_half_ratios")
    assert isinstance(peers, dict) and peers, (
        "the record carries no cross-half ratio for the admitted arms; without them a budget "
        "blocker is a threshold invented for one comparator")
    ratios = {k: v["ratio"] for k, v in peers.items() if isinstance(v, dict) and "ratio" in v}
    assert len(ratios) >= 3, f"only {len(ratios)} admitted arms carry the statistic: {list(peers)}"

    mine = d["budget"]["ratio"]
    budget_blockers = [b for b in d["blockers"] if "budget" in b["name"] or "output" in b["name"]]
    worst_peer = max(ratios.values(), key=lambda r: abs(r - 1.0))
    outside = abs(mine - 1.0) > abs(worst_peer - 1.0)
    assert bool(budget_blockers) is outside, (
        f"this delivery's halves differ by {mine:.4f} and the admitted arms span "
        f"{min(ratios.values()):.4f}-{max(ratios.values()):.4f}; a budget blocker should "
        f"{'stand' if outside else 'not stand'}, and the record has {len(budget_blockers)}")


def test_the_withdrawn_blocker_is_recorded_rather_than_reverted():
    """A claim made and dropped is a decision, and a decision that leaves no trace is the defect.

    The budget objection reached the manuscript's prose before it was refuted. Silently deleting it
    would leave a reader no way to see that it was ever asserted, which is exactly what this project
    refuses to accept from other people's papers. The record has to say what was claimed, and why it
    no longer stands, in enough detail to be argued with.
    """
    d = _blob()
    w = d.get("withdrawn_blockers")
    assert isinstance(w, list) and w, "no withdrawn blocker is recorded"
    for b in w:
        assert b.get("name") and b.get("was_claimed"), f"a withdrawal names nothing: {b}"
        why = b.get("why_it_was_withdrawn")
        assert isinstance(why, list) and len(why) >= 2, (
            f"{b.get('name')!r} is withdrawn on fewer than two stated grounds; a claim that "
            f"reached the prose deserves more than one line of retraction")
        assert not any(b["name"] == live["name"] for live in d["blockers"]), (
            f"{b['name']!r} is recorded as both withdrawn and standing")


def test_the_two_readings_are_both_computed_and_the_fork_is_reported_honestly():
    """Which file is the arm changes the answer, and the record has to say so either way.

    Only the supplier can say which delivered file is the rule-based run. Under one reading MetaTox
    covers the whole population; under the other it falls short of the floor. That the fork decides
    something is itself a finding, and an earlier version of this record claimed the opposite -- so
    the flag is checked against the readings rather than taken on trust.
    """
    d = _blob()
    rs = d.get("readings") or {}
    assert len(rs) >= 2, "only one reading is recorded; the choice of file was left implicit"

    which = d.get("which_file_is_the_arm") or {}
    names = {s["name"] for s in d["sources"]}
    assert which.get("chosen") in names, f"the chosen arm {which.get('chosen')!r} is not a source"
    assert which.get("on_what_evidence"), "the choice is recorded without the evidence for it"
    assert which.get("against_it"), (
        "the record gives no evidence against its own choice; a fork recorded with only the "
        "reasons for one side is a decision dressed as a finding")

    by_reading = d.get("eligible_by_reading") or {}
    assert set(by_reading) == set(rs), "eligible_by_reading does not cover every reading"
    for name, r in rs.items():
        c = r["coverage"]
        assert abs(c["share"] - c["union"] / c["evaluated_substrates"]) < 1e-9, \
            f"reading {name}: the share is not the union over the evaluated population"
        assert r["clears_the_floor"] is (c["share"] >= d["coverage_floor"]), \
            f"reading {name}: clears_the_floor disagrees with its own share"
        assert by_reading[name] is r["clears_the_floor"], \
            f"reading {name}: eligible_by_reading disagrees with the reading itself"
    assert d.get("the_choice_of_file_decides_eligibility") is (len(set(by_reading.values())) > 1), \
        "the record misstates whether the choice of file decides anything"


def test_eligibility_is_not_confused_with_being_in_the_axis():
    """Two different things, and collapsing them would overstate what has been done.

    With the budget objection withdrawn there is no ground left to exclude MetaTox. That does not
    put it in the axis: no ranked column over the 879 exists yet. The record has to hold both, so a
    reader is neither told the arm was excluded on principle nor told it is present when it is not.
    """
    d = _blob()
    assert "eligible_for_the_whole_population" in d and "in_the_whole_population_axis" in d, \
        "the record does not separate being eligible from being present"
    assert d["eligible_for_the_whole_population"] is (
        d["coverage_clears_the_floor"] and not d["blockers"]), \
        "eligibility disagrees with the floor and the blockers the record itself carries"
    if d["eligible_for_the_whole_population"] and not d["in_the_whole_population_axis"]:
        assert d.get("why_not_in_the_axis_yet"), (
            "the arm is eligible and absent, and the record does not say why; an absence without "
            "its reason is the thing this paper objects to")

    # Checked against the repository, not against the record's own other fields. A flag that says
    # the arm is in the axis is a claim about what population_definition.py reads, and a claim of
    # work done is exactly the kind that has to be verified somewhere other than where it is made.
    import population_definition as P
    named = {k.lower() for k in P.WHOLE_TEST}
    present = any("metatox" in k for k in named)
    if present:
        path = next(v for k, v in P.WHOLE_TEST.items() if "metatox" in k.lower())
        present = Path(path).exists()
    assert d["in_the_whole_population_axis"] is present, (
        f"the record says the arm is "
        f"{'in' if d['in_the_whole_population_axis'] else 'not in'} the whole-population axis, and "
        f"population_definition.WHOLE_TEST {'names it with a file on disk' if present else 'does not'}"
        f" (arms named: {sorted(named)})")


def test_the_disagreement_between_the_two_files_is_recorded_rather_than_resolved_silently():
    """Two artefacts disagree, so the record says so instead of quietly preferring one.

    The supervisor sent two files. They do not return the same substrates or the same number of
    records, and the counts above are taken from one of them. Which one, and what the other says,
    belongs in the record: a reader who is handed only the larger number cannot tell that a second
    file exists and answers differently.
    """
    d = _blob()
    src = d["sources"]
    assert len(src) >= 2, "only one file is recorded; the delivery was two"
    for s in src:
        for k in ("substrates", "records"):
            assert k in s, f"source {s.get('name')!r} does not record {k}"
    assert len({s["substrates"] for s in src}) > 1, (
        "the recorded files agree on substrate count, which contradicts the disagreement this "
        "record is supposed to carry; recount or drop the claim")
    dis = d.get("disagreement") or {}
    assert dis.get("counts_taken_from") in {s["name"] for s in src}, (
        f"the record does not name which file its counts come from "
        f"(got {dis.get('counts_taken_from')!r})")
    assert dis.get("why"), "the record does not say why that file was the one used"

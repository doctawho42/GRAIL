"""Tests for merging GLORYx's two runs into one arm, written before the producer exists.

Run: python -m pytest revision/tests/test_phase2_gloryx_merge.py -q

The published GLORYx run covers the 291-substrate comparison set. The Phase 2 run asks the service
only for what that run does not hold, so its output covers 879 substrates. Neither file covers the
evaluated population, and this is the trap: `phase1_tmain.arm_key_lists` reads a list arm with
`preds.get(s, [])`, so declaring either file as the arm on `evaluated1170` would hand back an empty
list for every substrate it lacks and score those as misses. GLORYx would be understated on a
quarter of the population in the one table built to make the systems comparable.

So the arm is a merge of both, and the merge has to refuse rather than fill. The tests below are
mostly refusals: incomplete coverage, an overlap between the two sources, a missing input. The
one-sided failure mode -- a merge that silently covers less than the population -- is the whole
reason this module exists, so it is the first test.
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
import phase2_gloryx_merge as M  # noqa: E402

PUBLISHED = ROOT / "results" / "gloryx_service_preds.json"
POPULATION = ROOT / "results" / "test_references.json"


def _published():
    return json.loads(PUBLISHED.read_text())["predictions"]


def _population():
    return sorted(json.loads(POPULATION.read_text()))


def _wider_file(path, substrates, per=2):
    """A stand-in for the service run's output, in the shape that producer writes."""
    preds = {s: [f"C{'C' * (i + 1)}O" for i in range(per)] for s in substrates}
    detail = {s: [{"rank": i + 1, "score": 1.0 - i / 10, "smiles": m}
                  for i, m in enumerate(v)] for s, v in preds.items()}
    path.write_text(json.dumps({"predictions": preds, "with_rank_and_score": detail,
                                "obtained_from": {"jobs": ["job-a", "job-b"]}}))
    return path


# --------------------------------------------------------------------------- the coverage refusal

def test_a_merge_that_does_not_cover_the_population_is_refused(tmp_path):
    """The defect this module exists to prevent: an arm short of the population reads as losses.

    One substrate is withheld from the wider file, so the union covers 1,169 of 1,170. That must
    raise, not return a map that `arm_key_lists` would happily turn into 1,169 scores and one
    silent zero.
    """
    pop = _population()
    missing_one = sorted(set(pop) - set(_published()))[:-1]
    wider = _wider_file(tmp_path / "wider.json", missing_one)
    try:
        M.merge(PUBLISHED, wider, population=pop)
    except Exception as e:
        assert "cover" in str(e).lower() or "1169" in str(e) or "missing" in str(e).lower(), (
            f"the refusal must say the population is not covered: {e!r}")
        return
    raise AssertionError("an incomplete merge must be refused, not returned")


def test_a_complete_merge_covers_the_population_exactly(tmp_path):
    pop = _population()
    pending = sorted(set(pop) - set(_published()))
    wider = _wider_file(tmp_path / "wider.json", pending)
    got = M.merge(PUBLISHED, wider, population=pop)
    assert set(got["predictions"]) == set(pop)
    assert len(got["predictions"]) == 1170
    assert got["n_from_published"] == 291
    assert got["n_from_wider"] == 879
    assert got["n_from_published"] + got["n_from_wider"] == 1170


# --------------------------------------------------------------------------- the overlap refusal

def test_an_overlap_between_the_two_sources_is_refused(tmp_path):
    """Two files claiming the same substrate means one of them is not the run it says it is.

    Silently preferring either would hide that, and the two runs were made under different
    submission sets, so a substrate in both is a fact about the artefacts and not a tie to break.
    """
    pop = _population()
    pending = sorted(set(pop) - set(_published()))
    overlapping = pending + sorted(_published())[:1]
    wider = _wider_file(tmp_path / "wider.json", overlapping)
    try:
        M.merge(PUBLISHED, wider, population=pop)
    except Exception as e:
        assert "overlap" in str(e).lower() or "both" in str(e).lower(), (
            f"the refusal must name the overlap: {e!r}")
        return
    raise AssertionError("an overlap between the sources must be refused")


def test_a_missing_input_is_refused_rather_than_treated_as_empty(tmp_path):
    pop = _population()
    try:
        M.merge(PUBLISHED, tmp_path / "does_not_exist.json", population=pop)
    except Exception as e:
        assert "exist" in str(e).lower() or "missing" in str(e).lower() or "refus" in str(e).lower()
        return
    raise AssertionError("a missing input must be refused, not read as an empty run")


# --------------------------------------------------------------------------- what is carried

def test_each_substrate_records_which_run_answered_it(tmp_path):
    """The two halves were obtained on different dates under different submission sets, so which
    run a substrate came from is part of the column's provenance."""
    pop = _population()
    pending = sorted(set(pop) - set(_published()))
    wider = _wider_file(tmp_path / "wider.json", pending)
    got = M.merge(PUBLISHED, wider, population=pop)
    src = got["source_of_each_substrate"]
    assert set(src) == set(pop)
    assert set(src.values()) == {"published", "wider"}
    assert sum(1 for v in src.values() if v == "published") == 291


def test_the_rank_and_score_detail_survives_both_halves(tmp_path):
    pop = _population()
    pending = sorted(set(pop) - set(_published()))
    wider = _wider_file(tmp_path / "wider.json", pending)
    got = M.merge(PUBLISHED, wider, population=pop)
    assert set(got["with_rank_and_score"]) == set(pop), (
        "dropping the detail for one half would make the column's ordering unexplainable")


def test_the_merged_predictions_are_the_sources_values_unchanged(tmp_path):
    """A merge must not re-key, re-order or re-truncate: those are the table's own steps."""
    pop = _population()
    pending = sorted(set(pop) - set(_published()))
    wider = _wider_file(tmp_path / "wider.json", pending)
    got = M.merge(PUBLISHED, wider, population=pop)
    pub = _published()
    for s in sorted(pub)[:20]:
        assert got["predictions"][s] == pub[s]
    wider_preds = json.loads(wider.read_text())["predictions"]
    for s in pending[:20]:
        assert got["predictions"][s] == wider_preds[s]

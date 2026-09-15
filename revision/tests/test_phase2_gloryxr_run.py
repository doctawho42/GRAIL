"""Tests for the local GLORYxR run over the evaluated population, written before it exists.

Run: python -m pytest revision/tests/test_phase2_gloryxr_run.py -q

GLORYxR needs Python >= 3.13 and is installed in an isolated environment, while this test suite
runs on the repository's own 3.10 interpreter. So the runner's `import gloryxr` has to be lazy,
inside the function that predicts, and everything else -- the population, the resume set, the
output paths, the ranking -- has to be testable here. The first test pins exactly that: importing
the module must not require gloryxr, or none of these tests can run at all.

Two runs are asked for, default and strict-SOM, and they differ only by `strict_soms` on the
Reactor. They must not be able to land on each other's output, and neither may land on the
service-derived files already in the repository.
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

# Hard import: until the runner exists these tests must fail, not skip.
import phase2_gloryxr_run as R  # noqa: E402


def test_the_module_imports_without_gloryxr_installed():
    """The runner is driven by a 3.13 interpreter; this suite is not.

    A module-level `import gloryxr` would make every test below a collection error on the
    repository's own interpreter, and the failure would look like a broken test rather than a
    misplaced import.
    """
    assert "gloryxr" not in sys.modules, (
        "importing the runner pulled in gloryxr; the import must be lazy")


def test_the_population_is_the_evaluated_set():
    subs = R.population()
    assert len(subs) == 1170
    truth = set(json.loads((ROOT / "results" / "test_references.json").read_text()))
    assert set(subs) == truth
    assert subs == sorted(subs), "the order must be deterministic across runs"


def test_the_two_modes_cannot_land_on_each_other_or_on_the_service_files():
    a, b = R.output_path("default"), R.output_path("strict")
    assert a != b
    for p in (a, b):
        name = Path(p).name
        assert "gloryxr" in name, f"{name} does not say which system produced it"
        assert name not in {"gloryx_service_preds.json",
                            "gloryx_service_preds_1170.json",
                            "gloryx_service_preds_evaluated1170.json"}
    assert R.checkpoint_path("default") != R.checkpoint_path("strict")


def test_an_unknown_mode_is_refused_rather_than_guessed():
    try:
        R.output_path("whatever")
    except TypeError as e:
        raise AssertionError(f"refused by signature rather than by choice: {e}")
    except Exception as e:
        assert "whatever" in str(e), f"the refusal must name what it rejected: {e!r}"
        return
    raise AssertionError("an unknown mode must raise, not fall back to a default")


def test_ranking_orders_by_score_and_keeps_the_best_duplicate():
    """One prediction is one product structure, ordered by the score the model gave it.

    Duplicates are collapsed to their highest score rather than dropped in file order: the
    service column is a ranked list of distinct metabolites, and a column that keeps whichever
    duplicate happened to come first is ordered by an accident.
    """
    pairs = [("CCO", 0.2), ("CCN", 0.9), ("CCO", 0.7), ("CCC", 0.5)]
    assert R.rank(pairs) == ["CCN", "CCO", "CCC"]


def test_ranking_is_stable_for_equal_scores():
    """Ties broken by the structure itself, so two runs of the same data agree."""
    pairs = [("CCO", 0.5), ("CCN", 0.5), ("CCC", 0.5)]
    assert R.rank(pairs) == R.rank(list(reversed(pairs)))


def test_pending_excludes_what_an_existing_output_already_holds(tmp_path):
    out = tmp_path / "partial.json"
    subs = R.population()
    out.write_text(json.dumps({"predictions": {s: [] for s in subs[:100]}}))
    left = R.pending(subs, out)
    assert len(left) == 1070
    assert not (set(left) & set(subs[:100]))
    assert left == R.pending(subs, out), "the resume set must be deterministic"


def test_pending_against_a_missing_output_is_the_whole_population(tmp_path):
    subs = R.population()
    assert R.pending(subs, tmp_path / "nope.json") == subs


def test_an_unprocessed_substrate_is_not_the_same_as_one_answered_with_nothing():
    """The defect this guards is the most dangerous one in the runner.

    Written AFTER the function, which is the wrong order and is recorded as such: the ground rule
    is a failing test first. It is kept because the semantics it pins are exactly what went wrong.

    The runner used to close with `for s in requested: flat.setdefault(s, [])`, so a run stopped
    halfway wrote an artifact carrying all 1,170 keys with empty lists for everything it never
    reached. `phase1_tmain.coverage_gaps` checks key PRESENCE only, so that artifact passes the
    coverage gate while three quarters of it is silence recorded as "predicted nothing" -- every
    recall figure downstream then understates the comparator.

    A substrate RDKit cannot parse is different: it is reached, deliberately recorded as an empty
    list inside the loop, and must NOT be reported as unprocessed.
    """
    requested = ["a", "b", "c", "d"]
    # 'b' was reached and answered with nothing (e.g. unparseable); 'c' and 'd' never ran.
    flat = {"a": ["CCO"], "b": []}
    assert R.unprocessed(flat, requested) == ["c", "d"]
    assert R.unprocessed({s: [] for s in requested}, requested) == []
    assert R.unprocessed({}, requested) == requested


def test_the_models_directory_is_required_and_its_absence_is_refused():
    """The delivered dumps are not in the repository, so the path is given rather than assumed."""
    try:
        R.load_models(Path("/nonexistent/models"))
    except Exception as e:
        assert "nonexistent" in str(e) or "exist" in str(e).lower(), repr(e)
        return
    raise AssertionError("a missing models directory must be refused")

"""The invariant T_main rests on: no declared list arm is short of its population.

Run: python -m pytest revision/tests/test_phase1_arm_coverage.py -q

`arm_key_lists` reads a list arm with `preds.get(s, [])`, so a substrate the file lacks arrives as
an empty list and is scored as a miss. Every arm declared today covers its population, which is
why the table is sound -- but nothing enforced it, and the GLORYx column over the evaluated
population came within one edit of breaking it: the service run's output holds only the 879
substrates the published run lacked, so declaring that file would have understated GLORYx on the
other 291.

That case was caught by reasoning about what the run would produce. This makes it structural.

An arm whose file is missing entirely is deliberately NOT a gap: `arm_key_lists` returns None and
the arm yields no cells at all, which is the correct treatment of a comparator that was never run.
The dangerous case is the file that exists and is short, because it looks like data.
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

# Hard import: until the check exists these tests must fail, not skip.
import phase1_tmain as T  # noqa: E402


def test_the_check_reports_a_partially_covering_arm(tmp_path):
    """Non-vacuous by construction: a file covering three of four substrates must be reported.

    Written against a synthetic arm rather than a real one, so this fails for the reason it names
    even on a day when every declared arm is complete. A guard whose pass depends on the data
    happening to be sound is not a guard.
    """
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"a": ["x"], "b": ["y"], "c": []}))
    arms = {"synthetic": {"short": ("list", str(partial.relative_to(ROOT))
                                    if partial.is_relative_to(ROOT) else str(partial), None)}}
    gaps = T.coverage_gaps(arms=arms, members={"synthetic": {"a", "b", "c", "d"}})
    assert len(gaps) == 1, gaps
    got = gaps[0]
    assert got["arm"] == "short"
    assert got["n_missing"] == 1
    assert got["n_covered"] == 3
    assert got["first_missing"] == "d"


def test_a_complete_arm_is_not_reported(tmp_path):
    full = tmp_path / "full.json"
    full.write_text(json.dumps({"a": ["x"], "b": [], "c": ["z"]}))
    arms = {"synthetic": {"ok": ("list", str(full), None)}}
    assert T.coverage_gaps(arms=arms, members={"synthetic": {"a", "b", "c"}}) == []


def test_an_arm_with_no_file_at_all_is_not_a_gap(tmp_path):
    """Absent is not short. The arm yields no cells, which is the right answer for a comparator
    that was never run; calling it a gap would demand a merge for something never attempted."""
    arms = {"synthetic": {"missing": ("list", str(tmp_path / "nope.json"), None)}}
    assert T.coverage_gaps(arms=arms, members={"synthetic": {"a", "b"}}) == []


def test_the_declared_accessor_is_honoured(tmp_path):
    """A file whose predictions sit under an envelope key must be read through it, or every
    substrate looks missing and the gap report is nonsense."""
    env = tmp_path / "env.json"
    env.write_text(json.dumps({"predictions": {"a": ["x"], "b": []},
                               "provenance": {"script": "whatever"}}))
    arms = {"synthetic": {"enveloped": ("list", str(env), "predictions")}}
    assert T.coverage_gaps(arms=arms, members={"synthetic": {"a", "b"}}) == []


def test_pool_arms_are_not_checked_this_way():
    """A pool arm is assembled from shards rather than read as a substrate map, and
    `arm_key_lists` indexes it directly, so a missing substrate raises there instead of turning
    into an empty list. Only list arms have the silent failure mode."""
    arms = {"synthetic": {"p": ("pool", "results/widepools_implicit/w*.json")}}
    assert T.coverage_gaps(arms=arms, members={"synthetic": {"a"}}) == []


def _declared_shortfalls() -> dict:
    """What the deployment table records about each comparator's coverage.

    A short arm still scores misses; that has not changed and is not defensible on its own. What
    this reads is whether the shortfall was WRITTEN DOWN, so the zeros are a stated treatment
    instead of an accident nobody counted. BioTransformer crashes deterministically on fifteen of
    the comparison set and the arm cannot answer for them; writing them into the file as empty
    lists would make a crash and a prediction of nothing the same record again, which is the
    defect the arm was replaced to remove.
    """
    f = ROOT / "results/deployment_table.json"
    if not f.exists():
        return {}
    cov = json.loads(f.read_text()).get("comparator_coverage") or {}
    return {a: v.get("scored_zero_for_absence", 0) for a, v in cov.items()}


def test_no_arm_is_short_of_its_population_without_saying_so():
    """The real invariant, over the arms T_main actually declares.

    Slow: it resolves both populations through the producer's own accessor rather than trusting a
    count, because three comparator files carry 291 keys and one of them is a different 291.

    A gap is admissible only when the deployment table records a shortfall of exactly that size
    for that arm. An undeclared gap -- the GLORYx case in the module docstring, where a file
    covering the wrong 879 would have been declared silently -- still fails here.
    """
    declared = _declared_shortfalls()
    undeclared = [g for g in T.coverage_gaps()
                  if declared.get(g["arm"]) != g["n_missing"]]
    assert undeclared == [], (
        "arms short of their population with no shortfall recorded in "
        f"results/deployment_table.json, which would score misses in silence: {undeclared}")


def test_a_shortfall_of_the_wrong_size_is_still_refused(monkeypatch):
    """The relaxation must not become a pass-through.

    Reading a declaration and comparing it to nothing would accept any gap from any arm that
    appears in the coverage record at all. The declared number has to MATCH the gap, so an arm
    that loses more substrates than anyone wrote down still fails, and so does one that appears
    in the record with a shortfall of zero.
    """
    import revision.tests.test_phase1_arm_coverage as M
    gap = [{"population": "comparison291", "arm": "biotransformer",
            "file": "x.json", "n_missing": 15, "n_covered": 276, "first_missing": "C"}]
    monkeypatch.setattr(T, "coverage_gaps", lambda *a, **k: gap)

    for declared, why in (({"biotransformer": 14}, "one fewer than the gap"),
                          ({"biotransformer": 0}, "recorded as complete"),
                          ({}, "not in the record at all")):
        monkeypatch.setattr(M, "_declared_shortfalls", lambda d=declared: d)
        undeclared = [g for g in T.coverage_gaps()
                      if M._declared_shortfalls().get(g["arm"]) != g["n_missing"]]
        assert undeclared, f"a shortfall {why} was accepted"

    monkeypatch.setattr(M, "_declared_shortfalls", lambda: {"biotransformer": 15})
    undeclared = [g for g in T.coverage_gaps()
                  if M._declared_shortfalls().get(g["arm"]) != g["n_missing"]]
    assert undeclared == [], "an exactly declared shortfall must be accepted"

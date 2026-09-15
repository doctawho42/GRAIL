"""No arm may cover the whole evaluated population and be absent from the population axis in silence.

Run: python -m pytest revision/tests/test_population_axis_names_every_whole_split_arm.py -q

Why this exists. scripts/typed_edit/population_definition.py decides which comparators can be read
on the whole evaluated test set from a literal dict, WHOLE_TEST, naming three files. It already has
a mechanism for rejecting an arm that does not cover the population: an arm present but covering
less than 99% of the substrates is dropped with a line to stderr, and the artifact records which
arms reached the population. What it has no mechanism for is an arm that covers the population and
was never offered to the dict.

That is what happened. The artifact was last written on 2026-09-03; the GLORYx column over all
1,170 substrates was added on 2026-09-15, twelve days later, and holds 1,170 of the 1,170 with
1,166 carrying at least one prediction. The literal dict could not know about a file that did not
exist when it was written, so \\numPopdefWholearms printed three while the repository held four.
The producer's own comment explains who is in the dict and why -- BioTransformer joined when it was
run there, MetaTox cannot follow because a second submission to a web service is not ours to make
-- and says nothing about GLORYx, because there was nothing to say yet.

So the defect is not the number. It is that nothing compared the dict against the repository. This
test is that comparison: every prediction file that covers the evaluated population must be either
in WHOLE_TEST or named in the producer as deliberately excluded, with a reason. An arm may be left
out; it may not be left out silently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import population_definition as P  # noqa: E402

POPULATION = ROOT / "results" / "test_references.json"

# Candidate columns: a prediction file is a candidate if it carries per-substrate predictions for
# the evaluated population. Named explicitly rather than globbed, because a glob over results/
# would sweep in pools, caches and shards and turn this gate into a survey of the directory.
CANDIDATES = {
    "sygma": "results/sygma_fulltest_predictions.json",
    "metapredictor": "artifacts/tier2_1170/metapredictor_preds.json",
    "biotransformer": "results/biotransformer_fulltest_preds.json",
    "gloryx": "results/gloryx_service_preds_evaluated1170.json",
    "metatox": "results/metatox_1170_preds.json",
}

COVERAGE_FLOOR = 0.99


def _population() -> set:
    return set(json.loads(POPULATION.read_text()))


def _substrates(rel: str) -> set:
    """The substrate keys a prediction file carries, read through its envelope if it has one."""
    path = ROOT / rel
    if not path.exists():
        return set()
    blob = json.loads(path.read_text())
    inner = blob.get("predictions") if isinstance(blob, dict) else None
    return set(inner if isinstance(inner, dict) else (blob if isinstance(blob, dict) else ()))


def test_the_population_is_readable_so_a_zero_below_means_something():
    """Positive control. Every count in this file is a share of this set, and a gate that cannot
    read it would pass by finding nothing."""
    pop = _population()
    assert len(pop) > 1000, f"the evaluated population reads as {len(pop)} substrates"


def test_every_arm_covering_the_population_is_offered_to_the_axis():
    """The gate that was missing.

    An arm covering the population must appear in WHOLE_TEST. If it does not, the axis measures a
    smaller comparison than the repository holds, and the count the manuscript prints is short by
    however many such arms exist.
    """
    pop = _population()
    offered = set(P.WHOLE_TEST)
    missing = []
    for name, rel in CANDIDATES.items():
        covered = len(_substrates(rel) & pop)
        if not covered:
            continue
        if covered >= COVERAGE_FLOOR * len(pop) and name not in offered:
            missing.append(f"{name} covers {covered} of {len(pop)} from {rel} but is not in "
                           f"WHOLE_TEST")
    assert not missing, ("an arm covers the evaluated population and the axis does not offer it: "
                         + "; ".join(missing))


def test_an_arm_left_out_is_left_out_with_a_reason():
    """Exclusion stays allowed, silence does not.

    MetaTox is the case this protects: its file does not exist, a second submission to a web
    service is not the authors' to make, and the producer says so in prose. A future exclusion must
    do the same, so this asserts the producer names every candidate it does not read.
    """
    source = (ROOT / "scripts" / "typed_edit" / "population_definition.py").read_text()
    unexplained = []
    for name in CANDIDATES:
        if name in P.WHOLE_TEST:
            continue
        if name not in source:
            unexplained.append(name)
    assert not unexplained, ("a candidate arm is neither read nor mentioned by the producer, so "
                             "its absence carries no reason: " + ", ".join(unexplained))


def test_the_coverage_floor_is_the_producer_s_own():
    """The floor this gate applies must be the floor the producer applies, or the two disagree
    about which arms qualify and the gate becomes advice."""
    source = (ROOT / "scripts" / "typed_edit" / "population_definition.py").read_text()
    assert "0.99" in source, ("the producer no longer uses a 0.99 coverage floor; this gate's "
                              "COVERAGE_FLOOR has to be re-read from it rather than assumed")

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
run there, MetaTox does not follow -- and says nothing about GLORYx, because there was nothing to
say yet. MetaTox's reason has since changed and the comment was rewritten with it: the second
submission was made, for the 879 substrates outside the comparison set, and what keeps that arm out
is no longer an unavailable submission. Two files came back and which is the rule-based run is
settled by nothing in the repository, so both are read; the arm falls short under either, on
coverage under one and on output budget under both. Measured in
results/metatox_outside_submission.json.

So the defect is not the number. It is that nothing compared the dict against the repository. This
test is that comparison: every prediction file that covers the evaluated population must be either
in WHOLE_TEST or named in the producer as deliberately excluded, with a reason. An arm may be left
out; it may not be left out silently.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import population_definition as P  # noqa: E402
import _population as shared  # noqa: E402

# The population comes from the shared helper, not from a file named here. results/
# test_references.json holds the references as structures and is deliberately not released
# (NOTICE.md, docs/ICLR_SUBMISSION.md): the source licences do not let this repository ship the
# corpus, so a clone has results/test_reference_descriptors.json instead. _population's own
# docstring records that reading the structural file directly is how every population became
# unloadable once the release stopped shipping it, and that the failure surfaced as a swallowed
# exception. This test read it directly in its first form, which would have made its positive
# control raise FileNotFoundError in any clone -- a completeness gate that cannot run is worse
# than no gate, because its silence reads as a pass.

# Candidate columns are DISCOVERED, and the first version of this file typed them into a dict.
# That dict was hand-maintained, which is the defect this gate exists to catch: the producer's
# WHOLE_TEST is hand-maintained, a column went missing from it in silence, and the gate written to
# notice reproduced the same blind spot one level up. It did not notice GLORYxR -- two files, both
# covering 1,170 of 1,170 -- because nobody added them to the list, which is precisely the failure
# mode it was written for. A gate whose completeness is maintained by the same hand as the thing it
# checks is not a completeness gate.
#
# The original objection to globbing was that a sweep of results/ would drag in pools, caches and
# shards. Measured rather than assumed: over results/**/*pred*.json and artifacts/**/*pred*.json,
# exactly seven files carry per-substrate predictions reaching the floor, and all seven are
# comparator columns. GRAIL's own pools are not named *pred*, so the naming convention answers the
# objection that the typed list was defending against.
GLOBS = ("results/**/*pred*.json", "artifacts/**/*pred*.json")

COVERAGE_FLOOR = 0.99


def _candidates() -> dict:
    """Every prediction file reaching the coverage floor, as repo-relative path -> substrates.

    Compared against WHOLE_TEST by PATH and never by a name derived from the filename: two of the
    seven differ only by a mode suffix, and a derived name would have collapsed them into one and
    hidden the second.
    """
    pop = _evaluated_population()
    out = {}
    for pat in GLOBS:
        for f in sorted(glob.glob(pat, recursive=True)):
            rel = str(Path(f).relative_to(ROOT) if Path(f).is_absolute() else f)
            try:
                blob = json.loads((ROOT / rel).read_text())
            except Exception:
                continue
            inner = blob.get("predictions") if isinstance(blob, dict) else None
            d = inner if isinstance(inner, dict) else (blob if isinstance(blob, dict) else None)
            if not isinstance(d, dict):
                continue
            covered = len(set(d) & pop)
            if covered >= COVERAGE_FLOOR * len(pop):
                out[rel] = covered
    return out


def _evaluated_population() -> set:
    """The evaluated test set, through the accessor that works with or without the structures.

    load_population("clean_test") reads the reference counts from whichever of the two files is
    present and asserts the split is the 1,170 substrates / 2,597 references the paper reports, so
    a disagreement about the population fails here rather than silently changing what the counts
    below are a share of.
    """
    return set(shared.load_population("clean_test"))


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
    pop = _evaluated_population()
    assert len(pop) > 1000, f"the evaluated population reads as {len(pop)} substrates"


def test_every_arm_covering_the_population_is_offered_to_the_axis():
    """The gate that was missing.

    An arm covering the population must appear in WHOLE_TEST. If it does not, the axis measures a
    smaller comparison than the repository holds, and the count the manuscript prints is short by
    however many such arms exist.
    """
    pop = _evaluated_population()
    cands = _candidates()
    assert cands, "no prediction file reaches the coverage floor, so this gate verified nothing"
    offered = {str(Path(v).resolve()) for v in P.WHOLE_TEST.values()}
    missing = [f"{rel} covers {n} of {len(pop)} but is not in WHOLE_TEST"
               for rel, n in sorted(cands.items()) if str((ROOT / rel).resolve()) not in offered]
    assert not missing, ("an arm covers the evaluated population and the axis does not offer it: "
                         + "; ".join(missing))


def test_an_arm_left_out_is_left_out_with_a_reason():
    """Exclusion stays allowed, silence does not.

    MetaTox is the case this protects, and it is now the case that shows why the test is worth
    having. Its whole-population file does not exist and the producer says in prose why not -- and
    the reason it gives is no longer the one it gave when this test was written, because the
    submission that was said to be impossible was made. If that file is ever built from the
    delivery, this test will see an arm covering the population that WHOLE_TEST does not name, and
    the budget difference will have to be written down as the exclusion rather than assumed. A
    future exclusion must do the same, so this asserts the producer names every candidate it does
    not read.
    """
    source = (ROOT / "scripts" / "typed_edit" / "population_definition.py").read_text()
    offered = {str(Path(v).resolve()) for v in P.WHOLE_TEST.values()}
    unexplained = []
    for rel in _candidates():
        if str((ROOT / rel).resolve()) in offered:
            continue
        if Path(rel).stem not in source and Path(rel).name not in source:
            unexplained.append(rel)
    assert not unexplained, ("a candidate arm is neither read nor mentioned by the producer, so "
                             "its absence carries no reason: " + ", ".join(unexplained))


def test_the_coverage_floor_is_the_producer_s_own():
    """The floor this gate applies must be the floor the producer applies, or the two disagree
    about which arms qualify and the gate becomes advice."""
    source = (ROOT / "scripts" / "typed_edit" / "population_definition.py").read_text()
    assert "0.99" in source, ("the producer no longer uses a 0.99 coverage floor; this gate's "
                              "COVERAGE_FLOOR has to be re-read from it rather than assumed")

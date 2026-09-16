"""The caveats printed beside the symmetric counter must be measured, and must be able to go false.

Run: python -m pytest revision/tests/test_the_caveats_beside_the_counter_are_measured_not_typed.py -q

Why this exists. The population axis prints a count of contrast cells in both directions, and beside
it two caveats that decide how that count reads:

  the favourable side is one comparator's -- removing BioTransformer leaves the unfavourable totals
  untouched and takes cells off the favourable side only;

  MetaTox is in neither total, and on the comparison set it separates for this work more often than
  against it, so its absence removes more from the favourable side than from the other.

The keys behind both were added to scripts/paper2_numbers.py without a test, which is the gap this
closes. Two of the assertions here are not about the producer at all but about the claims. The
Supporting Information now contains the sentence "The unfavourable totals do not move", which is
true of this artifact and is not true by construction: a single unfavourable separating cell from
BioTransformer would falsify it. Nothing in the repository noticed that, so the sentence could have
gone on being printed after it stopped being true. That is what test_removing_biotransformer_leaves
_the_unfavourable_side_untouched is for -- it fails rather than letting the paper print it.

The recounts below do not import the producer's helpers; they walk the artifacts directly. That
catches a key that stopped being emitted, a change to the set of budgets audited, and an artifact
that moved under the producer. It does not make them independent in the strong sense, and saying so
would be the overclaim this file exists to avoid: the MetaTox recount reads the same file, the same
"gap" field and the same two arm names the producer reads, so a producer reading the wrong field
would be matched by a test reading the wrong field, and the two would agree.

What covers that case is the pair of claim tests -- test_removing_biotransformer_leaves_the
_unfavourable_side_untouched and test_metatox_absence_removes_more_from_the_favourable_side. They
assert the sentences printed in the Supporting Information rather than the keys behind them, so
they go red when a sentence stops being true whatever the producer happens to say.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper2_macros  # noqa: E402
import paper2_numbers  # noqa: E402

ARMS = ("deployed_minus_comparator", "exhaustive_minus_comparator")
POPULATIONS = {"the comparison set": "comp", "the whole evaluated test set": "whole"}
# The budgets the producer audits MetaTox over, repeated rather than imported: see the docstring of
# test_the_metatox_caveat_matches_an_independent_recount for why reading the producer's own tuple
# would make the recount agree with it by construction.
MTX_BUDGETS = ("5", "10", "15", "30", "50")


@lru_cache(maxsize=None)
def _numbers() -> dict:
    """The producer's own output, computed in this process rather than read from the artifact.

    Memoised because build() is the expensive call in this repository's test suite and five of the
    tests below want the same dictionary. Callers only read it; nothing here mutates the result.

    Reading results/paper2_numbers.json instead would pass while the producer was broken and the
    artifact merely stale, which is the failure this whole family of gates exists to catch.
    """
    return paper2_numbers.build()


@lru_cache(maxsize=None)
def _population_rows() -> dict:
    """The artifact's contrast rows. The one caller that mutates them deep-copies first."""
    return json.loads((ROOT / "results/population_definition.json").read_text())["contrasts"]


@lru_cache(maxsize=None)
def _deployment_contrasts() -> dict:
    return json.loads((ROOT / "results/deployment_table.json").read_text())["contrasts"]


def _tally(row: dict, skip: str = "") -> tuple:
    """Separating cells in a population row, counted for and against this work.

    A positive difference favours this work. Cells whose interval covers zero are not counted on
    either side, so the two returned numbers do not have to sum to the number of cells.
    """
    favour = against = 0
    for comparator, block in row.items():
        if not isinstance(block, dict) or not any(a in block for a in ARMS):
            continue
        if comparator == skip:
            continue
        for arm in ARMS:
            for cell in (block.get(arm) or {}).values():
                if cell["excludes_zero"]:
                    favour += cell["difference"] > 0
                    against += cell["difference"] < 0
    return favour, against


def test_the_artifacts_are_readable_so_a_zero_below_means_something():
    """Positive control. Every count here is drawn from these two files, and a test that silently
    read nothing would pass by finding no disagreement."""
    rows = _population_rows()
    assert set(POPULATIONS) <= set(rows), f"population rows present: {sorted(rows)}"
    for name in POPULATIONS:
        favour, against = _tally(rows[name])
        assert favour + against > 0, f"{name} yields no separating cells at all"
    assert _deployment_contrasts(), "the deployment table carries no contrasts"


def test_the_recount_is_sensitive_to_the_data_it_reads():
    """Perturbation control for the recount itself.

    Flipping the sign of one separating cell in a copy of the artifact must move the tally. Without
    this, every agreement below could be two identical readings of a constant.
    """
    rows = copy.deepcopy(_population_rows())
    row = rows["the whole evaluated test set"]
    before = _tally(row)
    for block in row.values():
        if not isinstance(block, dict) or not any(a in block for a in ARMS):
            continue
        for arm in ARMS:
            for cell in (block.get(arm) or {}).values():
                if cell["excludes_zero"]:
                    cell["difference"] = -cell["difference"]
                    assert _tally(row) != before, (
                        "flipping a separating cell did not change the tally, so the tally is not "
                        "reading the cells")
                    return
    raise AssertionError("no separating cell was found to perturb")


def test_the_without_biotransformer_keys_match_an_independent_recount():
    """The producer's caveat keys against a recount that shares none of its code."""
    n, rows = _numbers(), _population_rows()
    for name, short in POPULATIONS.items():
        favour, against = _tally(rows[name], skip="biotransformer")
        assert n[f"popdef.withoutbtfor{short}"] == favour, (
            f"{name}: producer says {n[f'popdef.withoutbtfor{short}']} for this work without "
            f"BioTransformer, recount says {favour}")
        assert n[f"popdef.withoutbtagainst{short}"] == against, (
            f"{name}: producer says {n[f'popdef.withoutbtagainst{short}']} against this work "
            f"without BioTransformer, recount says {against}")


def test_removing_biotransformer_leaves_the_unfavourable_side_untouched():
    """The claim the Supporting Information prints, not the key behind it.

    "The unfavourable totals do not move" is a statement about this artifact and can stop being
    true: one separating cell in which a GRAIL arm trails BioTransformer would break it. If that
    ever happens the sentence has to change, and this is what says so.
    """
    n = _numbers()
    for short in POPULATIONS.values():
        assert n[f"popdef.withoutbtagainst{short}"] == n[f"popdef.againstthiswork{short}"], (
            f"BioTransformer now contributes a separating cell against this work on the {short} "
            f"population: without it {n[f'popdef.withoutbtagainst{short}']} against, with it "
            f"{n[f'popdef.againstthiswork{short}']}. The sentence in paper2/si.tex saying the "
            f"unfavourable totals do not move is now false and has to be rewritten.")


def test_the_metatox_caveat_matches_an_independent_recount():
    """MetaTox's contrast cells, recounted over the same budgets the producer audits.

    MTX_BUDGETS is repeated here rather than imported on purpose: a recount that read the producer's
    own tuple would agree with it by construction and could not catch a change to it. A change of
    length fails the guard below; a change of membership at the same length fails the counts.

    The first form of this test compared the producer's total against a recount over EVERY budget
    the table holds, with a <= between them. The producer counts five budgets and the table holds
    nine, so that assertion could not fail for any artifact -- a gate that cannot go red, written
    into the very test meant to catch gates that cannot go red.
    """
    n = _numbers()
    assert n["popdef.metatoxbudgets"] == len(MTX_BUDGETS), (
        f"the producer now audits {n['popdef.metatoxbudgets']} budgets while this recount walks "
        f"{len(MTX_BUDGETS)}; the two have to be brought back together before the counts below "
        f"mean anything")
    contrasts = _deployment_contrasts()
    favour = against = seen = 0
    for k in MTX_BUDGETS:
        for arm in ("whole bank", "trained budget"):
            cell = contrasts.get(k, {}).get(f"{arm} - metatox")
            if cell is None:
                continue
            seen += 1
            if cell["excludes_zero"]:
                favour += cell["gap"] > 0
                against += cell["gap"] < 0
    assert seen, "the deployment table carries no MetaTox contrasts over the audited budgets"
    assert n["popdef.metatoxfor"] == favour, (
        f"producer says {n['popdef.metatoxfor']} MetaTox cells for this work, recount says "
        f"{favour} over {seen} cells")
    assert n["popdef.metatoxagainst"] == against, (
        f"producer says {n['popdef.metatoxagainst']} MetaTox cells against this work, recount says "
        f"{against} over {seen} cells")


def test_metatox_absence_removes_more_from_the_favourable_side():
    """The claim the Supporting Information prints, separately from the keys behind it."""
    n = _numbers()
    assert n["popdef.metatoxfor"] > n["popdef.metatoxagainst"], (
        f"MetaTox no longer separates for this work more often than against it "
        f"({n['popdef.metatoxfor']} for, {n['popdef.metatoxagainst']} against). The sentence in "
        f"paper2/si.tex saying its absence removes more from the favourable side than from the "
        f"other is now false and has to be rewritten.")


def test_every_population_macro_the_prose_cites_is_one_the_producer_still_emits():
    """From the prose to the producer, which is the direction that breaks the manuscript.

    The first form of this test asked whether ANY caveat key was cited and passed on one hit, so
    dropping every macro but one would have passed it. It checked the harmless direction.

    The damaging direction is si.tex citing a macro the producer no longer emits: the build fails
    with "macros used and not defined", which does not say that a producer key was the cause. The
    word forms are where this actually bites, because they exist only for integers inside
    paper2_macros.WORDS. A count that walks out of that range loses its \\...Word macro silently --
    popdef.leadstotal moving from eleven to thirteen would have deleted \\numPopdefLeadstotalWord
    while si.tex cited it twice. So the emitted set below is built under the generator's own
    conditions rather than by assuming every key has every form.
    """
    n = _numbers()
    emitted = set()
    for k, v in n.items():
        macro = paper2_macros.name(k)
        emitted.add(macro)
        if isinstance(v, float) and 0.0 <= v <= 1.0:
            emitted.add(macro + "Pct")
        if isinstance(v, int) and not isinstance(v, bool) and v in paper2_macros.WORDS:
            emitted.add(macro + "Word")
    si = (ROOT / "paper2" / "si.tex").read_text()
    cited = sorted(set(re.findall(r"\\(numPopdef[A-Za-z]*)", si)))
    assert len(cited) > 10, (
        f"only {len(cited)} population macros are found cited in paper2/si.tex, so this scan is "
        f"probably not matching them and a pass here would mean nothing")
    missing = [m for m in cited if m not in emitted]
    assert not missing, (
        "paper2/si.tex cites population macros the producer no longer emits, so the manuscript "
        "will fail to build and the message will not name the producer as the cause: "
        + ", ".join(missing))

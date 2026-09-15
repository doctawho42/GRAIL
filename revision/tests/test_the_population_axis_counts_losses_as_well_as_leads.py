"""The population axis must count what it lost, not only what it kept.

Run: python -m pytest revision/tests/test_the_population_axis_counts_losses_as_well_as_leads.py -q

Why this exists. The count the manuscript prints for the move to the wider population is built in
scripts/paper2_numbers.py from one direction only. It walks the cells where the exhaustive arm was
positive AND separating on the comparison set, and asks whether each still separates on the whole
evaluated test set; the survivors are leadskept and the rest leadslost. A cell that separates
against this work on the wider population and did not separate on the comparison set is therefore
not a lost lead -- it is outside the counter's domain, and no number reports it.

That is not hypothetical. GLORYx over the whole test set gives exhaustive minus comparator
-0.0293 [-0.0522, -0.0067] at a budget of ten, where the same cell on the comparison set is
+0.0150 and does not separate. The counter cannot lose a lead it never claimed, so leadslost does
not move, while eight separating cells run against this work and four run for it. Printing the
kept/lost pair alone would put the two favourable additions in a headline number and leave the
eight unfavourable ones with no counter at all -- the asymmetry this paper's own argument is
against.

Two literals feed the same defect. paper2_numbers.py emits the interval macros for a hardcoded
pair of comparators and counts leads over a hardcoded triple, so a comparator added to the
artefact is measured and then silently left out of both. The comment above the triple says the
list "follows what the wider population actually holds rather than being named here"; it is named
there, and the population can only remove a name from it, never add one.

These tests are written to fail on the current producer. Each asserts a property of the numbers
rather than a value, so none of them has to be rewritten when a measurement changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARTEFACT = ROOT / "results" / "population_definition.json"
WHOLE = "the whole evaluated test set"
COMPARISON = "the comparison set"
BOOKKEEPING = ("n_substrates", "n_references", "arms_present")
OUR_ARMS = ("grail_deployed_recall", "grail_exhaustive_recall")
ARMS = ("deployed_minus_comparator", "exhaustive_minus_comparator")


def _artefact() -> dict:
    return json.loads(ARTEFACT.read_text())


def _comparators(blob: dict, population: str) -> list:
    row = blob["contrasts"].get(population, {})
    return [c for c in row if c not in BOOKKEEPING and c not in OUR_ARMS]


def _numbers() -> dict:
    """The numbers dict as the manuscript's macros are generated from.

    build() is the pure half of paper2_numbers.py: the write to results/paper2_numbers.json lives
    in main(), so calling build() reads artefacts and returns the dict without touching the
    repository. A test that ran the script instead would restamp that artefact as a side effect.
    """
    import paper2_numbers

    return paper2_numbers.build()


def test_the_artefact_carries_the_axis_so_a_zero_below_means_something():
    """Positive control. Every count here is taken from this artefact, and a test that could not
    read it would pass by finding nothing to count."""
    blob = _artefact()
    assert _comparators(blob, WHOLE), "the wider population carries no comparator at all"
    assert _comparators(blob, COMPARISON), "the comparison set carries no comparator at all"


def test_every_comparator_on_the_wider_population_is_in_the_lead_count():
    """The count must be over the comparators the artefact holds, not over a literal.

    Recomputed here from the artefact by the producer's own rule -- positive and separating on the
    comparison set, then checked on the whole set -- and compared against the number the
    manuscript prints. A comparator the literal omits shows up as a disagreement.
    """
    blob, n = _artefact(), _numbers()
    kept = lost = 0
    for comp in _comparators(blob, WHOLE):
        near = (blob["contrasts"][COMPARISON].get(comp) or {}).get(ARMS[1]) or {}
        far = (blob["contrasts"][WHOLE].get(comp) or {}).get(ARMS[1]) or {}
        for k, cell in near.items():
            if cell["difference"] > 0 and cell["excludes_zero"]:
                if far.get(k, {}).get("excludes_zero") and far[k]["difference"] > 0:
                    kept += 1
                else:
                    lost += 1
    assert (n["popdef.leadskept"], n["popdef.leadslost"]) == (kept, lost), (
        f"the printed count is {n['popdef.leadskept']} kept and {n['popdef.leadslost']} lost; "
        f"recomputed over every comparator the artefact holds it is {kept} kept and {lost} lost, "
        f"so a comparator is measured and left out of the count")


def test_a_cell_that_separates_against_this_work_is_counted_somewhere():
    """The counter's blind spot.

    A cell separating against this work on the wider population, where the comparison set did not
    separate, is reported by nothing: it is not a lost lead, and leadslost stays put. This asserts
    that some number carries the count of cells running against this work, so that the unfavourable
    side of the move is as countable as the favourable side.
    """
    blob, n = _artefact(), _numbers()
    against = 0
    for comp in _comparators(blob, WHOLE):
        cell = blob["contrasts"][WHOLE].get(comp) or {}
        for arm in ARMS:
            for d in (cell.get(arm) or {}).values():
                if d["excludes_zero"] and d["difference"] < 0:
                    against += 1
    assert against, "no cell separates against this work, so this test cannot fail usefully"
    keys = [k for k in n if k.startswith("popdef.") and "against" in k.lower()]
    assert keys, (f"{against} cells separate against this work on the wider population and no "
                  f"popdef number counts them, while popdef.leadskept counts the cells that run "
                  f"in its favour")


# How paper2_numbers.py spells each family of cell macros, by population and arm. Four families,
# because the producer emits them in four separate loops, and every one of those loops names its
# comparators in a literal -- two of them naming their budgets in a literal as well. A test that
# checked only one family would report the other three printable without looking at them, which is
# how the arm this test exists for came to be measured and unprintable at the same time.
SPELLINGS = {
    (COMPARISON, "deployed_minus_comparator"): "popdef.comp.{comp}.{k}",
    (WHOLE, "deployed_minus_comparator"): "popdef.whole.{comp}.{k}",
    (COMPARISON, "exhaustive_minus_comparator"): "popdef.exhComp{comp}{k}",
    (WHOLE, "exhaustive_minus_comparator"): "popdef.exhWhole{comp}{k}",
}


def test_every_separating_cell_has_its_interval_printable():
    """A cell whose value no macro carries cannot be checked by a reader.

    Every family of interval macros is emitted over a literal list of comparators, and two of the
    four over a literal list of budgets as well, so a measured cell can have no number anywhere on
    the page. This asserts it for every cell that separates -- in either direction, because a
    reader has no more access to an unfavourable cell than to a favourable one.

    Keys are matched exactly rather than by prefix: a budget of 5 is a suffix of a budget of 15,
    and a loose match would call the wrong cell printable.
    """
    blob, n = _artefact(), _numbers()
    missing = []
    for (population, arm), spelling in SPELLINGS.items():
        for comp in _comparators(blob, population):
            cells = (blob["contrasts"][population].get(comp) or {}).get(arm) or {}
            for k, d in cells.items():
                if not d["excludes_zero"]:
                    continue
                if spelling.format(comp=comp, k=k) not in n:
                    side = "against" if d["difference"] < 0 else "for"
                    missing.append(f"{population[:14]}/{arm.split('_')[0]}/{comp}@{k} "
                                   f"({d['difference']:+.4f}, {side} this work)")
    assert not missing, ("these cells separate and no macro carries their value, so a reader "
                         "cannot check them:\n  " + "\n  ".join(missing))

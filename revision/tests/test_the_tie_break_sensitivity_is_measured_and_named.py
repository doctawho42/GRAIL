"""How much of MetaTox's column is the method, and how much is the order its supplier wrote.

Run: python -m pytest revision/tests/test_the_tie_break_sensitivity_is_measured_and_named.py -q

MetaTox's column is ordered by the method's own Pa, descending. That decides about half the entries:
PASS writes a spectrum only above its own threshold, and everything below it carries no score at all
and keeps the order the delivery gave it. Equal Pa values leave ties among the scored ones too.

Inside those classes the method has expressed no preference, so any order within them is an equally
faithful reading of what MetaTox said, and the one that survives is the supplier's file. If recall@k
moves when those classes are permuted, then part of what the column reports at k is the file rather
than the method, and any contrast cell smaller than the movement is within reach of an arbitrary
choice nobody made deliberately.

It does move, and not symmetrically, which is why this is measured rather than waved at: on the
wider half the delivered order is worse than almost every permutation at every budget, so MetaTox is
understated there and the cells in which this work leads it are flattered by exactly that much.

So the tests below hold three things:

  the measurement exists, is computed under a recorded seed, and permutes ONLY within classes the
  method itself ties -- a permutation that crossed a real score difference would be measuring
  something else and would make the arm look arbitrary when it is not;
  the shift is reported per budget and per half, signed, so a reader can see which direction it
  runs and at which k;
  every contrast cell the shift could overturn is NAMED. A sensitivity reported as a summary
  statistic, with no list of what it reaches, is the kind of disclosure this paper objects to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ART = ROOT / "results" / "metatox_tie_break_sensitivity.json"
AXIS = ROOT / "results" / "population_definition.json"


def _blob() -> dict:
    assert ART.exists(), (
        f"{ART.relative_to(ROOT)} does not exist. Half of the MetaTox column carries no score and "
        f"keeps the order the delivery gave it, so how much recall@k depends on that order has to "
        f"be measured before the column's cells are reported.")
    d = json.loads(ART.read_text())
    # The producer banks the permutations before computing the cheap half, so a crash in the cheap
    # half leaves a partial file rather than throwing away half an hour. A partial file must not be
    # read as a finished measurement, which is the failure mode that banking introduces.
    assert not d.get("incomplete"), (
        "the artefact is the producer's partial write: the permutations completed and the rest did "
        "not. Re-run scripts/metatox_tie_break_sensitivity.py rather than reading this.")
    return d


def test_the_permutation_stays_inside_the_classes_the_method_ties():
    """The measurement is only honest if every permutation is a faithful reading of the method.

    Permuting across a real difference in Pa would scramble the method's own ordering and report a
    sensitivity the method does not have. The record has to say what was held fixed, and the count
    of classes has to be consistent with a column that is half unscored: far fewer classes than
    entries, and more than one per substrate.
    """
    d = _blob()
    how = d.get("what_was_permuted") or ""
    assert "tie" in how.lower() or "equal" in how.lower(), \
        "the record does not say that only tied classes were permuted"
    assert d.get("permutations", 0) >= 30, f"only {d.get('permutations')} permutations"
    assert d.get("seed") is not None, "no seed is recorded, so the measurement is not reproducible"
    c = d.get("classes") or {}
    assert c.get("entries") and c.get("tie_classes"), "the tie classes are not counted"
    assert c["tie_classes"] < c["entries"], "every entry is its own class, so nothing was permuted"
    assert c["tie_classes"] > c["substrates"], (
        "there is at most one class per substrate, which would mean the whole list was permuted "
        "rather than only the parts the method ties")


def test_the_shift_is_reported_per_budget_and_per_half_and_signed():
    """One summary number would hide the thing that matters: the direction changes with k.

    On the comparison set the delivered order is unusually bad at small budgets and unusually good
    at thirty. A single figure would average that away, and the averaged figure would be small and
    reassuring.
    """
    d = _blob()
    halves = d.get("halves") or {}
    assert set(halves) >= {"comparison_set", "wider"}, f"halves recorded: {sorted(halves)}"
    for name, rows in halves.items():
        assert rows, f"{name} carries no budgets"
        for k, row in rows.items():
            for f in ("delivered", "permuted_mean", "permuted_lo", "permuted_hi",
                      "shift", "percentile"):
                assert f in row, f"{name} k={k} does not record {f}"
            # 2e-6, not 1e-9: the three fields are each rounded to six places before they are
            # written, so the identity can only hold to the rounding, and a tighter tolerance tests
            # the rounding rather than the arithmetic.
            assert abs(row["shift"] - (row["delivered"] - row["permuted_mean"])) < 2e-6, (
                f"{name} k={k}: the recorded shift is not delivered minus the permuted mean")
            assert row["permuted_lo"] <= row["permuted_mean"] <= row["permuted_hi"], (
                f"{name} k={k}: the permuted mean is outside its own range")
            assert 0.0 <= row["percentile"] <= 100.0


def test_every_cell_the_shift_could_overturn_is_named():
    """The disclosure that makes this worth measuring.

    A contrast cell whose margin is smaller than the movement the supplier's file order produces is
    a cell an arbitrary choice could have decided. Those have to be listed, with their margins, not
    summarised -- and the list has to be checked against the axis rather than typed, or it will stop
    being true the next time the axis is rebuilt.
    """
    d = _blob()
    reach = d.get("cells_within_reach")
    assert isinstance(reach, list), "the record does not list the cells the shift could overturn"
    axis = json.loads(AXIS.read_text())

    for cell in reach:
        for f in ("population", "arm", "budget", "difference", "ci95", "shift", "separates"):
            assert f in cell, f"a named cell does not record {f}: {cell}"
        row = ((axis["contrasts"].get(cell["population"], {}).get("metatox") or {})
               .get(cell["arm"]) or {}).get(str(cell["budget"]))
        assert row is not None, (
            f"the named cell {cell['population']}/{cell['arm']}/k={cell['budget']} is not in the "
            f"axis; this list was typed rather than read")
        assert abs(row["difference"] - cell["difference"]) < 1e-9, (
            f"the named cell's difference {cell['difference']} is not the axis's "
            f"{row['difference']}")
        assert row["excludes_zero"] is cell["separates"]
        # A cell separates because zero is outside its interval, so what an arbitrary re-ordering
        # could unmake is measured from the interval's near edge, not from the point estimate.
        lo, hi = cell["ci95"]
        near = lo if cell["difference"] > 0 else -hi
        assert abs(near - cell["margin_to_zero"]) < 1e-6, (
            f"the recorded margin_to_zero {cell['margin_to_zero']} is not the near edge of "
            f"{cell['ci95']}")
        assert near <= abs(cell["shift"]) + 1e-12, (
            f"a cell is listed as within reach whose interval clears zero by {near}, more than "
            f"the shift {cell['shift']}")

    # The list must be complete, not merely correct: every separating MetaTox cell whose margin is
    # under the shift at its own budget and population has to appear.
    missed = []
    for pop_key, half in (("the comparison set", "comparison_set"),
                          ("the whole evaluated test set", "wider")):
        rows = (axis["contrasts"].get(pop_key, {}).get("metatox") or {})
        # Only the two contrast arms. The axis stores `recall` and `substrates_with_no_prediction`
        # beside them under the same comparator, so walking every key here reaches a dict of floats
        # and an int; this is the same shape assumption that cost the producer half an hour of
        # permutations, made a second time in the test that was supposed to check the producer.
        for arm in ("deployed_minus_comparator", "exhaustive_minus_comparator"):
            for k, c in (rows.get(arm) or {}).items():
                if not isinstance(c, dict) or "excludes_zero" not in c:
                    continue
                sh = (d["halves"].get(half) or {}).get(str(k))
                if not sh or not c["excludes_zero"]:
                    continue
                lo, hi = c["ci95"]
                near = lo if c["difference"] > 0 else -hi
                if near <= abs(sh["shift"]):
                    if not any(x["population"] == pop_key and x["arm"] == arm
                               and str(x["budget"]) == str(k) for x in reach):
                        missed.append(f"{pop_key}/{arm}/k={k} margin {c['difference']:+.4f} "
                                      f"against shift {sh['shift']:+.4f}")
    assert not missed, "separating cells within reach of the shift that the record does not name: " \
                       + "; ".join(missed)

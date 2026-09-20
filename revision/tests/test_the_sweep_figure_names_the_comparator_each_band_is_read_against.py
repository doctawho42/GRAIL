"""The sweep figure names the comparator each shaded band is read against.

The figure shades three regions -- GRAIL trails, neither separates, GRAIL leads -- and each
shading is computed against *the strongest comparator at that budget*, not against a fixed one.
Four different systems hold that position across the sweep and it changes hands three times, so
two bands side by side can be verdicts about two different methods. Nothing in either document
said which: a reader checking a band had to scan nine columns of Table 2 for a row maximum and
then trust that the figure had taken the same maximum.

This is also the division of labour between the table and the figure, which otherwise print the
same eighty-one numbers: the table is the levels, the figure is the verdict and what the verdict
is against.

The runs are derived from the artifact, not typed. A typed list beside an artifact that has its
own keys is the defect this project has met in eleven files -- correct the day it is written and
silently wrong one comparator later -- so the test does not merely assert today's names. It
perturbs the artifact so a different arm is strongest and requires the labels to follow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

ART = ROOT / "results/deployment_table.json"

pytestmark = pytest.mark.skipif(not ART.exists(),
                                reason="deployment_table.json is not in this checkout")


def _expected(rec, style, ours=("whole bank", "trained budget")):
    """Today's answer, computed here independently of the producer."""
    runs = []
    for k in sorted(rec, key=int):
        comp = [a for a in rec[k] if a not in ours and a in style]
        best = max(comp, key=lambda a: rec[k][a])
        if runs and runs[-1] == best:
            continue
        runs.append(best)
    return runs


def test_the_runs_name_the_strongest_comparator_of_every_budget():
    import paper2_figures as F
    d = json.loads(ART.read_text())
    rec = d["recall_micro"]
    runs = F.strongest_comparator_runs(rec)
    assert [r["arm"] for r in runs] == _expected(rec, F.SWEEP_STYLE)
    # every budget is covered by exactly one run, so no band is left unnamed
    covered = [k for r in runs for k in r["budgets"]]
    assert covered == sorted((int(k) for k in rec), key=int)


def test_the_labels_are_drawn_on_the_figure():
    import matplotlib
    matplotlib.use("Agg")
    import paper2_figures as F
    d = json.loads(ART.read_text())
    runs = F.strongest_comparator_runs(d["recall_micro"])
    drawn = F.fig_sweep()["comparator_labels"]
    assert drawn == [r["label"] for r in runs], f"figure drew {drawn}"


def test_the_runs_follow_the_artifact_rather_than_a_typed_list():
    """Make a different arm strongest and require the runs to say so.

    Without this the test passes against a hardcoded list that happens to be right today --
    the shape of gate this project has written three times and that was vacuous each time.
    """
    import paper2_figures as F
    d = json.loads(ART.read_text())
    rec = d["recall_micro"]
    ours = ("whole bank", "trained budget")
    leaders = set(_expected(rec, F.SWEEP_STYLE))
    never = [a for a in rec[next(iter(rec))]
             if a not in ours and a in F.SWEEP_STYLE and a not in leaders]
    assert never, "every comparator already leads somewhere; the perturbation has no target"
    target = never[0]
    for k in rec:
        rec[k][target] = max(rec[k].values()) + 1.0

    runs = F.strongest_comparator_runs(rec)
    assert [r["arm"] for r in runs] == [target], (
        f"the artifact now makes {target} strongest at every budget and the runs read "
        f"{[r['arm'] for r in runs]}: the runs are not derived from the artifact")


def test_an_arm_the_style_map_does_not_know_is_refused():
    """A comparator the sweep computes and the figure cannot name must stop the build.

    The same silence put a fifth method in the abstract's count and nowhere on the plot.
    """
    import paper2_figures as F
    d = json.loads(ART.read_text())
    rec = d["recall_micro"]
    for k in rec:
        rec[k]["a_comparator_nobody_named"] = 0.9
    with pytest.raises(SystemExit):
        F.strongest_comparator_runs(rec)

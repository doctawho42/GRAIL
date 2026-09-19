"""A bound quoted because a join was not run must become a point once the join is run.

Run: python -m pytest revision/tests/test_the_join_that_made_it_a_bound_is_reported.py -q

paper2/si.tex said, of the share of the shortfall that is chemistry the corpus never held: "it is a
bound, not a point, because the join itself is not reported here." That is an honest statement of a
missing measurement and the correct thing to write while it is missing. It is also a standing
promise: the moment the join exists, a paper that keeps quoting the bound is quoting an interval it
no longer needs, and the sentence explaining why becomes an explanation of something that has been
done.

The join is scripts/typed_edit/containment_on_the_uncovered.py, run over eight shards. Its producer
refuses to write unless the run reproduces the published decomposition of the shortfall, so a
partial merge cannot become the number: a four-shard merge was tried and refused, reporting 253
uncovered against the committed 475.

This holds three things:

  the artifact exists and its population is the one the bound is quoted over, so the two are
  comparable at all;
  the measured share falls INSIDE the bound the cross-tabulation's marginals give, because a point
  outside its own bound means one of the two is wrong and neither may be printed;
  and both manuscripts quote the point where they used to quote only the interval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ART = ROOT / "results" / "containment_on_the_uncovered.json"
NUMBERS = ROOT / "results" / "paper2_numbers.json"
BODY = ROOT / "paper2" / "body.tex"
SI = ROOT / "paper2" / "si.tex"


def _numbers() -> dict:
    return json.loads(NUMBERS.read_text())["numbers"]


def test_the_join_was_run_over_the_population_the_bound_is_quoted_over():
    """Two populations that differ make the comparison meaningless, however close the shares."""
    assert ART.exists(), (
        f"{ART.relative_to(ROOT)} does not exist. The Supporting Information says the figure is a "
        f"bound because the join is not reported; the join is this file.")
    art = json.loads(ART.read_text())
    n = _numbers()
    assert art["population"]["of_absent_type"] == n["ceiling.novel"], (
        f"the join covers {art['population']['of_absent_type']} references of absent type and the "
        f"census counts {n['ceiling.novel']}")
    assert art["population"]["uncovered"] == n["ceiling.uncovered"], (
        f"the join covers {art['population']['uncovered']} uncovered references and the census "
        f"counts {n['ceiling.uncovered']}")


def test_the_measured_share_falls_inside_the_bound():
    """A point outside its own bound means one of the two is wrong, and neither may be printed."""
    n = _numbers()
    lo, hi = n["bound.shortfallsharemin"], n["bound.shortfallsharemax"]
    point = n["containment.shortfallshare"]
    assert lo <= point <= hi, (
        f"the measured share {point} falls outside the bound [{lo}, {hi}] the cross-tabulation's "
        f"marginals give. The bound is derived from one cell of that table and the point from a "
        f"per-reference join; they disagree, so one of the two is wrong.")
    assert n["containment.absent"] + n["containment.present"] == n["ceiling.novel"], (
        "the join's two cells do not sum to the references of absent type")


def test_the_manuscripts_no_longer_explain_why_it_is_only_a_bound():
    """The sentence that justified the interval is an explanation of a measurement now made."""
    stale = "because the join itself is not reported here"
    assert stale not in SI.read_text(), (
        f"paper2/si.tex still says {stale!r} while "
        f"{ART.name} reports exactly that join")
    for doc in (BODY, SI):
        t = doc.read_text()
        if "numBoundShortfallsharemin" in t:
            assert "numContainmentShortfallshare" in t or "numContainmentShare" in t, (
                f"{doc.name} quotes the bound and never the point that has since been measured")

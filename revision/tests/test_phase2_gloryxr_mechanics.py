"""Tests for the GLORYxR mechanics record, written before the producer exists.

Run: python -m pytest revision/tests/test_phase2_gloryxr_mechanics.py -q

Why this record exists. Two committed artifacts said "Reactor.react_one applies each rule once and
the package has no recursion". The second half is true; the first half is false, and it is a
statement about someone else's code, published in a record a reader can download. RunReactants
enumerates every match of a template, so one rule yields several products: on a 20-substrate cost
probe the worst single rule gave 25 products and 171 of 377 rule firings gave more than one. This
producer measures that over the whole population and writes the number down, so the claim in the
table's provenance is backed by a measurement rather than by a reading of the source.

The same record carries two numbers the table asserted with nothing behind them: the 0.2 factor
applied to rules of "uncommon" priority, and how many of the rule table's rules carry that
priority. A number in a provenance record with no producer is the defect class this revision has
paid for repeatedly.

The measurement needs gloryxr, which requires Python >= 3.13 while this repository runs 3.10, so
the producer imports it lazily and these tests exercise the pure parts under either interpreter.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "revision")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Hard import: until the producer exists these tests must fail, not skip.
import phase2_gloryxr_mechanics as M  # noqa: E402


# --------------------------------------------------------------------------- the pure summary

def test_summarise_reports_the_worst_rule_and_the_share_of_multi_product_firings():
    """The quantity that refutes the old wording is "products from one firing of one rule".

    Synthetic input, so this fails for the reason it names even when the package is absent.
    Two substrates: the first has a rule giving 3 products and one giving 1, the second a rule
    giving 2. So 3 firings, 2 of them multi-product, worst 3.
    """
    got = M.summarise([
        {"substrate": "CCO", "firings": {"aliphatic hydroxylation": 3, "O-demethylation": 1}},
        {"substrate": "CCN", "firings": {"aromatic hydroxylation": 2}},
    ])
    assert got["n_substrates"] == 2
    assert got["n_rule_firings"] == 3
    assert got["n_firings_yielding_more_than_one_product"] == 2
    assert got["max_products_from_one_rule_firing"] == 3


def test_a_rule_that_fires_once_is_not_counted_as_multi_product():
    """The boundary the old wording got wrong is exactly one-versus-more-than-one."""
    got = M.summarise([{"substrate": "CCO", "firings": {"only one": 1}}])
    assert got["n_firings_yielding_more_than_one_product"] == 0
    assert got["max_products_from_one_rule_firing"] == 1
    assert got["one_application_per_rule_holds"] is True


def test_the_summary_states_whether_one_application_per_rule_holds():
    """The record must answer the question the retracted sentence answered wrongly, in a field,
    not only in prose a reader has to parse."""
    got = M.summarise([{"substrate": "CCO", "firings": {"a": 4}}])
    assert got["one_application_per_rule_holds"] is False


def test_an_empty_measurement_is_refused_rather_than_summarised_as_zero():
    """Zero substrates would report "one application per rule holds" vacuously, which is how a
    failed run turns into a false claim."""
    try:
        M.summarise([])
    except TypeError as e:
        raise AssertionError(f"refused by signature rather than by choice: {e}")
    except Exception as e:
        assert "empty" in str(e).lower() or "no substrate" in str(e).lower(), (
            f"the refusal must say what was missing: {e!r}")
    else:
        raise AssertionError("an empty measurement must be refused, not summarised")


# --------------------------------------------------------------------------- the rule table

def test_the_rule_table_counts_come_from_a_named_path_and_its_absence_is_recorded():
    """The rule table ships with GLORYxR, not with this repository, so a clean checkout has no
    copy of it. Absence must be recorded rather than fatal, and never filled in from memory."""
    rec = M.rule_table(Path("/nonexistent/gloryx_reactionrules_connect.csv"))
    assert rec["present"] is False
    assert rec["checked_path"] == "/nonexistent/gloryx_reactionrules_connect.csv"
    assert rec.get("n_rules") is None
    assert rec.get("n_uncommon") is None


def test_the_priority_factor_is_quoted_with_its_location_not_asserted():
    """0.2 is read out of someone else's source, so the record must carry the file and line it was
    read from. A bare number here is the thing this producer exists to stop."""
    rec = M.priority_factor()
    assert rec["factor"] == 0.2
    assert "fame3r.py" in rec["source"]
    assert ":" in rec["source"], "the source must name a line, not only a file"
    assert rec["provider"] == "MultiFAME3RModelProvider"


# --------------------------------------------------------------------------- the retraction

def test_the_record_names_the_sentence_it_retracts():
    """A correction that does not say what it corrects leaves the old sentence reachable in the
    history with nothing pointing at it. The blockers in phase2_comparators carry `retracts` for
    the same reason."""
    prov = M.provenance()
    text = prov.get("retracts", "")
    assert text, "the retracted wording is not named"
    assert "applies each rule once" in text
    for where in ("T_gloryxr_provenance.json", "phase2_gloryxr_table.py"):
        assert where in text, f"the record must name where the wrong wording was published: {where}"


def test_the_record_carries_the_interpreter_and_the_package_versions():
    """The mechanics are a property of a build. cdpkit recomputes descriptors at run time and the
    package set here is not the repository's own, so a record without versions cannot be placed."""
    prov = M.provenance()
    for field in ("python", "packages"):
        assert field in prov, f"{field} missing from the record"
    assert "cdpkit" in prov["packages"]

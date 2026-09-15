"""Tests for the GLORYxR comparison table, written before the producer exists.

Run: python -m pytest revision/tests/test_phase2_gloryxr_table.py -q

Why this table exists instead of a gloryxr arm in phase1_tmain.ARMS. The local column emits no
stereochemistry at all -- 0 of 14,755 predictions carry a stereo InChIKey block against 725 of
14,784 in the service-derived column -- so beside the service arm it disagrees by 25-26% under
`exact`, 12-14% under `canonical` and 9-10% under `inchikey`, always against the local arm, and by
~1.2% under `inchi_no_stereo` and `inchikey_tautomer`. ARMS cannot restrict an arm to a subset of
criteria, so declaring one there would emit four knowingly invalid rows per mode. The table is
therefore separate and restricted, and the restriction is a measurement rather than a preference:
asking it for a stereo-sensitive criterion must be refused, not silently computed.

The second thing pinned here is the invariant the two modes satisfy. default and strict produce
identical product sets on all 1,170 substrates and differ only in order, so any coverage number
must come out equal between them; if it does not, either a column was regenerated against
different inputs or the modes stopped meaning what they mean, and the table must refuse rather
than publish two coverage figures that cannot both be right.
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

# Hard import: until the producer exists these tests must fail, not skip.
import phase2_gloryxr_table as T  # noqa: E402

DEFAULT = ROOT / "results" / "gloryxr_local_preds_default.json"
STRICT = ROOT / "results" / "gloryxr_local_preds_strict.json"


# --------------------------------------------------------------------------- the restriction

def test_only_the_two_stereo_insensitive_criteria_are_admitted():
    """Exactly two, and named. The headline criterion is among them, which is why the headline
    comparison is sound while the supplementary rows would not have been."""
    assert set(T.CRITERIA) == {"inchi_no_stereo", "inchikey_tautomer"}
    assert "inchikey_tautomer" in T.CRITERIA


def test_a_stereo_sensitive_criterion_is_refused_with_its_reason():
    """Refused, not quietly computed: the number would be a measurement of an alphabet
    difference rather than of either system."""
    for criterion in ("exact", "canonical", "inchikey"):
        try:
            T.rows(criterion=criterion)
        except TypeError as e:
            raise AssertionError(f"refused by signature rather than by choice: {e}")
        except Exception as e:
            msg = str(e).lower()
            assert criterion in str(e), f"the refusal must name the criterion: {e!r}"
            assert "stereo" in msg, f"the refusal must give the reason: {e!r}"
        else:
            raise AssertionError(f"{criterion} must be refused, not computed")


# --------------------------------------------------------------------------- the invariant

def test_the_two_modes_must_agree_on_coverage_or_the_table_refuses():
    """They produce identical product sets, so every coverage figure is equal by construction.

    Checked against synthetic columns so it fails for the reason it names even when the real
    artifacts are sound.
    """
    same = {"s1": ["A", "B"], "s2": ["C"]}
    reordered = {"s1": ["B", "A"], "s2": ["C"]}
    assert T.coverage_disagreement(same, reordered) == []

    differing = {"s1": ["A", "B"], "s2": ["C", "D"]}
    got = T.coverage_disagreement(same, differing)
    assert got, "a genuine set difference must be reported"
    assert "s2" in got


def test_the_real_columns_satisfy_the_invariant():
    """The measured claim, over the artifacts as committed: 1170/1170 identical sets."""
    d = json.loads(DEFAULT.read_text())["predictions"]
    s = json.loads(STRICT.read_text())["predictions"]
    assert T.coverage_disagreement(d, s) == []


# --------------------------------------------------------------------------- assembly

def test_the_arms_are_assembled_through_the_certified_helper():
    """Reuse, not re-expression. phase1_tmain.arm_key_lists is the assembly the reproduction gate
    certifies; a second implementation of dedup, parent-drop and truncation is how two readings of
    one column drift apart, and this repository has paid for that already."""
    import phase1_tmain
    assert T.arm_key_lists is phase1_tmain.arm_key_lists


def test_every_row_names_its_system_criterion_budget_and_population():
    rows = T.rows(criterion="inchikey_tautomer")
    assert rows, "no rows produced"
    for r in rows:
        for field in ("system", "criterion", "k", "population", "recall", "n_substrates"):
            assert field in r, f"{field} missing from {r}"
        assert r["criterion"] == "inchikey_tautomer"
        assert r["population"] == "evaluated1170"
    systems = {r["system"] for r in rows}
    assert {"gloryxr default", "gloryxr strict"} <= systems


def test_the_service_column_is_included_for_comparison_but_marked():
    """The service-derived GLORYx column is the point of comparison, and the row has to say that
    the two were obtained by different routes -- one generation locally against the service's
    cascade, and a different score resolution."""
    rows = T.rows(criterion="inchikey_tautomer")
    service = [r for r in rows if r["system"] == "gloryx"]
    assert service, "the service column must be present to compare against"
    for r in service:
        assert r.get("route"), "the service rows must record how they were obtained"


# --------------------------------------------------------------------------- provenance

def test_the_recall_profile_of_the_mode_is_recorded_not_left_to_one_budget():
    """The k=15 row alone says the mode changes nothing, and that reading is wrong.

    strict differs from default on 9 of 9 budgets under inchi_no_stereo and 8 of 9 under
    inchikey_tautomer, better at small budgets and worse at large ones, with the crossing at
    k=15 -- the single budget where the difference is exactly zero. A record that omits the
    profile invites exactly the conclusion the crossing point produces.
    """
    prov = T.provenance()
    text = prov.get("what_the_mode_costs_and_buys", "")
    assert text, "the recall profile of the mode is not recorded"
    assert "k=15" in text, "the crossing budget must be named"
    for token in ("small budget", "large"):
        assert token in text.lower(), f"{token!r} missing from the profile"


def test_the_comparison_with_grail_records_that_the_sign_changes():
    """Neither system dominates; the ordering depends on the budget."""
    text = T.provenance().get("against_grail_the_sign_changes_with_budget", "")
    assert text, "the budget-dependent reversal against GRAIL is not recorded"
    assert "5 of 9" in text or "4 of 9" in text
    assert "k=30" in text and "k=50" in text


def test_the_limitations_travel_with_the_table():
    """Six measured limitations, each with its number. A table that carries the column without
    them invites exactly the comparison the restriction exists to prevent.

    This check is presence-only, and that was not enough: it stayed green for two commits while
    the `generation` limitation published a false sentence about GLORYxR's code, because the word
    "generation" was in the text either way. The checks below pin the content.
    """
    prov = T.provenance()
    text = json.dumps(prov).lower()
    for token in ("stereo", "generation", "resolution", "heavy", "tautomer", "duplicate"):
        assert token in text, f"limitation {token!r} not recorded"
    assert prov.get("why_not_an_arm_in_t_main")


def _limitation(prov, ident):
    return {row["id"]: row["detail"] for row in prov["limitations"]}[ident]


def test_the_generation_limitation_agrees_with_the_measurement_artefact():
    """The numbers live in one place and the prose must match it.

    The retracted wording said `Reactor.react_one` applies each rule once. One firing of one rule
    returns several products, because RunReactants enumerates every match of the template, so the
    claim was false about someone else's code in a record a reader can download. The measurement
    is results/gloryxr_mechanics.json; this asserts the limitation quotes that file rather than a
    reading of the source, and fails if the two ever drift apart.
    """
    mech = json.loads((ROOT / "results" / "gloryxr_mechanics.json").read_text())
    measured = mech["measured"]
    assert measured["one_application_per_rule_holds"] is False, (
        "the artefact no longer refutes the retracted wording; re-read it before trusting this")

    detail = _limitation(T.provenance(), "generation")
    assert "applies each rule once" not in detail, "the retracted wording is back in the record"
    assert "RunReactants" in detail, "the mechanism that refutes it must be named"
    for value in (f"{measured['n_firings_yielding_more_than_one_product']:,}",
                  str(measured["max_products_from_one_rule_firing"])):
        assert value in detail, f"the measured value {value!r} is missing from the limitation"


def test_the_generation_limitation_claims_nothing_about_the_service_that_was_not_measured():
    """It used to assert the service emits products needing two applications. Nothing here
    measured that, and an unmeasured clause beside measured ones borrows their standing."""
    detail = _limitation(T.provenance(), "generation")
    assert "two applications" not in detail, (
        "an unmeasured claim about the service is back in the record")


def test_the_duplicate_limitation_scopes_the_delivery_it_counts():
    """Nine dumps were delivered, eight per-subset plus single_model.joblib, and the repository's
    own record says so in n_models. Saying "eight delivered" contradicts a field of its own
    artefact and understates what arrived."""
    detail = _limitation(T.provenance(), "duplicate_model")
    assert "per-subset" in detail, "the duplicate pair must be scoped to the per-subset dumps"
    assert "eight delivered" not in detail

"""Tests for the population switch GLORYx's service run needs, written before it exists.

Run: python -m pytest revision/tests/test_phase2_gloryx_population.py -q

Phase 2 asks for GLORYx on all 1,170 evaluated substrates. The existing run covers 291 and its
output is a published artefact, so the new run has to do two things the current script cannot: take
the population by name, and submit only what is not already held. Both are assertions about which
molecules go to somebody else's service, so both are tested rather than trusted.

The consistency assertion is the important one. `population("comparison291")` must be exactly the
set the published GLORYx artefact already holds -- not merely the same size. Three comparator files
in this repository carry 291 keys and two of them are the same set while a third, MetaTox's
non-SMIRKS file, is a 248-subset; equal counts have already stood in for equal sets twice here, so
the test compares sets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Hard import: until the switch exists these tests must fail, not skip.
import gloryx_via_service as G  # noqa: E402

HELD = ROOT / "results" / "gloryx_service_preds.json"


def _held_substrates():
    return set(json.loads(HELD.read_text())["predictions"])


def test_the_comparison_population_is_the_set_the_published_artefact_holds():
    """Same set, not same size: a coincident count is not a population."""
    got = G.population("comparison291")
    assert len(got) == 291
    assert set(got) == _held_substrates()


def test_the_evaluated_population_is_the_1170_and_contains_the_291():
    got = set(G.population("evaluated1170"))
    assert len(got) == 1170
    assert _held_substrates() <= got, "the published 291 must lie inside the wider population"


def test_an_unknown_population_is_refused_rather_than_guessed():
    """A deliberate refusal that names the bad input, not merely "something raised".

    Written first as `except Exception: return`, this test passed against no implementation at
    all: an unimplemented `population()` rejects the argument with TypeError, which a bare
    `except` reads as a refusal. So TypeError is excluded explicitly and the message has to name
    what was refused. A gate whose pass does not depend on the code under test is not a gate.
    """
    try:
        G.population("whatever")
    except TypeError as e:                       # the signature, not a decision about the name
        raise AssertionError(f"refused by signature rather than by choice: {e}")
    except Exception as e:
        assert "whatever" in str(e), f"the refusal must name the population it rejected: {e!r}"
        return
    raise AssertionError("an unknown population name must raise, not fall back to a default")


def test_pending_excludes_what_is_already_held_and_keeps_order_deterministic():
    """The submission set is what the output does not already carry, so a resumed run asks for
    nothing it holds and two runs ask for the same molecules in the same order."""
    subs = G.population("evaluated1170")
    pending = G.pending(subs, HELD)
    held = _held_substrates()
    assert len(pending) == 1170 - 291
    assert not (set(pending) & held), "a held substrate must never be submitted again"
    assert pending == G.pending(subs, HELD), "the submission order must be deterministic"


def test_pending_against_a_missing_output_is_the_whole_population():
    subs = G.population("comparison291")
    pending = G.pending(subs, ROOT / "results" / "does_not_exist_gloryx.json")
    assert set(pending) == set(subs)

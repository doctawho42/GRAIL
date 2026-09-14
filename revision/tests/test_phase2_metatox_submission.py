"""Tests for MetaTox's submission set on the evaluated population, written before it exists.

Run: python -m pytest revision/tests/test_phase2_metatox_submission.py -q

MetaTox is a web service with no programmatic interface recorded in this repository, so Phase 2
writes what would be submitted and stops. That makes the submission files the deliverable, and the
two things that can go silently wrong with them are the population they were drawn from and the key
the returned batch joins back on.

The existing builder, scripts/make_metatox_input.py, takes the substrate column of a GRAIL
prediction CSV. Every full-test CSV here holds 1,169 of the 1,170: one CoA thioester is in the
references and in no CSV. So this producer hands the builder the population instead, and the tests
below pin that the missing structure is present and that the join key is the corpus string rather
than the natural-tautomer form actually submitted.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "revision"), str(ROOT / "scripts"),
           str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Hard import: until the producer exists these tests must fail, not skip.
import phase2_metatox_submission as M  # noqa: E402

# The one structure absent from every full-test prediction CSV, which is why the set is not
# sourced from one.
COA = ("CC(O)C(C)C(=O)SCCN=C(O)CCN=C(O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)"
       "C(O)C1OP(=O)(O)O")


def test_the_set_is_the_population_minus_what_metatox_already_holds():
    subs = M.to_submit()
    assert len(subs) == 879
    held = set(json.loads((ROOT / "results" / "metatox_smirks_preds.json").read_text())
               ["predictions"])
    assert not (set(subs) & held), "a substrate MetaTox already answered must not be resubmitted"


def test_the_structure_no_predictions_csv_carries_is_in_the_set():
    """Sourcing from a CSV would drop this one silently and shorten the column's denominator."""
    assert COA in set(M.to_submit())


def test_the_set_is_deterministic():
    assert M.to_submit() == M.to_submit()


def test_writing_produces_a_csv_the_existing_builder_can_consume(tmp_path):
    """One builder for every batch: a second submission format is how the join key drifts."""
    path = M.write_substrate_csv(tmp_path / "subs.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert "substrate" in (rows[0] if rows else {}), "the builder reads a 'substrate' column"
    assert len(rows) == 879
    assert {r["substrate"] for r in rows} == set(M.to_submit())


def test_the_manifest_records_the_source_and_the_baseline_by_name():
    man = M.manifest()
    assert man["source"].endswith("test_references.json")
    assert man["baseline"].endswith("metatox_smirks_preds.json")
    assert man["to_submit"] == 879
    assert man["held"] == 291
    assert man["held"] + man["to_submit"] == 1170
    assert not man["source"].endswith(".csv")

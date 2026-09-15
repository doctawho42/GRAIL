"""The comparator record must describe the file the published numbers come from.

Run: python -m pytest revision/tests/test_comparator_record_names_the_scored_artefact.py -q

Why this file exists, and why it is not redundant with the provenance auditor.
scripts/audit_artifact_provenance.py already covers results/comparator_provenance.json, but what
it checks is freshness: whether the artifact carries the digest of its producer as that producer
now stands. It cannot notice that a fresh, correctly stamped record describes the WRONG file.
That is what happened. The record named results/metatox_preds.json with the configuration "layer 1
only, without the SMIRKS-rule variant", while every published MetaTox number is read from
results/metatox_smirks_preds.json, whose own config field says "SMIRKS rules, all 291 parents
returned". The defect survived a session in which it was diagnosed and reported as fixed, because
nothing compared the record against the source.

So this gate compares the record with the source of truth rather than with a copy of the text:

  - the predictions path must be the one scripts/typed_edit/deployment_table.py declares for
    metatox, since that is the module the deployment numbers flow through;
  - the recorded configuration must agree with the scored artifact's own `config.variant`, read
    out of that artifact rather than restated here.

A test that hardcoded the expected strings would pass against a record that agrees with the test
and disagrees with the data, which is the failure mode this project has paid for repeatedly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RECORD = ROOT / "results" / "comparator_provenance.json"


def _record():
    return json.loads(RECORD.read_text())


def _declared_metatox_source() -> str:
    """The path deployment_table declares, imported rather than retyped."""
    import deployment_table as D
    rel, accessor = D.COMPARATORS["metatox"]
    assert accessor == "predictions", (
        f"the accessor changed to {accessor!r}; this gate assumes the envelope it reads")
    return rel


def test_the_record_names_the_file_the_deployment_numbers_are_read_from():
    """Two MetaTox artifacts exist and only one feeds the paper. Naming the other one described a
    side analysis as though it were the column."""
    declared = _declared_metatox_source()
    recorded = _record()["comparators"]["MetaTox"]["predictions"]
    assert recorded == declared, (
        f"the record names {recorded!r} but the deployment numbers are read from {declared!r}")


def test_the_recorded_configuration_agrees_with_the_scored_artefact():
    """The configuration is read out of the artifact that was scored, not restated here."""
    declared = _declared_metatox_source()
    scored = json.loads((ROOT / declared).read_text())
    variant = (scored.get("config") or {}).get("variant")
    assert variant, f"{declared} carries no config.variant to check against"

    configuration = _record()["comparators"]["MetaTox"]["configuration"]
    assert "SMIRKS" in variant, (
        f"the scored artefact's variant changed to {variant!r}; re-read it before trusting this")
    assert "SMIRKS" in configuration, (
        f"the record says {configuration!r}, which does not describe the scored variant "
        f"{variant!r}")
    assert "without the SMIRKS" not in configuration, (
        "the record still describes the configuration the published numbers did NOT come from")


def test_the_other_metatox_artefact_is_named_as_a_side_analysis_not_dropped():
    """There are two runs and the record should say so. Replacing one wrong file name with one
    right one, and saying nothing about the other, trades a false record for an incomplete one."""
    blob = json.dumps(_record()["comparators"]["MetaTox"])
    assert "metatox_preds.json" in blob, (
        "the second MetaTox run is no longer mentioned at all; it exists and feeds nothing, and "
        "that is worth recording rather than deleting")


def test_the_counters_the_manuscript_prints_are_still_derived_not_typed():
    """paper2_numbers.py reads n_comparators and n_carrying_a_version_string straight into macros,
    so they must keep matching the rows they summarise."""
    rec = _record()
    rows = rec["comparators"]
    assert rec["n_comparators"] == len(rows)
    assert rec["n_carrying_a_version_string"] == sum(1 for r in rows.values() if r.get("version"))

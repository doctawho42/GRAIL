"""The preregistration's MetaTox claim must be exactly what the artifact produces.

The hand-written version of this sentence conflated two arms within an hour of being written.
It is now derived, and this holds the file to the derivation.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def test_the_prereg_contains_the_generated_claim_verbatim():
    from sweep_claim import TABLE, claim
    if not TABLE.exists():
        pytest.skip("the deployment table is not in this checkout")
    text = (ROOT / "paper2/preregistration.md").read_text()
    assert claim() in text, (
        "section 0.2b has drifted from results/deployment_table.json; regenerate it with "
        "scripts/sweep_claim.py rather than editing the prose")


def test_the_claim_changes_when_the_artifact_does():
    """A check that cannot fail is not a check."""
    import json
    import tempfile

    from sweep_claim import TABLE, claim
    if not TABLE.exists():
        pytest.skip("the deployment table is not in this checkout")
    d = json.loads(TABLE.read_text())
    d["contrasts"]["15"]["whole bank - metatox"]["excludes_zero"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(d, fh)
        path = Path(fh.name)
    try:
        assert claim(path) != claim(), "the sentence ignored a change in the artifact"
    finally:
        path.unlink()

"""The manuscript may not print a number that is not read from an artifact.

Prose is the only part of this project that passes through no gate, and writing it produced five
wrong figures in an hour the last time it was checked afterwards. The manuscript therefore reaches
its numbers only through macros generated from results, and these hold that arrangement together.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(script):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                          capture_output=True, text=True, cwd=ROOT)


@pytest.mark.skipif(not (ROOT / "paper2/grail_service.tex").exists(),
                    reason="the manuscript is not in this checkout")
def test_every_number_in_the_manuscript_is_a_macro_from_an_artifact():
    r = _run("check_paper2_numbers.py")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not (ROOT / "results/paper2_numbers.json").exists(),
                    reason="the numbers artifact is not in this checkout")
def test_the_macros_regenerate_identically():
    """A stale numbers.tex is a stale paper; regenerating must be a no-op."""
    before = (ROOT / "paper2/numbers.tex").read_text()
    r = _run("paper2_macros.py")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (ROOT / "paper2/numbers.tex").read_text() == before, (
        "paper2/numbers.tex is out of date with results/paper2_numbers.json; "
        "re-run scripts/paper2_numbers.py then scripts/paper2_macros.py")


@pytest.mark.skipif(not (ROOT / "paper2/claim.tex").exists(),
                    reason="the claim block is not in this checkout")
def test_the_sweep_claim_block_matches_the_artifact():
    sys.path.insert(0, str(ROOT / "scripts"))
    from sweep_claim import claim
    got = (ROOT / "paper2/claim.tex").read_text()
    for line in claim().splitlines():
        assert line.replace("> ", "") in got, (
            "paper2/claim.tex has drifted from results/deployment_table.json; regenerate it")

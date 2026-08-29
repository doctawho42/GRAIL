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


def _run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=ROOT)


@pytest.mark.skipif(not (ROOT / "paper2/body.tex").exists(),
                    reason="the manuscript is not in this checkout")
def test_every_number_in_the_manuscript_is_a_macro_from_an_artifact():
    """The NAR manuscript reaches every figure through a macro generated from results/."""
    r = _run("check_paper2_numbers.py")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not (ROOT / "paper2/manuscript_draft.md").exists(),
                    reason="the markdown draft is not in this checkout")
def test_the_draft_numbers_can_be_traced():
    """The markdown manuscript has no macro path, so this reports rather than gates.

    grail_service.tex reached every figure through a generated macro and a checker refused any
    literal that was not one. The markdown draft cannot do that, and the loss is real: this runs
    the tracer and fails only if it cannot run, which is a weaker guarantee and is named as one
    in paper2/superseded/README.md.
    """
    r = _run("check_draft_numbers.py")
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


@pytest.mark.skipif(not (ROOT / "paper2/figures.sha256").exists(),
                    reason="the figures are not in this checkout")
def test_the_figures_are_drawn_from_the_current_artifacts():
    """Matplotlib does not produce byte-identical PDFs, so this compares the data instead.

    A figure is stale when the artifact behind it has moved, not when its timestamp has, and the
    first draft's comparison table went stale exactly that way.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    mod = importlib.import_module("paper2_figures")
    importlib.reload(mod)
    want = (ROOT / "paper2/figures.sha256").read_text().strip()
    assert mod.digest() == want, (
        "the figures are behind their artifacts; re-run scripts/paper2_figures.py")


def test_the_tracer_survives_a_non_numeric_register_entry():
    """Not every entry of the number register is a number.

    Some carry the literal text a sentence prints, such as the list of ranks the worked example
    reports. The tracer formatted every value with a thousands separator, which raises on a
    string rather than returning something useless, and the whole check died the first time such
    an entry was added. The variants helper is the place that has to tolerate it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_draft_numbers", ROOT / "scripts" / "check_draft_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.variants("8, 13, 16, 21") == {"8, 13, 16, 21"}
    assert "1,170" in module.variants(1170)
    assert "0.0035" in module.variants(0.0035)


def test_every_registered_prediction_is_accounted_for_in_the_paper():
    """The backwards direction of the preregistration check, run on the JCIM manuscript.

    This gate existed and had never been run against these documents: it was written for an
    earlier paper that numbered its predictions H1 upward, and this one numbers them P1 upward
    with the register's identifier beside each in a table. Given only H-identifiers the checker
    read the manuscript as mentioning no hypothesis at all and reported all sixteen absent,
    which is a gate that has stopped gating rather than a paper with sixteen defects.

    The forward direction runs in report mode because the register covers the deployed choices
    and the paper also reports measurements that were never deployed choices. The backwards
    direction is the hard one, and it is the one preregistration exists for: a prediction that
    was made, did not hold, and then quietly left.
    """
    r = _run("check_prereg.py", "--prereg", "paper2/preregistration.md", "--forward", "report",
             "--text", "paper2/body.tex", "paper2/si.tex", "paper2/table_hypotheses.tex")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "absent from the text" not in r.stdout, r.stdout

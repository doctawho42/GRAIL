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


def test_the_manuscript_describes_every_section_of_its_supporting_information():
    """The contents list is what a reader uses to find an argument, and it went stale silently.

    ACS requires the manuscript to say what the supporting information contains. Three sections
    were absent from that list, one of which the body twice sends the reader to, because a section
    can be added to the supporting information without anything noticing. A section whose title
    words are largely absent from the list is not described by it.
    """
    import re
    from pathlib import Path

    si = Path("paper2/si.tex").read_text()
    body = Path("paper2/body.tex").read_text()
    block = re.search(r"\\begin\{suppinfo\}(.*?)\\end\{suppinfo\}", body, re.S)
    assert block, "the manuscript has no suppinfo block"
    # Compare on word stems, so a section called Comparators is matched by "comparator versions".
    listing = {w[:6] for w in re.findall(r"[a-z]{4,}", block.group(1).lower())}

    undescribed = []
    for title in re.findall(r"\\section\{([^}]*)\}", si):
        stems = {w[:6] for w in re.findall(r"[a-z]{4,}", title.lower())}
        if not stems:
            continue
        hit = len(stems & listing)
        if hit < max(1, len(stems) // 2):
            undescribed.append(f"{title!r} ({hit} of {len(stems)} words in the list)")
    assert not undescribed, (
        "these supporting-information sections are not described in the manuscript's contents "
        "list: " + "; ".join(undescribed))


def test_every_artifact_a_number_comes_from_is_released_or_named_as_the_deposit():
    """The paper says the evaluation harness and its results are released; 25 were not there.

    results/ is ignored wholesale and its artifacts are tracked one at a time, so an analysis
    added later is released only if somebody remembers to force-add it. Twenty-five of the eighty
    artifacts the printed numbers come from had not been, including the matched-length control and
    the comparison table's own multiplicity accounting. A reader cloning the repository could not
    see them. The one artifact that legitimately stays out is the candidate pool, which is too
    large to commit and is the Zenodo deposit the availability statement names.
    """
    import json
    import subprocess
    from pathlib import Path

    sources = json.loads(Path("results/number_sources.json").read_text())
    srcs = sources if isinstance(sources, list) else sources.get("artifacts") or list(sources)
    names = {s if isinstance(s, str) else s.get("artifact") for s in srcs}
    names = {n for n in names if n}
    tracked = set(subprocess.run(["git", "ls-files", "results/", "paper2/"],
                                 capture_output=True, text=True).stdout.split())
    # Too large to commit, and named in the availability statement as the deposit.
    DEPOSITED = {"results/val_pools.json"}
    missing = sorted(n for n in names - tracked - DEPOSITED if Path(n).exists())
    assert not missing, (
        "these artifacts carry numbers the paper prints and are not in the repository, so the "
        "release does not contain what the availability statement says it does: "
        + ", ".join(missing))


def test_no_artifact_a_number_comes_from_is_older_than_the_pools_it_is_read_from():
    """A producer that fails silently leaves last week's answer sitting where this week's belongs.

    The comparison pools were rebuilt and one comparator's producer was invoked without a
    required argument, so it exited at once, its output went to /dev/null, and its artifact stayed
    at the previous week's date. Every other column moved to the new pools and that one did not,
    which is the mixed state the rebuild existed to remove. Nothing noticed, because an artifact
    that is stale is still stamped, still pinned, and still parses.
    """
    import glob
    import json
    import os
    from pathlib import Path

    pools = sorted(glob.glob("results/widepools_implicit/w*.json"))
    if not pools:
        return
    pool_time = max(os.path.getmtime(f) for f in pools)

    sources = json.loads(Path("results/number_sources.json").read_text())
    srcs = sources if isinstance(sources, list) else sources.get("artifacts") or list(sources)
    names = {s if isinstance(s, str) else s.get("artifact") for s in srcs}

    stale = []
    for name in sorted(n for n in names if n):
        path = Path(name)
        if not path.exists():
            continue
        try:
            blob = json.loads(path.read_text())
            source = Path(blob["provenance"]["script_path"]).read_text()
        except Exception:
            continue
        # What the producer could read is not what this artifact did read. A producer serving two
        # populations mentions both, and the validation sweep is not stale because the comparison
        # pools moved: it reads the validation pools, which did not. The artifact's own record of
        # its inputs wins over the producer's source wherever it has one.
        named = " ".join(str(row.get("path", "")) for row in (blob.get("inputs") or []))
        recorded = json.dumps(blob.get("population", ""))
        if named:
            depends = "widepools_implicit" in named
        elif "widepools_implicit" in recorded or "val_pools" in recorded:
            depends = "widepools_implicit" in recorded
        else:
            depends = "widepools_implicit" in source
        if depends and os.path.getmtime(path) < pool_time:
            stale.append(name)
    assert not stale, (
        "these artifacts are read from the comparison pools and are older than the pools "
        "themselves, so they answer a question about a build that no longer exists: "
        + ", ".join(stale))


def test_no_tracked_file_is_the_size_of_a_candidate_pool():
    """results/ is gitignored in full, so every pinned artifact is force-added by name.

    A force-add with a glob does not distinguish the artifacts from the two candidate pools,
    which are 45 MB each and are the entire content of the Zenodo deposit. One such glob put
    220 MB into the history and the only symptom was a push that hung up mid-transfer.
    """
    r = _run("check_tracked_sizes.py")
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_artifact_the_numbers_come_from_is_pinned_and_verifiable():
    """The provenance guarantee, checked by machine rather than maintained by hand.

    The paper asserted that every printed number came from a pinned artifact. Asked how many fell
    outside that, five readers produced five different counts, which is the argument for a gate
    rather than for any one of the counts. The gate runs the number generator with its artifact
    reader instrumented, so it counts files actually opened, and requires each to be pinned and to
    have a producer the provenance sweep can identify.
    """
    r = _run("check_number_provenance.py")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not (ROOT / "paper2/grail_jcim.aux").exists(),
                    reason="the manuscript has not been built in this checkout")
def test_every_cited_reference_can_be_found():
    """A reader can locate every reference the two documents cite.

    Round thirteen's referee found two incomplete entries by eye, which is the instrument this
    project replaced everywhere else. An entry passes if it carries a DOI, URL or eprint, or is
    a complete journal citation; an article missing a volume or a locator is reported as
    incomplete without failing, since a DOI already makes it findable.
    """
    r = _run("check_bibliography.py")
    assert r.returncode == 0, r.stdout + r.stderr

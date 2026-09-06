"""The paper's own gates have to run, or they rot without saying so.

`audit_claim_words.py` had been exiting non-zero for some time before this file existed: the
survey moved from the appendices into the body, the sentences it keys on moved with it, and
nothing ran the script, so a generated table stayed in the manuscript while the check that
produces it was failing. A gate nobody runs is not a gate.

These are slow enough to be worth naming and fast enough to keep in the default suite. They
need the manuscript and the committed artifacts, not the dataset.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

GATES = [
    ("audit_claim_words.py", []),      # every `certified' traces to a declared family
    ("check_prereg.py", ["--self-test"]),
    # the registry itself has to stay valid: a hypothesis that loses its failure condition or
    # its family size stops being registered, and nothing else would say so
    ("check_prereg.py", ["--prereg", "paper2/preregistration.md"]),
    ("polish_audit.py", []),           # no injected instruction, reader address or diary prose
    # every number the registration leans on still traces to the code that wrote it
    ("audit_artifact_provenance.py", []),
    # a swept row whose numbers move with the clock names which ones, and the SI says so
    ("check_wallclock_fields.py", []),
    # the word around a number makes a claim too: superlatives and separation verdicts
    ("check_quantifiers.py", []),
    # Both documents compile, resolve every reference and overrun no column. This gate was
    # written, was correct, and was not on this list, so a manuscript shipped a cross-reference
    # that printed ?? through a green suite: three referees found it and the checker that would
    # have caught it had never been run. A gate nothing invokes is not a gate.
    ("check_paper2_build.py", []),
    # LICENSE, NOTICE.md and the package configuration describe the bank that ships, not
    # the one the paper measures. Both files went on describing the wrong one for a week.
    ("check_licence_files.py", []),
    # An input digest has to be the digest a READER computes. git normalises line endings on
    # commit and reports nothing, so eight tracked data files carried bytes here that no clone
    # would ever have, and three artifacts verified in this tree and reported a moved input in a
    # fresh one. The guarantee the manuscript quotes has to hold on the reader's side.
    ("check_working_tree_matches_index.py", []),
    # The manuscript's voice packs a claim, its qualification and its evidence into one sentence,
    # which is why it read heavily; splitting them cost nothing and does not stay done, because
    # every new paragraph is written in the same voice and no single edit ever looks wrong.
    ("check_prose_density.py", []),
    # no document may claim the withheld templates are not redistributed while a tracked file
    # carries them
    ("check_no_withheld_templates.py", []),
    # the ignore rules name what the repository deposits, instead of being overridden by a
    # flag 372 times, which is how a released artifact goes missing from a release
    ("sync_tracked_artifacts.py", ["--check"]),
]


# What a gate reads that a fresh clone may not have. A missing input is a skip, not a failure.
NEEDS = {
    "check_paper2_build.py": ("paper2/si.log", "paper2/grail_jcim.log"),
}


@pytest.mark.parametrize("script,args", GATES,
                         ids=[f"{g[0]}{'-' + g[1][0].lstrip('-') if g[1] else ''}"
                              for g in GATES])
def test_paper_gate_exits_zero(script, args):
    path = ROOT / "scripts" / script
    if not path.exists():
        pytest.skip(f"{script} is not in this checkout")
    for a in args:
        if a.endswith(".md") and not (ROOT / a).exists():
            pytest.skip(f"{a} is not in this checkout")
    # Some gates read a build the repository does not carry: the manuscripts' LaTeX logs are not
    # committed, so a clone that has never compiled them has nothing for the build gate to check.
    # Skipping there is right -- the gate has no input, it has not passed -- and failing there
    # would make continuous integration red for a reason that says nothing about the code.
    for needed in NEEDS.get(script, ()):
        if not (ROOT / needed).exists():
            pytest.skip(f"{script} needs {needed}, which this checkout has not built")
    run = subprocess.run([sys.executable, str(path), *args], cwd=ROOT,
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, (
        f"{script} exited {run.returncode}\n"
        f"--- stdout ---\n{run.stdout[-4000:]}\n--- stderr ---\n{run.stderr[-2000:]}")


STRATA = [
    # (artifact, in-arm file, complement file, the row flag that means `in the arm')
    ("h1_stratum.json", "sparse_at_rule_dense_at_type.txt",
     "sparse_at_rule_dense_at_type_complement.txt", "in_stratum"),
    ("h6_stratum.json", "trivial_automorphism.txt", "nontrivial_automorphism.txt", "trivial"),
]

ORBIT_ARM = ("h6_stratum.json", "orbit_ge3.txt", "orbit_ge3_complement.txt")


def test_h6_treatment_arm_is_the_orbit_three_substrates():
    """H6 registers its treatment prediction on this file, so it must be exactly that arm."""
    import json

    art = ROOT / "results" / ORBIT_ARM[0]
    txt, comp = ROOT / "strata" / ORBIT_ARM[1], ROOT / "strata" / ORBIT_ARM[2]
    if not art.exists() or not txt.exists():
        pytest.skip("the H6 arms have not been built in this checkout")
    d = json.loads(art.read_text())
    wanted = sorted(r["substrate"] for r in d["rows"] if r["largest_orbit"] >= 3)
    listed = [l for l in txt.read_text().splitlines() if l.strip()]
    other = [l for l in comp.read_text().splitlines() if l.strip()]
    assert listed == wanted, f"the arm holds {len(listed)}, orbit>=3 is {len(wanted)}"
    assert len(listed) == d["n_orbit_ge3"]
    assert len(listed) + len(other) == d["n_substrates"], "the arm and its complement do not partition"


@pytest.mark.parametrize("artifact,arm,complement,flag", STRATA,
                         ids=[s[1].replace(".txt", "") for s in STRATA])
def test_stratum_file_matches_its_artifact(artifact, arm, complement, flag):
    """A membership file and the run that produced it cannot drift apart.

    Each hypothesis is registered on a file of substrate SMILES. If one were edited, or
    regenerated under a changed definition, the hypothesis would quietly become a different
    one. Each file has to be exactly the in-arm substrates of its committed artifact, and the
    two files have to partition the split.
    """
    import json

    art = ROOT / "results" / artifact
    txt, comp = ROOT / "strata" / arm, ROOT / "strata" / complement
    if not art.exists() or not txt.exists() or not comp.exists():
        pytest.skip(f"{artifact} has not been built in this checkout")
    d = json.loads(art.read_text())
    listed = [l for l in txt.read_text().splitlines() if l.strip()]
    other = [l for l in comp.read_text().splitlines() if l.strip()]
    from_rows = sorted({r["substrate"] for r in d["rows"] if r[flag]})
    assert listed == from_rows, (
        f"{arm} holds {len(listed)} substrates, the artifact {len(from_rows)}")
    assert len(listed) + len(other) == d["n_substrates"], f"{arm} and {complement} do not partition"
    assert not set(listed) & set(other), f"{arm} and {complement} overlap"


def test_split_manifest_still_matches_the_data():
    """The freeze is a claim about data the repository does not hold; verify it holds anyway.

    If the external dataset is absent this skips, which is the honest outcome: nothing was
    checked. If it is present and any fingerprint moved, the preregistration is registered
    against a split that no longer exists.
    """
    manifest = ROOT / "paper2" / "split_manifest.json"
    data = ROOT / "grail_metabolism" / "data" / "test_triples_clean.txt"
    if not manifest.exists() or not data.exists():
        pytest.skip("the split manifest or the external dataset is not in this checkout")
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "typed_edit" / "freeze_split.py"),
                          "--verify"], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, f"the split moved since the freeze\n{run.stdout[-3000:]}"


def test_h1_primary_stratum_is_the_intersection():
    """The primary membership file must be exactly the pairs every join key agrees on.

    H1 names this file as its primary definition in advance, so it cannot be the file that
    happened to come out best. The two files also have to partition the substrates that carry
    at least one typeable reference -- not the whole split, because 14 substrates carry none.
    """
    import json

    art = ROOT / "results" / "h1_join_sensitivity.json"
    txt = ROOT / "strata" / "sparse_at_rule_dense_at_type_intersection.txt"
    comp = ROOT / "strata" / "sparse_at_rule_dense_at_type_intersection_complement.txt"
    if not art.exists() or not txt.exists():
        pytest.skip("the join sensitivity has not been computed in this checkout")
    d = json.loads(art.read_text())
    keys = d["primary"]["keys"]
    wanted = sorted({r["substrate"] for r in d["rows"]
                     if all(r["in_stratum"][k] for k in keys)})
    listed = [l for l in txt.read_text().splitlines() if l.strip()]
    other = [l for l in comp.read_text().splitlines() if l.strip()]
    assert listed == wanted, f"the primary file holds {len(listed)}, the intersection {len(wanted)}"
    assert len(listed) == d["primary"]["substrates"]
    typeable = {r["substrate"] for r in d["rows"]}
    assert len(listed) + len(other) == len(typeable), "the two files do not partition"
    assert not set(listed) & set(other)
    # the ceiling the registration quotes has to be the one the arithmetic gives
    assert d["feasibility"]["max_gain_for_K"]["2.5"] == int(d["n_typeable_pairs"] // 2.5)

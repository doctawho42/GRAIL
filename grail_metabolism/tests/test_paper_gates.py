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
]


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

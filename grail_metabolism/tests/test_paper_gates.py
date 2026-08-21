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


def test_h1_stratum_file_matches_its_artifact():
    """The membership file and the run that produced it cannot drift apart.

    H1 is registered on a file of substrate SMILES. If that file were edited, or regenerated
    under a changed definition, the hypothesis would quietly become a different one. The file
    has to be exactly the in-stratum substrates of the committed artifact.
    """
    import json

    art = ROOT / "results" / "h1_stratum.json"
    txt = ROOT / "strata" / "sparse_at_rule_dense_at_type.txt"
    if not art.exists() or not txt.exists():
        pytest.skip("the H1 stratum has not been built in this checkout")
    d = json.loads(art.read_text())
    listed = [l for l in txt.read_text().splitlines() if l.strip()]
    from_rows = sorted({r["substrate"] for r in d["rows"] if r["in_stratum"]})
    assert listed == from_rows, (
        f"the stratum file holds {len(listed)} substrates, the artifact {len(from_rows)}")
    assert len(listed) == d["n_stratum_substrates"]
    complement = [l for l in (ROOT / "strata" /
                              "sparse_at_rule_dense_at_type_complement.txt").read_text()
                  .splitlines() if l.strip()]
    assert len(listed) + len(complement) == d["n_substrates"], "the two files do not partition"

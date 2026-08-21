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

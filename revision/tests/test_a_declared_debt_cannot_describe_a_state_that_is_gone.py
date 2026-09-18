"""A debt entry that stamps a digest must stamp the file's current one, or it is dating a ghost.

Run: python -m pytest revision/tests/test_a_declared_debt_cannot_describe_a_state_that_is_gone.py -q

DECLARED_DEBT in grail_metabolism/tests/test_paper_gates.py records, for each gate that is red on
purpose, what it measures and how far off it is. Several of those figures are stamped with the
sha256 of the file they were read from -- a mechanism the entry itself explains was introduced
because two readers of one file, minutes apart, reported 79 and 82, and only a file identity settles
which state was read.

The stamp makes a reading placeable. It does not make it current, and the difference bit.

The prose-rate entry carried "paper2/si.tex 259 of 1097 (23.61%) against 24 at digest 83ac498e".
That digest is si.tex three commits back. The very next commit took the same file to 30 carriers and
the working tree stood at 31 against a ceiling of 24 -- over by seven, where the entry reported an
excess of 235. The number was not merely old. It described the state BEFORE the work that fixed it,
and it overstated the live gap by a factor of thirty-three in the direction that makes the debt look
unclosable. In the same sentence the body.tex half was current, so the entry was not a deliberate
historical record; half of it tracked the live file and half of it had rotted.

An entry like that is the exact failure this project objects to in other people's papers: the work
was done and the passage explaining how far there was to go stayed. It is also self-refuting, since
the entry's own text says "A number a reader cannot re-derive is a number that stays wrong" and this
one re-derives in under a second.

So: every digest a debt entry stamps must be the current digest of the file it names. The entry may
still be red, and may still be long, and may still argue -- but it may not argue from a state that
is gone.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATES_FILE = ROOT / "grail_metabolism" / "tests" / "test_paper_gates.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# "paper2/si.tex 259 of 1097 (23.61%) against 24 at digest 83ac498e"
STAMPED = re.compile(r"(?P<file>paper2/[\w./-]+\.\w+)[^\"]{0,120}?at digest (?P<digest>[0-9a-f]{8})")


def _debt() -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_paper_gates", GATES_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "DECLARED_DEBT", {})


def test_every_digest_a_debt_entry_stamps_is_the_file_s_current_one():
    """The check that would have caught three commits of rot in under a second.

    Read from the entries rather than from a list here, so a debt entry added later is covered
    without anyone remembering to add it.
    """
    debt = _debt()
    assert debt, "DECLARED_DEBT is empty or could not be read"
    stale, checked = [], 0
    for gate, text in debt.items():
        for m in STAMPED.finditer(text if isinstance(text, str) else str(text)):
            path = ROOT / m.group("file")
            if not path.exists():
                stale.append(f"{gate}: names {m.group('file')}, which does not exist")
                continue
            now = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
            checked += 1
            if now != m.group("digest"):
                stale.append(
                    f"{gate}: stamps {m.group('file')} at {m.group('digest')} and the file is now "
                    f"{now}; the figure beside that stamp describes a state that is gone")
    assert checked, "no debt entry stamps a digest, so this check verified nothing"
    assert not stale, (
        "a declared debt argues from a file state that no longer exists:\n  "
        + "\n  ".join(stale)
        + "\n\nRe-run the gate, put its live figures in the entry, and re-stamp. A debt may stay "
          "red; it may not stay red on yesterday's numbers.")


def test_a_debt_entry_does_not_promise_work_in_a_window_that_has_closed():
    """A plan is a claim about the future and expires like any other.

    The prose-rate entry promised its Supporting-Information half "for W4" after the commit titled
    "W4 closes, and its entry leaves the declared-debt list" had landed. A window named in a debt
    entry has to be one that is still open, or the entry is describing work as pending that its own
    history says is over.
    """
    debt = _debt()
    log = ROOT / ".git"
    if not log.exists():
        return  # nothing to check against outside a checkout
    import subprocess
    closed = set()
    r = subprocess.run(["git", "log", "--format=%s", "-200"], cwd=ROOT,
                       capture_output=True, text=True, timeout=30)
    for line in r.stdout.splitlines():
        m = re.search(r"\b(W\d+)\b.{0,40}\bcloses\b", line)
        if m:
            closed.add(m.group(1))
    offenders = []
    for gate, text in debt.items():
        for w in re.findall(r"\bW\d+\b", text if isinstance(text, str) else str(text)):
            if w in closed:
                offenders.append(f"{gate}: still names {w}, which the history records as closed")
    assert not offenders, (
        "a declared debt schedules work into a window that has already closed:\n  "
        + "\n  ".join(offenders))

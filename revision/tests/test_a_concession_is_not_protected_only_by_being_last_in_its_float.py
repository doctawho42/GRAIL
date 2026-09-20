"""A short concession stays in the registry when a sentence is appended after it.

The inventory keys a concession by its normalised sentence. Inside a float, the last sentence
absorbs the environment names that follow it -- `tablenotes threeparttable table` -- because
nothing splits there. That tail made short concessions long enough to clear a 40-character floor,
so their protection came from their position in the LaTeX and not from their prose.

Appending one sentence to a table's notes took the tail off the sentence before it, dropped that
sentence under the floor, and `--check` reported a retraction for a sentence that is still in the
document, verbatim. Measured, the floor was protecting nothing else: at 40 characters the
inventory holds 244 keys, and at every value from 30 down to 0 it holds the same 249, because the
six-word floor is the only one that discriminates. Four of the five extra keys are concessions in
the Supporting Information that had never been protected at all.

So the character floor is gone and the word floor stays. These tests pin the property the removal
was for rather than the constant: a concession that stands on its own prose must not depend on
standing last in its float. They drive `collect()` itself, because a helper that re-implements the
floors is a gate checking its own copy of the thing under test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import disclosure_inventory as D  # noqa: E402

# Short enough that the old 40-character floor rejected it, and a real concession.
SHORT = "It is not the released configuration."
FRAGMENT = "MetaTox was not."

FLOAT = """\\begin{table}[t]
\\begin{threeparttable}
\\caption{Recall at each budget.}
\\begin{tablenotes}\\item The last column re-ranks the pool under another aggregation. BODY
\\end{tablenotes}
\\end{threeparttable}
\\end{table}
"""


def _collect(tmp_path, monkeypatch, body: str) -> dict:
    """Run the real collector over one synthetic float."""
    rel = "paper2/fixture_table.tex"
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(FLOAT.replace("BODY", body))
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setattr(D, "SOURCES", [rel])
    return D.collect()


def test_the_fixture_speaks_the_collector_s_dialect(tmp_path, monkeypatch):
    """Before anything else: the float must reach the collector at all.

    A fixture whose markup the extractor strips would make every other test here vacuous --
    green because nothing was collected rather than because the right thing was.
    """
    got = _collect(tmp_path, monkeypatch, SHORT)
    assert got, "the collector saw no concession in the fixture; the fixture is the bug"


def test_the_short_concession_is_collected_when_it_stands_last(tmp_path, monkeypatch):
    """Collected either way. Its KEY still differs by position -- see the xfail below."""
    got = _collect(tmp_path, monkeypatch, SHORT)
    assert any(k.startswith(D._norm(SHORT)) for k in got), sorted(got)


@pytest.mark.xfail(strict=True, reason=(
    "Declared debt, measured not guessed. The last sentence of a float still absorbs the "
    "environment names after it, so this concession's key is 'it is not the released "
    "configuration tablenotes threeparttable table' when it stands last and the bare sentence "
    "when it does not. Stripping \\begin/\\end WITH their argument in _plain fixes it and "
    "re-keys 13 of 249 entries: 7 pair as new-key-plus-tail, 6 shift at the FRONT instead "
    "(\\begin{abstract} puts 'abstract' at the head of the key), and none collapse into another. "
    "That is a re-baseline of its own with its own pairing to read, not a side effect of a "
    "caption edit, so it is declared here rather than taken tonight."))
def test_the_key_does_not_depend_on_position_in_the_float(tmp_path, monkeypatch):
    last = _collect(tmp_path, monkeypatch, SHORT)
    followed = _collect(tmp_path, monkeypatch, SHORT + " A sentence after it, and another.")
    assert (set(last) & set(followed)) == set(last)


def test_it_is_still_collected_when_a_sentence_follows_it(tmp_path, monkeypatch):
    """The failure this test exists for: one appended sentence read as a retraction."""
    after = SHORT + " Figure 1 plots these recalls and marks where they separate."
    got = _collect(tmp_path, monkeypatch, after)
    assert D._norm(SHORT) in got, (
        "the concession is keyed on standing last in its float, so appending a sentence after "
        f"it reads as a deletion; collected {sorted(got)}")


def test_the_word_floor_still_rejects_a_fragment(tmp_path, monkeypatch):
    """Removing the character floor must not admit the tail fragments the floors are for."""
    got = _collect(tmp_path, monkeypatch, FRAGMENT + " A sentence after it.")
    assert D._norm(FRAGMENT) not in got


def test_the_live_document_carries_it():
    """And the real file, so the fixture cannot drift into agreeing with itself."""
    p = ROOT / "paper2/table_sweep.tex"
    if not p.exists():
        pytest.skip("table_sweep.tex is not in this checkout")
    assert SHORT in p.read_text()
    assert D._norm(SHORT) in D.collect()

"""The whole-population contrast grid exists as a table, because the paper twice said it did not.

Run: python -m pytest revision/tests/test_the_wider_population_has_the_table_the_paper_says_it_lacks.py -q

The comparison set's contrasts are laid out in Table~\\ref{tab:si-intervals}: every difference
between a GRAIL arm and a comparator, with its paired interval, five comparators against two arms.
The whole evaluated test set has the same grid and had no table at all, and the Supporting
Information admitted it in those words twice -- once to say the wider cells "are in no table in this
paper", and once to explain that GLORYx's wider cells are "among those no table carries, so they are
set out here". Sixty-seven cell macros were being read out in sentences as a result, and admitting
MetaTox to that axis added eleven more.

A prose enumeration is not a neutral substitute for a grid. It is where a reader cannot see the
shape, where an omission is invisible, and where -- as the quantifier gate caught in this very
section -- a claim can bind to the wrong cell.

So the tests below hold the table to what the paper's own convention already demands of its sibling:

  it carries EVERY comparator the axis holds, taken from the artifact rather than from a list, so a
  sixth arm cannot be silently dropped from a table whose caption says it holds them all;
  it carries both GRAIL arms and every budget the axis sweeps;
  every cell it prints matches the artifact, sign and interval and separation mark alike;
  and the two admissions that the wider cells appear in no table are gone from the manuscript,
  because a paper that prints the table and still says it has none is wrong either way.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TABLE = ROOT / "paper2" / "si_table_intervals_whole.tex"
AXIS = ROOT / "results" / "population_definition.json"
SI = ROOT / "paper2" / "si.tex"

ARMS = (("whole bank", "exhaustive_minus_comparator"),
        ("trained budget", "deployed_minus_comparator"))


def _table() -> str:
    assert TABLE.exists(), (
        f"{TABLE.relative_to(ROOT)} does not exist. The comparison set has a table of every "
        f"contrast with its interval and the wider population has none, which the Supporting "
        f"Information states twice rather than fixes.")
    return TABLE.read_text()


def _axis() -> dict:
    return json.loads(AXIS.read_text())["contrasts"]["the whole evaluated test set"]


def test_the_table_carries_every_comparator_the_axis_holds():
    """A caption that says "every" has to mean it.

    Its sibling refuses to build when the deployment table carries a comparator the table has no
    column for, on the stated ground that printing four of five while the text reads a verdict off
    the fifth is worse than not building. The same has to be true here, and it is checked against
    the axis rather than against a list that was correct when it was written.
    """
    t = _table()
    axis = _axis()
    comps = sorted(a for a, v in axis.items()
                   if isinstance(v, dict) and any(k in v for _, k in ARMS))
    assert len(comps) >= 5, f"the axis holds {len(comps)} comparators: {comps}"
    LABELS = {"metatox": "MetaTox", "sygma": "SyGMa", "metapredictor": "MetaPredictor",
              "biotransformer": "BioTransformer", "gloryx": "GLORYx",
              "gloryxr_default": "GLORYxR def.", "gloryxr_strict": "GLORYxR str."}
    for c in comps:
        assert c in LABELS, f"the axis holds {c!r} and this test has no label for it"
        assert LABELS[c] in t, (
            f"the axis carries contrasts against {LABELS[c]} and the table has no column for it")


def test_both_arms_and_every_budget_are_present():
    """The grid is two arms by five budgets; a table missing either is a different claim."""
    t = _table()
    axis = _axis()
    ks = sorted({int(k) for v in axis.values() if isinstance(v, dict)
                 for _, key in ARMS for k in (v.get(key) or {})})
    assert len(ks) >= 5, f"budgets in the axis: {ks}"
    for k in ks:
        assert re.search(rf"\${k}\$", t), f"the table has no row for a budget of {k}"
    assert "exhaustive" in t and "interactive" in t, "the table does not name both GRAIL arms"


def test_every_printed_cell_is_the_artifact_s_cell():
    """Read back from the table and compared with the axis, so a hand-edit shows up.

    The values are trimmed for width -- +0.0496 prints as +.050 -- so the comparison is to three
    decimals, which is what the page carries.
    """
    t = _table()
    axis = _axis()
    LABELS = {"metatox": "MetaTox", "sygma": "SyGMa", "metapredictor": "MetaPredictor",
              "biotransformer": "BioTransformer", "gloryx": "GLORYx",
              "gloryxr_default": "GLORYxR def.", "gloryxr_strict": "GLORYxR str."}
    order = [c for c in LABELS if c in axis]
    # every numeric cell the table prints, in reading order
    printed = re.findall(r"(?:\$-\$|\+)\.\d{3}", t)
    # Seven comparator columns do not fit a one-column page, so the grid is stacked in blocks of
    # four and the reading order is block by block: within a block, both arms over every budget.
    # Reading it as one wide table gave the right multiset in the wrong order and failed here,
    # which is the gate working: a cell read in the wrong place is a cell attributed to the wrong
    # comparator.
    PER_BLOCK = 4
    chunks = [order[i:i + PER_BLOCK] for i in range(0, len(order), PER_BLOCK)]
    expected = []
    for group in chunks:
        for _, key in ARMS:
            ks = sorted({int(k) for c in order for k in (axis[c].get(key) or {})})
            for k in ks:
                for c in group:
                    cell = (axis[c].get(key) or {}).get(str(k))
                    if cell is None:
                        continue
                    for v in (cell["difference"], cell["ci95"][0], cell["ci95"][1]):
                        expected.append(("$-$" if v < 0 else "+") + f"{abs(v):.3f}".lstrip("0"))
    assert printed == expected, (
        f"the table prints {len(printed)} numbers and the axis gives {len(expected)}; first "
        f"disagreement at "
        f"{next((i for i, (a, b) in enumerate(zip(printed, expected)) if a != b), 'the end')}")


def test_the_admissions_that_no_table_carries_these_cells_are_gone():
    """A paper that prints the table and still says it has none is wrong either way.

    Both sentences were honest when written. Leaving either in place after the table exists would
    turn an honest admission into a false one, which is the same defect running the other way.
    """
    si = SI.read_text()
    for stale in ("the wider population's cells are in no table in this paper",
                  "among those no table carries"):
        assert stale not in si, (
            f"paper2/si.tex still says {stale!r} while "
            f"{TABLE.name} exists")
    assert "si_table_intervals_whole" in si, "the table is generated and never included"

"""What a fresh clone does not have, and why a check that needs it must say so.

The release withholds three kinds of file on purpose: the measured rule bank, because 611 of its
templates are BioTransformer's and the shipped bank is the one without them; the candidate pools,
because a single one is 46 MB and a git history is not the place for them; and the corpus, whose
sources' terms do not combine.

Checks that read those files were failing in a clone. That is the wrong answer twice over. The
check has not found a defect -- there is nothing wrong with a tree that lacks a file the release
never offered -- and a reader who runs the documented entry point should not be told the software
is broken when it is not. Passing silently would be worse: the check would look as though it had
run.

So they skip, and the skip names the file and the reason, which is the only outcome that is honest
in both trees: in ours the check runs, in a clone the reader is told exactly what did not run and
why it could not.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

WHY = {
    "grail_metabolism/resources/extended_smirks.txt":
        "the measured bank is not redistributed; the released bank beside it is what ships",
    "results/val_pools.json":
        "a 46 MB candidate pool, kept out of the history by scripts/check_tracked_sizes.py",
}


def requires(*relpaths):
    """Skip, naming the file and why the release does not carry it, when it is absent."""
    missing = [p for p in relpaths if not (ROOT / p).exists()]
    if not missing:
        return
    reasons = "; ".join(
        f"{p} ({WHY.get(p, 'not redistributed')})" for p in missing)
    pytest.skip(f"not checkable in this tree: {reasons}. This is a clone of the release, which "
                f"withholds it by design, not a defect in the check.")


def any_missing(prefix="results/valpools_"):
    """Whether a family of untracked artifacts is absent, for checks that read a whole family."""
    return not list(ROOT.glob(prefix + "*"))

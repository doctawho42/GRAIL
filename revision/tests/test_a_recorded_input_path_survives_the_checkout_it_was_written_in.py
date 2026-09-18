"""An input path an artifact records has to name the file in ANY checkout, not in the one that ran.

Run: python -m pytest revision/tests/test_a_recorded_input_path_survives_the_checkout_it_was_written_in.py -q

scripts/_provenance.record_inputs writes, for each file a producer read, a path and a digest, so an
artifact can be checked against what it was actually pointed at. check_inputs reads those rows back
as ROOT / row["path"], which only works if the path is repo-relative.

record_inputs relativises with `path.resolve().relative_to(ROOT)`. resolve() follows symlinks. In a
git worktree the large dataset under grail_metabolism/data/ is symlinked to the main checkout, so
resolve() lands OUTSIDE ROOT, relative_to raises, and the fallback records `str(path)` -- an
absolute path naming a directory that exists on one machine.

Nothing caught this because no producer had ever recorded a file from that directory: all 64
artifacts carrying an inputs list record paths under results/ or grail_metabolism/resources/, which
are real files, and every one of those 64 is repo-relative. The first producer to record a dataset
file would have been the first to write a machine-specific path, and check_inputs would then report
"input gone" in every clean checkout while reporting nothing on the machine that wrote it -- a check
that passes exactly where it cannot fail.

The fix is to try the UNRESOLVED path against ROOT first. For a real file under ROOT that gives the
same answer resolve() gives; for a symlink under ROOT it gives the repo-relative name, which is what
the artifact needs and what check_inputs looks for.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _provenance as pv  # noqa: E402


def test_a_symlinked_input_under_root_records_a_repo_relative_path(tmp_path):
    """The case the worktree actually produces, planted so it can be checked anywhere.

    A file outside ROOT, a symlink to it inside ROOT, and the symlink handed to record_inputs. The
    row must name the symlink's place in the repository, because that is the name a reader in a
    different checkout can resolve.
    """
    target = tmp_path / "elsewhere.txt"
    target.write_bytes(b"the dataset lives outside this checkout")
    link = ROOT / "results" / "_provenance_symlink_probe.tmp"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(target, link)
    try:
        rows = pv.record_inputs([link])
    finally:
        link.unlink()

    assert len(rows) == 1
    path = rows[0]["path"]
    assert not path.startswith("/"), (
        f"record_inputs wrote an absolute path {path!r}. It names a directory on one machine, so "
        f"check_inputs resolves ROOT / that path to nothing in every other checkout and reports the "
        f"input gone. resolve() followed the symlink out of ROOT; the unresolved path was already "
        f"inside it.")
    assert path == "results/_provenance_symlink_probe.tmp", path
    assert rows[0]["exists"] is True
    assert rows[0]["sha256_16"] == pv._digest(target.read_bytes())[:16], (
        "the digest must be of the file the symlink points at, since that is what the producer read")


def test_an_ordinary_file_under_root_is_unchanged_by_the_fix():
    """The 64 artifacts that already carry relative paths must keep recording the same string."""
    real = ROOT / "scripts" / "_provenance.py"
    rows = pv.record_inputs([real])
    assert rows[0]["path"] == "scripts/_provenance.py", rows[0]["path"]
    assert rows[0]["exists"] is True


def test_check_inputs_can_read_back_what_record_inputs_wrote(tmp_path):
    """The pair has to close: whatever record_inputs writes, check_inputs must find.

    Written because the two halves live in one file and were never exercised against each other on
    a symlink, which is how a path that only one machine can resolve got past both.
    """
    target = tmp_path / "elsewhere.txt"
    target.write_bytes(b"payload")
    link = ROOT / "results" / "_provenance_symlink_probe2.tmp"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(target, link)
    try:
        rec = {"inputs": pv.record_inputs([link])}
        problems = pv.check_inputs(rec)
    finally:
        link.unlink()
    assert problems == [], (
        f"check_inputs could not read back what record_inputs just wrote: {problems}")

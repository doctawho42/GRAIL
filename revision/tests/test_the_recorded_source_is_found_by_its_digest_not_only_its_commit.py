"""A stamp records WHICH source ran. Recovering it should use that, not only the commit beside it.

Run: python -m pytest revision/tests/test_the_recorded_source_is_found_by_its_digest_not_only_its_commit.py -q

scripts/_provenance.stamp writes three things about the producer: its source_sha256, the HEAD commit
at write time, and whether the tree was dirty. recorded_source recovers the old source by asking git
for the producer AT THE RECORDED COMMIT -- and nothing else. When the recorded commit does not carry
the producer, it gives up, and verify() can then only report producer_changed: a bare "the digest
moved", with no way to say whether the change was cosmetic.

That combination is not rare here. Measured over results/**.json: 271 artifacts carry a stamp, 252
of them were written from a dirty tree, and 62 name a commit that does not contain their own
producer -- a script written and run before it was committed stamps the HEAD it was run against,
which predates itself. For those 62, the cosmetic-versus-substantive diagnosis is permanently
unavailable, and it becomes unavailable exactly when the producer changes, which is when it is
needed. DECLARED_DEBT records this as a property of three artifacts.

The recorded digest is the stronger key. A blob that was ever committed can be found by its digest
anywhere in history, whatever commit the stamp happens to name. Both cases that prompted this were
recoverable that way: results/metatox_outside_submission.json stamps commit b0fb71e, which does not
carry scripts/metatox_outside_ingest.py at all, while the blob it records sits in c38e7b6 -- and
comparing that blob with the current file proves the only change is a docstring.

So: when the recorded commit cannot produce the source, recorded_source must look for the recorded
digest in history before giving up.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _provenance as pv  # noqa: E402


def _a_tracked_script_with_history() -> tuple:
    """Any committed script, with its HEAD blob. Chosen from git so this does not name a file."""
    listed = subprocess.run(["git", "ls-files", "scripts/*.py"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30).stdout.split()
    for rel in listed:
        blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)
        if blob.returncode == 0 and blob.stdout.strip():
            return rel, blob.stdout
    raise AssertionError("no tracked script under scripts/ could be read at HEAD")


def _root_commit() -> str:
    out = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, timeout=30).stdout.split()
    assert out, "the repository has no root commit"
    return out[0]


def test_the_source_is_recovered_when_the_stamped_commit_predates_the_producer():
    """The 62-artifact case, planted: a right digest beside a wrong commit.

    The commit is the repository's root, which is guaranteed not to carry a script added later --
    the same situation a producer creates by stamping the HEAD it was run against before it was
    itself committed.
    """
    rel, blob_at_head = _a_tracked_script_with_history()
    rec = {"script": Path(rel).name, "script_path": rel,
           "source_sha256": pv._digest(blob_at_head.encode()),
           "git_commit": _root_commit(), "git_dirty": True}

    source, how = pv.recorded_source(rec, ROOT / rel)
    assert source is not None, (
        f"recorded_source gave up on {rel} because the stamped commit does not carry it, although "
        f"the digest it records identifies a blob that IS in history. Reported: {how}")
    assert pv._digest(source.encode()) == rec["source_sha256"], (
        f"recorded_source returned a source that is not the one the stamp records ({how})")
    assert "digest" in how.lower() or "history" in how.lower(), (
        f"the recovery route has to say it came from the digest and not from the stamped commit, "
        f"so a reader can tell the two apart; it said: {how}")


def test_a_correct_commit_is_still_used_and_still_says_so():
    """The ordinary path must not change: 209 of the 271 stamps name a usable commit."""
    rel, blob_at_head = _a_tracked_script_with_history()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, timeout=30).stdout.strip()
    rec = {"script": Path(rel).name, "script_path": rel,
           "source_sha256": pv._digest(blob_at_head.encode()),
           "git_commit": head, "git_dirty": False}
    source, how = pv.recorded_source(rec, ROOT / rel)
    assert source == blob_at_head
    assert head[:12] in how, how


def test_a_digest_that_is_nowhere_in_history_still_reports_that_it_cannot_be_recovered():
    """The refusal has to survive. A search that always finds something answers nothing.

    A producer that ran uncommitted and was then edited leaves a digest matching no blob anywhere;
    recorded_source must say so rather than returning the nearest thing it found.
    """
    rel, _ = _a_tracked_script_with_history()
    rec = {"script": Path(rel).name, "script_path": rel,
           "source_sha256": "0" * 64, "git_commit": _root_commit(), "git_dirty": True}
    source, how = pv.recorded_source(rec, ROOT / rel)
    assert source is None, (
        f"recorded_source claimed to recover a source for a digest that is in no commit: {how}")

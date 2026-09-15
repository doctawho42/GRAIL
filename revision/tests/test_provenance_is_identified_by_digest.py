"""A provenance record must identify its subject by digest, not by a path that will not exist.

Run: python -m pytest revision/tests/test_provenance_is_identified_by_digest.py -q

Why this file exists. The GLORYxR model archive was delivered out of band into a session
scratchpad, and two records wrote that absolute path into their provenance: revision/
provenance_gloryx.json (the archive) and results/gloryxr_mechanics.json (the rule table read out
of the clone). The scratchpad is keyed to one session and will not exist for a reader, so those
records pointed at a location nobody else can ever inspect, in the field that exists to make them
checkable. A digest does not have that problem: it identifies the bytes without needing the path.

Two things are pinned here.

  1. When the archive is present, the record carries a digest of what it found and does not carry
     the session path. The digest is computed from the dumps themselves, so it stays checkable by
     anyone holding a copy even when the original .tar.gz is gone.
  2. The claim and its evidence do not drift apart. The blocker gloryxr_duplicate_model asserts a
     byte-identical pair; if the archive record cannot show that pair, the record must say so
     rather than leave the assertion standing with nothing under it. That is the gate that was
     missing: the archive record silently lost its nine digests during a regeneration without the
     environment variable, and a diff caught it where no test could.

Scope, stated rather than quietly narrowed: results/gloryxr_local_preds_default.json and
results/gloryxr_local_preds_strict.json also record scratchpad paths. They are not covered here.
Rewriting their provenance means re-running 1,170 substrates in two modes, and an exception list
would make this gate unable to fail. They remain an open defect of the same class.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "revision"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import phase2_comparators as P  # noqa: E402
import phase2_gloryxr_mechanics as M  # noqa: E402

SCRATCH_MARKERS = ("/scratchpad", "claude-501", "/private/tmp", "/var/folders")


def _write_fake_archive(tmp_path: Path, payloads: dict) -> Path:
    """A models/ directory with the shape the real one has: dumps under multi_models/."""
    root = tmp_path / "models"
    (root / "multi_models").mkdir(parents=True)
    for name, blob in payloads.items():
        (root / "multi_models" / name).write_bytes(blob)
    return root


# --------------------------------------------------------------------------- the digest

def test_a_present_archive_is_identified_by_a_digest(tmp_path):
    """The record must carry a digest of the set it found, not only per-file hashes."""
    root = _write_fake_archive(tmp_path, {"a.joblib": b"aaa", "b.joblib": b"bbb"})
    rec = P.model_archive(root)
    assert rec["present"] is True
    digest = rec.get("manifest_sha256")
    assert digest, "a present archive carries no manifest digest"
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_the_manifest_digest_is_derived_from_the_dumps_and_not_from_the_path(tmp_path):
    """Two copies of the same dumps in different directories must agree.

    This is the property that makes the digest a replacement for the path: it identifies the
    bytes, so it survives the directory being gone.
    """
    payloads = {"a.joblib": b"aaa", "b.joblib": b"bbb"}
    one = P.model_archive(_write_fake_archive(tmp_path / "one", payloads))
    two = P.model_archive(_write_fake_archive(tmp_path / "two", payloads))
    assert one["manifest_sha256"] == two["manifest_sha256"]

    other = P.model_archive(_write_fake_archive(
        tmp_path / "three", {"a.joblib": b"aaa", "b.joblib": b"CHANGED"}))
    assert other["manifest_sha256"] != one["manifest_sha256"], (
        "the digest does not change when a dump changes, so it identifies nothing")


def test_the_manifest_digest_is_computable_by_hand(tmp_path):
    """Stated so a reader can check it without running this code: it is the sha256 of the sorted
    "name:sha256" lines of the dumps, newline separated."""
    payloads = {"a.joblib": b"aaa", "b.joblib": b"bbb"}
    rec = P.model_archive(_write_fake_archive(tmp_path, payloads))
    lines = sorted(f"{m['name']}:{m['sha256']}" for m in rec["models"])
    expected = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    assert rec["manifest_sha256"] == expected


def test_a_present_archive_does_not_record_the_session_path(tmp_path):
    """The field that was wrong. A path under a session scratchpad is unusable to a reader, and
    writing it where the record promises checkability is the defect this file closes."""
    root = _write_fake_archive(tmp_path, {"a.joblib": b"aaa"})
    rec = P.model_archive(root)
    blob = json.dumps(rec)
    for marker in SCRATCH_MARKERS:
        assert marker not in blob, f"the record carries a session path fragment {marker!r}"


def test_an_absent_archive_still_says_where_it_looked(tmp_path):
    """Absence keeps its diagnostic path: someone regenerating the record needs to know which
    location came up empty. Only the present branch replaces the path with a digest."""
    rec = P.model_archive(tmp_path / "definitely" / "not" / "here")
    assert rec["present"] is False
    assert rec.get("checked_path"), "an absent archive must still say where it looked"
    assert rec.get("manifest_sha256") is None


# --------------------------------------------------------------------------- claim versus evidence

def test_the_duplicate_blocker_is_not_left_asserting_what_the_record_cannot_show():
    """The gate that was missing.

    A regeneration without GRAIL_GLORYXR_MODELS rewrote the archive record as present=false and
    dropped all nine digests, while the blocker went on asserting a byte-identical pair. The two
    must not drift: either the record shows the pair, or it states that the evidence is
    unavailable in this checkout.
    """
    archive = json.loads((ROOT / "revision" / "provenance_gloryx.json").read_text())["model_archive"]
    blockers = {b["id"]: b for b in P.blockers()}
    asserts_pair = "byte-identical" in json.dumps(blockers["gloryxr_duplicate_model"])
    if not asserts_pair:
        return
    if archive.get("present"):
        groups = archive.get("byte_identical_groups")
        assert groups, ("the blocker asserts a byte-identical pair and the archive is present, "
                        "but the record shows no such group")
        assert any(len(g) > 1 for g in groups)
    else:
        assert archive.get("note"), (
            "the archive is absent and the blocker still asserts a byte-identical pair, with "
            "nothing in the record saying the evidence is unavailable here")


# --------------------------------------------------------------------------- the mechanics record

def test_the_rule_table_record_is_identified_by_digest_not_by_the_clone_path(tmp_path):
    """The rule table is read out of a GLORYxR clone, which is as transient as the archive."""
    csv = tmp_path / "gloryx_reactionrules_connect.csv"
    csv.write_text("Reaction name,SMIRKS,Priority level,Name of rule subset,Rule source\n"
                   "r1,[C:1]>>[C:1],common,Phase 1 SyGMa rules,SyGMa\n"
                   "r2,[N:1]>>[N:1],uncommon,Other phase 2 rules,GLORY\n")
    rec = M.rule_table(csv)
    assert rec["present"] is True
    assert rec["n_rules"] == 2 and rec["n_uncommon"] == 1
    digest = rec.get("sha256")
    assert digest, "the rule table record carries no digest"
    assert digest == hashlib.sha256(csv.read_bytes()).hexdigest()
    blob = json.dumps(rec)
    for marker in SCRATCH_MARKERS:
        assert marker not in blob, f"the record carries a session path fragment {marker!r}"


def test_the_committed_mechanics_artefact_carries_no_session_path():
    """The artefact as committed, not only the function that writes it."""
    blob = (ROOT / "results" / "gloryxr_mechanics.json").read_text()
    for marker in SCRATCH_MARKERS:
        assert marker not in blob, f"the committed artefact carries {marker!r}"


def test_the_committed_comparator_record_carries_no_session_path():
    blob = (ROOT / "revision" / "provenance_gloryx.json").read_text()
    for marker in SCRATCH_MARKERS:
        assert marker not in blob, f"the committed artefact carries {marker!r}"

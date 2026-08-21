#!/usr/bin/env python3
"""Which version of which script wrote a number, checked when the number is read.

Artifacts here are gated on WRITE: a harness refuses to emit until it reproduces a committed
figure. They are read blind. That asymmetry produced a defect this project hit for real --
`bank_without_selection.json` carried `ceiling_on_this_subset: 0.7284` long after a correction
took the same quantity to 0.8007, and the stale literal went on looking committed because
nothing that read it asked which version of the code had written it.

This is the missing half. Writers stamp; readers verify.

  stamp(__file__)        the writing script's name, the digest of its own source, and the
                         commit and dirty flag of the tree it ran in
  verify(artifact)       recomputes whether the producer has changed since it wrote this
  read_checked(path)     load a JSON artifact and raise if its producer has moved

Verification works retroactively on the 84 artifacts that already record a commit but no source
digest: the producer's source AT that commit is recovered with `git show` and hashed. Anything
stamped from now on carries its own digest and needs no git at all.

A changed producer is not proof that a number is wrong. It is proof that nobody has checked,
which is the state the three defects in this project's own history all shared.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CURRENT = "current"
CHANGED = "producer_changed"
UNSTAMPED = "unstamped"
UNKNOWN = "producer_unknown"


def _git(*args, cwd=ROOT):
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stamp(file: str | Path) -> dict:
    """The provenance block a writer embeds. `stamp(__file__)` at the call site."""
    path = Path(file).resolve()
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        rel = path.name
    out = {"script": path.name, "script_path": rel,
           "source_sha256": _digest(path.read_bytes()) if path.exists() else None,
           "git_commit": (_git("rev-parse", "HEAD") or "").strip() or None,
           "git_dirty": bool((_git("status", "--porcelain") or "").strip())}
    return out


def _find_stamp(obj, depth=0):
    """Locate a provenance block wherever a writer put it: top level, or under `config`."""
    if depth > 2 or not isinstance(obj, dict):
        return None
    if "script" in obj and isinstance(obj.get("script"), str):
        return obj
    for key in ("provenance", "config", "code_version", "meta"):
        got = _find_stamp(obj.get(key), depth + 1)
        if got:
            return got
    return None


def _producer_path(rec) -> Path | None:
    for cand in (rec.get("script_path"), f"scripts/{rec.get('script')}", rec.get("script")):
        if cand and (ROOT / cand).exists():
            return ROOT / cand
    return None


def verify(artifact: str | Path) -> dict:
    """Has the code that wrote this artifact changed since it wrote it?"""
    path = Path(artifact)
    out = {"artifact": str(path.relative_to(ROOT) if path.is_absolute() else path)}
    try:
        obj = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        return {**out, "status": UNKNOWN, "detail": f"unreadable: {e}"}
    rec = _find_stamp(obj)
    if not rec:
        return {**out, "status": UNSTAMPED,
                "detail": "no provenance block: nothing records which code wrote this"}
    out["script"] = rec.get("script")
    producer = _producer_path(rec)
    if producer is None:
        return {**out, "status": UNKNOWN, "detail": "the named script is not in this checkout"}
    now = _digest(producer.read_bytes())

    recorded = rec.get("source_sha256")
    if recorded:
        out["how"] = "recorded source digest"
    elif rec.get("git_commit"):
        # retroactive: recover the producer as it stood at the commit the artifact names
        rel = str(producer.relative_to(ROOT))
        blob = _git("show", f"{rec['git_commit']}:{rel}")
        if blob is None:
            return {**out, "status": UNKNOWN,
                    "detail": f"commit {rec['git_commit'][:12]} does not carry {rel}"}
        recorded = _digest(blob.encode())
        out["how"] = f"source at {rec['git_commit'][:12]}"
        out["dirty_when_written"] = rec.get("git_dirty")
    else:
        return {**out, "status": UNSTAMPED, "detail": "no digest and no commit to recover one"}

    return _finish(out, recorded, now)

def _finish(out, recorded, now):
    out["status"] = CURRENT if recorded == now else CHANGED
    if out["status"] == CHANGED:
        out["detail"] = f"producer was {recorded[:12]}, is now {now[:12]}"
    return out


def infer(artifact: str | Path, producer: str | Path) -> dict:
    """A weaker check for an artifact written before stamping existed.

    The commit that ADDED the artifact is recovered from the log, and the producer as it stood
    at that commit is hashed. That is the producer at write time only if the script did not
    change between the run and the commit, which is an assumption about how the work was done
    and not a fact the artifact records. Reported as inferred, and never as `current' evidence
    of the same weight as a recorded digest.
    """
    path = Path(artifact)
    rel_a = str(path.relative_to(ROOT) if path.is_absolute() else path)
    rel_p = str(Path(producer).relative_to(ROOT) if Path(producer).is_absolute() else producer)
    out = {"artifact": rel_a, "script": Path(rel_p).name, "inferred": True}
    log = _git("log", "--diff-filter=A", "--format=%H", "--", rel_a)
    commits = [c for c in (log or "").split() if c]
    if not commits:
        return {**out, "status": UNKNOWN, "detail": "no commit adds this artifact"}
    intro = commits[-1]
    blob = _git("show", f"{intro}:{rel_p}")
    if blob is None:
        return {**out, "status": UNKNOWN,
                "detail": f"{rel_p} does not exist at {intro[:12]}"}
    if not (ROOT / rel_p).exists():
        return {**out, "status": UNKNOWN, "detail": f"{rel_p} is not in this checkout"}
    out["how"] = f"producer at {intro[:12]}, the commit that added the artifact"
    return _finish(out, _digest(blob.encode()), _digest((ROOT / rel_p).read_bytes()))


def read_checked(artifact: str | Path, *, require: bool = True):
    """Load a JSON artifact, refusing a number whose producer has moved under it."""
    v = verify(artifact)
    if require and v["status"] != CURRENT:
        raise RuntimeError(
            f"{v['artifact']}: {v['status']} -- {v.get('detail', '')}. "
            f"Regenerate it, or read it with require=False and say so.")
    return json.loads(Path(artifact).read_text())

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

import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CURRENT = "current"
COSMETIC = "cosmetic_only"
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


class _StripDocstrings(ast.NodeTransformer):
    """Remove docstrings, which are the one part of the AST that carries no behaviour."""

    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_Module = visit_FunctionDef = _strip
    visit_AsyncFunctionDef = visit_ClassDef = _strip


def _shape(src: str):
    """The parse tree with docstrings removed. Comments and formatting never enter it."""
    try:
        tree = _StripDocstrings().visit(ast.parse(src))
        return ast.dump(ast.fix_missing_locations(tree))
    except SyntaxError:
        return None


def semantic_equal(before: str, after: str) -> bool | None:
    """True when two versions differ only in comments, docstrings or layout.

    This is a proof and not a heuristic: what it compares is the tree the interpreter runs.
    A changed constant, a moved call, a renamed variable all survive into it. Returns None
    when either version does not parse, because then nothing has been established.
    """
    a, b = _shape(before), _shape(after)
    return None if a is None or b is None else a == b


def producer_diff(before: str, after: str, name: str = "producer", limit: int = 40) -> str:
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                      f"{name} (as recorded)", f"{name} (now)",
                                      lineterm="", n=2))
    return "\n".join(lines[:limit]) + ("\n  ... diff truncated" if len(lines) > limit else "")


# How far back to look for a recorded blob. A script here has a handful of revisions; the cap only
# stops a pathological file from making the sweep unbounded, and exhausting it is reported as a
# failure to recover rather than as an absence.
_MAX_HISTORY = 300


def _source_by_digest(rel: str, want: str) -> tuple:
    """The committed version of `rel` whose sha256 is `want`, anywhere in its history.

    The digest is a stronger key than the commit beside it. A stamp records the HEAD the producer
    was run against, which is the wrong commit whenever the producer was written and run before it
    was committed -- measured here at 62 of 271 stamped artifacts. The blob it actually ran is
    findable by content wherever it eventually landed.
    """
    revs = (_git("rev-list", "--all", "--", rel) or "").split()
    seen = set()
    for r in revs[:_MAX_HISTORY]:
        blob = _git("show", f"{r}:{rel}")
        if blob is None:
            continue
        d = _digest(blob.encode())
        if d in seen:
            continue
        seen.add(d)
        if d == want:
            return blob, r
    return None, None


def recorded_source(rec: dict, producer: Path) -> tuple:
    """(source at write time, how it was recovered) or (None, why not)."""
    commit = rec.get("git_commit")
    rel = str(producer.relative_to(ROOT))
    want = rec.get("source_sha256")
    blob = _git("show", f"{commit}:{rel}") if commit else None
    if blob is not None and (not want or _digest(blob.encode()) == want):
        return blob, f"source at {commit[:12]}"
    # Either the stamped commit does not carry the producer, or it carries a different version
    # because the tree was dirty. In both cases the recorded digest still names the exact source.
    if want:
        found, at = _source_by_digest(rel, want)
        if found is not None:
            return found, f"the recorded digest, found in history at {at[:12]}"
    if blob is not None:
        return blob, (f"the tree was dirty when written, the recorded digest is in no commit that "
                      f"touches {rel}, so the source at {commit[:12]} is the nearest committed "
                      f"version and not exactly what ran")
    if not commit:
        return None, "the artifact records no commit, so the old source cannot be recovered"
    return None, (f"{rel} is not in commit {commit[:12]}, and the digest it records matches no "
                  f"committed version of it")


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


def record_inputs(paths) -> list:
    """The files a producer read, by path and digest, for the artifact to carry.

    A stamp says which script wrote a file and a verify says whether that script has changed
    since. Neither can say what the script was pointed at. A perturbation test run against the
    real results directory once planted a pool, wrote a real artifact from it, and left the
    artifact in place: the stamp matched, the sweep passed, and the recorded curve held a budget
    that does not exist. An artifact that names its inputs can be checked against them, and one
    whose input has vanished says so instead of reading as current.
    """
    out = []
    for path in paths:
        path = Path(path)
        # The UNRESOLVED path first. resolve() follows symlinks, and in a git worktree the large
        # dataset under grail_metabolism/data/ is symlinked to the main checkout, so resolving a
        # dataset input lands outside ROOT and the row falls back to an absolute path naming one
        # machine. check_inputs reads rows as ROOT / row["path"], and pathlib discards the left
        # side of that join when the right is absolute -- so the round trip closes on the machine
        # that wrote it and reports "input gone" everywhere else. For a real file under ROOT the
        # two agree, so this changes nothing for the 64 artifacts that already record relative
        # paths; it only stops a symlinked input from recording a path no reader can resolve.
        try:
            rel = str(path.absolute().relative_to(ROOT))
        except ValueError:
            try:
                rel = str(path.resolve().relative_to(ROOT))
            except ValueError:
                rel = str(path)
        row = {"path": rel, "exists": path.exists()}
        if path.exists():
            row["sha256_16"] = _digest(path.read_bytes())[:16]
            row["bytes"] = path.stat().st_size
        out.append(row)
    return out


def check_inputs(rec: dict) -> list:
    """Every input an artifact names that has since vanished or moved, as readable lines."""
    problems = []
    for row in (rec.get("inputs") or []):
        path = ROOT / row["path"]
        if not path.exists():
            problems.append(f"input gone: {row['path']}")
            continue
        if "sha256_16" in row and _digest(path.read_bytes())[:16] != row["sha256_16"]:
            problems.append(f"input moved: {row['path']}")
    return problems


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

    return _finish(out, recorded, now, rec, producer)


def _finish(out, recorded, now, rec=None, producer=None):
    if recorded == now:
        out["status"] = CURRENT
        return out
    out["status"] = CHANGED
    out["detail"] = f"producer was {recorded[:12]}, is now {now[:12]}"
    if rec is None or producer is None:
        return out
    before, how = recorded_source(rec, producer)
    if before is None:
        out["diff_note"] = how
        return out
    after = producer.read_text()
    same = semantic_equal(before, after)
    out["diff_recovered_by"] = how
    out["diff"] = producer_diff(before, after, producer.name)
    if same is True:
        out["status"] = COSMETIC
        out["detail"] = ("the producer changed only in comments, docstrings or layout: the parse "
                         "tree with docstrings removed is identical, so the numbers cannot have "
                         "moved")
    elif same is None:
        out["detail"] += "; one of the two versions does not parse, so nothing is established"
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
    how = "the commit that added the artifact"
    if blob is None:
        # An artifact can predate the script that now produces it: the first version was written
        # by something else, or by hand. The artifact's CURRENT content was written at its last
        # modification, so that commit is the better place to look for the producer at write
        # time, and looking there is honest as long as the report says which was used.
        touched = _git("log", "--format=%H", "--", rel_a)
        for candidate in [c for c in (touched or "").split() if c]:
            blob = _git("show", f"{candidate}:{rel_p}")
            if blob is not None:
                intro, how = candidate, ("the last commit touching the artifact at which the "
                                         "producer existed")
                break
    if blob is None:
        return {**out, "status": UNKNOWN,
                "detail": f"{rel_p} does not exist at {intro[:12]}"}
    if not (ROOT / rel_p).exists():
        return {**out, "status": UNKNOWN, "detail": f"{rel_p} is not in this checkout"}
    out["how"] = f"producer at {intro[:12]}, {how}"
    return _finish(out, _digest(blob.encode()),
                   _digest((ROOT / rel_p).read_bytes()),
                   {"git_commit": intro}, ROOT / rel_p)


def read_checked(artifact: str | Path, *, require: bool = True):
    """Load a JSON artifact, refusing a number whose producer has moved under it."""
    v = verify(artifact)
    if require and v["status"] != CURRENT:
        raise RuntimeError(
            f"{v['artifact']}: {v['status']} -- {v.get('detail', '')}. "
            f"Regenerate it, or read it with require=False and say so.")
    return json.loads(Path(artifact).read_text())

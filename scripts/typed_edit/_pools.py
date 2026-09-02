#!/usr/bin/env python3
"""Refuse to read a pool that was not scored by the pair this repository releases.

Pools are built by one script and read by several. When the release moved from one training run
to another, the builders that named a checkpoint by hand went stale, and a reader merging shards
by `setdefault` cannot see that half of them came from a superseded model. That is what happened
to the standardised pools: the drawing sweep compared pools scored by one filter against pools
scored by another, and reported the difference as an effect of the drawing.

A pool built after this was written records the checkpoints it used, by path and by digest. This
checks them against the release, and refuses on a pool that records none rather than trusting it,
because a file that cannot say what scored it is exactly the file the defect lived in.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def released_pair() -> dict:
    """{stage: path} for the checkpoints this repository tracks, read from git."""
    try:
        tracked = subprocess.run(["git", "ls-files", "artifacts/"], cwd=ROOT,
                                 capture_output=True, text=True, timeout=30).stdout.splitlines()
    except Exception:
        tracked = []
    found: dict = {}
    for rel in tracked:
        parts = rel.split("/")
        if len(parts) >= 4 and parts[-2] == "checkpoints" and parts[-1].endswith(".pt"):
            stage = parts[-1][:-3]
            if stage in ("generator", "filter"):
                found.setdefault(stage, set()).add(rel)
    out = {}
    for stage, paths in found.items():
        if len(paths) == 1:
            out[stage] = next(iter(paths))
    return out


def _digest16(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def check(paths, *, require_record: bool = True) -> list:
    """Every reason the given pool files are not all on the released pair, as readable lines."""
    pair = released_pair()
    want = {stage: _digest16(ROOT / rel) for stage, rel in pair.items()}
    problems = []
    for path in paths:
        path = Path(path)
        try:
            blob = json.loads(path.read_text())
        except Exception as exc:
            problems.append(f"{path.name}: cannot be read ({exc})")
            continue
        record = blob.get("checkpoints")
        if not record:
            if require_record:
                problems.append(f"{path.name}: records no checkpoints, so what scored it is "
                                f"unknown and it cannot be shown to be the released pair")
            continue
        for stage, expected in want.items():
            got = (record.get(stage) or {}).get("sha256_16")
            if expected is not None and got != expected:
                problems.append(f"{path.name}: {stage} {got} is not the released "
                                f"{pair[stage]} ({expected})")
    return problems


def assert_released(paths, *, require_record: bool = True) -> None:
    """Raise with every reason, rather than the first, so one run names the whole problem."""
    problems = check(paths, require_record=require_record)
    if problems:
        raise SystemExit("these pools are not the released pair's:\n  " + "\n  ".join(problems))

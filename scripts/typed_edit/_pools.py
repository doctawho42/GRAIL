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


# One release can be tracked as two directories. scripts/build_released_checkpoint.py writes
# artifacts/full5000_released/checkpoints FROM artifacts/full5000_implicit/checkpoints: the
# generator with its per-rule rows subset to the released rule bank, the filter copied byte for
# byte. So the deploy's pair is a derivative of the measured pair and not a rival to it, and the
# pair every pool in this repository was scored by is the BASE.
#
# Declared here rather than imported from that script, which imports torch at module level and
# would put a torch import in every consumer of this module. A guard test asserts this mapping
# still matches that script's SRC and DST, so the declaration cannot drift in silence.
DERIVED_FROM = {"full5000_released": "full5000_implicit"}


def _tracked_checkpoints() -> dict:
    """{stage: {run: path}} for every generator and filter checkpoint git tracks."""
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
                found.setdefault(stage, {})[parts[1]] = rel
    return found


def derivative_runs() -> dict:
    """{run: base} for the declared derivatives this repository actually tracks.

    Named rather than merely dropped: a reader asking what the tree ships should be told that the
    deploy's pair is here and what it was made from, not shown a single pair and left to discover
    the other by listing files.
    """
    runs = {run for byrun in _tracked_checkpoints().values() for run in byrun}
    return {run: base for run, base in DERIVED_FROM.items()
            if run in runs and base in runs}


def _resolve(by_run: dict, stage: str) -> dict:
    """Drop a tracked derivative when its base is tracked too, proving what can be proved cheaply.

    For the filter the claim is byte-for-byte copying, so it is checked here: if the two files
    differ, the declaration is stale and resolving on it would hide a real ambiguity about what
    ships. For the generator the claim is a row subset, which cannot be established without
    loading both checkpoints; that half is carried by the deploy guard that scores the released
    generator against the full one on the kept rules, and this function does not pretend to it.
    """
    out = dict(by_run)
    for run, base in DERIVED_FROM.items():
        if run in out and base in out:
            if stage == "filter":
                a, b = _digest16(ROOT / out[run]), _digest16(ROOT / out[base])
                if a is None or b is None or a != b:
                    raise SystemExit(
                        f"{run}/{stage} is declared a copy of {base}/{stage} and is not: "
                        f"{a} against {b}. Either the derivation changed or the declaration in "
                        f"{Path(__file__).name} is stale; do not resolve the ambiguity on it.")
            out.pop(run)
    return out


def released_runs() -> dict:
    """{stage: run} for the run each stage's released checkpoint comes from.

    Ambiguity that survives resolution raises. It used to return nothing for the affected stage,
    which is worse than an error: `check` then compared every pool against an empty expectation
    and could not fail, so a pool scored by any model at all passed as released.
    """
    out = {}
    for stage, by_run in _tracked_checkpoints().items():
        resolved = _resolve(by_run, stage)
        if not resolved:
            continue
        if len(resolved) > 1:
            raise SystemExit(
                f"the repository tracks more than one {stage} checkpoint and they are not one "
                f"release: {sorted(resolved)}. Track one per stage, or declare the derivation.")
        out[stage] = next(iter(resolved))
    return out


def released_pair() -> dict:
    """{stage: path} for the checkpoint each stage ships, read from git."""
    runs = released_runs()
    tracked = _tracked_checkpoints()
    return {stage: tracked[stage][run] for stage, run in runs.items()}


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

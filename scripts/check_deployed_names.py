#!/usr/bin/env python3
"""A constant may not call a checkpoint deployed unless the repository ships it.

Eleven scripts defined DEPLOYED_GEN as the full5000_priors generator and nine defined
DEPLOYED_FILTER as the full5000_single filter. The released pair is full5000_implicit for both
stages. results/pool_checkpoints.json establishes that by fingerprinting each pool's own scores
against every trained checkpoint in the tree, and it says in its own words that it follows what
the repository tracks "rather than a name written beside it" -- which is what those constants
were. Reading them instead of the audit costs two hours of compute and produces a comparison of
the wrong model against the right comparators, with every gate passing, because nothing a gate
looked at was wrong.

This is the gate that was missing. It reads the released run per stage from the audit, finds every
constant whose name claims deployment or release, and refuses any whose path names a different run.

A constant that names its run honestly -- PRIORS_GEN, SINGLE_FILTER -- is not this check's
business, whatever it points at. The claim is what is checked, not the value.

    python scripts/check_deployed_names.py
    python scripts/check_deployed_names.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "pool_checkpoints.json"

# Every module-level constant, with the claim tested in Python rather than in the pattern. The
# first version spelled the claim into the regex as [A-Z][A-Z0-9_]*DEPLOYED, which requires a
# character before the word and so matched no constant actually called DEPLOYED_GEN. It would have
# passed the live scan by matching nothing, which is the failure this file exists to prevent.
CONSTANT = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.+)$", re.M)
CLAIM = re.compile(r"DEPLOYED|RELEASED")
# The run and the stage a checkpoint path names, written either as a string or as / operands.
RUN = re.compile(r"artifacts[\"'/\s]+[\"']?([A-Za-z0-9_]+)[\"']?")
STAGE = re.compile(r"(generator|filter)\.pt")


def released(audit_path: Path) -> dict:
    d = json.loads(audit_path.read_text())
    by_stage = d.get("released_runs_by_stage") or {}
    if not by_stage:
        raise SystemExit(f"{audit_path} carries no released_runs_by_stage; the gate has nothing "
                         f"to check against and refuses rather than passing vacuously")
    return by_stage


def _show(p: Path) -> str:
    """Repository-relative where that means anything, and the raw path in the self-test."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def offences(paths, by_stage) -> list:
    out = []
    for p in paths:
        text = p.read_text()
        for m in CONSTANT.finditer(text):
            name, value = m.group(1), m.group(2)
            if not CLAIM.search(name):
                continue
            stage = STAGE.search(value)
            run = RUN.search(value)
            if not stage or not run:
                continue
            want = by_stage.get(stage.group(1))
            if want and run.group(1) != want:
                line = text[: m.start()].count("\n") + 1
                out.append(f"{_show(p)}:{line}: {name} claims deployment but names "
                           f"{run.group(1)}; the released {stage.group(1)} is {want}")
    return out


def _self_test() -> int:
    """Both directions: a lying constant is caught, an honestly named one is not."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        audit = td / "audit.json"
        audit.write_text(json.dumps({"released_runs_by_stage":
                                     {"generator": "full5000_implicit",
                                      "filter": "full5000_implicit"}}))
        by_stage = released(audit)

        bad = td / "bad.py"
        bad.write_text('ROOT = "."\n'
                       'DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"\n')
        got = offences([bad], by_stage)
        assert len(got) == 1 and "full5000_priors" in got[0], got
        print("  a constant claiming deployment of a run that is not released -> CAUGHT")

        good = td / "good.py"
        good.write_text('ROOT = "."\n'
                        'PRIORS_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"\n')
        assert offences([good], by_stage) == [], offences([good], by_stage)
        print("  the same path under a name that does not claim it -> PASSES")

        right = td / "right.py"
        right.write_text('ROOT = "."\n'
                         'DEPLOYED_FILTER = ROOT / "artifacts" / "full5000_implicit" / "checkpoints" / "filter.pt"\n')
        assert offences([right], by_stage) == [], offences([right], by_stage)
        print("  a constant claiming deployment of the released run -> PASSES")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()

    by_stage = released(AUDIT)
    paths = sorted(set(ROOT.glob("scripts/*.py")) | set(ROOT.glob("scripts/**/*.py")))
    paths = [p for p in paths if p.name != Path(__file__).name]
    bad = offences(paths, by_stage)
    print(f"  released per stage: {by_stage}")
    print(f"  scanned {len(paths)} scripts")
    for b in bad:
        print(f"  {b}")
    if bad:
        print(f"\n{len(bad)} constant(s) claim a checkpoint the repository does not ship.")
        return 1
    print("  no constant claims a checkpoint the repository does not ship")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

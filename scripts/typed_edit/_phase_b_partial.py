#!/usr/bin/env python3
"""Run phase B over whatever phase-A shards have landed, as a dry run of the machinery.

This deliberately skips the merge gate, which is meant to fail on a partial set. The number
it produces is not the answer -- the answer is phase B over the full 98 -- but the control
arm and the monotonicity check are exercised on real pairs here rather than at the end.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from grail_metabolism.utils.preparation import load_default_rules  # noqa: E402
from known_type_recovery import phase_b  # noqa: E402

pairs, slices = [], []
for p in sorted(glob.glob(str(ROOT / "results/typed_edit_gapshards/a*.json"))):
    blob = json.loads(Path(p).read_text())["phase_a"]
    pairs.extend(blob["known_pairs"])
    slices.append(blob["slice"])
print(f"partial: {len(pairs)} known-type pairs from slices {sorted(slices)}", file=sys.stderr)

out = phase_b(load_default_rules(), pairs)
out["partial"] = True
out["slices"] = sorted(slices)
Path(ROOT / "results" / "typed_edit_recovery_partial.json").write_text(
    json.dumps(out, indent=1))
short = json.loads(json.dumps(out))
for a in short["arms"].values():
    a.pop("recovered_pairs", None)
print(json.dumps(short, indent=1))

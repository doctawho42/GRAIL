#!/usr/bin/env python3
"""What each comparator in the closed list can actually be run as, checked rather than assumed.

The registration closes the comparator list and says an unobtainable one is reported with its
reason. This re-derives the local half of that report: whether the artifact is on this machine,
whether its weights load, and whether the entry point the paper describes can be imported. The
remote half -- what a repository publishes, what a licence permits, what an author said -- is
recorded in `results/comparator_acquisition.json` with the URL and the date it was read,
because a claim about someone else's repository has to name where it was read.

    python scripts/typed_edit/check_comparators.py
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Where the comparator checkouts live. A sibling of the repository by default; set
# GRAIL_BASELINES to point elsewhere. It was an absolute path into one machine.
BASELINES = Path(os.environ.get("GRAIL_BASELINES", ROOT.parent / "GRAIL_baselines"))
RECORD = ROOT / "results" / "comparator_acquisition.json"


def check_deepmetab() -> dict:
    d = BASELINES / "DeepMetab"
    deps = BASELINES / "deepmetab-deps"
    out = {"checkout": str(d), "present": d.exists()}
    if not d.exists():
        return out
    ck = sorted(glob.glob(str(d / "Model" / "**" / "*.pt"), recursive=True))
    out["checkpoints"] = len(ck)
    out["checkpoints_are_lfs_pointers"] = sum(
        1 for p in ck if Path(p).stat().st_size < 400)
    # the module the paper's metabolite generation goes through
    out["generation_module_present"] = (d / "SOM" / "Reaction.py").exists()
    out["generation_bytecode_present"] = bool(
        glob.glob(str(d / "SOM" / "__pycache__" / "Reaction.*.pyc")))
    loaded, err = 0, None
    if ck and deps.exists():
        try:
            sys.path.insert(0, str(deps))
            import torch
            torch.load = functools.partial(torch.load, weights_only=False)
            from chemprop.utils import load_checkpoint
            for p in ck[:5]:
                load_checkpoint(p)
                loaded += 1
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:200]
    out["sampled_checkpoints_loaded"] = f"{loaded}/5"
    if err:
        out["load_error"] = err
    out["runnable_end_to_end"] = bool(
        out["generation_module_present"] and loaded == 5)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "comparator_local_state.json"))
    args = ap.parse_args()

    state = {"deepmetab": check_deepmetab()}
    if RECORD.exists():
        rec = json.loads(RECORD.read_text())
        state["remote_record"] = {"source": str(RECORD.relative_to(ROOT)),
                                  "read_on": rec.get("read_on"),
                                  "runnable": [k for k, v in rec["comparators"].items()
                                               if v.get("runnable_as_released")]}
    Path(args.out).write_text(json.dumps(state, indent=1))
    print(json.dumps(state, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

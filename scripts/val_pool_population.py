#!/usr/bin/env python3
"""The four scalars that describe the validation draw, split out of the pool they live in.

`results/val_pools.json` is a 46 MB candidate pool and is not tracked, for the reason
`scripts/check_tracked_sizes.py` exists: a pool of that size does not belong in a git history. The
manuscript takes four numbers from it -- the cap the draw was made under, the seed, the number of
substrates declared and the number actually paired -- and nothing else.

Reading a 46 MB untracked file for four scalars is what stopped the whole number chain from running
in a fresh clone. `paper2_numbers.build()` raised on the missing file, so six tests failed for a
reader who had done nothing wrong, and the provenance guarantee the manuscript quotes could not be
reproduced by anyone who was not us.

This writes those four numbers into a tracked artifact and records the pool as its input by digest,
so the split-out cannot drift: change the pool and this artifact fails verification rather than
going on describing the draw it used to describe.

    python scripts/val_pool_population.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import record_inputs, stamp  # noqa: E402

POOL = "results/val_pools.json"
OUT = ROOT / "results" / "val_pool_population.json"
# Exactly what the manuscript takes. Naming them here rather than copying the block wholesale keeps
# a 46 MB file's other contents from arriving in the tracked one by accident.
FIELDS = ("cap", "seed", "declared_n", "n")


def main() -> int:
    pool = ROOT / POOL
    if not pool.exists():
        print(f"REFUSING: {POOL} is not in this checkout, so the draw it describes cannot be "
              f"read. It is untracked by design; rebuild it with "
              f"scripts/typed_edit/build_val_pools.py or copy it from a checkout that has it.",
              file=sys.stderr)
        return 1

    blob = json.loads(pool.read_text())
    pop = blob.get("population") or {}
    missing = [f for f in FIELDS if f not in pop]
    if missing:
        print(f"REFUSING: {POOL} records no {', '.join(missing)} for its population, so the draw "
              f"is not described by the file it was made in.", file=sys.stderr)
        return 1

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([POOL]),
        "question": ("what the validation draw was: the cap it was made under, the seed, and how "
                     "many substrates it declared against how many were paired"),
        "split": blob.get("split"),
        "match": blob.get("match"),
        "population": {f: pop[f] for f in FIELDS},
        "why_this_is_separate": (
            "the pool it comes from is 46 MB and is not tracked, so a manuscript number taken "
            "straight out of it could not be read in a clone and the number chain would not run "
            "there at all"),
        "reading": ("the draw is a sample and not the whole validation split; a figure quoted on "
                    "it that names neither the cap nor the seed is not checkable"),
    }
    OUT.write_text(json.dumps(report, indent=1))
    p = report["population"]
    print(f"  validation draw: cap {p['cap']}, seed {p['seed']}, "
          f"{p['n']} paired of {p['declared_n']} declared")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

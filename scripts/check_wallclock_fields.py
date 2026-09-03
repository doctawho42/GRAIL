#!/usr/bin/env python3
"""An artifact whose numbers depend on the clock says which ones, and the paper agrees with it.

The SyGMa scenario sweep applies a ninety-second wall-clock deadline per substrate, and what does
not finish keeps the shallower scenario's list. So how many substrates finish is a property of how
loaded the machine was, and two numbers the manuscript prints move with it: the count of unfinished
substrates and the mean emission, which rises when a substrate that used to time out contributes
its longer list instead.

Re-running the producer moved exactly three fields -- mean_emitted, unfinished, seconds -- and left
every recall, every gap and every bootstrap interval bit-identical. That is the distinction worth
keeping: a sweep with a timeout is not non-reproducible, it is reproducible in the quantities the
argument rests on and load-dependent in three that describe the run. Without the distinction the
next re-run looks like drift and gets either explained away or chased.

So the producer names those three fields in the artifact, and this holds the rest of the paper to
that naming:

  * the artifact declares the fields, and declares exactly the ones the supporting information says
    it does;
  * no OTHER numeric field of the swept row is treated as load-dependent, which is what stops the
    declaration from growing to cover a real change;
  * the manuscript may print a declared field -- it does, twice -- but the supporting information
    has to say that it is load-dependent where it does, which is checked by requiring the sentence
    that says so.

    python scripts/check_wallclock_fields.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# artifact -> (the fields its producer may legitimately move between runs, the sentence in the
# supporting information that has to disclose it). Both halves are named so neither can drift
# without the other: a field added to the artifact with no sentence fails, and a sentence removed
# while the field stays fails too.
DECLARED = {
    "results/sygma_scenario_sweep.json": (
        {"mean_emitted", "unfinished", "seconds"},
        r"That deadline is wall-clock, so this\s+count and the mean emission it produces depend "
        r"on how loaded the machine is",
    ),
}


def main() -> int:
    bad, checked = [], 0
    si = (ROOT / "paper2" / "si.tex").read_text()

    for name, (fields, sentence) in DECLARED.items():
        path = ROOT / name
        if not path.exists():
            bad.append(f"{name}: missing")
            continue
        blob = json.loads(path.read_text())
        rows = blob.get("by_scenario") or {}
        if not rows:
            bad.append(f"{name}: no swept rows to check")
            continue

        for label, row in rows.items():
            # A row read from a frozen prediction file never invokes the tool, so no deadline
            # applies and nothing in it moves. That is a property the row already records -- it
            # names its frozen source and carries no timing -- so it is read rather than assumed,
            # and such a row is required to declare an empty list rather than omit the key: an
            # absent key cannot be told from a key nobody thought about.
            frozen = str(row.get("source", "")).startswith("results/") and "seconds" not in row
            if frozen:
                # Determinism is verified from the row rather than taken on trust, and the two
                # halves of it are checked: it names a frozen prediction file as its source and it
                # records no elapsed time, so no deadline ran and nothing in it can move. A row
                # that started invoking the tool would lose both properties and fall through to
                # the branch below, which demands the declaration.
                if row.get("wall_clock_dependent") not in (None, []):
                    bad.append(f"{name} [{label}]: reads a frozen file yet declares "
                               f"{row['wall_clock_dependent']} as load-dependent")
                checked += 1
                continue
            if "wall_clock_dependent" not in row:
                bad.append(f"{name} [{label}]: ran the tool under a deadline and does not say "
                           f"which of its fields a re-run may move; expected {sorted(fields)}. "
                           f"Re-run the producer rather than writing the key by hand.")
                continue
            declared = set(row["wall_clock_dependent"])
            if declared != fields:
                bad.append(f"{name} [{label}]: declares {sorted(declared)}, expected "
                           f"{sorted(fields)}")
            missing = [f for f in declared if f not in row]
            if missing:
                bad.append(f"{name} [{label}]: declares {missing} which the row does not carry")
            checked += 1

        if not re.search(sentence, re.sub(r"[ \t]+", " ", si)):
            bad.append(f"{name}: the supporting information does not say that its figures are "
                       f"load-dependent, so a reader meets a number that moves with no warning")

    if bad:
        print("REFUSING:")
        for line in bad:
            print("   " + line)
        return 1
    print(f"{checked} swept rows declare exactly the fields a re-run may move, and the supporting "
          f"information says so")
    return 0


if __name__ == "__main__":
    sys.exit(main())

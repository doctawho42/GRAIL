#!/usr/bin/env python3
"""The comparison set, written out, so a reader does not have to hold the repository to see it.

Every comparator number in this work is measured on 291 substrates whose membership is derived at
run time from the intersection of four prediction files. That is the right way to derive it and
the wrong way to publish it: a referee reading the two documents cannot see which molecules they
are, and the population's provenance is already the weakest link in the comparison.

This writes the members out under the key everything here is scored by, so the Supporting
Information can name its own population.

    python scripts/typed_edit/comparison_set_members.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "comparison_set_members.json"))
    args = ap.parse_args()

    from bank_without_selection import _key as tautkey
    from vs_metatox import population

    subs, truth, _ = population()
    subs = sorted(s for s in subs if truth.get(s))
    members = {}
    for s in subs:
        key = tautkey(s)
        if key:
            members[key] = len(truth[s])

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([ROOT / "results/test_references.json"]),
        "key": "tautomer-aware InChIKey, the key every comparison here is scored under",
        "n_substrates": len(subs),
        "n_keys": len(members),
        "references_per_member": members,
        "note": ("membership is derived from the intersection of the four prediction files at "
                 "run time and written here rather than maintained by hand; a substrate whose "
                 "key cannot be computed would be missing from the list and the two counts "
                 "above would differ, which is why both are given"),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"{len(subs)} substrates, {len(members)} distinct keys")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

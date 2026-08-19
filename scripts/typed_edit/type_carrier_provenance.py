#!/usr/bin/env python3
"""For each bank type, who carries it and whether the relaxation can reach any carrier.

A known-type miss is a reference whose reaction type the bank holds and whose rule did not
fire. Loosening the convention-dependent primitives can only recover it if some rule
carrying that type carries such a primitive, so the ceiling on recovery is not the bank-wide
2.8% of mined rules the census reports: it is the share of TYPES with at least one carrier
the relaxation touches, which is larger whenever a type is carried by several rules or by a
curated one.

Types are keyed by `reaction_types.canonical_type`, the same signature `coverage_gap_types`
uses to decide that the bank holds a type. Keying them by the step-0 signature instead would
be a different partition and would answer a question the 98 are not defined by.

This is the unconditional mixture, over every type in the bank. The mixture conditional on
being a type the 98 need is what phase A dumps, and the two differ if a type's chance of
failing to fire correlates with how constrained its carriers are.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from grail_metabolism.model.reaction_types import canonical_type  # noqa: E402

from primitive_census import relaxed_reactant  # noqa: E402
from step0 import load_rules  # noqa: E402

BANK = ROOT / "grail_metabolism" / "resources" / "extended_smirks.txt"
CATALOG = ROOT / "results" / "mined_rule_catalog_v2.json"


def carrier_table(rules, counts):
    """type -> {carriers, curated, relaxable, relaxable_curated, relaxable_mined}."""
    table = defaultdict(lambda: {"carriers": 0, "curated": 0, "relaxable": 0,
                                 "relaxable_curated": 0, "relaxable_mined": 0})
    untyped = 0
    for smirks, _ in rules:
        t = canonical_type(smirks)
        if t is None:
            untyped += 1
            continue
        relaxable = relaxed_reactant(smirks, True, True) != smirks
        curated = smirks not in counts
        row = table[t]
        row["carriers"] += 1
        row["curated"] += curated
        row["relaxable"] += relaxable
        if relaxable:
            row["relaxable_curated" if curated else "relaxable_mined"] += 1
    return table, untyped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "typed_edit_type_carriers.json"))
    args = ap.parse_args()

    rules, load_stats = load_rules(str(BANK))
    counts = {k: int(v.get("count", 0)) for k, v in json.loads(CATALOG.read_text()).items()}
    table, untyped = carrier_table(rules, counts)

    n_types = len(table)
    any_relaxable = sum(1 for r in table.values() if r["relaxable"])
    any_curated = sum(1 for r in table.values() if r["curated"])
    only_mined = sum(1 for r in table.values() if r["curated"] == 0)
    only_mined_no_relax = sum(1 for r in table.values()
                              if r["curated"] == 0 and r["relaxable"] == 0)
    carriers = [r["carriers"] for r in table.values()]
    curated_share = [r["curated"] / r["carriers"] for r in table.values()]

    out = {
        "rule_bank": {"path": str(BANK.relative_to(ROOT)), **load_stats},
        "keyed_by": "grail_metabolism.model.reaction_types.canonical_type",
        "n_types": n_types,
        "n_rules_without_a_type": untyped,
        "carriers_per_type": {
            "mean": round(statistics.mean(carriers), 2),
            "median": statistics.median(carriers),
            "max": max(carriers),
            "share_with_one_carrier": round(sum(1 for c in carriers if c == 1) / n_types, 4),
        },
        "types_with_any_curated_carrier": any_curated,
        "share_types_with_any_curated_carrier": round(any_curated / n_types, 4),
        "types_carried_only_by_mined_rules": only_mined,
        "types_with_any_relaxable_carrier": any_relaxable,
        "share_types_with_any_relaxable_carrier": round(any_relaxable / n_types, 4),
        "types_only_mined_and_no_relaxable_carrier": only_mined_no_relax,
        "mean_curated_share_of_carriers": round(statistics.mean(curated_share), 4),
        "note": "unconditional over the bank; the mixture conditional on the 98 comes from "
                "the phase-A dump and can differ if failing to fire correlates with how "
                "constrained a carrier is",
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

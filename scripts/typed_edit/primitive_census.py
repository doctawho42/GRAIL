#!/usr/bin/env python3
"""Which rules the convention-dependent relaxation can reach at all, by provenance.

The relaxation ladder measures what loosening the hydrogen-count and connectivity primitives
costs in candidates. It says nothing about which rules those primitives are even written in,
and that decides what the loosening can recover: a rule carrying no such primitive is
untouched by it, so a reference reachable only through that rule stays out of reach whatever
the relaxation does.

The split by provenance is the point. The leaderboard paper reports that the bank's curated
fifth and its mined majority reverse places under the hydrogen convention. This is the same
division seen from the other side: the constructs whose meaning depends on that convention
are concentrated in the half that was written by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from relaxation_ladder import _strip  # noqa: E402
from step0 import load_rules  # noqa: E402

BANK = ROOT / "grail_metabolism" / "resources" / "extended_smirks.txt"
CATALOG = ROOT / "results" / "mined_rule_catalog_v2.json"


def relaxed_reactant(smirks: str, drop_h: bool, drop_deg: bool) -> str:
    parts = smirks.split(">")
    parts[0] = _strip(parts[0], drop_h, drop_deg)
    return ">".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "typed_edit_primitive_census.json"))
    args = ap.parse_args()

    rules, load_stats = load_rules(str(BANK))
    counts = {k: int(v.get("count", 0)) for k, v in json.loads(CATALOG.read_text()).items()}

    rows = {"mined": {"rules": 0, "touched": 0, "pairs": 0, "pairs_touched": 0},
            "curated": {"rules": 0, "touched": 0, "pairs": 0, "pairs_touched": 0}}
    n_h = n_deg = n_either = 0
    for smirks, _ in rules:
        h = relaxed_reactant(smirks, True, False) != smirks
        d = (relaxed_reactant(smirks, True, True)
             != relaxed_reactant(smirks, True, False))
        n_h += h
        n_deg += d
        n_either += (h or d)
        prov = "mined" if smirks in counts else "curated"
        rows[prov]["rules"] += 1
        rows[prov]["touched"] += (h or d)
        rows[prov]["pairs"] += counts.get(smirks, 0)
        rows[prov]["pairs_touched"] += counts.get(smirks, 0) if (h or d) else 0

    for prov, r in rows.items():
        r["share_touched"] = round(r["touched"] / max(r["rules"], 1), 4)
        r["share_pairs_touched"] = round(r["pairs_touched"] / max(r["pairs"], 1), 4)

    out = {
        "rule_bank": {"path": str(BANK.relative_to(ROOT)), **load_stats},
        "rules_with_a_hydrogen_count_primitive": n_h,
        "rules_with_a_connectivity_primitive": n_deg,
        "rules_the_relaxation_touches": n_either,
        "share_of_bank_touched": round(n_either / max(len(rules), 1), 4),
        "by_provenance": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

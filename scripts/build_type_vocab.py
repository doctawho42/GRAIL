#!/usr/bin/env python3
"""Build the coarse radius-0 reaction-type vocabulary + rule->type map.

Reads `results/mined_rule_catalog_v2.json` (5,856 mined SMIRKS, each with a `count` = train-pair
support), groups rules by their radius-0 `canonical_type` signature (see
`grail_metabolism/model/reaction_types.py`), keeps the types whose pooled support is >=
`min_pairs`, and writes `grail_metabolism/resources/coarse_type_vocab.json`:

    {"types": {type_id: {"signature": str, "n_rules": int, "n_pairs": int}},
     "rule_to_type": {smirks: type_id}}

Prints `K types cover P% of train pairs` for the kept (dense, type_id >= 0) vocabulary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.model.reaction_types import build_type_vocab

CATALOG = ROOT / "results" / "mined_rule_catalog_v2.json"
OUT = ROOT / "grail_metabolism" / "resources" / "coarse_type_vocab.json"
MIN_PAIRS = 5


def main() -> int:
    catalog = json.loads(CATALOG.read_text())
    type_id_to_sig, rule_to_type = build_type_vocab(catalog, min_pairs=MIN_PAIRS)

    total_pairs = sum(int(entry.get("count", 0)) for entry in catalog.values())
    covered_pairs = sum(t["n_pairs"] for t in type_id_to_sig.values())
    coverage_pct = 100.0 * covered_pairs / total_pairs if total_pairs else 0.0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"types": type_id_to_sig, "rule_to_type": rule_to_type}, indent=2))

    n_types = len(type_id_to_sig)
    print(f"{n_types} types cover {coverage_pct:.1f}% of train pairs")
    print(
        f"({covered_pairs}/{total_pairs} pairs; {len(catalog)} rules in catalog, "
        f"{sum(1 for t in rule_to_type.values() if t == -1)} mapped to the 'other' bucket)"
    )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

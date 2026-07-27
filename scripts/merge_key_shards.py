#!/usr/bin/env python3
"""Merge sharded key tables into one lookup per criterion, failing loudly on a missing shard."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "key_tables"


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "inchikey_tautomer"
    shards = sorted(D.glob(f"{mode}.shard*of*.json"))
    if not shards:
        raise SystemExit(f"ERROR: no shards for '{mode}'")
    total = int(shards[0].stem.split("of")[-1])
    if len(shards) != total:
        raise SystemExit(f"ERROR: {len(shards)} of {total} shards present for '{mode}'; "
                         "a shard died and merging would silently drop its structures.")
    table = {}
    for s in shards:
        table.update(json.loads(s.read_text()))
    out = D / f"{mode}.json"
    out.write_text(json.dumps(table))
    print(f"merged {len(shards)} shards -> {out} ({len(table):,} keys)", flush=True)
    for s in shards:
        s.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

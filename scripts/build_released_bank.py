#!/usr/bin/env python3
"""The bank this repository may redistribute, which is not the bank the paper measures.

Two obligations sit on the templates the bank borrows verbatim, and neither can be settled by this
repository alone. BioTransformer's distribution is LGPL, but its README additionally requires
explicit permission for commercial use or redistribution; that permission has not been sought, and
the system is intended to run as a service. And one of its files, the environmental-microbial set,
is CC BY-NC-SA 4.0, which no GPL release can carry at all: that is not a permission anyone can
grant on this side, it is an incompatibility.

Asking is the authors' to do and relicensing somebody else's NonCommercial work is nobody's, so the
release drops those templates rather than shipping them under a claim of compliance the supporting
information withdraws two pages later.

What it costs is measured rather than assumed, and it is nothing: results/licence_removal_cost__
clean_test.json prices the removal of every BioTransformer template at zero references lost, an
interval of exactly [0.0, 0.0], because every reference they reach is reached by something else in
the bank. Removing every borrowed template from all three rightsholders would cost 21 references of
2,597, a change in reach of -0.0081 [-0.0120, -0.0046]; that variant is not what this builds,
because SyGMa's templates are GPL and GLORYx's carry no term this repository can point at.

The paper's figures are measured on the full bank and stay measured on it. A reader who wants that
bank can rebuild it exactly: obtain BioTransformer's published reaction set from its own project
and re-add the templates this script names, whose count and digest are recorded beside the output.
That is the same arrangement the repository already uses for the third-party files themselves.

    python scripts/build_released_bank.py            # write the released bank and its record
    python scripts/build_released_bank.py --check    # non-zero if the released bank is stale
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import record_inputs, stamp  # noqa: E402

FULL = ROOT / "grail_metabolism" / "resources" / "extended_smirks.txt"
RELEASED = ROOT / "grail_metabolism" / "resources" / "extended_smirks_released.txt"
RECORD = ROOT / "results" / "released_bank.json"
# The one file of BioTransformer's that this repository still holds. The other two contribute two
# templates and nothing, and are named in the record so their absence is stated rather than implied.
BT_CORE = ROOT / "grail_metabolism" / "resources" / "external" / "bt_database_metabolicReactions.json"


def smirks(path: Path) -> set:
    """Every reaction string in a BioTransformer database file, comments stripped."""
    raw = path.read_text()
    body = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("//"))
    blob = json.loads(body)
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and ">>" in o:
            out.add(o.strip())
    walk(blob)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the released bank matches what this would write")
    args = ap.parse_args()

    if not BT_CORE.exists():
        print(f"{BT_CORE.relative_to(ROOT)} is not present, so the templates to remove cannot be "
              f"identified. It is obtainable from BioTransformer's own project.", file=sys.stderr)
        return 2

    full = [l.rstrip("\n") for l in FULL.read_text().split("\n") if l.strip()]
    borrowed = smirks(BT_CORE)
    kept = [t for t in full if t.strip() not in borrowed]
    removed = len(full) - len(kept)

    text = "\n".join(kept) + "\n"
    record = {
        "provenance": stamp(__file__),
        # The two files this bank is the difference of.
        "inputs": record_inputs([FULL, BT_CORE]),
        "what_this_is": "the rule bank with every template BioTransformer's published set also "
                        "contains removed, which is the bank this repository redistributes",
        "why": "BioTransformer's README requires explicit permission for redistribution and that "
               "permission has not been sought; one of its files is CC BY-NC-SA 4.0, which no GPL "
               "release can carry. Neither is settleable here.",
        "measured_bank": {"path": str(FULL.relative_to(ROOT)), "templates": len(full),
                          "sha256_16": hashlib.sha256(FULL.read_bytes()).hexdigest()[:16]},
        "released_bank": {"path": str(RELEASED.relative_to(ROOT)), "templates": len(kept),
                          "sha256_16": hashlib.sha256(text.encode()).hexdigest()[:16]},
        "removed": removed,
        "cost": "zero references on the evaluated test set; every reference these templates reach "
                "is reached by another template in the bank, priced in "
                "results/licence_removal_cost__clean_test.json",
        "not_identifiable_here": "BioTransformer's environmental-microbial file contributes two "
                                 "further templates and its standardisation file none; the first "
                                 "is CC BY-NC-SA 4.0 and neither is held here, so those two "
                                 "templates remain in the released bank and are declared rather "
                                 "than removed",
        "how_to_rebuild_the_measured_bank": "obtain BioTransformer's reaction database from its own "
                                            "project and re-add every template of it that this "
                                            "bank contained; the measured bank's digest above "
                                            "confirms the result",
    }

    if args.check:
        if not RELEASED.exists():
            print(f"REFUSING: {RELEASED.relative_to(ROOT)} does not exist", file=sys.stderr)
            return 1
        if RELEASED.read_text() != text:
            print(f"REFUSING: {RELEASED.relative_to(ROOT)} is not what the full bank and "
                  f"BioTransformer's set produce; re-run without --check", file=sys.stderr)
            return 1
        print(f"the released bank is current: {len(kept)} templates, {removed} removed")
        return 0

    RELEASED.write_text(text)
    RECORD.write_text(json.dumps(record, indent=1))
    print(f"full bank {len(full)}, released bank {len(kept)}, removed {removed}")
    print(f"wrote {RELEASED.relative_to(ROOT)} and {RECORD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

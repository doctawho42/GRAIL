"""What the measured bank is made of, counted once so the numbers do not need the bank itself.

The manuscript reports the bank's composition: how many templates it holds, how many parse, how
many are mined and how many curated, and how many of the curated half come from each of the three
collections that ship with the code. Those counts are about
`grail_metabolism/resources/extended_smirks.txt`, which is the MEASURED bank and is not what the
release ships.

Counting them inside the number generator meant the generator opened a file no reader has, so the
whole chain from artifacts to macros raised in a fresh clone and five tests failed for someone who
had done nothing wrong. The counts are about a withheld file and cannot stop being; what they can
stop doing is requiring that file at the moment a number is printed.

So they are counted here, into a tracked artifact that records the bank and every collection it
compared against by digest. A reader gets the numbers; a change to the bank makes this artifact
fail verification rather than letting a stale composition go on being printed.

    python scripts/bank_composition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import record_inputs, stamp  # noqa: E402

BANK = "grail_metabolism/resources/extended_smirks.txt"
MINED = "grail_metabolism/resources/mined_only_v2.txt"
# The three curated collections that ship with the code. The rest of the curated half comes from a
# fourth whose file is not in the repository, which is why `unnamed` is reported rather than
# attributed.
CURATED = {
    "hydroxylation": "grail_metabolism/data/smirks.txt",
    "merged": "grail_metabolism/data/merged_smirks.txt",
    "notebooks": "grail_metabolism/resources/notebooks_rules.txt",
}


def _rules(rel: str) -> set:
    return {ln.strip() for ln in (ROOT / rel).read_text().splitlines() if ln.strip()}


def main() -> int:
    absent = [p for p in [BANK, MINED, *CURATED.values()] if not (ROOT / p).exists()]
    if absent:
        print(f"REFUSING: this counts the measured bank's composition and cannot see "
              f"{', '.join(absent)}. The measured bank is not redistributed; rebuild it with "
              f"scripts/build_released_bank.py from BioTransformer's own reaction set, or run "
              f"this in a checkout that has it.", file=sys.stderr)
        return 1

    bank = [ln for ln in (ROOT / BANK).read_text().splitlines() if ln.strip()]
    bankset, minedset = set(bank), _rules(MINED)

    from rdkit import RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    def _parses(smirks: str) -> bool:
        # ReactionFromSmarts raises on a malformed template rather than returning None, so the
        # count has to catch as well as test; one template in the bank does exactly this.
        try:
            return AllChem.ReactionFromSmarts(smirks.strip()) is not None
        except Exception:
            return False

    named, by_collection = set(), {}
    for tag, rel in CURATED.items():
        r = _rules(rel) & bankset
        by_collection[tag] = len(r)
        named |= r

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([BANK, MINED, *CURATED.values()]),
        "question": ("what the measured bank is made of, counted where the bank is rather than "
                     "where the numbers are printed"),
        "rules": len(bank),
        "parses": sum(1 for r in bank if _parses(r)),
        "curated_total": len(bankset - minedset),
        "curated_named": len(named),
        "curated_unnamed": len(bankset - minedset - named),
        "curated_by_collection": by_collection,
        "why_this_is_separate": (
            "the bank is not redistributed, so a number generator that opened it could not run in "
            "a clone; these counts are about a withheld file and are traceable to it by digest"),
    }
    out = ROOT / "results" / "bank_composition.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"  {report['rules']} templates, {report['parses']} parse; "
          f"{report['curated_total']} curated of which {report['curated_named']} are named")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

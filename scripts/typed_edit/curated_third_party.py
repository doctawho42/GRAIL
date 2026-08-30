#!/usr/bin/env python3
"""How much of the curated half of the bank is somebody else's rules, verbatim.

The bank's curated portion is three collections that ship with this code, and the manuscript has
carried an open marker where their published sources and licences should be. The question is not
only bibliographic. Two of the comparators in this paper distribute rule sets under terms of their
own: SyGMa's are GPL, and BioTransformer's require permission before redistribution. A bank that
contains either verbatim inherits the term, and a repository released without knowing which it
contains has not made a licensing decision but skipped one.

String equality is the right instrument and the only honest one. A rule that has been rewritten is
a different object whose provenance cannot be established this way, so what follows is a lower
bound on borrowing and never an upper one. Every count here is of templates present character for
character in both places, measured against the deployed bank rather than the collection files, so
what is reported is what ships.

    python scripts/typed_edit/curated_third_party.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

BANK = ROOT / "grail_metabolism/resources/extended_smirks.txt"
MINED = ROOT / "grail_metabolism/resources/mined_only.txt"
COLLECTIONS = {
    "hydroxylation": "grail_metabolism/data/smirks.txt",
    "merged": "grail_metabolism/data/merged_smirks.txt",
    "notebooks": "grail_metabolism/resources/notebooks_rules.txt",
}
BIOTRANSFORMER = "artifacts/tier2/biotransformer/database/metabolicReactions.json"


def rules_of(path: Path) -> set:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(errors="replace").splitlines()
            if line.strip() and ">>" in line and not line.strip().startswith("//")}


def biotransformer_rules() -> set:
    """Every reaction SMIRKS the shipped reaction database quotes, however it is nested."""
    path = ROOT / BIOTRANSFORMER
    if not path.exists():
        return set()
    return set(re.findall(r'"([^"\n]*>>[^"\n]*)"', path.read_text(errors="replace")))


def sygma_rules() -> set:
    """SyGMa's published rules, taken from the installed package's own files."""
    try:
        import sygma
    except Exception:
        return set()
    base = Path(os.path.dirname(sygma.__file__))
    found = set()
    for path in base.rglob("*"):
        if path.is_file() and path.suffix in (".txt", ".json", ".py", ".dat"):
            try:
                found |= set(re.findall(r"([^\s\"']*>>[^\s\"']*)", path.read_text(errors="replace")))
            except Exception:
                continue
    return {r for r in found if ">>" in r}


def main() -> int:
    bank = rules_of(BANK)
    mined = rules_of(MINED)
    curated = bank - mined
    sources = {"SyGMa": sygma_rules(), "BioTransformer": biotransformer_rules()}

    collections = {}
    for name, rel in COLLECTIONS.items():
        rules = rules_of(ROOT / rel)
        collections[name] = {"file": rel, "n": len(rules),
                             "in_the_deployed_bank": len(rules & bank)}
        for src, pool in sources.items():
            collections[name][f"verbatim_in_{src}"] = len(rules & pool)
            collections[name][f"verbatim_in_{src}_and_shipped"] = len(rules & pool & bank)

    shipped = {}
    for src, pool in sources.items():
        hits = curated & pool
        shipped[src] = {
            "third_party_rules_read": len(pool),
            "of_them_in_the_bank": len(hits),
            "share_of_the_curated_half": round(len(hits) / max(len(curated), 1), 4),
            "share_of_the_whole_bank": round(len(hits) / max(len(bank), 1), 4),
            "also_among_the_mined_templates": len(mined & pool),
        }
    union = curated & (sources["SyGMa"] | sources["BioTransformer"])

    report = {
        "provenance": stamp(__file__),
        "instrument": ("verbatim string equality against the deployed bank; a rewritten rule "
                       "cannot be traced this way, so every count is a lower bound"),
        "bank": {"templates": len(bank), "mined": len(mined), "curated": len(curated)},
        "third_party_in_the_curated_half": shipped,
        "curated_templates_traceable_to_either_source": len(union),
        "share_of_the_curated_half_so_traceable": round(len(union) / max(len(curated), 1), 4),
        "collections": collections,
        "terms": {
            "SyGMa": "GPL, per the distributed package",
            "BioTransformer": ("distributed with BioTransformer, whose terms require permission "
                               "for redistribution"),
        },
        "reading": (
            "The curated half is not original to this work in the part measured here. What that "
            "obliges is a licensing decision and not a correction: the templates are somebody "
            "else's, the terms differ between the two sources, and both must be satisfied by "
            "whatever licence the repository carries. Removing them is measurable in the same "
            "instrument that found them, so the cost of that option can be computed rather than "
            "guessed."),
    }
    (ROOT / "results/curated_third_party.json").write_text(json.dumps(report, indent=1))
    print(f"bank {len(bank)} templates: {len(mined)} mined, {len(curated)} curated")
    for src, row in shipped.items():
        print(f"  {src:16s} {row['of_them_in_the_bank']:4d} of its {row['third_party_rules_read']:4d} "
              f"rules ship here verbatim, {row['share_of_the_curated_half']:.1%} of the curated half")
    print(f"  together {len(union)} curated templates, "
          f"{report['share_of_the_curated_half_so_traceable']:.1%} of that half, are traceable")
    for name, row in collections.items():
        print(f"  {name:14s} n={row['n']:5d}  SyGMa {row['verbatim_in_SyGMa']:4d}  "
              f"BioTransformer {row['verbatim_in_BioTransformer']:4d}")
    print("wrote results/curated_third_party.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

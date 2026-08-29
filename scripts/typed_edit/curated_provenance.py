#!/usr/bin/env python3
"""Where the 492 unattributed templates come from, established rather than left open.

The bank is described as 1,725 curated templates and 5,856 mined ones, and 492 of the curated
half are carried as unattributed because no file in the repository was known to hold them. That
is not a resting position for redistributed rules, and it turns out not to be necessary: the 492
identify themselves three separate ways, and all three say the same thing.

  membership   477 of the 492 are verbatim lines of grail_metabolism/data/xtracted.txt, a file
               whose one and only commit says "added extracted reaction rules"
  syntax       the 492 are written in element notation with multi-component reactant groups,
               like xtracted.txt and unlike every hand-written collection in the bank
  dialect      the 492 carry imidic-drawn amides at the rate the mined half does and at fifty
               times the rate the named expert collections do, which is the fingerprint of a
               template extracted from this corpus rather than written by a chemist

The conclusion is that the curated half is not all curated. Two machine extractions from the same
annotated pairs are in the bank: the deployed mined half, written in atomic-number notation, and
an earlier extraction in element notation that was filed with the expert rules.

    python scripts/typed_edit/curated_provenance.py
"""
from __future__ import annotations

import argparse
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

NAMED = {
    "hydroxylation": "grail_metabolism/data/smirks.txt",
    "merged": "grail_metabolism/data/merged_smirks.txt",
    "notebooks": "grail_metabolism/resources/notebooks_rules.txt",
}
EXTRACTED = "grail_metabolism/data/xtracted.txt"


def rules(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def sygma_rules() -> tuple[set[str], str | None]:
    """SyGMa's published set, read from the installed package."""
    try:
        import sygma  # noqa: F401
        base = Path(sygma.__file__).parent / "rules"
    except Exception:
        env = os.environ.get("SYGMA_RULES")
        if not env:
            return set(), None
        base = Path(env)
    if not base.exists():
        return set(), None
    out = set()
    for name in ("phase1.txt", "phase2.txt"):
        for line in (base / name).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(line.split("\t")[0].strip())
    return out, str(base)


def syntax(collection: set[str]) -> dict:
    """The notation a collection is written in, which separates hand-written from extracted."""
    n = max(len(collection), 1)
    atomic = sum(1 for s in collection if "[#" in s)
    element = sum(1 for s in collection
                  if "[#" not in s and re.search(r"\[[A-Za-z]{1,2}[;:]", s))
    multicomponent = sum(1 for s in collection if s.startswith("("))
    heavily_mapped = sum(1 for s in collection
                         if len(re.findall(r":(\d+)\]", s.split(">>")[0])) >= 6)
    return {"n": len(collection),
            "atomic_number_notation": round(atomic / n, 4),
            "element_notation": round(element / n, 4),
            "multi_component_reactants": round(multicomponent / n, 4),
            "six_or_more_mapped_reactant_atoms": round(heavily_mapped / n, 4)}


def dialect(collection: set[str]) -> dict:
    """The imidic fingerprint, using the same instrument as the dialect census."""
    sys.path.insert(0, str(HERE))
    from dialect_census import _reactant_motifs
    parsed = amide = imidic = lactim = 0
    for smirks in collection:
        motifs = _reactant_motifs(smirks)
        if motifs is None:
            continue
        parsed += 1
        amide += motifs["amide"]
        imidic += motifs["imidic"]
        lactim += motifs["aromatic_lactim"]
    return {"parsed": parsed, "amide_requiring": amide, "imidic_requiring": imidic,
            "aromatic_lactim_requiring": lactim,
            "imidic_share": round(imidic / max(parsed, 1), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "curated_provenance.json"))
    args = ap.parse_args()

    bank = rules(ROOT / "grail_metabolism/resources/extended_smirks.txt")
    mined = rules(ROOT / "grail_metabolism/resources/mined_only_v2.txt")
    curated = bank - mined
    named, per_collection = set(), {}
    for tag, rel in NAMED.items():
        hit = rules(ROOT / rel) & bank
        per_collection[tag] = len(hit)
        named |= hit
    unnamed = curated - named
    extracted = rules(ROOT / EXTRACTED)
    sygma, sygma_path = sygma_rules()

    collections = {
        "curated_named": named,
        "curated_unattributed": unnamed,
        "mined": mined,
        "xtracted_file": extracted,
    }
    if sygma:
        collections["sygma_published"] = sygma

    report = {
        "provenance": stamp(__file__),
        "bank": {"templates": len(bank), "mined": len(mined), "curated": len(curated),
                 "named": len(named), "unattributed": len(unnamed),
                 "named_by_collection": per_collection},
        "identification": {
            "unattributed_verbatim_in_xtracted": len(unnamed & extracted),
            "unattributed_total": len(unnamed),
            "xtracted_total": len(extracted),
            "xtracted_inside_bank": len(extracted & bank),
            "xtracted_in_mined_half": len(extracted & mined),
            "xtracted_in_named_half": len(extracted & named),
            "xtracted_commit": "f588cff  added extracted reaction rules",
        },
        "syntax": {name: syntax(items) for name, items in collections.items()},
        "dialect": {name: dialect(items) for name, items in collections.items()},
    }
    if sygma:
        report["sygma_containment"] = {
            "rules_read_from": sygma_path,
            "sygma_rules": len(sygma),
            "verbatim_in_bank": len(sygma & bank),
            "share_of_sygma_inside": round(len(sygma & bank) / max(len(sygma), 1), 4),
            "in_curated_half": len(sygma & curated),
            "in_mined_half": len(sygma & mined),
            "in_named_collections": len(sygma & named),
            "in_unattributed": len(sygma & unnamed),
            "share_of_curated_half_that_is_sygma": round(
                len(sygma & curated) / max(len(curated), 1), 4),
            "by_named_collection": {tag: len(sygma & (rules(ROOT / rel) & bank))
                                    for tag, rel in NAMED.items()},
            "test": "verbatim string membership in extended_smirks.txt",
        }

    report["reading"] = (
        "the unattributed 492 are an earlier machine extraction from the same annotated pairs, "
        "not an expert collection: they sit verbatim in a file committed as extracted rules, "
        "they are written in that file's notation and in no hand-written collection's, and they "
        "carry the corpus's imidic drawing at the mined half's rate. The bank therefore holds "
        "two extractions and one curated body, and the curated body is 1,233 templates rather "
        "than 1,725. SyGMa's containment is a separate fact and does not touch the 492: every "
        "one of the contained SyGMa rules falls in the named collections.")

    Path(args.out).write_text(json.dumps(report, indent=1))
    b = report["bank"]
    print(f"bank {b['templates']}: mined {b['mined']}, curated {b['curated']} "
          f"(named {b['named']}, unattributed {b['unattributed']})")
    i = report["identification"]
    print(f"unattributed verbatim in {EXTRACTED}: {i['unattributed_verbatim_in_xtracted']} "
          f"of {i['unattributed_total']}")
    print(f"\n{'collection':<24}{'n':>6}{'[#N]':>8}{'[El]':>8}{'multi':>8}{'imidic':>9}")
    for name in collections:
        s, d = report["syntax"][name], report["dialect"][name]
        print(f"{name:<24}{s['n']:>6}{s['atomic_number_notation']:>8.2f}"
              f"{s['element_notation']:>8.2f}{s['multi_component_reactants']:>8.2f}"
              f"{d['imidic_share']:>9.3f}")
    if sygma:
        c = report["sygma_containment"]
        print(f"\nSyGMa: {c['verbatim_in_bank']} of {c['sygma_rules']} verbatim in the bank "
              f"({c['share_of_sygma_inside']:.1%}), {c['in_named_collections']} of them in the "
              f"named collections and {c['in_unattributed']} in the unattributed 492")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Whose rules this repository holds, and which of them it redistributes, counted two ways.

The bank's curated portion is described as three collections that ship with this code, and the
manuscript carried an open marker where their sources and licences should be. The question is not
bibliographic. Several tools in this field distribute rule sets under terms of their own, a bank
containing them verbatim inherits the term, and a repository released without knowing which it
contains has not made a licensing decision but skipped one.

There are two questions and an earlier version of this file answered only the first.

    templates   how much of the bank is somebody else's rules. Measured by string equality
                against every published rule set on disk, which is a lower bound: a template that
                was rewritten is a different string and cannot be traced this way.

    files       which third-party files this repository itself tracks and therefore redistributes
                verbatim. That is a separate obligation and a stricter one -- a file carries the
                terms of everything in it, including the parts the bank never uses.

The second question is the one that was missed. Answering only the first found two rightsholders;
the tracked files name four, and one of them carries a NonCommercial licence that a permissive or
GPL release could not absorb.

    python scripts/typed_edit/curated_third_party.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
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
EXTERNAL = ROOT / "grail_metabolism/resources/external"
COLLECTIONS = {
    "hydroxylation": "grail_metabolism/data/smirks.txt",
    "merged": "grail_metabolism/data/merged_smirks.txt",
    "notebooks": "grail_metabolism/resources/notebooks_rules.txt",
}

# What each source's own distribution says about reuse, quoted from a file in this repository
# where one exists. Where no licence text is held, that is recorded as the answer rather than
# filled in from elsewhere.
TERMS = {
    "BioTransformer core": (
        "LGPL, per artifacts/tier2/biotransformer/LICENSE.md; its prose says version 2.1 and the "
        "text appended to it is version 3, which the distribution does not reconcile. Copying and "
        "redistribution are granted with credit, a link to the licence, an indication of changes "
        "and the notices retained. Commercial use or redistribution needs the authors' permission."),
    "BioTransformer ENVMICRO": (
        "CC BY-NC-SA 4.0, per the header of the file itself and "
        "artifacts/tier2/biotransformer/database/ENVMICRO/LICENSE: EAWAG data licensed by "
        "EnviPath. NonCommercial and ShareAlike, so it cannot be absorbed into a GPL or permissive "
        "release."),
    "BioTransformer standardisation": "LGPL, as the core file.",
    "SyGMa": "GPL, unversioned: the installed distribution's metadata says only 'License: GPL' "
             "and ships no licence text, so which version governs is not settled on disk.",
    "GLORYx": "no licence text for it is held in this repository.",
    "RetroSim": "no licence text for it is held in this repository.",
}


def rules_of(path: Path) -> set:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(errors="replace").splitlines()
            if line.strip() and ">>" in line and not line.strip().startswith("//")}


def json_smirks(path: Path) -> set:
    """Every reaction SMIRKS a JSON file quotes, however it is nested."""
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


def gloryx_rules() -> dict:
    """{declared source: templates}. The GLORYx file attributes every rule in a column of its own."""
    path = EXTERNAL / "gloryx_reactionrules.csv"
    if not path.exists():
        return {}
    out: dict = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            smirks = (row.get("SMIRKS") or "").strip()
            if smirks:
                out.setdefault((row.get("Rule source") or "unattributed").strip(),
                               set()).add(smirks)
    return out


def digest(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked(rel: str) -> bool:
    try:
        out = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
        return out.returncode == 0
    except Exception:
        return False


def main() -> int:
    bank = rules_of(BANK)
    mined = rules_of(MINED)
    curated = bank - mined

    pools = {
        "BioTransformer core": json_smirks(EXTERNAL / "bt_database_metabolicReactions.json"),
        "BioTransformer ENVMICRO":
            json_smirks(EXTERNAL / "bt_database_ENVMICRO_metabolicReactions.json"),
        "BioTransformer standardisation":
            json_smirks(EXTERNAL / "bt_database_standardizationReactions.json"),
        "SyGMa": sygma_rules(),
    }
    retrosim = EXTERNAL / "retrosim_templates_general.json"
    if retrosim.exists():
        try:
            pools["RetroSim"] = set(json.loads(retrosim.read_text()))
        except Exception:
            pools["RetroSim"] = json_smirks(retrosim)
    for source, rules in gloryx_rules().items():
        pools[f"GLORYx file, attributed to {source}"] = rules

    by_pool = {}
    for name, pool in pools.items():
        hits = curated & pool
        by_pool[name] = {
            "rules_in_the_published_set": len(pool),
            "of_them_in_the_curated_half": len(hits),
            "also_among_the_mined_templates": len(mined & pool),
            "share_of_the_curated_half": round(len(hits) / max(len(curated), 1), 4),
        }
    union = set()
    for pool in pools.values():
        union |= curated & pool

    # Rightsholders rather than files: the GLORYx distribution attributes part of its own file to
    # SyGMa, so counting files would count one party's rules twice and miss that GLORY is a party.
    holders = {
        "BioTransformer": ["BioTransformer core", "BioTransformer ENVMICRO",
                           "BioTransformer standardisation"],
        "SyGMa": ["SyGMa", "GLORYx file, attributed to SyGMa"],
        "GLORYx": ["GLORYx file, attributed to GLORY",
                   "GLORYx file, attributed to Current work"],
        "RetroSim": ["RetroSim"],
    }
    by_holder = {}
    for holder, names in holders.items():
        hits = set()
        for name in names:
            hits |= curated & pools.get(name, set())
        by_holder[holder] = {"templates_in_the_curated_half": len(hits),
                             "terms": TERMS.get(holder, TERMS.get(names[0], "not recorded"))}

    # The second question. A file this repository tracks is redistributed by it, whether or not the
    # bank uses what is inside.
    files = {}
    for path in sorted(EXTERNAL.glob("*")) if EXTERNAL.exists() else []:
        rel = str(path.relative_to(ROOT))
        files[rel] = {
            "bytes": path.stat().st_size,
            "tracked_by_git": tracked(rel),
            "sha256": digest(path),
            "identical_to_an_upstream_copy_in_this_checkout": None,
        }
    # Name the upstream copies explicitly rather than guessing them from the file name.
    upstreams = {
        "grail_metabolism/resources/external/bt_database_metabolicReactions.json":
            "artifacts/tier2/biotransformer/database/metabolicReactions.json",
        "grail_metabolism/resources/external/bt_database_ENVMICRO_metabolicReactions.json":
            "artifacts/tier2/biotransformer/database/ENVMICRO/metabolicReactions.json",
        "grail_metabolism/resources/external/bt_database_standardizationReactions.json":
            "artifacts/tier2/biotransformer/database/standardizationReactions.json",
    }
    for rel, up in upstreams.items():
        if rel in files:
            files[rel]["identical_to_an_upstream_copy_in_this_checkout"] = (
                digest(ROOT / up) == files[rel]["sha256"])
            files[rel]["upstream"] = up

    # A tracked file whose templates the bank never uses is a redistribution obligation with no
    # scientific return, which is the cheapest thing in this report to act on.
    for rel, row in files.items():
        name = {v: k for k, v in {
            "BioTransformer core": "grail_metabolism/resources/external/bt_database_metabolicReactions.json",
            "BioTransformer ENVMICRO": "grail_metabolism/resources/external/bt_database_ENVMICRO_metabolicReactions.json",
            "BioTransformer standardisation": "grail_metabolism/resources/external/bt_database_standardizationReactions.json",
            "RetroSim": "grail_metabolism/resources/external/retrosim_templates_general.json",
        }.items()}.get(rel)
        if name and name in pools:
            row["templates_of_this_file_used_by_the_bank"] = len(curated & pools[name])
        if rel.endswith("gloryx_reactionrules.csv"):
            row["templates_of_this_file_used_by_the_bank"] = len(
                curated & set().union(*(v for k, v in pools.items() if k.startswith("GLORYx"))))

    collections = {}
    for name, rel in COLLECTIONS.items():
        rules = rules_of(ROOT / rel)
        collections[name] = {"file": rel, "n": len(rules),
                             "in_the_deployed_bank": len(rules & bank),
                             "traceable_to_a_published_set": len(rules & union)}

    report = {
        "provenance": stamp(__file__),
        "instrument": ("verbatim string equality against the deployed bank for templates, and "
                       "git tracking plus SHA-256 for files; a rewritten template cannot be traced, "
                       "so every template count is a lower bound"),
        "bank": {"templates": len(bank), "mined": len(mined), "curated": len(curated)},
        "by_published_set": by_pool,
        "by_rightsholder": by_holder,
        "curated_templates_traceable_to_a_published_set": len(union),
        "share_of_the_curated_half_so_traceable": round(len(union) / max(len(curated), 1), 4),
        "curated_templates_traceable_to_nothing_measured": len(curated) - len(union),
        "third_party_files_this_repository_tracks": files,
        "collections": collections,
        "reading": (
            "Two obligations, not one. The templates put the bank under the terms of four "
            "rightsholders; the tracked files put the repository under the terms of everything "
            "inside them, including a NonCommercial ShareAlike set whose templates the bank barely "
            "uses. A file that is redistributed and not used is the cheapest of these to resolve, "
            "and the report says which those are."),
    }
    (ROOT / "results/curated_third_party.json").write_text(json.dumps(report, indent=1))

    print(f"bank {len(bank)}: {len(mined)} mined, {len(curated)} curated\n")
    print("templates, by rightsholder:")
    for holder, row in by_holder.items():
        print(f"  {holder:16s} {row['templates_in_the_curated_half']:4d} in the curated half")
    print(f"  {'union':16s} {len(union):4d} = "
          f"{report['share_of_the_curated_half_so_traceable']:.1%} of the curated half; "
          f"{report['curated_templates_traceable_to_nothing_measured']} trace to nothing measured")
    print("\nfiles this repository redistributes:")
    for rel, row in files.items():
        used = row.get("templates_of_this_file_used_by_the_bank")
        print(f"  {'tracked ' if row['tracked_by_git'] else 'untracked'} "
              f"{Path(rel).name:44s} {row['bytes']:>8d} B  "
              f"bank uses {used if used is not None else '?'}")
    print("\nwrote results/curated_third_party.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

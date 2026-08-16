#!/usr/bin/env python3
r"""How often the criterion the paper prefers fails to reach a canonical form at all.

The tautomer-aware criterion is the one this paper recommends quoting a leaderboard under, so what
it does when it cannot finish is a property of the recommendation and not an aside. RDKit's
enumerator walks a transform graph and stops when it exhausts a budget; a molecule that stops that
way is left at whatever form the search reached, which is not canonical and need not agree between
two molecules that ought to match.

This counts that on the external reference set --- the population the external appendix reports ---
by enumerating each unique reference metabolite and reading the enumerator's own status. Nothing is
matched, scored or compared here: the question is only how many references the criterion can
canonicalise at all.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

GLORYX_JSON = ROOT / "docs" / "benchmark" / "data" / "gloryx_test.json"


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def _flatten(ms) -> list:
    out = []
    for m in ms or []:
        if isinstance(m, dict):
            if m.get("smiles"):
                out.append(m["smiles"])
            out += _flatten(m.get("metabolites"))
        elif isinstance(m, str):
            out.append(m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "tautomer_limit_audit.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")

    # the same escape fix and flattening the external ceiling's loader uses, so the population is
    # the population that appendix reports on
    raw = GLORYX_JSON.read_text()
    data = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw))
    refs = set()
    for parent in data:
        refs |= set(_flatten(parent.get("metabolites", [])))

    enumerator = rdMolStandardize.TautomerEnumerator()
    reached, at_limit, unparseable, stuck = 0, 0, 0, []
    for smiles in sorted(refs):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            unparseable += 1
            continue
        status = str(getattr(enumerator.Enumerate(mol), "status", ""))
        if "Max" in status:
            at_limit += 1
            stuck.append({"smiles": smiles, "status": status})
        else:
            reached += 1

    rep = {"config": {**_code_version(), "source": str(GLORYX_JSON.relative_to(ROOT)),
                      "rdkit_version": rdBase.rdkitVersion,
                      "note": "the enumerator's own status decides; a molecule whose enumeration "
                              "stops on a budget is left at the form the search reached"},
           "unique_reference_metabolites": len(refs),
           "reached_a_canonical_form": reached,
           "stopped_on_a_limit": at_limit,
           "unparseable": unparseable,
           "share_stopped_percent": round(100.0 * at_limit / max(len(refs), 1), 4),
           "the_ones_that_stopped": stuck}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"  {len(refs)} unique reference metabolites")
    print(f"  {reached} reach a canonical form, {at_limit} stop on a limit "
          f"({rep['share_stopped_percent']:.2f}%), {unparseable} do not parse")
    print(f"  rdkit {rep['config']['rdkit_version']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

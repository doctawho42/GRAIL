#!/usr/bin/env python3
"""When the system produces the right metabolite, is the site it names the site that changed?

The system's selling point is that every prediction carries the rule that produced it and the
atoms it fired on, and the paper has never tested the second half. A rule can reach the right
product from the wrong place: an ester hydrolysis template firing on an amide can give a structure
whose key matches by coincidence of formula and connectivity, and the prediction would be counted
correct while the mechanism it offers a chemist is wrong.

This measures that. For every generated candidate whose key matches an annotated reference, the
atoms the template fired on are compared with the atoms that actually differ between substrate and
reference.

What the reference centre is, exactly, matters and is not an annotation. The corpus records a
substrate and a product; it does not record a site. The centre used here is derived: the maximum
common substructure between the two is computed, the substrate atoms it does not cover are taken as
changed, and the substrate atoms bonded to a part of the product the substructure does not cover
are added, which is what catches a pure addition. That is an inference from structure and it is
reported as one. What this can therefore establish is agreement between the site the system names
and the structural change it claims to explain, not agreement with an experimentally determined
site of metabolism, for which this corpus carries no labels.

    python scripts/typed_edit/site_agreement.py --substrates 291
    python scripts/typed_edit/site_agreement.py --substrates 20 --out /tmp/probe.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402


def reference_centre(sub_mol, ref_mol):
    """Substrate atom indices the reference differs at, inferred from the common substructure.

    Returns None when no common substructure is found, which is a fragmentation the comparison
    cannot localise; those are counted separately rather than scored as a disagreement.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    result = rdFMCS.FindMCS([sub_mol, ref_mol], timeout=5,
                            atomCompare=rdFMCS.AtomCompare.CompareElements,
                            bondCompare=rdFMCS.BondCompare.CompareAny,
                            ringMatchesRingOnly=False, completeRingsOnly=False)
    if result.canceled or not result.smartsString:
        return None
    core = Chem.MolFromSmarts(result.smartsString)
    if core is None:
        return None
    sub_match = sub_mol.GetSubstructMatch(core)
    ref_match = ref_mol.GetSubstructMatch(core)
    if not sub_match or not ref_match:
        return None

    # Atoms of the substrate the common core does not cover: the part that was removed or altered.
    centre = set(range(sub_mol.GetNumAtoms())) - set(sub_match)
    # A pure addition leaves the substrate wholly covered. The attachment point is then the
    # substrate atom whose counterpart in the product carries a neighbour outside the core.
    to_sub = {r: s for s, r in zip(sub_match, ref_match)}
    outside = set(range(ref_mol.GetNumAtoms())) - set(ref_match)
    for idx in outside:
        for nbr in ref_mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetIdx() in to_sub:
                centre.add(to_sub[nbr.GetIdx()])
    # Neighbours of a removed atom are part of the change as a chemist would draw it.
    for idx in list(centre):
        if idx < sub_mol.GetNumAtoms():
            for nbr in sub_mol.GetAtomWithIdx(idx).GetNeighbors():
                centre.add(nbr.GetIdx())
    return centre


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", type=int, default=291)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--gen-ckpt",
                    default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "site_agreement.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    import torch  # noqa: F401

    from bank_without_selection import _key, _load
    from grail_metabolism.config import GeneratorConfig
    from grail_metabolism.workflows.factory import build_generator

    refs = {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_k30/all.json"))):
        refs.update(json.loads(Path(f).read_text())["references"])
    structures = json.loads((ROOT / "results/test_references.json").read_text())
    subs = sorted(s for s in refs if refs[s] and s in structures)[: args.substrates]

    generator = _load(Path(args.gen_ckpt),
                      lambda a, r: build_generator(GeneratorConfig(**a), r))

    tally = Counter()
    per_substrate, t0 = [], time.time()
    for i, s in enumerate(subs, 1):
        sub_mol = Chem.MolFromSmiles(s)
        if sub_mol is None:
            continue
        wanted = set(refs[s])
        ref_mols = {}
        for met in structures.get(s, []):
            mol = Chem.MolFromSmiles(met)
            if mol is None:
                continue
            try:
                ref_mols[_key(met)] = mol
            except Exception:
                continue
        try:
            det = generator.generate_scored_with_details(s, top_k=args.top_k, threshold=None,
                                                         compute_sites=True)
        except Exception:
            tally["substrate_failed"] += 1
            continue
        seen, rows = set(), []
        for smiles, _score, _rule, sites in det:
            key = _key(smiles)
            if not key or key in seen or key not in wanted:
                continue
            seen.add(key)
            ref_mol = ref_mols.get(key)
            if ref_mol is None:
                tally["reference_structure_absent"] += 1
                continue
            centre = reference_centre(sub_mol, ref_mol)
            tally["matched"] += 1
            if centre is None:
                tally["centre_not_derivable"] += 1
                continue
            fired = {int(a) for a in sites if 0 <= int(a) < sub_mol.GetNumAtoms()}
            if not fired:
                tally["no_site_reported"] += 1
                continue
            tally["scored"] += 1
            hit = bool(fired & centre)
            inside = fired <= centre
            tally["centre_hit"] += hit
            tally["fired_inside_centre"] += inside
            rows.append({"key": key, "fired": sorted(fired), "centre": sorted(centre),
                         "intersects": hit, "contained": inside})
        if rows:
            per_substrate.append({"substrate": s, "matches": rows})
        # after the substrate's own work, not before it, so the line reports what has been done
        if i % 25 == 0 or i == len(subs):
            print(f"  {i}/{len(subs)} ({time.time() - t0:.0f}s)  "
                  f"matched {tally['matched']} agree {tally['centre_hit']}", flush=True)

    scored = max(tally["scored"], 1)
    report = {
        "provenance": stamp(__file__),
        "population": {"substrates_attempted": len(subs), "rule_budget": args.top_k,
                       "note": "the generator's own candidates, before the filter and before any "
                               "budget, so this measures localisation and not ranking"},
        "reference_centre": (
            "inferred from the maximum common substructure between substrate and reference, not "
            "annotated; the corpus carries no site labels"),
        "counts": dict(tally),
        "share_of_scored_where_the_reported_site_touches_the_centre": round(
            tally["centre_hit"] / scored, 4),
        "share_of_scored_where_the_reported_site_lies_wholly_inside_it": round(
            tally["fired_inside_centre"] / scored, 4),
        "reading": (
            "A correct structure reached from the wrong atoms is a prediction whose explanation is "
            "wrong, and this is the rate at which that happens. The centre is inferred from "
            "structure, so this establishes agreement between the site named and the change it "
            "claims to explain, not agreement with an experimental site of metabolism."),
        "per_substrate": per_substrate,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nmatched {tally['matched']} references, scored {tally['scored']}")
    print(f"  site touches the inferred centre : "
          f"{report['share_of_scored_where_the_reported_site_touches_the_centre']:.4f}")
    print(f"  site lies wholly inside it       : "
          f"{report['share_of_scored_where_the_reported_site_lies_wholly_inside_it']:.4f}")
    for key in ("centre_not_derivable", "no_site_reported", "reference_structure_absent",
                "substrate_failed"):
        if tally[key]:
            print(f"  {key}: {tally[key]}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Free first stage of queue point 5: does the MCS correspondence differ from the template's map?

The merged substrate-product filter lost with cross-edges placed by MCS. The template that produced
a candidate carries the exact atom map, so the loss may have been the MCS correspondence rather than
the architecture. This measures how often the two disagree: apply a rule (RunReactants keeps each
product atom's react_atom_idx, the true substrate atom it came from), align the same pair by the MCS
route from_pair uses, and count the substrate atoms the two routes send to different product atoms.

If they rarely disagree, the exact map cannot rescue the pair filter and point 5 is closed without
training. If they often disagree, the earlier loss is plausibly the correspondence, and the exact
map is a different experiment from the one that lost.

    python scripts/typed_edit/mcs_vs_template_map.py --substrates 60 --rules 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, rdFMCS  # noqa: E402
RDLogger.DisableLog("rdApp.*")
import bank_without_selection as B  # noqa: E402
from _provenance import stamp  # noqa: E402


def mcs_correspondence(sub, prod):
    """{substrate_atom_idx: product_atom_idx} from the MCS route from_pair uses."""
    try:
        m = rdFMCS.FindMCS([sub, prod], timeout=5, matchValences=False,
                           ringMatchesRingOnly=True, completeRingsOnly=True,
                           bondCompare=rdFMCS.BondCompare.CompareAny,
                           atomCompare=rdFMCS.AtomCompare.CompareElements)
    except Exception:
        return None
    if m.canceled or m.numAtoms == 0:
        return None
    core = Chem.MolFromSmarts(m.smartsString)
    if core is None:
        return None
    ms = sub.GetSubstructMatch(core)
    mp = prod.GetSubstructMatch(core)
    if not ms or not mp or len(ms) != len(mp):
        return None
    return dict(zip(ms, mp))


def template_correspondence(prod):
    """{substrate_atom_idx: product_atom_idx} from RunReactants provenance."""
    out = {}
    for a in prod.GetAtoms():
        if a.HasProp("react_atom_idx"):
            out[int(a.GetProp("react_atom_idx"))] = a.GetIdx()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", type=int, default=60)
    ap.add_argument("--rules", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results" / "mcs_vs_template_map.json"))
    args = ap.parse_args()

    bank = [ln.split()[0] for ln in
            (ROOT / "grail_metabolism/resources/extended_smirks.txt").read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")][: args.rules]
    rxns = [(r, AllChem.ReactionFromSmarts(r)) for r in bank]
    rxns = [(r, x) for r, x in rxns if x is not None]

    # substrates from the frozen comparison pools, not load_dataset_bundle: the bundle rebuilds the
    # whole split's preprocessing (the expensive standardisation) even for a handful, which hangs.
    import glob as _glob
    wide = {}
    for f in sorted(_glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        wide.update(json.loads(Path(f).read_text())["pools"])
    subs = sorted(wide)[: args.substrates]
    print(f"{len(subs)} substrates x {len(rxns)} rules", flush=True)

    pairs = agree = disagree = no_mcs = 0
    atoms_total = atoms_diff = 0
    t0 = time.time()
    for i, s in enumerate(subs, 1):
        if i % 10 == 0:
            print(f"  {i}/{len(subs)} ({time.time()-t0:.0f}s) pairs={pairs} disagree={disagree}",
                  flush=True)
        m0 = Chem.MolFromSmiles(s)
        if m0 is None or m0.GetNumHeavyAtoms() > 32:   # the big substrates are the ones that blow up
            continue
        sub = Chem.AddHs(m0)
        for _, rxn in rxns:
            try:
                outs = rxn.RunReactants((sub,))
            except Exception:
                continue
            for pset in (outs[:2] if outs else []):
                prod = pset[0]
                try:
                    Chem.SanitizeMol(prod)
                except Exception:
                    continue
                tmpl = template_correspondence(prod)
                if not tmpl:
                    continue
                mcs = mcs_correspondence(sub, prod)
                if mcs is None:
                    no_mcs += 1
                    continue
                pairs += 1
                shared = set(tmpl) & set(mcs)
                if not shared:
                    continue
                d = sum(1 for a in shared if tmpl[a] != mcs[a])
                atoms_total += len(shared)
                atoms_diff += d
                if d:
                    disagree += 1
                else:
                    agree += 1

    rep = {"provenance": stamp(__file__),
           "substrates": len(subs), "rules": len(rxns),
           "pairs_compared": pairs, "no_mcs": no_mcs,
           "pairs_agree": agree, "pairs_disagree": disagree,
           "pair_disagreement_rate": round(disagree / pairs, 4) if pairs else None,
           "atoms_compared": atoms_total, "atoms_misaligned": atoms_diff,
           "atom_misalignment_rate": round(atoms_diff / atoms_total, 4) if atoms_total else None,
           "reading": (
               "the MCS correspondence and the template's exact map are compared on the substrate "
               "atoms both align. A high pair-disagreement rate means the exact map sends atoms "
               "somewhere the MCS does not, so the pair filter's cross-edges were on an inferred "
               "correspondence that was often wrong, and the exact-map re-run is a different "
               "experiment from the one that lost; a low rate means the MCS was already right and "
               "point 5 cannot be rescued by the exact map.")}
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\n  pairs {pairs}: agree {agree}, disagree {disagree} "
          f"({disagree/pairs:.1%})" if pairs else "  no pairs")
    print(f"  atoms {atoms_total}: misaligned {atoms_diff} "
          f"({atoms_diff/atoms_total:.1%})" if atoms_total else "  no atoms")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

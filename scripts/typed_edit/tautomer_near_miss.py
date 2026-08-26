"""References the pool contains and the key did not credit.

The matching key is a tautomer-canonical InChIKey: prediction and reference both go through
`standardize_mol`, whose canonicaliser is meant to send every tautomer of one molecule to one
representative. On enumerated tautomers it reaches that for 0.9652 of products at the shipped
budget. Enumerated tautomers include forms a rule engine would not emit, so that is an upper
bound on how often the shortfall costs anything.

This measures the cost directly. For every annotated reference no candidate matched, it asks
whether some candidate is the same molecule anyway. The test is two invariants a proton shift
cannot change and little else preserves together:

  skeleton   the heavy-atom connectivity with every bond order flattened and charges dropped
  formula    the molecular formula, hydrogens included

A tautomer pair agrees on both by construction. An alkene against its alkane shares the skeleton
and fails the formula; a constitutional isomer with a different connectivity fails the skeleton.

That screen is not sufficient, and reading its output is what showed it: of the first ten pairs
it produced, several were double-bond positional isomers -- two fatty acids whose unsaturation
sits at different carbons, a piperazine whose double bond had moved into a cyclopropyl ring.
Those share a formula and a flattened skeleton and are not tautomers of anything. So the screen
is only a screen, and each pair it proposes is then confirmed with the machinery the matching
itself uses: one molecule counts as a tautomer of the other only if it appears in the other's
enumerated tautomer set. That is exactly the statement "the canonicaliser should have merged
these two and did not".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402


def skeleton(smiles, Chem):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    rw = Chem.RWMol(m)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in rw.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetIsAromatic(False)
    try:
        return Chem.MolToSmiles(rw.GetMol(), isomericSmiles=False)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "results/val_pools.json"))
    ap.add_argument("--out", default=str(ROOT / "results/tautomer_near_miss.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize

    from build_val_pools import population

    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(200)

    def is_tautomer_of(a: str, b: str) -> bool:
        """Confirm with the enumerator, not with the screen that proposed the pair."""
        ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
        if ma is None or mb is None:
            return False
        target = Chem.MolToSmiles(mb)
        try:
            for t in enumerator.Enumerate(ma):
                if Chem.MolToSmiles(t) == target:
                    return True
        except Exception:
            return False
        return False

    d = json.loads(Path(args.pools).read_text())
    pools, ref_keys = d["pools"], d["references"]
    _, vmap = population()          # reference SMILES, which the pool artifact stores only as keys

    def fp(s):
        m = Chem.MolFromSmiles(s)
        sk = skeleton(s, Chem)
        return (sk, rdMolDescriptors.CalcMolFormula(m)) if (m and sk) else None

    subs = sorted(s for s in pools if ref_keys.get(s) and s in vmap)
    n_ref = n_unmatched = n_screened = n_recovered = n_rejected = 0
    found, rejected = [], []
    for i, s in enumerate(subs, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(subs)}", file=sys.stderr, flush=True)
        have = {c["key"] for c in pools[s]}
        cand_fp = {}
        for c in pools[s]:
            f = fp(c["smiles"])
            if f:
                cand_fp.setdefault(f, []).append(c["smiles"])
        from bank_without_selection import _key
        for r in vmap[s]:
            k = _key(r)
            if not k:
                continue
            n_ref += 1
            if k in have:
                continue
            n_unmatched += 1
            f = fp(r)
            if not (f and f in cand_fp):
                continue
            n_screened += 1
            hit = next((c for c in cand_fp[f] if is_tautomer_of(r, c)), None)
            if hit is None:
                n_rejected += 1
                if len(rejected) < 8:
                    rejected.append({"reference": r[:80], "candidate": cand_fp[f][0][:80]})
                continue
            n_recovered += 1
            if len(found) < 15:
                found.append({"substrate": s[:70], "reference": r[:80], "candidate": hit[:80]})

    rep = {"provenance": stamp(__file__), "split": "validation",
           "population": {"n": len(subs)},
           "references": n_ref,
           "references_the_key_did_not_match": n_unmatched,
           "passed_the_skeleton_and_formula_screen": n_screened,
           "rejected_by_the_enumerator": n_rejected,
           "of_those_present_in_the_pool_as_a_tautomer": n_recovered,
           "share_of_all_references": round(n_recovered / n_ref, 4) if n_ref else None,
           "share_of_unmatched": round(n_recovered / n_unmatched, 4) if n_unmatched else None,
           "screen": "same flattened heavy-atom skeleton and same molecular formula",
           "confirmation": "the candidate appears in the reference's enumerated tautomer set; "
                           "the screen alone admits double-bond positional isomers, which is "
                           "why the confirmation exists",
           "examples": found,
           "examples_the_screen_proposed_and_the_enumerator_rejected": rejected}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\n{len(subs)} validation substrates, {n_ref} references")
    print(f"  not matched by any candidate key      {n_unmatched}")
    print(f"  passed the skeleton/formula screen    {n_screened}")
    print(f"  of those, rejected by the enumerator  {n_rejected}")
    print(f"  confirmed present as a tautomer       {n_recovered}"
          f"  ({rep['share_of_all_references']:.2%} of all references,"
          f" {rep['share_of_unmatched']:.2%} of the unmatched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Are the mined rules one enzymatic edit, measured on molecules rather than on templates?

A first attempt read the SMIRKS directly and miscounted twice: a cleavage whose bridging atom is
unmapped is invisible to a comparison of mapped bonds, and a cyclisation joins two distant loci in
one event. Parsing arbitrary templates with partial atom mapping is the wrong instrument.

This uses the pair the rule was mined from. Every mined template comes from an annotated
substrate-product pair, so the edit can be measured on real molecules by the same
maximum-common-substructure route the paper's transformation types use. The reaction centre is
the set of substrate atoms outside the MCS or whose environment changes across it; its connected
components, counted in the substrate graph, are the loci the edit touches.

One locus is one edit. Two loci joined by a bond that appears or disappears are one event, which
is what a cyclisation or a cleavage looks like. Two loci joined by nothing are two independent
edits, and a rule that performs two independent edits in one application is not one enzymatic
step.

    python scripts/typed_edit/cascade_by_pairs.py --limit 800
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="results/mined_rule_catalog_v2.json")
    ap.add_argument("--limit", type=int, default=0, help="rules to score, 0 for all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/cascade_by_pairs.json"))
    args = ap.parse_args()

    import random
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from scripts.coverage_gap_types import pair_to_type  # noqa: F401  (shared MCS settings)
    from scripts.mine_rules import MCS_TIMEOUT_SECONDS, find_reaction_center
    from rdkit.Chem import rdFMCS

    catalog = json.loads((ROOT / args.catalog).read_text())
    keys = sorted(catalog)
    if args.limit:
        random.Random(args.seed).shuffle(keys)
        keys = keys[:args.limit]

    stats = defaultdict(int)
    examples = []
    for smk in keys:
        pair = (catalog[smk].get("source_pairs") or [None])[0]
        if not pair:
            stats["no_pair"] += 1
            continue
        sub, prod = Chem.MolFromSmiles(pair[0]), Chem.MolFromSmiles(pair[1])
        if sub is None or prod is None:
            stats["unparseable_pair"] += 1
            continue
        try:
            mcs = rdFMCS.FindMCS([sub, prod], timeout=MCS_TIMEOUT_SECONDS, matchValences=False,
                                 ringMatchesRingOnly=True, completeRingsOnly=True,
                                 bondCompare=rdFMCS.BondCompare.CompareAny,
                                 atomCompare=rdFMCS.AtomCompare.CompareElements)
            if mcs.canceled or mcs.numAtoms == 0:
                stats["mcs_failed"] += 1
                continue
            core = Chem.MolFromSmarts(mcs.smartsString)
            sm = sub.GetSubstructMatch(core)
            pm = prod.GetSubstructMatch(core)
            if not sm or not pm:
                stats["no_match"] += 1
                continue
            cs, _ = find_reaction_center(sub, prod, sm, pm)
        except Exception:
            stats["error"] += 1
            continue
        if not cs:
            stats["no_centre"] += 1
            continue

        # connected components of the centre inside the substrate
        centre = set(cs)
        seen, comps = set(), 0
        for a in centre:
            if a in seen:
                continue
            comps += 1
            stack = [a]
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                for nb in sub.GetAtomWithIdx(u).GetNeighbors():
                    if nb.GetIdx() in centre and nb.GetIdx() not in seen:
                        stack.append(nb.GetIdx())
        stats[f"loci_{min(comps, 4)}"] += 1
        stats["scored"] += 1
        stats["multi_locus"] += comps > 1
        d_heavy = prod.GetNumHeavyAtoms() - sub.GetNumHeavyAtoms()
        if comps > 1 and len(examples) < 20:
            examples.append({"loci": comps, "heavy_delta": d_heavy,
                             "substrate": pair[0][:90], "product": pair[1][:90],
                             "smirks": smk[:150]})

    rep = {"provenance": stamp(__file__),
           "n_rules_considered": len(keys), "limit": args.limit, "seed": args.seed,
           "criterion": ("the reaction centre of the pair the rule was mined from, by the mining "
                         "MCS route, split into connected components inside the substrate"),
           "counts": dict(stats),
           "multi_locus_share": round(stats["multi_locus"] / max(stats["scored"], 1), 4),
           "examples": examples,
           "caveat": ("the MCS runs under a wall-clock timeout, so the count is load-dependent "
                      "at the margin; and a rule is scored on one of its source pairs, not all"),
           "reading": ("a centre in one piece is one edit; two pieces are two loci, which is a "
                       "cascade only when no bond appears or disappears between them, and this "
                       "measurement does not separate those two cases")}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"scored {stats['scored']} of {len(keys)} mined rules")
    for k in sorted(stats):
        print(f"  {k:<18} {stats[k]}")
    print(f"\nmore than one locus: {stats['multi_locus']} ({rep['multi_locus_share']:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

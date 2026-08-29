#!/usr/bin/env python3
"""Are the bank's templates single edits, or do some compress a cascade into one rule?

The paper says predictions are single-step and reports a one-step coverage ceiling. Both
sentences mean "one application of one template". Whether that is also one enzymatic step depends
on the templates: the miner derives a rule from an annotated substrate-product pair, and if the
corpus annotates a metabolite two enzymatic steps from its parent -- a glucuronide of a
hydroxylated metabolite, say -- the rule it yields performs both edits at once.

The test is the connectivity of the reaction centre. A single enzymatic step changes one region:
a hydroxylation adds an oxygen at one site, a dealkylation cleaves one bond, a conjugation
attaches one group at one site. Two changed regions separated by unchanged atoms are two
independent edits.

A bond is CHANGED when its order differs between the mapped reactant and product, or when it
exists on one side only. The changed bonds are projected onto the reactant graph and their
connected components counted, joining components whose atoms lie within `--slack` bonds of one
another so that a single edit reported as two adjacent bond changes is not miscounted.

    python scripts/typed_edit/cascade_audit.py
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


def bonds_by_map(mol):
    """{(mapA, mapB): bond order} for every bond whose two atoms are both mapped."""
    out = {}
    for b in mol.GetBonds():
        a, z = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
        if a and z:
            out[tuple(sorted((a, z)))] = b.GetBondTypeAsDouble()
    return out


def _regions_map(changed_pairs, adjacency, slack):
    """atom -> region id, by the same union-find `components` counts with."""
    atoms = sorted({a for p in changed_pairs for a in p})
    parent = {a: a for a in atoms}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def near(x, y):
        seen, frontier = {x}, [x]
        for _ in range(slack + 1):
            nxt = []
            for u in frontier:
                for v in adjacency.get(u, ()):
                    if v == y:
                        return True
                    if v not in seen:
                        seen.add(v); nxt.append(v)
            frontier = nxt
        return False
    for p in changed_pairs:
        parent[find(p[0])] = find(p[1])
    for i, a in enumerate(atoms):
        for b in atoms[i + 1:]:
            if find(a) != find(b) and near(a, b):
                parent[find(a)] = find(b)
    return {a: find(a) for a in atoms}


def components(changed_pairs, adjacency, slack):
    """Connected components of the changed bonds, joined through up to `slack` unchanged bonds."""
    atoms = sorted({a for p in changed_pairs for a in p})
    if not atoms:
        return 0
    # distance between changed atoms through the reactant graph, capped at slack + 1
    def near(x, y):
        seen, frontier = {x}, [x]
        for _ in range(slack + 1):
            nxt = []
            for u in frontier:
                for v in adjacency.get(u, ()):
                    if v == y:
                        return True
                    if v not in seen:
                        seen.add(v); nxt.append(v)
            frontier = nxt
        return False

    parent = {a: a for a in atoms}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for p in changed_pairs:
        parent[find(p[0])] = find(p[1])
    for i, a in enumerate(atoms):
        for b in atoms[i + 1:]:
            if find(a) != find(b) and near(a, b):
                parent[find(a)] = find(b)
    return len({find(a) for a in atoms})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="grail_metabolism/resources/extended_smirks.txt")
    ap.add_argument("--mined", default="grail_metabolism/resources/mined_only_v2.txt")
    ap.add_argument("--slack", type=int, default=2,
                    help="bonds of separation still counted as one region")
    ap.add_argument("--out", default=str(ROOT / "results/cascade_audit.json"))
    args = ap.parse_args()

    from rdkit import RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    bank = [l.strip() for l in (ROOT / args.bank).read_text().splitlines() if l.strip()]
    mined = {l.strip() for l in (ROOT / args.mined).read_text().splitlines() if l.strip()}

    stats = defaultdict(int)
    per_rule = []
    for i, smk in enumerate(bank):
        try:
            rx = AllChem.ReactionFromSmarts(smk)
        except Exception:
            stats["unparseable"] += 1
            continue
        if rx is None:
            stats["unparseable"] += 1
            continue
        rmol = {}
        adjacency = defaultdict(set)
        rb = {}
        for t in rx.GetReactants():
            rb.update(bonds_by_map(t))
            for b in t.GetBonds():
                a, z = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
                if a and z:
                    adjacency[a].add(z); adjacency[z].add(a)
            for at in t.GetAtoms():
                if at.GetAtomMapNum():
                    rmol[at.GetAtomMapNum()] = at.GetAtomicNum()
        pb = {}
        for t in rx.GetProducts():
            pb.update(bonds_by_map(t))
        changed = [k for k in set(rb) | set(pb) if rb.get(k) != pb.get(k)]

        # An added group is unmapped in the product -- hydroxylation is [C:1][H:2] >> [C:1][O][H:2]
        # with no map on the O -- so a criterion that only compares mapped bonds sees no change at
        # all and misses the single commonest transformation in the bank. Each unmapped fragment
        # attached to a mapped atom is one edit, anchored at that atom.
        anchors = set()
        for t in rx.GetProducts():
            for b in t.GetBonds():
                x, y = b.GetBeginAtom(), b.GetEndAtom()
                for m, u in ((x, y), (y, x)):
                    if m.GetAtomMapNum() and not u.GetAtomMapNum():
                        anchors.add(m.GetAtomMapNum())
        for a in anchors:
            changed.append((a, a))          # a self-pair marks an edit at that atom

        if not changed:
            stats["no_change_detected"] += 1
            continue
        n = components(changed, adjacency, args.slack)

        # Regions are not steps. A cleavage, a cyclisation and an acyl migration each change two
        # distant places in ONE enzymatic event, and the event itself links them: a bond appears
        # between the regions, or a bond between them disappears. An independent pair of edits has
        # no such link. Only the unlinked case is a candidate cascade.
        made = {k for k in pb if k not in rb}
        broke = {k for k in rb if k not in pb}
        linked = False
        if n > 1:
            comp = {}
            for grp, atom in enumerate(sorted({a for c in changed for a in c})):
                comp[atom] = grp
            # recompute membership through the same union-find the count used
            def region_of(a):
                return _region_lookup.get(a)
            _region_lookup = _regions_map(changed, adjacency, args.slack)
            for a, b in made | broke:
                ra, rb_ = _region_lookup.get(a), _region_lookup.get(b)
                if ra is not None and rb_ is not None and ra != rb_:
                    linked = True
                    break
            if linked:
                stats["multi_region_one_event"] += 1
        stats[f"regions_{min(n, 4)}"] += 1
        stats["multi_region"] += n > 1
        stats["mined" if smk in mined else "curated"] += 1
        if n > 1 and not linked:
            stats["multi_mined" if smk in mined else "multi_curated"] += 1
            stats["independent_edits"] += 1
            if len(per_rule) < 25:
                per_rule.append({"index": i, "regions": n, "changed_bonds": len(changed),
                                 "source": "mined" if smk in mined else "curated",
                                 "smirks": smk[:220]})

    total = sum(v for k, v in stats.items() if k.startswith("regions_"))
    rep = {
        "provenance": stamp(__file__),
        "bank": args.bank, "n_templates": len(bank), "slack_bonds": args.slack,
        "criterion": ("a bond is changed when its order differs between mapped reactant and "
                      "product or exists on one side only; changed bonds are projected onto the "
                      "reactant graph and their connected components counted, joining components "
                      "within `slack` bonds so one edit reported as two adjacent bond changes is "
                      "not miscounted"),
        "counts": dict(stats),
        "n_scored": total,
        "multi_region_share": round(stats["multi_region"] / max(total, 1), 4),
        "examples": per_rule,
        "reading": ("a template with more than one region performs more than one independent "
                    "edit, so one application of it is not one enzymatic step"),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"scored {total} of {len(bank)} templates at slack {args.slack}")
    for k in sorted(stats):
        print(f"  {k:<20} {stats[k]}")
    print(f"\nmore than one region: {stats['multi_region']} ({rep['multi_region_share']:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

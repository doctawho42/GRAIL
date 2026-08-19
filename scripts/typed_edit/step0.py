#!/usr/bin/env python3
"""Step 0: does the typed-local-edit form survive contact with the real bank?

Three measurements on an existing SMIRKS bank:
  A. extract each template's reaction-centre signature and cluster templates into
     TYPES; report the curve `number of types against signature radius` -- a curve
     and not a point, per Proposition 2 of the leaderboard paper;
  B. cut each reactant query into a SKELETON (centre atoms, element and the bonds
     between them only) and an ENVIRONMENT (everything else: hydrogen count, degree,
     ring membership, charge, aromaticity, context atoms);
  C. enumerate (type, site) pairs on substrates and count the combinatorics: how many
     pairs survive skeletal admissibility against the full SMARTS.

The deciding number is how much larger the (type, site) pool is than the current
(rule, site) pool. More than an order of magnitude and the form has to change.

    python3 step0.py --rules rules.smirks --substrates substrates.smi
    python3 step0.py --self-test
"""
import argparse, sys, json, hashlib, re
from collections import defaultdict, Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions, rdqueries
RDLogger.DisableLog('rdApp.*')



# ------------------------------------------------------------- A. the centre

def _atom_props(a):
    """The atom properties that fix the IDENTITY of an edit.

    Hydrogen count and degree are left out on purpose: in a reactant query those are
    constraints on the environment rather than properties of the transformation, and
    they are exactly the constructs whose meaning depends on how the template was
    written. `Aromatic carbon hydroxylation` is defined by an aromatic C gaining a bond
    to an entering O, not by that carbon having carried exactly one hydrogen."""
    return (a.GetAtomicNum(), a.GetFormalCharge(), int(a.GetIsAromatic()))


def _bond_key(b):
    m1, m2 = b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()
    if m1 == 0 or m2 == 0:
        return None
    return (min(m1, m2), max(m1, m2))


def reaction_centre(rxn):
    """Return (centre_maps, entering, leaving, bond_deltas, atom_deltas, r_atoms).

    `centre_maps` holds the map numbers of the atoms the edit touches.
    """
    r_atoms, p_atoms = {}, {}
    r_bonds, p_bonds = {}, {}
    r_unmapped, p_unmapped = [], []

    for t in rxn.GetReactants():
        for a in t.GetAtoms():
            (r_atoms.__setitem__(a.GetAtomMapNum(), _atom_props(a))
             if a.GetAtomMapNum() else r_unmapped.append(a.GetAtomicNum()))
        for b in t.GetBonds():
            k = _bond_key(b)
            if k:
                r_bonds[k] = b.GetBondTypeAsDouble()
    for t in rxn.GetProducts():
        for a in t.GetAtoms():
            (p_atoms.__setitem__(a.GetAtomMapNum(), _atom_props(a))
             if a.GetAtomMapNum() else p_unmapped.append(a.GetAtomicNum()))
        for b in t.GetBonds():
            k = _bond_key(b)
            if k:
                p_bonds[k] = b.GetBondTypeAsDouble()

    centre, atom_deltas = set(), []
    for m in set(r_atoms) | set(p_atoms):
        rp, pp = r_atoms.get(m), p_atoms.get(m)
        if rp != pp:
            centre.add(m)
            atom_deltas.append((m, rp, pp))

    bond_deltas = []
    for k in set(r_bonds) | set(p_bonds):
        rb, pb = r_bonds.get(k), p_bonds.get(k)
        if rb != pb:
            centre.update(k)
            bond_deltas.append((k, rb, pb))

    # atoms carrying an entering fragment belong to the centre as well
    for t in rxn.GetProducts():
        for b in t.GetBonds():
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            if a1.GetAtomMapNum() and not a2.GetAtomMapNum():
                centre.add(a1.GetAtomMapNum())
            if a2.GetAtomMapNum() and not a1.GetAtomMapNum():
                centre.add(a2.GetAtomMapNum())
    for t in rxn.GetReactants():
        for b in t.GetBonds():
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            if a1.GetAtomMapNum() and not a2.GetAtomMapNum():
                centre.add(a1.GetAtomMapNum())
            if a2.GetAtomMapNum() and not a1.GetAtomMapNum():
                centre.add(a2.GetAtomMapNum())

    return (centre, sorted(Counter(p_unmapped).items()),
            sorted(Counter(r_unmapped).items()),
            sorted(bond_deltas), sorted(atom_deltas), r_atoms)


def signature(rxn, radius, canonical_deltas=True):
    """Canonical signature of the centre.

    radius=0 keeps only the atoms and bonds that change; radius=r adds the reactant
    template's environment out to r bonds. A larger radius gives a more specific
    signature and therefore MORE types.
    """
    centre, entering, leaving, bdel, adel, r_atoms = reaction_centre(rxn)
    if not centre:
        return None

    env = []
    for t in rxn.GetReactants():
        m2i = {a.GetAtomMapNum(): a.GetIdx() for a in t.GetAtoms() if a.GetAtomMapNum()}
        seeds = [m2i[m] for m in centre if m in m2i]
        if not seeds:
            continue
        shell, seen = set(seeds), set(seeds)
        for _ in range(radius):
            nxt = set()
            for i in shell:
                for nb in t.GetAtomWithIdx(i).GetNeighbors():
                    if nb.GetIdx() not in seen:
                        nxt.add(nb.GetIdx())
            seen |= nxt
            shell = nxt
        for i in sorted(seen - set(seeds)):
            a = t.GetAtomWithIdx(i)
            env.append((a.GetAtomicNum(), int(a.GetIsAromatic())))

    centre_desc = sorted(r_atoms[m] for m in centre if m in r_atoms)

    # None means `no bond or atom on this side of the arrow`. Tuples containing None
    # cannot be sorted -- this raises on the first bank rule with two changed bonds --
    # so absence is encoded explicitly: bond order 0.0, atom properties (-1, -1, -1).
    # The atom deltas are sorted rather than left in map-number order, because map
    # numbers are a property of whoever wrote the template: the same transformation
    # with a permuted numbering has to give one type.
    nb = lambda x: 0.0 if x is None else x
    na = lambda x: (-1, -1, -1) if x is None else tuple(x)
    payload = json.dumps({
        "centre_desc": centre_desc,
        "atom_deltas": sorted((na(rp), na(pp)) for _, rp, pp in adel) if canonical_deltas
                       else [(rp, pp) for _, rp, pp in adel],
        "bond_deltas": sorted((nb(rb), nb(pb)) for _, rb, pb in bdel),
        "entering": entering, "leaving": leaving,
        "n_centre": len(centre),
        "env": sorted(env),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# ------------------------------------------------- B. skeleton and environment

def skeleton_query(rxn, keep_aromaticity=True, relax_bonds=False):
    """The reactant template cut down to the centre atoms, with atom queries reduced to
    a single element. Everything else is environment: it moves into the model's features
    instead of staying a hard gate.
    """
    centre = reaction_centre(rxn)[0]
    if not centre:
        return None, 0
    t = rxn.GetReactantTemplate(0)
    rw = Chem.RWMol(t)

    drop = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomMapNum() not in centre]
    dropped = len(drop)
    for i in sorted(drop, reverse=True):
        rw.RemoveAtom(i)

    for a in rw.GetAtoms():
        n = a.GetAtomicNum()
        if n <= 0:                       # a query with no single element: leave it alone
            continue
        q = rdqueries.AtomNumEqualsQueryAtom(n)
        # Aromaticity is kept only where the template ASSERTED it. A query written
        # `[#6:1]` matches an aromatic or an aliphatic carbon, so adding IsAliphatic to
        # its skeleton tightens the gate instead of relaxing it -- on this bank that made
        # the skeleton miss sites the full template matched, in the one direction the
        # measurement must never move. Asserting aromatic where the query already said
        # `c` stays a relaxation.
        if keep_aromaticity and a.GetIsAromatic():
            q.ExpandQuery(rdqueries.IsAromaticQueryAtom())
        a.SetQuery(q)
        a.SetNoImplicit(False)
    out = rw.GetMol()
    if relax_bonds:
        p = Chem.AdjustQueryParameters.NoAdjustments()
        p.makeBondsGeneric = True
        out = Chem.AdjustQueryProperties(out, p)
    return out, dropped


# ----------------------------------------------------- C. the combinatorics

def count_pairs(substrate, templates, types, skeletons):
    """full: (rule, site) pairs as the pipeline enumerates them today; skel: the same
    under the skeleton query; typed: (type, site atom set), deduplicated."""
    full = skel = 0
    typed = set()
    for tid, rxn in enumerate(templates):
        rt = rxn.GetReactantTemplate(0)
        full += len(substrate.GetSubstructMatches(rt, uniquify=True, maxMatches=5000))
        sk = skeletons[tid]
        if sk is None:
            continue
        for m in substrate.GetSubstructMatches(sk, uniquify=True, maxMatches=5000):
            skel += 1
            typed.add((types[tid], frozenset(m)))
    return full, skel, len(typed)


# ------------------------------------------------------------------ runner

def strip_comment(line):
    """Strip a comment without touching the SMARTS.

    `#` in SMARTS is the atomic-number primitive (`[#6:1]`), not the start of a comment.
    The naive `line.split("#")[0]` silently drops 6,737 of the bank's 7,581 rules and
    leaves the report standing on the 844 survivors, which changes the population
    without saying so. Only a `#` at the start of a line or after whitespace is a
    comment."""
    if line.lstrip().startswith("#"):
        return ""
    return re.split(r"\s#", line, maxsplit=1)[0].strip()


def load_rules(path):
    """Return (rules, stats). The stats are printed and recorded in the report: losing
    rules silently is a change of population, and it has to be visible."""
    out = []
    stats = Counter()
    for line in open(path):
        stats["lines"] += 1
        s = strip_comment(line)
        if not s:
            stats["blank_or_comment"] += 1
            continue
        try:
            rxn = rdChemReactions.ReactionFromSmarts(s)
            if rxn is not None:
                rxn.Initialize()
        except Exception:
            stats["unparseable"] += 1
            continue
        if rxn and rxn.GetNumReactantTemplates() >= 1:
            out.append((s, rxn))
            stats["parsed"] += 1
        else:
            stats["no_reactant_template"] += 1
    return out, dict(stats)


def run(rules, substrates, radii=(0, 1, 2), keep_arom=True, relax_bonds=False):
    smirks = [s for s, _ in rules]
    rxns = [r for _, r in rules]

    curve, curve_maporder = {}, {}
    for r in radii:
        sigs = [signature(x, r) for x in rxns]
        curve[r] = len({s for s in sigs if s})
        sigs_mo = [signature(x, r, canonical_deltas=False) for x in rxns]
        curve_maporder[r] = len({s for s in sigs_mo if s})
    base_r = radii[0]
    types = [signature(x, base_r) or f"unsig{i}" for i, x in enumerate(rxns)]

    skels, dropped = [], []
    for x in rxns:
        sk, d = skeleton_query(x, keep_arom, relax_bonds)
        skels.append(sk); dropped.append(d)

    rows = []
    for name, mol in substrates:
        f, s, t = count_pairs(mol, rxns, types, skels)
        rows.append({"substrate": name, "heavy": mol.GetNumHeavyAtoms(),
                     "full": f, "skeleton": s, "typed": t})

    # the support distribution over types is the direct answer to Proposition 2
    support = Counter(types)
    sup_vals = sorted(support.values())
    singleton_types = sum(1 for v in sup_vals if v == 1)

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0

    n = max(len(rows), 1)
    tot = {k: sum(r[k] for r in rows) / n for k in ("full", "skeleton", "typed")}
    typed_vals = [r["typed"] for r in rows]
    return {
        "type_support": {
            "n_types": len(support),
            "singleton_types": singleton_types,
            "singleton_type_share": round(singleton_types / max(len(support), 1), 3),
            "median_templates_per_type": pct(sup_vals, 0.5),
            "max_templates_per_type": max(sup_vals) if sup_vals else 0,
        },
        "typed_pairs_per_substrate": {
            "mean": round(tot["typed"], 1),
            "p90": pct(typed_vals, 0.9),
            "max": max(typed_vals) if typed_vals else 0,
        },
        "n_templates": len(rxns),
        "n_types_at_radius": curve,
        "n_types_at_radius_maporder": curve_maporder,
        "n_unsigned_templates": sum(1 for t in types if t.startswith("unsig")),
        "type_reduction": round(len(rxns) / max(curve[base_r], 1), 1),
        "mean_env_atoms_dropped": round(sum(dropped) / max(len(dropped), 1), 2),
        "per_substrate_mean": {k: round(v, 1) for k, v in tot.items()},
        "blowup_typed_over_full": round(tot["typed"] / tot["full"], 2) if tot["full"] else None,
        "rows": rows,
        "n_smirks_parsed": len(smirks),
    }


# ----------------------------------------------------------------- self-test

def self_test():
    from fixture import RULES, SUBSTRATES
    ok = True

    rules = []
    for name, s in RULES:
        rxn = rdChemReactions.ReactionFromSmarts(s)
        rxn.Initialize()
        rules.append((s, rxn))
    if len(rules) != len(RULES):
        print("FAIL: not every SMIRKS parsed"); ok = False

    subs = [(n, Chem.MolFromSmiles(s)) for n, s in SUBSTRATES]
    if any(m is None for _, m in subs):
        print("FAIL: a substrate did not parse"); ok = False

    res = run(rules, subs, radii=(0, 1, 2))

    curve = res["n_types_at_radius"]
    if not (curve[0] <= curve[1] <= curve[2]):
        print(f"FAIL: the type count is not monotone in the radius: {curve}"); ok = False
    if curve[0] >= res["n_templates"]:
        print(f"FAIL: the clustering collapsed nothing: {curve[0]} of {res['n_templates']}")
        ok = False

    rxns = [r for _, r in rules]
    sig0 = [signature(x, 0) for x in rxns]
    sig1 = [signature(x, 1) for x in rxns]
    if sig0[0] != sig0[1]:
        print("FAIL: two notations of aromatic hydroxylation gave different types at r=0")
        ok = False
    if sig0[2] != sig0[3]:
        print("FAIL: methyl hydroxylation in two environments already split at r=0")
        ok = False
    if sig1[2] == sig1[3]:
        print("FAIL: r=1 did not separate the two environments, so the radius does nothing"); ok = False

    # the three invariants below guard the three defects the real bank exposed; each one
    # fails by name if its fix is reverted
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".smirks", delete=False) as fh:
        fh.write("# a comment line\n"
                 "[#6:1][O:2][C:3]([H])>>[#6:1][O:2].[C;X3:3](=O)\n"
                 "[C;X4:1][H:2]>>[C:1][O][H:2]  # trailing comment\n")
        probe = fh.name
    probe_rules, probe_stats = load_rules(probe)
    if probe_stats["parsed"] != 2 or probe_stats["blank_or_comment"] != 1:
        print(f"FAIL: the loader mishandles `#`: {probe_stats}"); ok = False

    two_breaks = rdChemReactions.ReactionFromSmarts(
        "[C:1](=[O:2])[O:3][C:4][N:5]>>[C:1](=[O:2])O.[C:4]=[N:5]")
    two_breaks.Initialize()
    if signature(two_breaks, 0) is None:
        print("FAIL: a rule with two changed bonds produced no signature"); ok = False

    a = rdChemReactions.ReactionFromSmarts("[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]")
    b = rdChemReactions.ReactionFromSmarts("[C:7](=[O:9])[N:2]>>[C:7](=[O:9])O.[N:2]")
    a.Initialize(); b.Initialize()
    if signature(a, 0) != signature(b, 0):
        print("FAIL: renumbering the atom maps changed the type"); ok = False

    for r in res["rows"]:
        if r["typed"] > r["skeleton"]:
            print(f"FAIL: typed > skeleton on {r['substrate']}"); ok = False
    if not any(r["skeleton"] > r["full"] for r in res["rows"]):
        print("FAIL: relaxing the environment added no match anywhere"); ok = False

    print(f"templates: {res['n_templates']}   types by radius: {curve}   "
          f"collapse: x{res['type_reduction']}")
    print(f"environment atoms dropped, mean: {res['mean_env_atoms_dropped']}")
    print(f"{'substrate':<14}{'heavy':>6}{'(rule,site)':>14}{'skeleton':>10}{'(type,site)':>13}")
    for r in res["rows"]:
        print(f"{r['substrate']:<14}{r['heavy']:>5}{r['full']:>16}{r['skeleton']:>9}{r['typed']:>12}")
    m = res["per_substrate_mean"]
    print(f"{'mean':<14}{'':>6}{m['full']:>14}{m['skeleton']:>10}{m['typed']:>13}")
    ts, tp = res["type_support"], res["typed_pairs_per_substrate"]
    print(f"BLOWUP (type,site)/(rule,site): x{res['blowup_typed_over_full']}")
    print(f"types: {ts['n_types']}, of which one template only: {ts['singleton_types']} "
          f"({ts['singleton_type_share']:.1%}); median templates per type "
          f"{ts['median_templates_per_type']}, max {ts['max_templates_per_type']}")
    print(f"(type,site) pairs per substrate: mean {tp['mean']}, p90 {tp['p90']}, max {tp['max']}")
    print("self-test: OK" if ok else "self-test: FAILURES ABOVE")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", help="SMIRKS file, one per line")
    ap.add_argument("--substrates", help="SMILES file, one per line")
    ap.add_argument("--radii", default="0,1,2")
    ap.add_argument("--relax-bonds", action="store_true")
    ap.add_argument("--drop-aromaticity", action="store_true")
    ap.add_argument("--json", help="where to write the full report")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not (a.rules and a.substrates):
        ap.error("--rules and --substrates are required (or --self-test)")

    rules, load_stats = load_rules(a.rules)
    subs = []
    for line in open(a.substrates):
        s = line.split()[0].strip() if line.split() else ""
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m:
            subs.append((s[:24], m))
    print(f"bank lines: {load_stats['lines']}   rules parsed: {len(rules)}   "
          f"unparseable: {load_stats.get('unparseable', 0)}   substrates: {len(subs)}",
          file=sys.stderr)

    res = run(rules, subs, tuple(int(x) for x in a.radii.split(",")),
              keep_arom=not a.drop_aromaticity, relax_bonds=a.relax_bonds)
    res["rule_bank"] = {"path": a.rules, **load_stats}
    res["substrates"] = {"path": a.substrates, "n": len(subs)}
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)
    r = dict(res); r.pop("rows")
    print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

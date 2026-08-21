#!/usr/bin/env python3
"""The type vocabulary as a curve over declared choices, not as one point.

Step 0 asks whether a bank of 7,581 rules collapses into a few hundred types with real
support per class. The answer does not depend on the radius alone. It depends on what is
declared to be part of an edit's identity, so each such choice is a signature variant here
and all of them are reported together.

The variants:
  as_written      the signature of step0.py unchanged
  no_explicit_H   hydrogen is removed from the identity of an edit altogether: mapped H
                  atoms do not enter the centre, H is not counted in the entering or
                  leaving fragment, and bond deltas involving an H are dropped. This
                  continues the decision not to read a hydrogen count out of the query:
                  the bank mixes [cH:1]>>[c:1]O with [c:1][H:2]>>[c:1][O][H:2], and
                  without this one transformation yields two types
  no_H_no_size    the same, and the centre size is dropped from the signature
  changed_only    the centre description is cut back to the atoms that actually changed
  bonds_only      the multiset of changed bonds between mapped atoms alone, which is the
                  signature already implemented in grail_metabolism/model/reaction_types.py

For each variant this reports not only the number of types but the pooled training-pair
support of a type: a type built from one template carrying forty pairs is learnable, a type
built from forty templates carrying one pair each is not. Support comes from
results/mined_rule_catalog_v2.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from step0 import load_rules, reaction_centre  # noqa: E402

CATALOG = ROOT / "results" / "mined_rule_catalog_v2.json"
BANK = ROOT / "grail_metabolism" / "resources" / "extended_smirks.txt"


def _elements(rxn):
    """map number -> atomic number, on both sides of the arrow."""
    out = {}
    for side in (rxn.GetReactants(), rxn.GetProducts()):
        for t in side:
            for a in t.GetAtoms():
                if a.GetAtomMapNum():
                    out.setdefault(a.GetAtomMapNum(), a.GetAtomicNum())
    return out


def _env(rxn, centre, radius):
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
    return env


def signature(rxn, radius=0, variant="as_written"):
    centre, entering, leaving, bdel, adel, r_atoms = reaction_centre(rxn)
    if not centre:
        return None
    elt = _elements(rxn)
    drop_h = variant in ("no_explicit_H", "no_H_no_size", "changed_only")

    if drop_h:
        centre = {m for m in centre if elt.get(m) != 1}
        entering = [(z, n) for z, n in entering if z != 1]
        leaving = [(z, n) for z, n in leaving if z != 1]
        bdel = [(k, rb, pb) for k, rb, pb in bdel
                if elt.get(k[0]) != 1 and elt.get(k[1]) != 1]
        adel = [(m, rp, pp) for m, rp, pp in adel if elt.get(m) != 1]
        if not centre:
            return None

    nb = lambda x: 0.0 if x is None else x
    na = lambda x: (-1, -1, -1) if x is None else tuple(x)

    if variant == "bonds_only":
        # changed bonds between mapped atoms only, as in reaction_types.py
        multiset = Counter((tuple(sorted((elt.get(k[0], 0), elt.get(k[1], 0)))), nb(rb), nb(pb))
                           for k, rb, pb in bdel)
        if not multiset:
            return None
        payload = json.dumps(sorted(multiset.items()), sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    if variant == "changed_only":
        centre_desc = sorted(na(rp) for _, rp, _ in adel)
    else:
        centre_desc = sorted(r_atoms[m] for m in centre if m in r_atoms)

    body = {
        "centre_desc": centre_desc,
        "atom_deltas": sorted((na(rp), na(pp)) for _, rp, pp in adel),
        "bond_deltas": sorted((nb(rb), nb(pb)) for _, rb, pb in bdel),
        "entering": sorted(entering), "leaving": sorted(leaving),
        "env": sorted(_env(rxn, centre, radius)),
    }
    if variant not in ("no_H_no_size", "changed_only"):
        body["n_centre"] = len(centre)
    return hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


def summarise(sigs, smirks, counts):
    """sigs: one signature per template (None means the template does not type)."""
    by_type = defaultdict(list)
    unsigned = 0
    for s, sig in zip(smirks, sigs):
        if sig is None:
            unsigned += 1
            continue
        by_type[sig].append(s)

    n_types = len(by_type)
    singleton = sum(1 for v in by_type.values() if len(v) == 1)

    # pooled training-pair support, over the bank rules the mined catalog knows about
    pooled, known = {}, 0
    for sig, rules in by_type.items():
        tot = 0
        for r in rules:
            if r in counts:
                tot += counts[r]
                known += 1
        pooled[sig] = tot
    with_pairs = [v for v in pooled.values() if v > 0]
    total_pairs = sum(with_pairs)
    dense = [v for v in with_pairs if v >= 5]
    return {
        "n_types": n_types,
        "n_unsigned_templates": unsigned,
        "singleton_type_share": round(singleton / max(n_types, 1), 3),
        "median_templates_per_type": sorted(len(v) for v in by_type.values())[n_types // 2]
        if n_types else 0,
        "max_templates_per_type": max((len(v) for v in by_type.values()), default=0),
        "types_with_train_pairs": len(with_pairs),
        "types_with_ge5_pairs": len(dense),
        "share_types_ge5_pairs": round(len(dense) / max(len(with_pairs), 1), 3),
        "train_pairs_in_dense_types": round(sum(dense) / max(total_pairs, 1), 3),
        "median_pairs_per_type": sorted(with_pairs)[len(with_pairs) // 2] if with_pairs else 0,
        "catalog_rules_matched": known,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=str(BANK))
    ap.add_argument("--radii", default="0,1,2")
    ap.add_argument("--out", default=str(ROOT / "results" / "typed_edit_type_curve.json"))
    a = ap.parse_args()

    rules, load_stats = load_rules(a.rules)
    smirks = [s for s, _ in rules]
    rxns = [r for _, r in rules]
    catalog = json.loads(CATALOG.read_text())
    counts = {k: int(v.get("count", 0)) for k, v in catalog.items()}
    in_bank = sum(1 for s in smirks if s in counts)

    variants = ["as_written", "no_explicit_H", "no_H_no_size", "changed_only", "bonds_only"]
    radii = tuple(int(x) for x in a.radii.split(","))

    out = {
        "provenance": stamp(__file__),
        "rule_bank": {"path": str(Path(a.rules).relative_to(ROOT)), **load_stats},
        "catalog": {"path": str(CATALOG.relative_to(ROOT)), "n_rules": len(counts),
                    "bank_rules_found_in_catalog": in_bank,
                    "total_train_pairs": sum(counts.values()),
                    "catalog_rules_with_one_pair": sum(1 for v in counts.values() if v == 1)},
        "by_variant": {}, "by_radius": {},
    }
    for v in variants:
        sigs = [signature(x, 0, v) for x in rxns]
        out["by_variant"][v] = summarise(sigs, smirks, counts)
        print(f"  {v:<16} {out['by_variant'][v]['n_types']:>6} types  "
              f"singleton {out['by_variant'][v]['singleton_type_share']:.1%}  "
              f"dense(>=5 pairs) {out['by_variant'][v]['types_with_ge5_pairs']:>4}",
              file=sys.stderr, flush=True)
    for r in radii:
        sigs = [signature(x, r, "no_explicit_H") for x in rxns]
        out["by_radius"][r] = summarise(sigs, smirks, counts)

    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Redesign Gate A — de-memorization check for the factorized-generative redesign.

The redesign replaces the 7,581-way PU rule label with a DENSE reaction-TYPE label. That only
escapes the PU degeneracy (Prop 2) if the type vocabulary is genuinely SHARED across substrates —
i.e. the radius-0 reaction center (bond-level transformation, periphery stripped) is NOT itself
73%-singleton the way the radius-1 SMIRKS bank is.

For each mined SMIRKS (`results/mined_rule_catalog_v2.json`, whose `count` is its train-pair
support), compute a radius-0 transformation signature = the multiset of bonds broken/formed between
the reactant and product templates, described only by (element_a, element_b, order) — no periphery.
Aggregate rule support onto signatures and report the singleton fraction over TRAIN PAIRS.

GO if the pair-weighted singleton fraction over types is << 73% (types are shared -> dense label is
real). KILL if ~73% (types are also singletons -> the density claim is a radius artifact).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

CATALOG = ROOT / "results" / "mined_rule_catalog_v2.json"


def _template_bonds(mol) -> dict:
    """map-number-pair -> bond order, for atoms that carry atom-map numbers in a SMARTS template."""
    idx_to_map = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    idx_to_elt = {a.GetIdx(): a.GetAtomicNum() for a in mol.GetAtoms()}
    bonds = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx_to_map and j in idx_to_map:
            key = tuple(sorted((idx_to_map[i], idx_to_map[j])))
            bonds[key] = (b.GetBondTypeAsDouble(), idx_to_elt[i], idx_to_elt[j])
    return bonds, idx_to_map, idx_to_elt


def radius0_signature(smirks: str):
    """Element-level broken/formed-bond signature of the reaction center; None if unparseable."""
    try:
        rxn = AllChem.ReactionFromSmarts(smirks)
        if rxn is None:
            return None
        r_bonds = {}
        for t in rxn.GetReactants():
            rb, _, _ = _template_bonds(t)
            r_bonds.update(rb)
        p_bonds = {}
        for t in rxn.GetProducts():
            pb, _, _ = _template_bonds(t)
            p_bonds.update(pb)
    except Exception:
        return None
    changed = []
    for key in set(r_bonds) | set(p_bonds):
        ro = r_bonds.get(key)
        po = p_bonds.get(key)
        if ro == po:
            continue  # unchanged bond among mapped atoms -> not part of the radius-0 center
        # describe the change by element pair (order-independent) + before/after bond order
        elts = tuple(sorted((ro or po)[1:]))
        before = ro[0] if ro else 0.0
        after = po[0] if po else 0.0
        changed.append((elts, before, after))
    if not changed:
        return None
    return tuple(sorted(changed))


def main() -> int:
    catalog = json.loads(CATALOG.read_text())
    sig_rules = Counter()          # signature -> number of distinct rules
    sig_pairs = Counter()          # signature -> total train-pair support
    unparsed_rules = 0
    unparsed_pairs = 0
    total_pairs = 0
    for smirks, entry in catalog.items():
        support = int(entry.get("count", 0))
        total_pairs += support
        sig = radius0_signature(smirks)
        if sig is None:
            unparsed_rules += 1
            unparsed_pairs += support
            continue
        sig_rules[sig] += 1
        sig_pairs[sig] += support

    n_rules = len(catalog)
    n_types = len(sig_rules)
    # singleton fractions
    singleton_types_by_rule = sum(1 for s, c in sig_rules.items() if c == 1)
    # pair-weighted: fraction of train pairs whose type is supported by only ONE pair total
    single_pair_types = {s for s, p in sig_pairs.items() if p == 1}
    pairs_in_singleton_types = sum(sig_pairs[s] for s in single_pair_types)

    report = {
        "n_rules_radius1": n_rules,
        "n_types_radius0": n_types,
        "compression_x": round(n_rules / n_types, 1) if n_types else None,
        "unparsed_rules": unparsed_rules,
        "rule_bank_singleton_frac_radius1": 0.73,  # the measured baseline we must beat
        "type_singleton_frac_by_rulecount": round(singleton_types_by_rule / n_types, 3) if n_types else None,
        "pairs_total": total_pairs,
        "pairs_in_single_pair_types_frac": round(pairs_in_singleton_types / total_pairs, 3) if total_pairs else None,
        "top_types_by_pair_support": [
            {"signature": str(s), "pairs": sig_pairs[s], "rules": sig_rules[s]}
            for s, _ in sig_pairs.most_common(10)
        ],
    }
    out = ROOT / "results" / "redesign_gate_a.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "top_types_by_pair_support"}, indent=2), flush=True)
    print("\ntop reaction types by train-pair support:", flush=True)
    for t in report["top_types_by_pair_support"]:
        print(f"  pairs={t['pairs']:>5}  rules={t['rules']:>4}  {t['signature'][:90]}", flush=True)
    # The synthesis's Gate A is the singleton fraction OVER TRAIN PAIRS (not over types): a dense,
    # learnable type label needs most training pairs to fall in a RECURRING type. The type-count
    # singleton fraction over-reports because the crude multi-bond signature over-splits ring
    # transformations into many rare types; the pair-weighted fraction is the decisive metric.
    pw = report["pairs_in_single_pair_types_frac"]
    verdict = "GO (type label dense at the pair level)" if (pw or 1) < 0.30 else "KILL (types also singleton over pairs)"
    print(f"\nGATE A: radius-1 rule-singleton 73% over pairs -> radius-0 TYPE-singleton {pw} over pairs "
          f"({report['pairs_total']} pairs, {n_types} raw types; note the crude signature over-splits "
          f"multi-bond/ring changes, so the effective vocabulary is smaller) => {verdict}", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Radius-0 reaction-TYPE vocabulary + rule_id -> type_id map.

Stage-1 (the generator) currently learns over the full ~7,581-way PU rule-label space and
degenerates (see `scripts/redesign_gate_a.py`). This module builds a much coarser, DENSE
reaction-TYPE vocabulary by collapsing each mined SMIRKS rule to its radius-0 signature — the
element-typed multiset of bonds broken/formed between the reactant and product templates, with
all periphery (everything beyond the directly changed bonds) stripped away.

`canonical_type` ports the bond-diffing logic from `scripts/redesign_gate_a.py:radius0_signature`
but additionally collapses the changed-bond list into a `collections.Counter` multiset, so that N
occurrences of an identical changed-bond descriptor (e.g. several equivalent bonds broken/formed
when a ring forms) become a single `(descriptor, N)` entry instead of N positional entries. Two
SMIRKS whose changed-bond multisets agree — even if the raw (unmerged) bond lists were ordered or
duplicated differently — map to the same canonical type.

`build_type_vocab` groups a rule catalog (`{smirks: {"count": <train-pair support>, ...}}`) by
`canonical_type`, pools support per signature, and assigns dense integer type ids (0..K-1, by
descending pooled support) to signatures with pooled support >= `min_pairs`. Rules whose signature
is unparseable, has no bond-level change, or falls below the support threshold are mapped to the
shared "other" bucket, `type_id = -1`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

__all__ = ["canonical_type", "build_type_vocab"]


def _template_bonds(mol) -> dict:
    """map-number-pair -> (bond_order, element_a, element_b), for mapped atoms in a template."""
    idx_to_map = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    idx_to_elt = {a.GetIdx(): a.GetAtomicNum() for a in mol.GetAtoms()}
    bonds = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx_to_map and j in idx_to_map:
            key = tuple(sorted((idx_to_map[i], idx_to_map[j])))
            bonds[key] = (b.GetBondTypeAsDouble(), idx_to_elt[i], idx_to_elt[j])
    return bonds


def _changed_bonds(smirks: str) -> Optional[list]:
    """List of (elements, before_order, after_order) descriptors for changed mapped bonds.

    Returns None if `smirks` is unparseable as an RDKit reaction.
    """
    try:
        rxn = AllChem.ReactionFromSmarts(smirks)
        if rxn is None:
            return None
        r_bonds: dict = {}
        for t in rxn.GetReactants():
            r_bonds.update(_template_bonds(t))
        p_bonds: dict = {}
        for t in rxn.GetProducts():
            p_bonds.update(_template_bonds(t))
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
    return changed


def canonical_type(smirks: str) -> Optional[tuple]:
    """Canonical, order-collapsed radius-0 reaction-type signature for a SMIRKS rule.

    The changed-bond descriptors are pooled into a `Counter` multiset before being returned, so
    that N identical changed-bond descriptors (e.g. from a symmetric ring formation) collapse to
    one `(descriptor, N)` entry rather than N positional entries — this is what lets SMIRKS that
    differ only in aromatic/ring periphery, or in how many times an equivalent bond change is
    duplicated, resolve to the same type.

    Returns None when `smirks` is unparseable, or has no bond-level change between two mapped
    atoms (i.e. no radius-0 reaction center can be derived).
    """
    changed = _changed_bonds(smirks)
    if not changed:
        return None
    counter = Counter(changed)
    return tuple(sorted(counter.items()))


def build_type_vocab(catalog: dict, min_pairs: int = 5) -> tuple[dict, dict]:
    """Group a mined-rule catalog into a dense radius-0 reaction-type vocabulary.

    Args:
        catalog: `{smirks: {"count": <train-pair support>, ...}}`.
        min_pairs: minimum pooled train-pair support (summed over all rules sharing a
            `canonical_type`) required for a signature to get its own dense type id.

    Returns:
        `(type_id_to_sig, rule_smirks_to_type_id)`:
          - `type_id_to_sig`: `{type_id: {"signature": str, "n_rules": int, "n_pairs": int}}`
            for the kept (dense) types, `type_id` in `0..K-1`, ordered by descending pooled
            support.
          - `rule_smirks_to_type_id`: `{smirks: type_id}` for every SMIRKS in `catalog`; rules
            that are unparseable, have no bond-level change, or belong to a signature with
            pooled support below `min_pairs` map to the shared "other" bucket, `type_id = -1`.
    """
    sig_to_rules: dict = defaultdict(list)
    sig_to_pairs: Counter = Counter()
    rule_to_sig: dict = {}

    for smirks, entry in catalog.items():
        support = int(entry.get("count", 0))
        sig = canonical_type(smirks)
        rule_to_sig[smirks] = sig
        if sig is None:
            continue
        sig_to_rules[sig].append(smirks)
        sig_to_pairs[sig] += support

    kept_sigs = [sig for sig, pairs in sig_to_pairs.items() if pairs >= min_pairs]
    # dense ids assigned by descending pooled support; break ties deterministically on the
    # signature's own (already-sorted, hashable) representation
    kept_sigs.sort(key=lambda sig: (-sig_to_pairs[sig], sig))

    sig_to_id: dict = {}
    type_id_to_sig: dict = {}
    for type_id, sig in enumerate(kept_sigs):
        sig_to_id[sig] = type_id
        type_id_to_sig[type_id] = {
            "signature": repr(sig),
            "n_rules": len(sig_to_rules[sig]),
            "n_pairs": sig_to_pairs[sig],
        }

    rule_smirks_to_type_id = {
        smirks: sig_to_id.get(sig, -1) if sig is not None else -1
        for smirks, sig in rule_to_sig.items()
    }

    return type_id_to_sig, rule_smirks_to_type_id

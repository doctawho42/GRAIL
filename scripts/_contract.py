"""Contracting an expanded product, and why it is not one call.

An arm that expands the substrate has to put the hydrogens back before the product is read.
`Chem.RemoveHs` alone does not do that. `AddHs` marks every heavy atom as taking no implicit
hydrogens, so when a template consumes a mapped hydrogen and puts nothing in its place, removing the
drawn hydrogens leaves the atom one short and RDKit records an unpaired electron rather than
refilling the valence. On this bank that happens to 22.6% of the products that parse at all, and a
metabolite corpus contains no radicals, so those products cannot match a reference and are lost
silently rather than loudly.

Clearing the flag first restores the implicit-hydrogen capacity that AddHs suspended. It yields no
radicals at all and 4,069 more parseable products from the same firings. Which of the two an engine
uses is one more undeclared implementation choice of the kind this paper is about, and it is the
reason this module exists rather than the call being written inline in four places.
"""
from __future__ import annotations

from rdkit import Chem


def contract(product):
    """Put back the hydrogens AddHs drew, restoring implicit capacity first. May raise."""
    mol = Chem.Mol(product)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
    mol = Chem.RemoveHs(mol, sanitize=False)
    Chem.SanitizeMol(mol)
    return mol


def contract_by_removing_only(product):
    """The one-call version, kept so the difference between the two remains measurable."""
    mol = Chem.RemoveHs(Chem.Mol(product), sanitize=False)
    Chem.SanitizeMol(mol)
    return mol

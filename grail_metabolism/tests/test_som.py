"""Tests for the site-of-metabolism (SoM) regioselectivity prior (model/som.py)."""
from __future__ import annotations

import types

import numpy as np
from rdkit import Chem

from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.model.grail import summon_the_grail
from grail_metabolism.model.som import (
    SoMPredictor,
    _reacting_atoms,
    build_som_dataset,
    derive_som_labels,
)
from grail_metabolism.utils.seed import seed_everything

RULE = "[CH2:1][OH:2]>>[CH:1]=[O:2]"


def test_derive_som_labels_detects_bond_order_change():
    # Ethanol -> acetaldehyde changes only a bond order (C-O -> C=O) + H count, so the
    # permissive MCS keeps every atom in the core; the label routine must still flag the
    # carbonyl carbon (idx 1) and oxygen (idx 2) via the atom-environment change.
    labels = derive_som_labels("CCO", ["CC=O"])
    assert 1 in labels and 2 in labels


def test_score_atoms_shape_and_range():
    som = SoMPredictor()
    arr = som.score_atoms("CCO")
    assert arr.shape == (3,)  # ethanol has 3 heavy atoms
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


def test_som_beta_zero_is_identity():
    # beta=0 (and som_beta=None) must reproduce the exact non-SoM ranking (back-compat).
    seed_everything(0)
    model = summon_the_grail([RULE])
    model.som = SoMPredictor()
    model.filter.calibrated_threshold = 0.0

    def fake_scored(sub, top_k=None, threshold=None):
        return [("CCO", 0.9), ("CC=O", 0.8), ("CC(=O)O", 0.7)]

    model.generator.generate_scored = fake_scored
    out_off = model.generate("CCO", gate_by_filter=False)             # som_beta defaults to None -> off
    out_beta0 = model.generate("CCO", som_beta=0.0, gate_by_filter=False)
    assert out_off == out_beta0


def test_som_reweight_promotes_high_site_product():
    # Two equal-generator products at different sites; boosting one site's SoM must rank
    # that product first when beta>0.
    seed_everything(0)
    # para-cresol (not ortho) so prod_b's reaction site stays clear of the benzylic atom's
    # 1-hop neighborhood -- otherwise the boosted site would leak into both products.
    sub, prod_a, prod_b = "Cc1ccccc1", "OCc1ccccc1", "Cc1ccc(O)cc1"  # benzylic vs para-ring hydroxylation
    sub_mol = Chem.MolFromSmiles(sub)
    react_a = _reacting_atoms(sub_mol, Chem.MolFromSmiles(prod_a))
    assert react_a  # sanity: the benzylic site is localizable

    arr = np.full(sub_mol.GetNumAtoms(), 0.05, dtype=np.float32)
    for i in react_a:
        arr[i] = 0.95

    som = SoMPredictor()
    som.score_atoms = lambda s: arr  # type: ignore[assignment]
    model = summon_the_grail([RULE])
    model.som = som
    model.filter.calibrated_threshold = 0.0
    model.filter.score_batch = lambda sub, prods: [1.0] * len(prods)  # type: ignore[assignment]
    model.generator.gen_normalization = "canonical"
    model.generator.generate_scored = lambda s, top_k=None, threshold=None: [(prod_a, 0.5), (prod_b, 0.5)]

    out = model.generate(sub, som_beta=4.0, gate_by_filter=False)
    assert _tautomer_inchikey(out[0]) == _tautomer_inchikey(prod_a)


def test_build_som_dataset_labels_match_derive():
    stub = types.SimpleNamespace(map={"CCO": {"CC=O"}})
    samples = build_som_dataset(stub)
    assert len(samples) == 1
    data = samples[0]
    got = {int(i) for i in (data.y == 1).nonzero(as_tuple=True)[0].tolist()}
    assert got == derive_som_labels("CCO", {"CC=O"})

from scripts.ceiling_external_validity import composition_descriptors


def test_composition_descriptors_shape_and_values():
    d = composition_descriptors("c1ccccc1")  # benzene
    assert set(d) == {"mw", "n_rings", "n_aromatic", "n_hetero", "n_conj", "n_true_ph"}
    assert 77 < d["mw"] < 79  # ~78.11
    assert d["n_rings"] == 1 and d["n_aromatic"] == 6 and d["n_hetero"] == 0
    assert composition_descriptors("not_a_smiles") is None

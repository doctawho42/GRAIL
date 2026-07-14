"""Tests for the dense (type, site) training-label dataset builder (model/factorized_data.py)."""
from __future__ import annotations

import pandas as pd

from grail_metabolism.model.factorized_data import build_factorized_dataset
from grail_metabolism.utils.preparation import MolFrame, _standardize_smiles_cached

RULE = "[CH2:1][OH:2]>>[CH:1]=[O:2]"


def test_dense_type_and_site_labels():
    frame = MolFrame(pd.DataFrame([{"sub": "CCO", "prod": "CC=O", "real": 1}]))
    frame.full_setup(rules=[RULE], include_pair_graphs=False, include_morgan=False)
    rule_to_type = {RULE: 0}
    catalog = {RULE: {"count": 1, "source_pairs": [["CCO", "CC=O"]]}}

    ds = build_factorized_dataset(frame, rule_to_type, catalog=catalog)

    assert len(ds) == 1
    d = ds[0]
    assert d.y_type.sum() >= 1          # at least one positive type
    assert d.y_site.sum() >= 1          # at least one reacting atom
    assert d.y_site.numel() == d.x.size(0)
    assert d.y_type.numel() == 1        # num_types = 1 + max(rule_to_type.values()) = 1


def test_catalog_keys_are_standardized_to_match_molframe():
    # Catalog source_pairs hold RAW SMILES from the clean train SDF; MolFrame standardizes
    # its substrate keys. A raw substrate SMILES that differs textually from its
    # standardized form (but is the same molecule) must still align.
    raw_sub = "OCC"  # same molecule as CCO but different SMILES string
    standardized = _standardize_smiles_cached(raw_sub)
    assert standardized != raw_sub  # sanity: the raw string really is non-canonical

    frame = MolFrame(pd.DataFrame([{"sub": raw_sub, "prod": "CC=O", "real": 1}]))
    frame.full_setup(rules=[RULE], include_pair_graphs=False, include_morgan=False)
    rule_to_type = {RULE: 0}
    catalog = {RULE: {"count": 1, "source_pairs": [[raw_sub, "CC=O"]]}}

    ds = build_factorized_dataset(frame, rule_to_type, catalog=catalog)

    assert len(ds) == 1
    assert ds[0].y_type.sum() >= 1


def test_missing_catalog_entry_yields_all_zero_type():
    frame = MolFrame(pd.DataFrame([{"sub": "CCO", "prod": "CC=O", "real": 1}]))
    frame.full_setup(rules=[RULE], include_pair_graphs=False, include_morgan=False)
    rule_to_type = {RULE: 0}
    catalog = {RULE: {"count": 1, "source_pairs": [["c1ccccc1", "Oc1ccccc1"]]}}  # unrelated substrate

    ds = build_factorized_dataset(frame, rule_to_type, catalog=catalog)

    assert len(ds) == 1
    assert ds[0].y_type.sum() == 0

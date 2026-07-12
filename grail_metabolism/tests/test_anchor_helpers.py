"""Focused unit test for the anchor-certification per-substrate helpers (Task 4).

Runs under `make test` without the dataset: `per_substrate_recall` / `any_hit` are pure
tautomer-InChIKey set arithmetic over SMILES lists (RDKit only, no SyGMa, no torch model)."""
from scripts.anchor_certification import any_hit, per_substrate_recall


def test_per_substrate_recall_tautomer():
    trues = ["CCO", "c1ccccc1"]
    preds = ["OCC"]  # ethanol only -> recovers 1 of 2 tautomer-distinct trues
    assert per_substrate_recall(preds, trues) == 0.5
    assert any_hit(preds, trues) is True
    assert any_hit(["CCCC"], trues) is False


def test_per_substrate_recall_empty_trues():
    assert per_substrate_recall(["CCO"], []) == 0.0
    assert any_hit(["CCO"], []) is False

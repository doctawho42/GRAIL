"""Focused unit test for the anchor-certification per-substrate helpers (Task 4).

Runs under `make test` without the dataset: `per_substrate_recall` / `any_hit` are pure
tautomer-InChIKey set arithmetic over SMILES lists (RDKit only, no SyGMa, no torch model)."""
from scripts.anchor_certification import any_hit, per_substrate_recall


def test_per_substrate_recall_tautomer():
    trues = ["CCO", "c1ccccc1"]  # 2 tautomer-distinct trues -> caller passes u=2 (the shared U_i)
    preds = ["OCC"]  # ethanol only -> recovers 1 of 2
    assert per_substrate_recall(preds, trues, 2) == 0.5
    assert any_hit(preds, trues) is True
    assert any_hit(["CCCC"], trues) is False


def test_per_substrate_recall_zero_denominator():
    # u == 0 (no trues) -> 0.0, no division error
    assert per_substrate_recall(["CCO"], [], 0) == 0.0
    assert any_hit(["CCO"], []) is False

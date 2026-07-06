"""Dataset-free guard tests for grail_metabolism/eval/diversity.py.

Hand-built-fixture, dataset-free tests in the style of test_audit_fixes.py /
test_set_gflownet.py: small SMILES constants at module top, one assertion per
known-answer property, no external SDF/triples. Tests that depend on tautomer
collapse monkeypatch ``_tautomer_inchikey`` (following the ``_fake_taut_ik``
pattern in test_set_gflownet.py) rather than relying on the installed RDKit
build's actual tautomer-canonicalization coverage, keeping them
RDKit-version-independent.
"""

from __future__ import annotations

import grail_metabolism.eval.diversity as diversity
import grail_metabolism.metrics as metrics

# Hexane/heptane: genuinely distinct SMILES but high Tanimoto similarity
# (~0.875, verified against the installed RDKit build); benzene is
# structurally unrelated to both (Tanimoto ~0.0 to either). At
# threshold=0.4 (distance 0.6) hexane and heptane are close enough to
# collapse into one pick, leaving benzene separate -> 2 circles. At
# threshold=0.9 (distance 0.1) hexane/heptane's 0.875 similarity is NOT
# within the tighter distance bound, so all three survive -> 3 circles.
# This is the sign-conversion guard: a bug that passes `threshold` directly
# instead of `1 - threshold` would silently invert this relationship.
_HEXANE = "CCCCCC"
_HEPTANE = "CCCCCCC"
_BENZENE = "c1ccccc1"

# Two distinct, parseable molecules with DISTINCT plain InChIKeys (so a
# regression to plain-InChIKey identity would NOT collapse them); the
# monkeypatch below simulates a tautomer-canonicalizer collapsing them onto
# one shared key, mirroring test_set_gflownet.py's _fake_taut_ik pattern.
_TAUT_A = "CCO"
_TAUT_B = "CCCO"
_DISTINCT = "c1ccccc1"


def _fake_taut_ik(smiles: str) -> str:
    """Collapse _TAUT_A/_TAUT_B onto one key; everything else keys to itself."""
    return "SHARED_TAUT_KEY" if smiles in (_TAUT_A, _TAUT_B) else smiles


def test_circles_count_known_answer():
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    # Sanity-check the hand-picked similarity assumption still holds against
    # whatever RDKit build is installed (keeps the test's own premise honest).
    mols = [Chem.MolFromSmiles(s) for s in (_HEXANE, _HEPTANE, _BENZENE)]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    sim_hexane_heptane = DataStructs.TanimotoSimilarity(fps[0], fps[1])
    sim_hexane_benzene = DataStructs.TanimotoSimilarity(fps[0], fps[2])
    assert sim_hexane_heptane > 0.6  # close enough to collapse at threshold=0.4
    assert sim_hexane_benzene < 0.1  # far enough to survive at threshold=0.9

    smiles = [_HEXANE, _HEPTANE, _BENZENE]
    assert diversity.circles_count(smiles, threshold=0.4) == 2
    assert diversity.circles_count(smiles, threshold=0.9) == 3


def test_circles_count_degenerate_inputs():
    assert diversity.circles_count([], threshold=0.4) == 0
    assert diversity.circles_count(["CCO"], threshold=0.4) == 1
    for threshold in (0.1, 0.4, 0.7, 0.99):
        assert diversity.circles_count(["CCO", "CCO", "CCO"], threshold=threshold) == 1


def test_dedup_to_budget_collapses_tautomer_variants(monkeypatch):
    monkeypatch.setattr(diversity, "_tautomer_inchikey", _fake_taut_ik)
    monkeypatch.setattr(metrics, "_tautomer_inchikey", _fake_taut_ik)

    out = diversity.dedup_to_budget([_TAUT_A, _TAUT_B, _DISTINCT], k=2)
    assert len(out) == 2
    # The freed slot (from the A/B tautomer collapse) is filled by the third,
    # genuinely-distinct molecule.
    assert _DISTINCT in out
    # Only one of the tautomer-collapsed pair occupies a slot.
    assert sum(1 for s in out if s in (_TAUT_A, _TAUT_B)) == 1


def test_union_at_k_curve_monotonic():
    # A ranked stream of 5 distinct molecules; the first 3 are annotated true.
    smiles = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO"]
    annotated_ik = {metrics._tautomer_inchikey(s) for s in smiles[:3]}

    curve = diversity.union_at_k_curve(smiles, annotated_ik, ks=(2, 3, 5))
    values = [curve[k] for k in (2, 3, 5)]
    assert values == sorted(values)  # non-decreasing in K
    assert curve[3] == 1.0  # all 3 annotated hits found by k=3
    assert curve[5] == 1.0  # no further gain past k=3 (monotone plateau)
    assert 0.0 < curve[2] < 1.0  # partial coverage at k=2


def test_diversity_and_recall_agree_on_molecule_identity(monkeypatch):
    """EVAL-04 guard: both the recall path and circles_count treat a
    tautomer-collapsed pair as ONE molecule, proving the dedup pre-pass runs
    inside the diversity function itself (not merely upstream in some other
    caller's dedup)."""
    monkeypatch.setattr(diversity, "_tautomer_inchikey", _fake_taut_ik)
    monkeypatch.setattr(metrics, "_tautomer_inchikey", _fake_taut_ik)

    # Recall path: metrics._tautomer_inchikey collapses the pair to one key.
    assert metrics._tautomer_inchikey(_TAUT_A) == metrics._tautomer_inchikey(_TAUT_B)

    # Diversity path: circles_count must ALSO treat them as one molecule.
    assert diversity.circles_count([_TAUT_A, _TAUT_B], threshold=0.99) == 1


def test_annotated_coverage_count_renamed_matches_old_modes_discovered():
    # Regression guard for the "pure rename, zero behavior change" claim:
    # a hand-built input with a known union-intersect-annotated count.
    sets = [frozenset({"A", "X"}), frozenset({"B", "Y"}), frozenset({"A"})]
    annotated_ik = {"A", "B", "C"}
    # A and B are found across the sampled sets; C never appears.
    assert diversity.annotated_coverage_count(sets, annotated_ik) == 2


def test_modes_discovered_canonical_gates_on_reward_and_tanimoto():
    # Hand-built candidates: some fail the tau gate outright, some pass the
    # gate but are within delta of an already-accepted mode.
    candidates = [_HEXANE, _HEPTANE, _BENZENE, "not_annotated_smiles"]
    rewarded = {_HEXANE, _HEPTANE, _BENZENE}  # "not_annotated_smiles" fails tau

    def reward_fn(x: str) -> float:
        return 1.0 if x in rewarded else 0.0

    # tau=1: only the three rewarded candidates survive the reward gate.
    # delta=0.7 (distance 0.3): hexane/heptane (sim ~0.875) are within delta
    # of each other -> sphere-exclusion collapses them to one; benzene
    # (dissimilar to both) survives as a second mode.
    count = diversity.modes_discovered_canonical(candidates, reward_fn, tau=1.0, delta=0.7)
    assert count == 2

    # Raising tau above every candidate's reward excludes everything.
    count_none = diversity.modes_discovered_canonical(candidates, reward_fn, tau=2.0, delta=0.7)
    assert count_none == 0


def test_auc_of_curve_known_answer():
    # Exact D-EVAL02-KGRID non-uniform grid with hand-picked distinct values.
    curve = {5: 0.10, 10: 0.20, 15: 0.30, 20: 0.40, 30: 0.50, 50: 0.60}
    ks = (5, 10, 15, 20, 30, 50)

    expected = sum(
        0.5 * (curve[ks[i]] + curve[ks[i + 1]]) * (ks[i + 1] - ks[i])
        for i in range(len(ks) - 1)
    ) / (50 - 5)

    auc = diversity.auc_of_curve(curve, k_min=5, k_max=50)
    assert abs(auc - expected) < 1e-9

    # Positively distinguish the dx-weighted trapezoid from a naive
    # equal-weight mean, which would over-weight the sparse K=30/50 tail.
    naive_mean = sum(curve.values()) / len(curve)
    assert abs(auc - naive_mean) > 1e-6

    # 2-point edge case pins the (k_max - k_min) normalization divisor.
    two_point = {5: 0.2, 50: 0.8}
    assert diversity.auc_of_curve(two_point, k_min=5, k_max=50) == 0.5 * (0.2 + 0.8)

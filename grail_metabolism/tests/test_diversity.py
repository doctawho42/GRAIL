from grail_metabolism.eval.diversity import (
    annotated_coverage_count,
    mean_pairwise_tanimoto,
    n_unique_scaffolds,
    set_size_calibration,
)


def test_modes_discovered_counts_distinct_hits_across_sets():
    sets = [frozenset({"A", "X"}), frozenset({"B", "Y"}), frozenset({"A"})]
    assert annotated_coverage_count(sets, annotated_ik={"A", "B", "C"}) == 2  # A, B found; C never


def test_mean_pairwise_tanimoto_identical_is_one():
    assert abs(mean_pairwise_tanimoto(["CCO", "CCO"]) - 1.0) < 1e-6


def test_modes_discovered_empty_sets_finds_nothing():
    assert annotated_coverage_count([], annotated_ik={"A", "B"}) == 0
    assert annotated_coverage_count([frozenset()], annotated_ik={"A", "B"}) == 0


def test_modes_discovered_ignores_unannotated_hits():
    sets = [frozenset({"Z"})]
    assert annotated_coverage_count(sets, annotated_ik={"A", "B"}) == 0


def test_mean_pairwise_tanimoto_distinct_molecules_below_one():
    # Ethanol vs. a much larger, structurally distinct molecule.
    value = mean_pairwise_tanimoto(["CCO", "c1ccccc1"])
    assert 0.0 <= value < 1.0


def test_mean_pairwise_tanimoto_single_molecule_is_one():
    # No pairs exist; documented convention matches the "identical -> 1.0" intuition.
    assert mean_pairwise_tanimoto(["CCO"]) == 1.0


def test_mean_pairwise_tanimoto_empty_list_is_one():
    assert mean_pairwise_tanimoto([]) == 1.0


def test_mean_pairwise_tanimoto_skips_unparseable_smiles():
    # "not_a_smiles" should be dropped rather than raising; result equals the
    # identical-pair case for the two valid "CCO" entries.
    value = mean_pairwise_tanimoto(["CCO", "not_a_smiles", "CCO"])
    assert abs(value - 1.0) < 1e-6


def test_mean_pairwise_tanimoto_all_unparseable_is_one():
    assert mean_pairwise_tanimoto(["not_a_smiles", "also_bad"]) == 1.0


def test_n_unique_scaffolds_counts_distinct_bemis_murcko_scaffolds():
    # Benzene and toluene share the same aromatic-ring scaffold; phenol shares it too.
    smiles = ["c1ccccc1C", "c1ccccc1", "c1ccccc1O"]
    assert n_unique_scaffolds(smiles) == 1


def test_n_unique_scaffolds_distinguishes_different_ring_systems():
    smiles = ["c1ccccc1C", "c1ccc2ccccc2c1"]  # benzene ring vs. naphthalene
    assert n_unique_scaffolds(smiles) == 2


def test_n_unique_scaffolds_empty_list_is_zero():
    assert n_unique_scaffolds([]) == 0


def test_n_unique_scaffolds_skips_unparseable_smiles():
    smiles = ["c1ccccc1C", "not_a_smiles"]
    assert n_unique_scaffolds(smiles) == 1


def test_set_size_calibration_matches_expected_difference():
    sets = [frozenset({"A", "B"}), frozenset({"A"})]  # mean size = 1.5
    annotated = {"A", "B", "C"}  # size = 3
    assert set_size_calibration(sets, annotated_ik=annotated) == 1.5 - 3


def test_set_size_calibration_zero_when_matched():
    sets = [frozenset({"A", "B"})]
    annotated = {"X", "Y"}
    assert set_size_calibration(sets, annotated_ik=annotated) == 0.0


def test_set_size_calibration_empty_sets_is_negative_annotated_count():
    assert set_size_calibration([], annotated_ik={"A", "B"}) == -2.0

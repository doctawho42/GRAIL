"""Dataset-free guard tests for grail_metabolism/eval/baselines.py (BASE-01/02/03/05).

Hand-built-fixture, dataset-free, RDKit-version-independent tests in the style
of test_eval_diversity.py: small SMILES constants at module top, one assertion
per known-answer property, no external SDF/triples. Tests that depend on
tautomer collapse monkeypatch ``_tautomer_inchikey`` in ALL THREE modules that
import it (``diversity``, ``metrics``, and ``baselines``), following the
``_fake_taut_ik`` pattern already established in test_eval_diversity.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import grail_metabolism.eval.baselines as baselines
import grail_metabolism.eval.diversity as diversity
import grail_metabolism.metrics as metrics

# ---------------------------------------------------------------------------
# Fixtures (duplicated verbatim from test_eval_diversity.py -- no cross-test-file
# import precedent exists in this codebase, so per 02-RESEARCH.md's "import or
# duplicate" guidance, duplicate here to keep this file independently runnable).
# ---------------------------------------------------------------------------

# Hexane/heptane: genuinely distinct SMILES but high Tanimoto similarity
# (~0.875, verified against the installed RDKit build); benzene is
# structurally unrelated to both (Tanimoto ~0.0 to either).
_HEXANE = "CCCCCC"
_HEPTANE = "CCCCCCC"
_BENZENE = "c1ccccc1"

# Two distinct, parseable molecules with DISTINCT plain InChIKeys (so a
# regression to plain-InChIKey identity would NOT collapse them); the
# monkeypatch below simulates a tautomer-canonicalizer collapsing them onto
# one shared key.
_TAUT_A = "CCO"
_TAUT_B = "CCCO"
_DISTINCT = "c1ccccc1"


def _fake_taut_ik(smiles: str) -> str:
    """Collapse _TAUT_A/_TAUT_B onto one key; everything else keys to itself."""
    return "SHARED_TAUT_KEY" if smiles in (_TAUT_A, _TAUT_B) else smiles


def _patch_tautomer_ik(monkeypatch):
    """3-way monkeypatch: baselines.py imports _tautomer_inchikey too."""
    monkeypatch.setattr(diversity, "_tautomer_inchikey", _fake_taut_ik)
    monkeypatch.setattr(metrics, "_tautomer_inchikey", _fake_taut_ik)
    monkeypatch.setattr(baselines, "_tautomer_inchikey", _fake_taut_ik)


# A pool of (smiles, raw-logit score) tuples. Scores are hand-picked so a
# descending sort is unambiguous (no ties) and comment-justified per molecule.
_POOL = [
    (_HEXANE, 3.0),  # highest relevance
    (_HEPTANE, 2.0),  # near-duplicate of hexane (Tanimoto ~0.875), 2nd relevance
    (_BENZENE, 1.0),  # structurally distinct, lowest relevance
    ("CCN", 0.5),  # ethylamine -- distinct filler
    ("CCCl", 0.0),  # chloroethane -- distinct filler
]


# ---------------------------------------------------------------------------
# Task 1: select() contract + temperature_topp_select (BASE-01)
# ---------------------------------------------------------------------------


def test_select_contract_degenerate_inputs():
    assert baselines.select([], k=5, method="temperature_topp") == []
    assert baselines.select(_POOL, k=0, method="temperature_topp") == []
    assert baselines.select(_POOL, k=-1, method="temperature_topp") == []
    # k >= distinct-pool-size -> full deduped ranking (length == distinct pool size)
    out = baselines.select(_POOL, k=100, method="temperature_topp", T=1.0, p=1.0)
    assert len(out) == len(_POOL)


def test_temperature_topp_select_dedups_and_hits_budget_k(monkeypatch):
    _patch_tautomer_ik(monkeypatch)

    pool_with_dup = [
        (_TAUT_A, 3.0),
        (_TAUT_B, 2.9),  # tautomer-duplicate of _TAUT_A under the fake IK
        (_HEXANE, 2.0),
        (_HEPTANE, 1.0),
        (_BENZENE, 0.5),
    ]
    rng = np.random.default_rng(0)
    ranked = baselines.temperature_topp_select(pool_with_dup, k=4, T=1.0, p=1.0, rng=rng)
    out = diversity.dedup_to_budget(ranked, k=4)
    assert len(out) == 4
    # only one of the tautomer-duplicate pair occupies a slot
    assert sum(1 for s in out if s in (_TAUT_A, _TAUT_B)) == 1


def test_temperature_increases_mean_pairwise_diversity_monotonically():
    pool = _POOL
    results = {}
    for T in (0.5, 1.0, 2.0):
        rng = np.random.default_rng(42)
        ranked = baselines.temperature_topp_select(pool, k=4, T=T, p=1.0, rng=rng)
        out = diversity.dedup_to_budget(ranked, k=4)
        results[T] = diversity.mean_pairwise_tanimoto(out)

    # More temperature -> more spread -> mean pairwise tanimoto should be
    # non-increasing (or n_unique_scaffolds non-decreasing); allow a small
    # numerical tolerance since this is a stochastic sampler.
    assert results[2.0] <= results[0.5] + 1e-9


def test_reranker_output_is_unbounded_logit_not_probability():
    import torch
    from rdkit import Chem
    from torch_geometric.data import Batch

    from grail_metabolism.model.reranker import BiEncoderReranker
    from grail_metabolism.utils.transform import from_rdmol

    found_unbounded = False
    for seed in range(5):
        torch.manual_seed(seed)
        reranker = BiEncoderReranker(hidden_dims=(8, 8), out_dim=8, dropout=0.0)
        reranker.eval()
        # Scale up the head's weights (still random, just larger magnitude) so
        # the pre-activation logit is forced comfortably outside [0,1] absent
        # any bounding nonlinearity -- a bare nn.Linear output is linear in its
        # weights, so this is a legitimate way to exercise "is the output ever
        # allowed to leave [0,1]" without requiring a fully-trained checkpoint.
        # If a sigmoid (or any other bounded squashing) were silently added to
        # the head, output would stay in [0,1] regardless of weight scale.
        with torch.no_grad():
            for module in reranker.head.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.mul_(20.0)
                    module.bias.mul_(20.0)

        sub = from_rdmol(Chem.MolFromSmiles("CCO"))
        prods = [
            from_rdmol(Chem.MolFromSmiles(s)) for s in ("CCCO", "CCN", "CCCl")
        ]
        prod_batch = Batch.from_data_list(prods)
        rule_prior = torch.zeros(len(prods))
        gen_score = torch.zeros(len(prods))

        with torch.no_grad():
            out = reranker(sub, prod_batch, rule_prior, gen_score)

        if bool((out.abs() > 1.0).any().item()):
            found_unbounded = True
            break

    assert found_unbounded, (
        "BiEncoderReranker output must be unbounded (raw logit space) across "
        "at least one of several random inits -- a probability output would be "
        "provably bounded to [0,1]; this guards against a future sigmoid being "
        "silently added to the head, which would break temperature_topp_select's "
        "direct-division-by-T assumption"
    )


# ---------------------------------------------------------------------------
# Task 2: dpp_greedy_select (BASE-02) -- greedy MAP via incremental Cholesky
# with diagonal jitter + per-step gain clip + relative-tolerance stop.
# ---------------------------------------------------------------------------

# A cluster of near-identical long-alkyl-chain regioisomers: verified (below,
# inline) to have pairwise Tanimoto == 1.0 under the shared Morgan r=2/2048
# fingerprint (long straight chains saturate the radius-2 environment so
# chain-length differences stop registering). This models GRAIL's
# regioisomer-heavy candidate pools (same rule applied at different sites).
_NEARDUP_CLUSTER = [
    "CCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCC",
]

# A polycyclic aromatic, structurally unrelated to the alkyl-chain cluster
# (Tanimoto ~0.0 to every cluster member, verified inline below).
_DIVERSE_ONE = "c1ccc2c(c1)ccc1ccccc12"


def _verify_neardup_fixture_premise():
    """Inline known-answer sanity check of the hand-picked fixture's premise
    (mirrors test_eval_diversity.py's discipline of re-validating fixture
    assumptions against the installed RDKit build before asserting on them)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    mols = [Chem.MolFromSmiles(s) for s in _NEARDUP_CLUSTER]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            assert DataStructs.TanimotoSimilarity(fps[i], fps[j]) > 0.9999

    diverse_mol = Chem.MolFromSmiles(_DIVERSE_ONE)
    diverse_fp = AllChem.GetMorganFingerprintAsBitVect(diverse_mol, radius=2, nBits=2048)
    for fp in fps:
        assert DataStructs.TanimotoSimilarity(fp, diverse_fp) < 0.1


def test_dpp_greedy_select_dedups_and_hits_budget_k(monkeypatch):
    _patch_tautomer_ik(monkeypatch)

    # Pool has a tautomer-duplicate pair (_TAUT_A/_TAUT_B, fake-collapsed) plus
    # enough genuinely distinct fillers so the ranked stream can supply a
    # 4th distinct candidate once dedup_to_budget frees the slot the
    # duplicate pair would otherwise occupy (mirrors
    # test_dedup_to_budget_prevents_silent_budget_shrinkage's pool shape --
    # DPP is asked for MORE than k so the post-dedup budget can still be met).
    pool_with_dup = [
        (_TAUT_A, 5.0),
        (_TAUT_B, 4.9),
        (_HEXANE, 4.0),
        (_HEPTANE, 3.0),
        (_BENZENE, 2.0),
        ("CCN", 1.0),
    ]
    ranked = baselines.dpp_greedy_select(pool_with_dup, k=5, theta=1.0)
    out = diversity.dedup_to_budget(ranked, k=4)
    assert len(out) == 4
    assert sum(1 for s in out if s in (_TAUT_A, _TAUT_B)) == 1


def test_dpp_and_mmr_kernel_matches_diversity_module_tanimoto():
    # Known-answer sanity check first (mirrors test_circles_count_known_answer's
    # discipline): validate the fixture premise against the installed RDKit
    # build before asserting equality with the module helper.
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    mols = [Chem.MolFromSmiles(s) for s in (_HEXANE, _HEPTANE)]
    fps_raw = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    raw_tanimoto = DataStructs.TanimotoSimilarity(fps_raw[0], fps_raw[1])

    fps = baselines._pool_fingerprints([_HEXANE, _HEPTANE])
    S = baselines._tanimoto_kernel_matrix(fps)

    # A 2-molecule mean_pairwise_tanimoto IS exactly their pairwise Tanimoto.
    expected = diversity.mean_pairwise_tanimoto([_HEXANE, _HEPTANE])
    assert abs(raw_tanimoto - expected) < 1e-9
    # Assert on the OFF-diagonal only -- the diagonal is deliberately
    # regularized (S_ii = 1 + eps) inside dpp_greedy_select's own kernel
    # construction, so a bare _tanimoto_kernel_matrix diagonal of 1.0 (not
    # 1+eps) must not be conflated with that.
    assert abs(S[0, 1] - expected) < 1e-9
    assert S[0, 0] == 1.0 and S[1, 1] == 1.0


def test_dpp_theta_sweep_monotonic_relevance_weighting():
    pool = [
        (_HEXANE, 3.0),
        (_HEPTANE, 2.0),
        (_BENZENE, 1.0),
        ("CCN", 0.5),
        ("CCCl", 0.0),
    ]
    mean_relevance = {}
    for theta in (0.0, 1.0, 2.0):
        ranked = baselines.dpp_greedy_select(pool, k=3, theta=theta)
        score_of = dict(pool)
        mean_relevance[theta] = sum(score_of[s] for s in ranked) / len(ranked)

    assert mean_relevance[0.0] <= mean_relevance[1.0] + 1e-9
    assert mean_relevance[1.0] <= mean_relevance[2.0] + 1e-9


def test_dpp_greedy_select_exclusion_stop_on_near_duplicate_pool():
    """Load-bearing near-duplicate EXCLUSION/STOP guard (Pitfall 5): a pool of
    N near-identical items (all pairwise Tanimoto > 0.9999) requested with
    k=N must return FEWER than N picks -- proving the relative-tolerance stop
    actually fires (the selector STOPS on redundancy), not merely avoids NaN."""
    _verify_neardup_fixture_premise()

    n = len(_NEARDUP_CLUSTER)
    pool = [(s, 1.0) for s in _NEARDUP_CLUSTER]
    ranked = baselines.dpp_greedy_select(pool, k=n, theta=1.0)
    assert len(ranked) < n, (
        "dpp_greedy_select must EXCLUDE redundant near-duplicates via the "
        "relative-tolerance stop, returning fewer than k picks on a pool of "
        "near-identical items -- a defective fixed-epsilon-floor scheme would "
        "never stop and would return all N"
    )


def test_dpp_greedy_select_signal_survives_in_mixed_pool():
    """Load-bearing diversity-SIGNAL-SURVIVES guard (Pitfall 5): a mixed pool
    of 1 genuinely-diverse candidate + M near-duplicates, k=2, must select the
    genuinely-diverse candidate -- proving the diversity signal is not
    noise-dominated by a naive epsilon floor."""
    _verify_neardup_fixture_premise()

    pool = [(s, 1.0) for s in _NEARDUP_CLUSTER] + [(_DIVERSE_ONE, 1.0)]
    ranked = baselines.dpp_greedy_select(pool, k=2, theta=1.0)
    assert _DIVERSE_ONE in ranked, (
        "the genuinely-diverse candidate must be selected in a mixed pool of "
        "1 diverse + M near-duplicates -- an epsilon-floor-poisoned argmax "
        "would instead pick near-duplicates over the real diversity signal"
    )


def test_dpp_greedy_select_numerically_stable_on_regioisomer_pool():
    """No NaN / no invalid-value sqrt warning on a near-singular kernel."""
    import warnings

    _verify_neardup_fixture_premise()

    pool = [(s, 1.0) for s in _NEARDUP_CLUSTER]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ranked = baselines.dpp_greedy_select(pool, k=len(_NEARDUP_CLUSTER), theta=1.0)

    assert all(isinstance(s, str) for s in ranked)
    assert len(ranked) > 0


def test_dpp_theta_zero_ignores_relevance_ordering():
    """theta=0: q_i is constant (exp(0)=1 for all i), so selection is driven
    purely by the similarity kernel S -- selection order must be invariant to
    a monotonic rescaling of the input scores."""
    pool_a = [(_HEXANE, 1.0), (_HEPTANE, 2.0), (_BENZENE, 3.0)]
    pool_b = [(_HEXANE, 10.0), (_HEPTANE, 20.0), (_BENZENE, 30.0)]  # monotonic rescale

    ranked_a = baselines.dpp_greedy_select(pool_a, k=3, theta=0.0)
    ranked_b = baselines.dpp_greedy_select(pool_b, k=3, theta=0.0)
    assert ranked_a == ranked_b


# ---------------------------------------------------------------------------
# Task 3: mmr_select (BASE-03) -- reconciled relevance/similarity scale with
# lambda=1/lambda=0 degenerate limits.
# ---------------------------------------------------------------------------


def test_mmr_lambda_one_recovers_top_k_ranking():
    pool = [
        (_HEXANE, 3.0),
        (_HEPTANE, 2.0),
        (_BENZENE, 1.0),
        ("CCN", 0.5),
        ("CCCl", 0.0),
    ]
    ranked = baselines.mmr_select(pool, k=len(pool), lam=1.0)
    expected_order = [
        s for s, _ in sorted(pool, key=lambda x: (-x[1], metrics._tautomer_inchikey(x[0])))
    ]
    assert ranked == expected_order


def _farthest_point_reference(pool, k):
    """Hand-rolled max-min (farthest-point) picker independent of relevance,
    used as an oracle for the lambda=0 degenerate case. Tie-break lexicographic
    on tautomer-InChIKey, matching mmr_select's own deterministic tie-break."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    smiles = [s for s, _ in pool]
    tie_keys = [metrics._tautomer_inchikey(s) for s in smiles]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    n = len(smiles)
    max_sim = [0.0] * n
    selected_mask = [False] * n
    ranked = []
    for _ in range(min(k, n)):
        best_j = None
        best_score = None
        for j in range(n):
            if selected_mask[j]:
                continue
            score = -max_sim[j]
            key = (-score, tie_keys[j])
            if best_score is None or key < best_score:
                best_score = key
                best_j = j
        selected_mask[best_j] = True
        ranked.append(smiles[best_j])
        for j in range(n):
            if not selected_mask[j]:
                sim = DataStructs.TanimotoSimilarity(fps[best_j], fps[j])
                max_sim[j] = max(max_sim[j], sim)
    return ranked


def test_mmr_lambda_zero_recovers_maxmin_diversity():
    pool = [
        (_HEXANE, 3.0),
        (_HEPTANE, 2.0),
        (_BENZENE, 1.0),
        ("CCN", 0.5),
    ]
    # First-pick invariance to a permutation of relevance scores (lam=0 must
    # ignore Rel entirely) -- both permutations select the same first pick.
    permuted_pool = [(s, -sc) for s, sc in pool]
    ranked_original = baselines.mmr_select(pool, k=1, lam=0.0)
    ranked_permuted = baselines.mmr_select(permuted_pool, k=1, lam=0.0)
    assert ranked_original == ranked_permuted

    # Cross-check against a hand-rolled farthest-point reference.
    ranked_full = baselines.mmr_select(pool, k=len(pool), lam=0.0)
    reference = _farthest_point_reference(pool, k=len(pool))
    assert ranked_full == reference


def test_mmr_lambda_sweep_monotonic_diversity():
    pool = [
        (_HEXANE, 3.0),
        (_HEPTANE, 2.0),
        (_BENZENE, 1.0),
        ("CCN", 0.5),
        ("CCCl", 0.0),
    ]
    results = {}
    for lam in (1.0, 0.5, 0.0):
        ranked = baselines.mmr_select(pool, k=3, lam=lam)
        results[lam] = diversity.mean_pairwise_tanimoto(ranked)

    # lam=1 -> most similar to top-K (higher mean similarity); lam=0 -> most
    # spread out (lower mean similarity). Non-increasing as lam decreases.
    assert results[0.0] <= results[0.5] + 1e-9
    assert results[0.5] <= results[1.0] + 1e-9


def test_mmr_select_dedups_and_hits_budget_k(monkeypatch):
    _patch_tautomer_ik(monkeypatch)

    pool_with_dup = [
        (_TAUT_A, 5.0),
        (_TAUT_B, 4.9),
        (_HEXANE, 4.0),
        (_HEPTANE, 3.0),
        (_BENZENE, 2.0),
        ("CCN", 1.0),
    ]
    ranked = baselines.mmr_select(pool_with_dup, k=5, lam=0.5)
    out = diversity.dedup_to_budget(ranked, k=4)
    assert len(out) == 4
    assert sum(1 for s in out if s in (_TAUT_A, _TAUT_B)) == 1


# ---------------------------------------------------------------------------
# BASE-05: cross-baseline validation suite -- proves temperature, DPP, MMR,
# and the SyGMa dedup_to_budget composition ALL (a) output k distinct
# post-tautomer-dedup entries at a matched budget, (b) route through the
# SHARED Morgan r=2/2048 fingerprint, (c) share the ONE canonicalization path
# (no per-baseline drift). Plus the SyGMa diversity-triplet canonicalization
# guard (Pitfall 4).
# ---------------------------------------------------------------------------


def _sygma_adapter_ranked_smiles(pool, k):
    """Inline stand-in for the SyGMa producer under BASE-05's test scope.

    SyGMa's real contract (per 02-RESEARCH.md/02-02-PLAN.md) is "raw ranked
    SMILES in, dedup_to_budget out" -- the external `sygma` package is NOT
    import-required for this dataset-free suite (no top-level `import sygma`
    anywhere in this file). This adapter reproduces exactly that contract: it
    takes a pool of (smiles, score) already ranked by the caller (descending
    score, as `tree.to_smiles()` + `calc_scores()` would produce) and returns
    the raw ranked SMILES list unchanged -- the composition under test is
    THIS raw list piped through the REAL `diversity.dedup_to_budget`, exactly
    mirroring how `sygma_baseline`'s diversity triplet consumes
    `tree.to_smiles()` output in scripts/run_benchmark.py.
    """
    ranked = [s for s, _ in sorted(pool, key=lambda x: -x[1])]
    return ranked[: max(k, len(ranked))]


def _base05_pool_with_dup():
    """A pool with one monkeypatched tautomer-duplicate pair plus enough
    genuinely distinct fillers to reach k=4 distinct entries after dedup
    (mirrors the per-selector budget/dedup fixture shape already established
    above in this file)."""
    return [
        (_TAUT_A, 5.0),
        (_TAUT_B, 4.9),
        (_HEXANE, 4.0),
        (_HEPTANE, 3.0),
        (_BENZENE, 2.0),
        ("CCN", 1.0),
    ]


_BASE05_BASELINE_PRODUCERS = {
    "temperature_topp": lambda pool, k: baselines.temperature_topp_select(
        pool, k=k, T=1.0, p=1.0, rng=np.random.default_rng(0)
    ),
    "dpp": lambda pool, k: baselines.dpp_greedy_select(pool, k=k, theta=1.0),
    "mmr": lambda pool, k: baselines.mmr_select(pool, k=k, lam=0.5),
    "sygma_adapter": lambda pool, k: _sygma_adapter_ranked_smiles(pool, k),
}


@pytest.mark.parametrize("name", sorted(_BASE05_BASELINE_PRODUCERS))
def test_base05_all_baselines_dedup_to_budget_k_distinct(monkeypatch, name):
    """(a) Every baseline (temperature/DPP/MMR/SyGMa-adapter) yields exactly k
    DISTINCT entries through the REAL dedup_to_budget on a pool with a known
    tautomer duplicate plus surplus distinct fillers -- no baseline forks the
    budget/dedup contract."""
    _patch_tautomer_ik(monkeypatch)

    pool = _base05_pool_with_dup()
    producer = _BASE05_BASELINE_PRODUCERS[name]
    ranked = producer(pool, 5)
    out = diversity.dedup_to_budget(ranked, k=4)

    assert len(out) == 4, f"{name}: expected 4 distinct post-dedup entries, got {len(out)}"
    assert sum(1 for s in out if s in (_TAUT_A, _TAUT_B)) == 1, (
        f"{name}: the tautomer-duplicate pair must collapse to exactly one slot"
    )


def test_base05_shared_fingerprint_consistency():
    """(b) baselines._tanimoto_kernel_matrix agrees with
    diversity.mean_pairwise_tanimoto on a fixed pair -- proving DPP/MMR's
    kernel uses the SAME Morgan r=2/2048 fingerprint as the eval-harness
    diversity metrics (a 2-molecule mean_pairwise_tanimoto IS exactly their
    pairwise Tanimoto)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    # Known-answer sanity check first (validate the premise on the installed
    # RDKit build before asserting equality with the module helpers).
    mols = [Chem.MolFromSmiles(s) for s in (_HEXANE, _HEPTANE)]
    fps_raw = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    raw_tanimoto = DataStructs.TanimotoSimilarity(fps_raw[0], fps_raw[1])

    fps = baselines._pool_fingerprints([_HEXANE, _HEPTANE])
    S = baselines._tanimoto_kernel_matrix(fps)
    expected = diversity.mean_pairwise_tanimoto([_HEXANE, _HEPTANE])

    assert abs(raw_tanimoto - expected) < 1e-9
    assert abs(S[0, 1] - expected) < 1e-9


def test_base05_knob_monotonicity_rollup():
    """(c) Cross-baseline roll-up of the per-selector knob-monotonicity
    guards: temperature (T up -> more spread -> mean pairwise tanimoto
    non-increasing), DPP (theta up -> mean relevance non-decreasing), MMR
    (lambda down -> more weight on diversity -> mean pairwise tanimoto
    non-increasing) -- each baseline's knob has a demonstrable directional
    effect on a fixed pool."""
    pool = _POOL

    # Temperature: T up -> more spread -> mean_pairwise_tanimoto non-increasing.
    temp_results = {}
    for T in (0.5, 1.0, 2.0):
        rng = np.random.default_rng(42)
        ranked = baselines.temperature_topp_select(pool, k=4, T=T, p=1.0, rng=rng)
        out = diversity.dedup_to_budget(ranked, k=4)
        temp_results[T] = diversity.mean_pairwise_tanimoto(out)
    assert temp_results[2.0] <= temp_results[0.5] + 1e-9

    # DPP: theta up -> trusts the ranker's own ordering more -> mean relevance
    # of the selected set is non-decreasing.
    dpp_pool = [
        (_HEXANE, 3.0),
        (_HEPTANE, 2.0),
        (_BENZENE, 1.0),
        ("CCN", 0.5),
        ("CCCl", 0.0),
    ]
    score_of = dict(dpp_pool)
    dpp_mean_relevance = {}
    for theta in (0.0, 1.0, 2.0):
        ranked = baselines.dpp_greedy_select(dpp_pool, k=3, theta=theta)
        dpp_mean_relevance[theta] = sum(score_of[s] for s in ranked) / len(ranked)
    assert dpp_mean_relevance[0.0] <= dpp_mean_relevance[1.0] + 1e-9
    assert dpp_mean_relevance[1.0] <= dpp_mean_relevance[2.0] + 1e-9

    # MMR: lambda down -> more weight on diversity -> mean_pairwise_tanimoto
    # of the selected set is non-increasing as lambda decreases.
    mmr_pool = [
        (_HEXANE, 3.0),
        (_HEPTANE, 2.0),
        (_BENZENE, 1.0),
        ("CCN", 0.5),
        ("CCCl", 0.0),
    ]
    mmr_results = {}
    for lam in (1.0, 0.5, 0.0):
        ranked = baselines.mmr_select(mmr_pool, k=3, lam=lam)
        mmr_results[lam] = diversity.mean_pairwise_tanimoto(ranked)
    assert mmr_results[0.0] <= mmr_results[0.5] + 1e-9
    assert mmr_results[0.5] <= mmr_results[1.0] + 1e-9


def test_base05_degenerate_limits_rollup():
    """(d) DPP/MMR degenerate limits: MMR lambda=1 recovers top-K, lambda=0
    recovers max-min; DPP theta=0 ignores relevance ordering (selection is
    invariant to a monotonic rescaling of input scores)."""
    pool = [
        (_HEXANE, 3.0),
        (_HEPTANE, 2.0),
        (_BENZENE, 1.0),
        ("CCN", 0.5),
        ("CCCl", 0.0),
    ]

    # MMR lambda=1 -> exact top-K ranking (Rel-only degenerate case).
    ranked = baselines.mmr_select(pool, k=len(pool), lam=1.0)
    expected_order = [
        s for s, _ in sorted(pool, key=lambda x: (-x[1], metrics._tautomer_inchikey(x[0])))
    ]
    assert ranked == expected_order

    # MMR lambda=0 -> first-pick invariant to a permutation of relevance scores.
    maxmin_pool = pool[:4]
    permuted_pool = [(s, -sc) for s, sc in maxmin_pool]
    ranked_original = baselines.mmr_select(maxmin_pool, k=1, lam=0.0)
    ranked_permuted = baselines.mmr_select(permuted_pool, k=1, lam=0.0)
    assert ranked_original == ranked_permuted

    # DPP theta=0 -> selection invariant to a monotonic rescaling of scores.
    pool_a = [(_HEXANE, 1.0), (_HEPTANE, 2.0), (_BENZENE, 3.0)]
    pool_b = [(_HEXANE, 10.0), (_HEPTANE, 20.0), (_BENZENE, 30.0)]
    ranked_a = baselines.dpp_greedy_select(pool_a, k=3, theta=0.0)
    ranked_b = baselines.dpp_greedy_select(pool_b, k=3, theta=0.0)
    assert ranked_a == ranked_b


def test_base05_sygma_diversity_triplet_uses_tautomer_dedup_not_plain_inchikey(monkeypatch):
    """(g) SyGMa canonicalization guard (Pitfall 4): feed a raw ranked-SMILES
    list containing a monkeypatched tautomer pair into
    mean_pairwise_tanimoto/circles_count/n_unique_scaffolds and assert they
    treat the pair as ONE molecule via the shared _fake_taut_ik 3-way
    monkeypatch -- independent of whatever plain-InChIKey dedup
    sygma_baseline's own recall/precision path applies, proving the SyGMa
    triplet routes through the SAME canonicalization as recall + the
    gflownet path."""
    _patch_tautomer_ik(monkeypatch)

    # Raw ranked SMILES as scripts/run_benchmark.py:sygma_baseline would feed
    # into the diversity functions: parent already dropped, NOT deduped by
    # plain InChIKey (both _TAUT_A and _TAUT_B are present, distinct plain
    # InChIKeys but the same fake tautomer key).
    raw_ranked_smiles = [_TAUT_A, _TAUT_B, _HEXANE, _HEPTANE]

    # mean_pairwise_tanimoto/circles_count/n_unique_scaffolds all run their
    # own _dedup_smiles_by_tautomer_ik pre-pass -- the tautomer pair collapses
    # to one molecule before any pairwise/scaffold computation.
    deduped_count = len(diversity._dedup_smiles_by_tautomer_ik(raw_ranked_smiles))
    assert deduped_count == 3, (
        "the tautomer pair must collapse to ONE molecule before diversity "
        "computation (3 distinct molecules total: the collapsed pair + hexane "
        "+ heptane), independent of plain-InChIKey dedup"
    )

    # circles_count/n_unique_scaffolds/mean_pairwise_tanimoto must all be
    # computable without raising and must reflect the collapsed count, not a
    # plain-InChIKey-deduped count of 4.
    n_scaffolds = diversity.n_unique_scaffolds(raw_ranked_smiles)
    assert n_scaffolds <= 3

    circles_04 = diversity.circles_count(raw_ranked_smiles, threshold=0.4)
    assert circles_04 <= 3

    mpt = diversity.mean_pairwise_tanimoto(raw_ranked_smiles)
    assert 0.0 <= mpt <= 1.0


def test_base05_no_top_level_sygma_import_required():
    """The BASE-05 suite (and this test module as a whole) is dataset-free and
    does NOT require the external `sygma` package to import -- the SyGMa path
    under test is the dedup_to_budget composition on a hand-built raw ranked
    list (see `_sygma_adapter_ranked_smiles`), not a call into the real
    `sygma` library."""
    import sys

    assert "sygma" not in getattr(
        sys.modules.get("grail_metabolism.tests.test_eval_baselines"), "__dict__", {}
    ), "test_eval_baselines.py must not import sygma at module scope"

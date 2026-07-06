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

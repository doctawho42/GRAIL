"""Regression tests for the audit fixes (correctness, calibration, reproducibility)."""
from __future__ import annotations

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch

from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.model.filter import Filter
from grail_metabolism.model.grail import summon_the_grail
from grail_metabolism.model.train_model import PULoss
from grail_metabolism.utils.preparation import (
    MolFrame,
    iscorrect,
    load_default_rules,
    load_phase2_rules,
    resolve_default_rule_bank,
)
from grail_metabolism.utils.seed import seed_everything
from grail_metabolism.utils.transform import from_pair

RULE = "[CH2:1][OH:2]>>[CH:1]=[O:2]"


def _cross_edges(graph, n_sub):
    edges = set()
    for k in range(graph.edge_index.size(1)):
        a, b = int(graph.edge_index[0, k]), int(graph.edge_index[1, k])
        if (a < n_sub) != (b < n_sub) and float(graph.edge_attr[k].abs().sum()) == 0.0:
            edges.add((a, b) if a < b else (b, a))
    return edges


def test_mcs_cross_edges_connect_corresponding_elements():
    # aniline -> 4-aminophenol: every alignment edge must join same-element atoms.
    sub = Chem.MolFromSmiles("c1ccc(N)cc1")
    prod = Chem.MolFromSmiles("Nc1ccc(O)cc1")
    graph = from_pair(sub, prod)
    n_sub = sub.GetNumAtoms()
    cross = _cross_edges(graph, n_sub)
    assert cross, "expected MCS alignment cross-edges"
    for lo, hi in cross:
        assert sub.GetAtomWithIdx(lo).GetSymbol() == prod.GetAtomWithIdx(hi - n_sub).GetSymbol()


def test_iscorrect_keeps_small_metabolites_drops_lone_atoms():
    assert iscorrect("C=O")  # formaldehyde (2 heavy atoms)
    assert iscorrect("OC=O")  # formate
    assert iscorrect("CCO")  # ethanol
    assert not iscorrect("O")  # water (lone heavy atom)
    assert not iscorrect("[Cl-]")  # chloride leaving group


def test_filter_return_logits_is_logit_domain():
    graph = from_pair(Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC=O"))
    model = Filter(18, 18, [32, 64, 32, 64, 32, 16], mode="pair")
    model.eval()  # disable dropout so the two forwards are comparable
    batch = Batch.from_data_list([graph])
    prob = model(batch)
    logit = model(batch, return_logits=True)
    assert 0.0 <= float(prob) <= 1.0
    assert torch.allclose(torch.sigmoid(logit), prob, atol=1e-5)


def test_puloss_trains_on_logits():
    # Before the fix, probabilities were fed into a logit-domain surrogate (double
    # sigmoid), collapsing the loss range and killing the gradient.
    seed_everything(0)
    model = Filter(18, 18, [32, 64, 32, 64, 32, 16], mode="pair")
    crit = PULoss(0.5)
    pos = from_pair(Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC=O"))
    pos.y = torch.tensor([1.0])
    neg = from_pair(Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCO"))
    neg.y = torch.tensor([0.0])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = last = None
    for _ in range(30):
        batch = Batch.from_data_list([pos, neg])
        out = model(batch, return_logits=True)
        loss = crit(out, batch.y.view(-1, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
        last = float(loss)
    assert last < first  # the loss actually moves now


def test_score_batch_matches_per_item_score():
    seed_everything(0)
    model = Filter(18, 18, [32, 64, 32, 64, 32, 16], mode="pair")
    model.eval()
    prods = ["CC=O", "CCO", "CC(=O)O"]
    batched = model.score_batch("CCO", prods)
    per_item = [model.score("CCO", p) for p in prods]
    assert max(abs(a - b) for a, b in zip(batched, per_item)) < 1e-6


def test_seed_everything_makes_training_reproducible():
    frame = MolFrame(pd.DataFrame([{"sub": "CCO", "prod": "CC=O", "real": 1},
                                   {"sub": "CCO", "prod": "CCO", "real": 0}]))
    frame.full_setup(rules=[RULE], include_pair_graphs=False, include_morgan=False)

    def run():
        seed_everything(123)
        model = summon_the_grail([RULE])
        model.generator.fit(frame, eps=3, verbose=False)
        return list(model.generator.loss_history_)

    assert run() == run()


def test_default_rule_bank_is_consistent_across_entry_points():
    from grail_metabolism.experiments.presets import _default_rules_path

    bank = resolve_default_rule_bank()
    assert bank is not None
    assert _default_rules_path().endswith(bank.name)
    with open(bank) as handle:
        file_rules = [line.strip() for line in handle if line.strip()]
    assert load_default_rules() == file_rules


def test_phase2_rules_all_compile_and_fire():
    from rdkit.Chem import AllChem

    rules = load_phase2_rules()
    assert len(rules) >= 10
    for rule in rules:
        rxn = AllChem.ReactionFromSmarts(rule)
        assert rxn is not None and rxn.GetNumReactantTemplates() >= 1


def test_metrics_inchikey_matching_and_output_size():
    # Two equivalent SMILES for acetic acid: exact string match misses, InChIKey catches.
    preds = [{"predicted": ["OC(=O)C", "c1ccccc1"], "real": ["CC(O)=O"]}]
    exact = aggregate_prediction_metrics(preds, ks=[1], match="exact")
    inchi = aggregate_prediction_metrics(preds, ks=[1], match="inchikey")
    assert exact["recall"] == 0.0
    assert inchi["recall"] == 1.0
    assert inchi["mean_output_size"] == 2.0


def test_generate_respects_max_output_cap():
    seed_everything(0)
    model = summon_the_grail([RULE])
    model.filter.calibrated_threshold = 0.0  # accept everything

    def fake_scored(sub, top_k=None, threshold=None):
        return [("CCO", 0.9), ("CC=O", 0.8), ("CC(=O)O", 0.7)]

    model.generator.generate_scored = fake_scored
    assert len(model.generate("CCO", max_output=2)) == 2
    assert len(model.generate("CCO")) == 3


def test_eval_prior_strength_override():
    # The eval-time prior_strength override must set the generator's prior weight when given
    # and leave it untouched when None (the deploy of the prior-vs-learned finding).
    from grail_metabolism.config import EvaluationConfig
    from grail_metabolism.workflows.evaluation import _apply_prior_strength

    gen = summon_the_grail([RULE]).generator
    gen.prior_strength = 0.4
    _apply_prior_strength(gen, EvaluationConfig(prior_strength=8.0))
    assert gen.prior_strength == 8.0
    _apply_prior_strength(gen, EvaluationConfig(prior_strength=None))  # None = leave as-is
    assert gen.prior_strength == 8.0


def test_match_protocols_disagree_rank_flip():
    # The same prediction is scored "correct" or "wrong" depending purely on the match
    # protocol each paper uses -- the match-sensitivity phenomenon the benchmark is built on.
    # predicted = D-alanine + acetone enol; real = L-alanine + acetone keto.
    preds = [{"predicted": ["C[C@@H](N)C(=O)O", "CC(O)=C"], "real": ["C[C@H](N)C(=O)O", "CC(=O)C"]}]
    rec = lambda m: aggregate_prediction_metrics(preds, ks=[2], match=m)["recall"]
    assert rec("inchikey") == 0.0           # strict: stereo + tautomer both miss
    assert rec("inchi_no_stereo") == 0.5    # GLORYx stereo-blind: alanine matches, tautomer doesn't
    assert rec("tanimoto1") == 0.5          # MetaTrans Tanimoto=1: same
    assert rec("inchikey_tautomer") == 1.0  # tautomer+stereo collapse: both match


def test_rule_prior_logits_persist_through_state_dict():
    # Empirical per-rule priors (SyGMa-style log-odds, learned in _update_rule_statistics)
    # must survive save/reload. They were persistent=False, so state_dict dropped them and
    # reloaded models ran with zeroed priors (lost ~0.03 recall@15).
    seed_everything(0)
    gen = summon_the_grail([RULE]).generator
    with torch.no_grad():
        gen.rule_prior_logits.copy_(torch.full_like(gen.rule_prior_logits, 1.234))
    assert "rule_prior_logits" in gen.state_dict()  # now persisted
    gen2 = summon_the_grail([RULE]).generator
    gen2.load_state_dict(gen.state_dict())
    assert torch.allclose(gen2.rule_prior_logits, torch.full_like(gen2.rule_prior_logits, 1.234))


def test_rule_embedding_cache_consistent_and_invalidated():
    # Inference caches the encoded rule bank (the dominant per-substrate cost): scoring is
    # deterministic (eval mode, dropout off) and the cached tensor is reused across calls;
    # a grad-enabled (training) forward invalidates it so weight updates aren't masked.
    from torch_geometric.data import Batch

    seed_everything(0)
    gen = summon_the_grail([RULE]).generator
    s1 = gen.score_rules("CCO")
    assert gen._rule_embedding_cache is not None
    cached = gen._rule_embedding_cache
    s2 = gen.score_rules("CCO")
    assert gen._rule_embedding_cache is cached          # reused, not re-encoded
    assert (s1 == s2).all()                              # deterministic inference

    graph = gen._graph_for_substrate("CCO")[1]
    gen.train()
    with torch.enable_grad():
        gen(Batch.from_data_list([graph]))
    assert gen._rule_embedding_cache is None             # training forward invalidates the cache


def test_tautomer_match_recovers_hits_plain_inchikey_misses():
    # Acetone keto (CC(=O)C) vs its enol (CC(O)=C): standard InChI does NOT normalize
    # this keto-enol pair, so plain "inchikey" matching misses it; tautomer
    # canonicalization collapses both onto the same key. The rule engine routinely
    # emits a different tautomer of the reference, so this is the recall-correct mode.
    preds = [{"predicted": ["CC(O)=C"], "real": ["CC(=O)C"]}]
    plain = aggregate_prediction_metrics(preds, ks=[1], match="inchikey")
    taut = aggregate_prediction_metrics(preds, ks=[1], match="inchikey_tautomer")
    assert plain["recall"] == 0.0
    assert taut["recall"] == 1.0
    assert taut["top_1_recall"] == 1.0


def test_rank_only_policy_keeps_subthreshold_hits():
    # A true hit whose filter score sits BELOW the calibrated threshold is dropped by the
    # hard gate but kept (and ranked) by the rank-only policy. This guards the conclusion
    # that gating hurts recall@k while the filter is still useful as a ranker.
    seed_everything(0)
    model = summon_the_grail([RULE])
    model.filter.calibrated_threshold = 0.6

    def fake_scored(sub, top_k=None, threshold=None):
        return [("CCO", 0.9), ("CC=O", 0.8)]  # CCO is the sub-threshold true hit

    model.generator.generate_scored = fake_scored
    model.filter.score_batch = lambda sub, prods: [{"CCO": 0.3, "CC=O": 0.7}.get(p, 0.0) for p in prods]

    gated = model.generate("CCO", gate_by_filter=True)
    rank_only = model.generate("CCO", gate_by_filter=False)

    assert "CC=O" in gated and "CCO" not in gated          # gate discards the sub-threshold hit
    assert "CCO" in rank_only and "CC=O" in rank_only       # rank-only retains it
    assert rank_only[0] == "CC=O"                            # still ordered by filter*generator


def test_output_dedup_collapses_tautomer_variants_freeing_budget():
    # Acetone keto (CC(=O)C) and enol (CC(O)=C) are the SAME molecule (one tautomer-
    # InChIKey). With canonical normalization they keep distinct SMILES strings, so a
    # string-keyed dedup would let both occupy the 2-slot budget and crowd out ethanol.
    # The tautomer-keyed output dedup must collapse them and free a slot for the distinct
    # third molecule -- matching the key the structure metric uses.
    seed_everything(0)
    model = summon_the_grail([RULE])
    model.filter.calibrated_threshold = 0.0
    model.generator.gen_normalization = "canonical"  # force dedup (not normalization) to collapse tautomers
    model.filter.score_batch = lambda sub, prods: [1.0] * len(prods)  # rank by generator score

    def fake_scored(sub, top_k=None, threshold=None):
        return [("CC(=O)C", 0.9), ("CC(O)=C", 0.85), ("CCO", 0.8)]  # keto, enol, ethanol

    model.generator.generate_scored = fake_scored
    out = model.generate("CCO", max_output=2)
    keys = {_tautomer_inchikey(s) for s in out}
    assert len(out) == 2                            # two slots filled
    assert len(keys) == 2                           # with two DISTINCT molecules, not two acetone tautomers
    assert _tautomer_inchikey("CCO") in keys        # ethanol got in because the tautomer dup freed a slot


def test_tautomer_path_fails_loud_when_pair_stops_merging():
    # A broken standardize env silently makes _tautomer_inchikey == plain _inchikey, degrading
    # every tautomer number (0.735 ceiling -> plain 0.718) with NO error. The one-time canary must
    # raise instead. Simulate degradation: the no-fallback key returns the PLAIN inchikey.
    import pytest
    import grail_metabolism.metrics as m
    orig_flag, orig_raw = m._TAUTOMER_PATH_OK, m._taut_key_raw
    try:
        m._TAUTOMER_PATH_OK = None
        m._taut_key_raw = m._inchikey            # keto/enol no longer merge under a plain key
        with pytest.raises(RuntimeError, match="tautomer"):
            m._ensure_tautomer_path()
    finally:
        m._TAUTOMER_PATH_OK, m._taut_key_raw = orig_flag, orig_raw


def test_tautomer_path_fails_loud_when_standardize_throws():
    import pytest
    import grail_metabolism.metrics as m
    orig_flag, orig_raw = m._TAUTOMER_PATH_OK, m._taut_key_raw

    def _boom(_s):
        raise ImportError("numpy missing")

    try:
        m._TAUTOMER_PATH_OK = None
        m._taut_key_raw = _boom
        with pytest.raises(RuntimeError):
            m._ensure_tautomer_path()
    finally:
        m._TAUTOMER_PATH_OK, m._taut_key_raw = orig_flag, orig_raw


def test_tautomer_path_healthy_in_this_env():
    # Positive control: in a real env the canary passes and a per-molecule bad SMILES still falls
    # back gracefully (does NOT raise) — fail-fast is systemic-only.
    import grail_metabolism.metrics as m
    m._TAUTOMER_PATH_OK = None
    m._ensure_tautomer_path()  # must not raise
    assert m._tautomer_inchikey("CC(=O)CC(C)=O") == m._tautomer_inchikey("CC(=O)C=C(O)C")
    assert m._tautomer_inchikey("not_a_smiles") == m._tautomer_inchikey("not_a_smiles")  # per-mol fallback, no raise

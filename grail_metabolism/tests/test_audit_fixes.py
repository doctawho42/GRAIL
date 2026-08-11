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


def test_filter_precision_calibration():
    # Synthetic (substrate, product) pool where true positives score higher than the
    # unlabeled negatives, but the two bands overlap slightly -- so only a
    # sufficiently high threshold clears a precision floor, and it must cost recall.
    positives = ["CC=O", "CC(=O)O", "CCN", "CCCl", "CCBr", "CCF", "CCI", "CCS"]
    negatives = ["CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC"]
    rows = [{"sub": "CCO", "prod": p, "real": 1} for p in positives]
    rows += [{"sub": "CCO", "prod": p, "real": 0} for p in negatives]
    frame = MolFrame(pd.DataFrame(rows))

    sub_key = next(iter(frame.map.keys()))
    pos_keys = sorted(frame.map[sub_key])
    neg_keys = sorted(frame.gen_map[sub_key])
    assert len(pos_keys) == len(positives)
    assert len(neg_keys) == len(negatives)

    # High, distinct scores for true positives; lower, distinct scores for negatives.
    score_table = {}
    for i, key in enumerate(pos_keys):
        score_table[(sub_key, key)] = 0.90 - i * 0.01  # 0.90 .. 0.83
    for i, key in enumerate(neg_keys):
        score_table[(sub_key, key)] = 0.20 + i * 0.01  # 0.20 .. 0.24

    model = Filter(18, 18, [32, 64, 32, 64, 32, 16], mode="pair")
    model.score = lambda sub, prod, pca=False: score_table[(sub, prod)]

    threshold, _ = model.calibrate_threshold(frame, target="precision", min_precision=0.8, verbose=False)

    tp = sum(1 for key in pos_keys if score_table[(sub_key, key)] >= threshold)
    fp = sum(1 for key in neg_keys if score_table[(sub_key, key)] >= threshold)
    fn = len(pos_keys) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    assert precision >= 0.8
    assert recall > 0.0
    assert model.calibrated_threshold == threshold


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


def test_factorized_reranker_reshapes_rank_but_never_gates():
    """The factorized re-ranker multiplies a per-candidate type*site factor into the rank (the
    §10 hybrid re-rank, deployable form) without ever gating a candidate out; a uniform multiplier
    leaves the filter*generator order unchanged, and factorized=None is byte-identical."""
    from grail_metabolism.model.wrapper import ModelWrapper

    class _Gen:
        gen_normalization = "canonical"
        calibrated_threshold = None

        def generate_scored_with_details(self, sub, top_k=None, threshold=None, compute_sites=True):
            return [("CCO", 0.9, 0, ()), ("CCN", 0.5, 1, ())]

    class _Filter:
        mode = "single"
        calibrated_threshold = 0.0

        def score_batch(self, sub, prods):
            return [0.5 for _ in prods]  # equal filter -> rank set by generator * factorized

    class _Reranker:
        def __init__(self, mults):
            self._m = mults

        def multipliers(self, sub_mol, detailed):
            return self._m

    sub = "CCCCO"
    # Uniform type*site factor -> order follows the generator score (CCO 0.9 > CCN 0.5).
    base = ModelWrapper(_Filter(), _Gen(), rules=[], factorized=_Reranker([1.0, 1.0]))
    assert base.generate(sub, gate_by_filter=False) == ["CCO", "CCN"]
    # A large type*site factor on the low-generator candidate reranks it to the top,
    # and BOTH candidates still survive (rank-only, never gates).
    rr = ModelWrapper(_Filter(), _Gen(), rules=[], factorized=_Reranker([1.0, 10.0]))
    out_rr = rr.generate(sub, gate_by_filter=False)
    assert out_rr[0] == "CCN"
    assert set(out_rr) == {"CCO", "CCN"}


def test_propensity_weighting_upweights_rare_rules_and_is_off_by_default():
    """Propensity-scored positives (Jain et al. 2016) must invert the firing rate, not follow it.

    The selection diagnosis says a learner under constant weighting recovers a score dominated by
    each rule's marginal firing rate. The correction is only meaningful if the weight it applies is
    monotonically DECREASING in that rate; a weight that rises with frequency would amplify the
    pathology instead of countering it. This pins the direction, the normalisation, and the fact
    that the deployed default is unchanged.
    """
    import math
    import torch

    from grail_metabolism.config import GeneratorConfig
    from grail_metabolism.model.generator import GeneratorObjective

    assert GeneratorConfig().propensity_weighting is False, "deployed default must not change"

    # Reproduce the estimator on a synthetic label-frequency profile.
    positives = torch.tensor([1.0, 4.0, 16.0, 64.0, 256.0])
    n, a, b = 500.0, 0.55, 1.5
    c = (math.log(n) - 1.0) * (b + 1.0) ** a
    propensity = 1.0 / (1.0 + c * torch.exp(-a * torch.log(positives + b)))
    inverse = 1.0 / propensity.clamp_min(1e-6)
    weight = inverse / inverse.mean()

    diffs = weight[1:] - weight[:-1]
    assert torch.all(diffs < 0), f"weight must fall as a rule fires more often, got {weight.tolist()}"
    assert weight[0] > weight[-1] * 1.5, "rarest rule must be materially up-weighted against the commonest"

    # The objective must use it for positives and leave the PU down-weighting of negatives alone.
    obj = GeneratorObjective(rank_weight=0.0, unlabeled_weight=0.5)
    logits = torch.zeros(2, 5)
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
    mask = torch.ones(2, 5)
    pos_w = torch.ones(5)
    flat = obj(logits, targets, mask, pos_w)
    scored = obj(logits, targets, mask, pos_w, weight)
    assert not torch.isclose(flat, scored), "supplying propensities must change the loss"
    assert torch.isclose(obj(logits, targets, mask, pos_w, torch.ones(5)), flat), \
        "unit propensities must reproduce constant weighting exactly"


def test_return_scores_does_not_change_the_deployed_output():
    """The scored dump must be the deployed ranking, not a reimplementation of it.

    A per-candidate score dump is only usable for calibration work if it is the same list the
    pipeline returns, in the same order. Reimplementing the ranking in a script is exactly how this
    codebase has produced numbers that disagree with each other. This pins that return_scores is a
    projection of the default path: same candidates, same order, scores attached.
    """
    import inspect

    from grail_metabolism.model.wrapper import ModelWrapper

    src = inspect.getsource(ModelWrapper.generate)
    # one ranked list, one dedup loop, one truncation -- the scored branch must sit inside it
    assert src.count("for candidate, combined, filter_score, generator_score in ranked_candidates") == 1, \
        "scored output must read the same ranked list as the default path"
    assert src.count("if cap is not None and len(ranked) >= cap") == 1, \
        "a second truncation would let the scored dump diverge from the deployed output"
    assert inspect.signature(ModelWrapper.generate).parameters["return_scores"].default is False, \
        "return_scores must be opt-in so the deployed path is untouched"


def test_explicit_hydrogen_detector_is_a_token_test_not_a_substring_search():
    """A template needs explicit hydrogens iff a hydrogen ATOM appears on its reactant side.

    The first version of this detector was a substring search for "#1" and for "H" after a bracket.
    It counted sulfur ([#16:2]) and phosphorus ([#15:2]) as hydrogen because their atomic numbers
    begin with the same two characters, and it counted the negation [!#1] -- which asserts the
    atom is *not* hydrogen -- as hydrogen. Three published shares were wrong because of it. The
    distinction it must keep is chemical, not textual: RDKit matches [H] and [#1] only against a
    hydrogen atom, while the hydrogen-COUNT primitive inside [CH3] is unaffected by AddHs, which is
    the whole reason the convention matters.
    """
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from explicit_h_mechanism import needs_explicit_hydrogen as needs

    for token in ("[H]", "[#1]", "[H:1]", "[#1:3]", "[H+]", "[C,H]"):
        assert needs(token), f"{token} is a hydrogen atom and requires the substrate be expanded"
    for token in ("[#10]", "[#15:2]", "[#16:2]"):
        assert not needs(token), f"{token} is not hydrogen; its atomic number merely starts with 1"
    for token in ("[!#1]", "[!#6;!#1]", "[*;!#1]"):
        assert not needs(token), f"{token} asserts the atom is not hydrogen"
    for token in ("[CH3]", "[C;H2]", "[#7;H1]", "[C;X4;!H3]"):
        assert not needs(token), f"{token} counts attached hydrogens and is blind to the expansion"


def test_the_hydrogen_atom_primitive_only_matches_an_expanded_substrate():
    """The premise the convention result rests on, checked against RDKit rather than assumed.

    If [H] matched an implicit hydrogen the whole effect would be a different phenomenon, so this
    pins the semantics: the atom primitives find nothing until the substrate is expanded, and the
    count primitive is unmoved by expanding it.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    expanded = Chem.AddHs(Chem.Mol(mol))
    for pattern in ("[H]", "[#1]"):
        query = Chem.MolFromSmarts(pattern)
        assert not mol.GetSubstructMatches(query), f"{pattern} must not match implicit hydrogens"
        assert len(expanded.GetSubstructMatches(query)) == 6, f"{pattern} must match the six atoms"
    count_primitive = Chem.MolFromSmarts("[CH3]")
    assert len(mol.GetSubstructMatches(count_primitive)) == 1
    assert len(expanded.GetSubstructMatches(count_primitive)) == 1, \
        "the hydrogen-count primitive must be blind to the expansion"


def test_the_manuscript_agrees_with_the_artifact_it_cites():
    """Every quantity the paper derives from the decomposition must follow from the record.

    SELF_CLAIMS row 11 asserted this and nothing enforced it, which cost seventeen stale values the
    day the ceiling was corrected: the macro moved and the numbers derived from it did not. A
    conversion ratio, a truncation count, a paired difference that became arithmetically impossible
    and a figure caption all survived a reading and were caught only by a reader. This runs the same
    comparison mechanically, by name rather than by scanning for coincidences, so the next
    correction cannot leave a derived value behind.
    """
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(root / "scripts" / "verify_paper_numbers.py")],
                            capture_output=True, text=True, cwd=root, timeout=300)
    assert result.returncode == 0, (
        "a manuscript number does not follow from results/recall_factorization.json:\n"
        + result.stdout[-2000:])


def test_the_released_checker_and_the_paper_census_cannot_disagree():
    """scripts/declare_conventions.py is what a reader runs; it must count what the paper counted.

    The tool exists so the paper's recommendation can be acted on, which makes any drift between
    its census and the manuscript's worse than no tool at all -- a reader would get a different
    number from the same file and have no way to tell which is the paper's. The first version drifted
    immediately: it counted a template carrying both a hydrogen atom and a recursive SMARTS in two
    categories at once and reported 332 unclassifiable where the paper reports 126.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for p in (str(root), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from declare_conventions import classify_template
    from explicit_h_mechanism import hydrogen_convention
    from bank_engine_replication import load_bank

    census = hydrogen_convention()
    for bank in ("grail_full", "sygma_175", "biotransformer"):
        rules = load_bank(bank)
        tags = [classify_template(r) for r in rules]
        assert sum("wants_expanded" in t for t in tags) == census[bank]["with_explicit_hydrogen"], (
            f"{bank}: the checker and the paper disagree on how many templates want the expansion")
        assert sum("unclassifiable" in t for t in tags) == census[bank][
            "unclassified_recursive_smarts"], (
            f"{bank}: the checker and the paper disagree on the residual category, which is the one "
            f"a dispatch policy has to guess at")


def test_the_retrosynthesis_block_convention_is_proven_not_assumed():
    """Every cross-domain number rests on the first row of a block being the recorded answer.

    Seven files agreeing among themselves would agree just as well if all seven were read wrongly,
    so the reading is checked where an independent ground truth exists: the three systems whose
    reactions are in this repository's own USPTO-50k copy.
    """
    import csv, sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for p in (str(root), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    d = root / "grail_metabolism" / "data" / "evalretro"
    split = root / "grail_metabolism" / "data" / "USPTO_50k" / "test.csv"
    if not d.exists() or not split.exists():
        import pytest
        pytest.skip("the released prediction files are not present")

    from evalretro_ingest import canonical_set, parse_blocks

    # keyed on the PAIR: a product does not identify a reaction, since one product can be made more
    # than one way and this split contains such cases. Keying on the product alone made this check
    # fail on 22 reactions that are in fact read correctly.
    pairs = {(canonical_set(r["PRODUCT"]), canonical_set(r["REACTANT"]))
             for r in csv.DictReader(open(split))}
    products = {p for p, _ in pairs}
    for name in ("chemformer", "gta", "tiedtransformer"):
        blocks = parse_blocks(d / f"{name}_pred.csv")
        shared = [b for b in blocks if canonical_set(b["product"]) in products]
        agree = sum(1 for b in shared
                    if (canonical_set(b["product"]), canonical_set(b["true"])) in pairs)
        assert shared, f"{name}: none of its reactions is in our split"
        assert agree == len(shared), (
            f"{name}: the first row of a block is the recorded answer on only {agree} of "
            f"{len(shared)} reactions; the block convention does not hold and every cross-domain "
            f"number computed from these files is meaningless")


def test_released_checker_sees_a_degree_primitive_however_it_is_spelled():
    """The checker's own regex once required a semicolon before D.

    RDKit reads [CD1:1], [#6D2:1], [C&D1:1] and [C;D1:1] as the same degree constraint, and all
    four lose every match when the substrate is expanded with explicit hydrogens. A bank written
    in any of the first three got a clean bill of health from the tool this paper ships, which is
    the one artifact a practitioner is meant to run on a bank we have never seen.
    """
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from retro_template_convention import (COUNTS_EXPLICIT, DEGREE, IMPLICIT_H_COUNT,
                                           TOTAL_CONNECTIONS)

    for spelling in ("[C;D1:1]", "[CD1:1]", "[C&D1:1]", "[#6D2:1]"):
        assert COUNTS_EXPLICIT.search(spelling), f"{spelling} is a degree constraint and was missed"

    # the two constructs that were absent from the taxonomy: one moves, one does not
    assert IMPLICIT_H_COUNT.search("[c;h1:1]"), "the implicit-hydrogen count is not inert"
    assert not IMPLICIT_H_COUNT.search("[cH1:1]"), "the bracketed H count is a different primitive"
    assert TOTAL_CONNECTIONS.search("[c;X3:1]"), "X counts neighbours including implicit hydrogens"
    assert not COUNTS_EXPLICIT.search("[c;X3:1]"), "X is not D and must not be counted as it"

    # deletion is a different job from detection: it must leave a parseable bracket behind
    from rdkit import Chem
    for spelling, expected in (("[C;D1:1]", "[C:1]"), ("[CD1:1]", "[C:1]"),
                               ("[C&D1:1]", "[C:1]"), ("[#6D2:1]", "[#6:1]")):
        stripped = DEGREE.sub("", spelling)
        assert stripped == expected, f"{spelling} stripped to {stripped}"
        assert Chem.MolFromSmarts(stripped) is not None


def test_the_expansion_breaks_and_over_matches_as_the_paper_says():
    """The three directions a convention bites, checked against the toolkit rather than asserted."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles("O=C(O)c1ccccc1")
    expanded = Chem.AddHs(Chem.Mol(mol))

    def matches(smarts, m):
        return len(m.GetSubstructMatches(Chem.MolFromSmarts(smarts)))

    # inert: the bracketed hydrogen count and the total connection count
    assert matches("[cH1]", mol) == matches("[cH1]", expanded)
    assert matches("[c;X3]", mol) == matches("[c;X3]", expanded)
    # broken: the degree and the implicit-hydrogen count both read what the expansion removed
    assert matches("[c;D2]", mol) > 0 and matches("[c;D2]", expanded) == 0
    assert matches("[c;h1]", mol) > 0 and matches("[c;h1]", expanded) == 0
    # over-matching: a wildcard gains the drawn hydrogens as neighbours
    assert matches("[c]~[*]", expanded) > matches("[c]~[*]", mol)


def test_contracting_an_expanded_product_needs_more_than_removing_the_hydrogens():
    """The two ways of contracting an expanded product are not the same, and one makes radicals.

    An arm that expands the substrate has to put the hydrogens back before the product is read.
    Doing that with ``RemoveHs`` alone looks right on simple templates and is wrong in general:
    ``AddHs`` marks every heavy atom as taking no implicit hydrogens, so when a template consumes a
    mapped hydrogen and puts nothing in its place, removing the drawn ones leaves the atom one short
    and RDKit records an unpaired electron instead of refilling the valence. A metabolite corpus
    contains no radicals, so such a product cannot match any reference; it is lost silently.

    Clearing the flag first restores the capacity ``AddHs`` suspended. This asserts the difference
    rather than the equivalence, because the equivalence is what an earlier version of this file
    asserted on eight hand-picked templates and it does not hold on the bank.
    """
    import sys

    from rdkit import Chem
    from rdkit.Chem import AllChem

    sys.path.insert(0, "scripts")
    from _contract import contract, contract_by_removing_only

    # a template that takes a hydrogen off a mapped carbon and puts nothing back
    reaction = AllChem.ReactionFromSmarts("[C:1][C:2]=[C:3][C:4]>>[C:1][C:2][C:3][C:4]")
    substrate = Chem.AddHs(Chem.MolFromSmiles("CC=CCO"))
    radicals_one_call = radicals_reset = compared = 0
    for products in reaction.RunReactants((substrate,)):
        for product in products:
            try:
                a = contract_by_removing_only(product)
                b = contract(product)
            except Exception:
                continue
            compared += 1
            radicals_one_call += sum(x.GetNumRadicalElectrons() for x in a.GetAtoms())
            radicals_reset += sum(x.GetNumRadicalElectrons() for x in b.GetAtoms())
    assert compared, "the template fired on nothing, so the test asserts nothing"
    assert radicals_one_call > 0, "the one-call contraction is expected to strand a valence here"
    assert radicals_reset == 0, "restoring implicit capacity must not leave an unpaired electron"


def test_a_similarity_threshold_is_not_an_identity_relation():
    """Tanimoto equal to one identifies molecules that are not the same molecule.

    Published work compares Morgan fingerprints, so the criterion is reported; it is kept out of
    the grid of candidate conventions because a cell of that grid has to be a possible answer to
    "are these the same compound", and this one answers yes for a homologue and for an enantiomer.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprint(smiles):
        return generator.GetFingerprint(Chem.MolFromSmiles(smiles))

    collisions = [("CCCCCCCCCC", "CCCCCCCCCCC"),      # decane against undecane
                  ("CCCCCCCCC", "CCCCCCCCCCCC"),      # nonane against dodecane
                  ("C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O")]  # D- against L-alanine
    for left, right in collisions:
        assert DataStructs.TanimotoSimilarity(fingerprint(left), fingerprint(right)) >= 1.0
        assert Chem.MolToSmiles(Chem.MolFromSmiles(left)) != \
               Chem.MolToSmiles(Chem.MolFromSmiles(right))

    import sys
    sys.path.insert(0, "scripts")
    import robust_order_metabolite
    assert "tanimoto1" not in robust_order_metabolite.MODES
    import robust_order
    assert "tanimoto1" not in robust_order.MODES

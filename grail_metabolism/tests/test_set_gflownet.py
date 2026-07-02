from grail_metabolism.model.set_gflownet import ForestState


def test_forest_leaves_and_terminal_set():
    s = ForestState(root="R", max_depth=3, max_size=10)
    s = s.add("R", "A").add("R", "B").add("A", "C")   # R->A,R->B,A->C
    assert s.terminal_set() == frozenset({"A", "B", "C"})
    assert set(s.leaves()) == {"B", "C"}              # A has child C, so A is not a leaf; R is root
    assert s.depth_of("C") == 2

def test_forest_flat_set_all_leaves():
    s = ForestState(root="R", max_depth=3, max_size=10).add("R", "A").add("R", "B")
    assert set(s.leaves()) == {"A", "B"}              # flat single-step set: every member is a leaf


import math
from grail_metabolism.model.set_gflownet import set_coverage_logreward

def test_set_coverage_logreward_pu_and_size_penalty():
    annotated = {"A", "B"}
    # set {A, X}: TP=1 (A hits; X is unlabeled, NOT penalized as false), |S|=2
    lr = set_coverage_logreward(frozenset({"A", "X"}), annotated, beta=2.0, lam=0.1)
    assert math.isclose(lr, 2.0 * (1 - 0.1 * 2))          # 2*(1-0.2)=1.6
    # empty set: TP=0, |S|=0 -> logreward 0
    assert set_coverage_logreward(frozenset(), annotated, beta=2.0, lam=0.1) == 0.0


from grail_metabolism.model.set_gflownet import ForestState, log_pb_trajectory

def test_log_pb_flat_set_is_minus_log_factorial():
    s0 = ForestState(root="R", max_depth=3, max_size=10)
    s1 = s0.add("R", "A")               # 1 leaf
    s2 = s1.add("R", "B")               # 2 leaves
    s3 = s2.add("R", "C")               # 3 leaves
    lp = log_pb_trajectory([s1, s2, s3])
    assert abs(lp - (math.log(1/1) + math.log(1/2) + math.log(1/3))) < 1e-9


# --------------------------------------------------------------------------- #
# Task 5: SetGFlowNetTrainer -- reranker forward policy P_F over a forest rollout,
# Trajectory-Balance loss. Terminal = a set of metabolites; reward = PU set-coverage.
# --------------------------------------------------------------------------- #

import torch  # noqa: E402
from grail_metabolism.model.reranker import BiEncoderReranker  # noqa: E402
from grail_metabolism.utils.transform import SINGLE_NODE_DIM  # noqa: E402
from grail_metabolism.config import GFlowNetConfig  # noqa: E402
from grail_metabolism.model.set_gflownet import SetGFlowNetTrainer  # noqa: E402


class _MiniGen:
    rule_prior_logits = torch.zeros(8)

    def generate_scored_with_details(self, sub, top_k=200, max_pool=None, compute_sites=False):
        # R->A,B ; A->C ; others terminal. gen_scores arbitrary, rule_ids < 8.
        return {
            "CCO": [("CCO O", 0.9, 1), ("CC", 0.5, 2)],
            "CCO O": [("CCOO", 0.7, 3)],
        }.get(sub, [])


def test_tb_loss_is_finite_and_backprops():
    gen = _MiniGen()
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(
        max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1,
        lam=0.1, max_size=5, top_k=200,
    )
    trainer = SetGFlowNetTrainer(gen, rr, cfg, annotated_ik_fn=lambda root: set())
    torch.manual_seed(0)
    loss = trainer.tb_loss("CCO")
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in rr.parameters()
    )
    # Backbone-grad assertion: the reranker's GNN backbone (rr.encoder) must actually
    # train -- guards against a future regression that detaches the encoder so only the
    # head params get gradient.
    backbone_params = list(rr.encoder.parameters())
    assert backbone_params, "rr.encoder has no parameters to check"
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in backbone_params
    )


class _BadSmilesGen:
    """Emits one valid child and one RDKit-unparseable child for the same state."""

    rule_prior_logits = torch.zeros(8)

    def generate_scored_with_details(self, sub, top_k=200, max_pool=None, compute_sites=False):
        return {
            "CCO": [("CC", 0.9, 1), ("@@@bad", 0.5, 2)],
        }.get(sub, [])


def test_candidate_children_drops_unparseable_smiles():
    gen = _BadSmilesGen()
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(
        max_depth=1, beta=2.0, epsilon=0.0, batch_substrates=1,
        lam=0.1, max_size=5, top_k=200,
    )
    trainer = SetGFlowNetTrainer(gen, rr, cfg, annotated_ik_fn=lambda root: set())
    kids = trainer.candidate_children("CCO")
    assert [c[0] for c in kids] == ["CC"]   # bad SMILES silently dropped, valid one kept

    torch.manual_seed(0)
    loss = trainer.tb_loss("CCO")           # must not raise (TypeError from Batch.from_data_list)
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# Final-review fix: ``sample_forest``'s state identity must route through
# ``metrics._tautomer_inchikey`` (same key space as ``annotated_ik_fn`` / eval), not
# plain ``Chem.MolToInchiKey`` -- otherwise two tautomers of one true metabolite are
# counted as two distinct terminal-set members / two separate TPs instead of one.
#
# These tests must NOT depend on RDKit's real tautomer canonicalization, which is
# VERSION-DEPENDENT (a pair that collapses under one RDKit build may not under another --
# e.g. acetylacetone keto/enol merges under 2022.09 but not on some newer builds). We
# instead MONKEYPATCH ``set_gflownet._tautomer_inchikey`` with a controlled stub that
# collapses two chosen (plain-InChIKey-distinct) SMILES onto one key, and assert the
# forest collapses them -- proving ``sample_forest`` calls the tautomer-IK function it
# imports, independent of which real tautomers a given RDKit build actually merges.
# --------------------------------------------------------------------------- #

import grail_metabolism.model.set_gflownet as _sg  # noqa: E402
from grail_metabolism.metrics import _tautomer_inchikey  # noqa: E402

# Two distinct, parseable molecules with DISTINCT plain InChIKeys; the stub below
# collapses them onto one key (as a tautomer canonicalizer would for real tautomers).
_ROOT = "c1ccccc1"
_CH_A = "CCO"
_CH_B = "CCCO"


def _fake_taut_ik(smiles):
    """Collapse the two chosen children onto one key; everything else keys to itself."""
    return "SHARED_TAUT_KEY" if smiles in (_CH_A, _CH_B) else smiles


class _TwoChildrenGen:
    """Root emits two distinct children that the stubbed tautomer-IK collapses to one."""

    rule_prior_logits = torch.zeros(8)

    def generate_scored_with_details(self, sub, top_k=200, max_pool=None, compute_sites=False):
        return {_ROOT: [(_CH_A, 0.9, 1), (_CH_B, 0.5, 2)]}.get(sub, [])


def test_sample_forest_state_identity_routes_through_tautomer_inchikey(monkeypatch):
    """Version-independent guard: FAILS if ``sample_forest`` keyed state by plain
    ``Chem.MolToInchiKey`` (the two children have distinct plain keys, so both could enter
    the terminal set); PASSES because state identity routes through the (here monkeypatched)
    tautomer-IK, which collapses them to a single member."""
    from rdkit import Chem

    # The two children really are distinct under plain InChIKey, so a regression to plain
    # keying would let both into the terminal set -> this test would then fail.
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(_CH_A)) != Chem.MolToInchiKey(Chem.MolFromSmiles(_CH_B))

    monkeypatch.setattr(_sg, "_tautomer_inchikey", _fake_taut_ik)
    gen = _TwoChildrenGen()
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(max_depth=1, beta=2.0, epsilon=1.0, batch_substrates=1,
                         lam=0.1, max_size=3, top_k=200)
    trainer = SetGFlowNetTrainer(gen, rr, cfg, annotated_ik_fn=lambda root: set())

    added = False
    for seed in range(20):
        torch.manual_seed(seed)
        state, _sum_log_pf, _post_add = trainer.sample_forest(_ROOT)
        terminal = state.terminal_set()
        assert len(terminal) <= 1, (
            f"seed={seed}: terminal_set={terminal} has >1 member for tautomer-collapsed "
            "children -- state identity is not routed through _tautomer_inchikey"
        )
        if len(terminal) == 1:
            added = True
    assert added, "no seed ever added a child -- the ADD path was never exercised"


def test_set_coverage_reward_matches_terminal_across_tautomer_key_space(monkeypatch):
    """A forest terminal keyed in the tautomer-IK space must count as a TP against an
    annotation given in that same space -- reward-side proof terminal_set() and
    annotated_ik share one key space (monkeypatched, RDKit-version-independent)."""
    monkeypatch.setattr(_sg, "_tautomer_inchikey", _fake_taut_ik)
    gen = _TwoChildrenGen()
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(max_depth=1, beta=2.0, epsilon=1.0, batch_substrates=1,
                         lam=0.1, max_size=3, top_k=200)
    trainer = SetGFlowNetTrainer(gen, rr, cfg, annotated_ik_fn=lambda root: set())

    annotated_ik = {_fake_taut_ik(_CH_B)}  # the shared key
    hit = False
    for seed in range(20):
        torch.manual_seed(seed)
        state, _sum_log_pf, _post_add = trainer.sample_forest(_ROOT)
        if state.terminal_set() & annotated_ik:
            hit = True
            break
    assert hit, "terminal never matched the annotation in the shared tautomer key space"


def test_tautomer_inchikey_collapses_real_pair_where_rdkit_supports_it():
    """Documentation/sanity: where this RDKit build canonicalizes acetylacetone keto/enol
    to one tautomer, _tautomer_inchikey collapses them while plain InChIKey does not.
    SKIPPED on RDKit builds that don't merge this particular pair (version-dependent), so
    it never fails the suite -- the real invariant is covered by the two monkeypatch tests
    above, which don't depend on any specific RDKit tautomer behavior."""
    import pytest
    from rdkit import Chem

    keto, enol = "CC(=O)CC(=O)C", "CC(=O)C=C(O)C"
    if _tautomer_inchikey(keto) != _tautomer_inchikey(enol):
        pytest.skip("this RDKit build does not canonicalize acetylacetone keto/enol to one form")
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(keto)) != Chem.MolToInchiKey(Chem.MolFromSmiles(enol))


def test_trainer_unifies_reranker_stophead_logz_device():
    """Device-unification guard: the reranker (forward policy P_F), stop_head, and log_z
    must all live on ``trainer.device``. Otherwise ``_frontier_embed`` produces an embedding
    from the (un-moved) reranker encoder that the on-device StopHead rejects -- a hard
    ``Expected all tensors to be on the same device`` crash on GPU. Trivially true on a
    CPU-only box, but it locks the invariant against a regression that moves stop_head/log_z
    to self.device while leaving the reranker where the caller passed it."""
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(max_depth=1, beta=2.0, epsilon=0.0, batch_substrates=1,
                         lam=0.1, max_size=3, top_k=200)
    trainer = SetGFlowNetTrainer(_TwoChildrenGen(), rr, cfg, annotated_ik_fn=lambda root: set())
    assert next(trainer.reranker.parameters()).device == trainer.device
    assert next(trainer.stop_head.parameters()).device == trainer.device
    assert trainer.log_z.device == trainer.device

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
# Final-review fix: ``sample_forest``'s state identity must use
# ``metrics._tautomer_inchikey`` (same key space as ``annotated_ik_fn`` / eval), not
# plain ``Chem.MolToInchiKey`` -- otherwise two tautomers of the same true metabolite
# are counted as two distinct terminal-set members / two separate TPs instead of one.
# --------------------------------------------------------------------------- #

from grail_metabolism.metrics import _tautomer_inchikey  # noqa: E402

# Acetylacetone keto/enol tautomers: RDKit's plain InChIKey treats them as DIFFERENT
# molecules, but metrics._tautomer_inchikey (full tautomer canonicalization) collapses
# them onto the same key. This is exactly the pair that would expose a regression back
# to plain Chem.MolToInchiKey in sample_forest.
_KETO = "CC(=O)CC(=O)C"
_ENOL = "CC(=O)C=C(O)C"


def test_tautomer_pair_collapses_under_tautomer_inchikey_not_plain():
    from rdkit import Chem

    plain_keto = Chem.MolToInchiKey(Chem.MolFromSmiles(_KETO))
    plain_enol = Chem.MolToInchiKey(Chem.MolFromSmiles(_ENOL))
    assert plain_keto != plain_enol, "test fixture assumption broken: pick another tautomer pair"
    assert _tautomer_inchikey(_KETO) == _tautomer_inchikey(_ENOL)


class _TautomerGen:
    """Root emits BOTH tautomers of the same metabolite as separate children."""

    rule_prior_logits = torch.zeros(8)

    def generate_scored_with_details(self, sub, top_k=200, max_pool=None, compute_sites=False):
        return {
            "CCO": [(_KETO, 0.9, 1), (_ENOL, 0.5, 2)],
        }.get(sub, [])


def test_sample_forest_collapses_tautomer_children_to_one_terminal_member():
    """Guard: FAILS under the old plain-Chem.MolToInchiKey ``ik`` (terminal_set would be
    able to hold both _KETO and _ENOL as two distinct members); PASSES after routing
    ``ik`` through metrics._tautomer_inchikey (they can only ever be one member)."""
    gen = _TautomerGen()
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(
        max_depth=1, beta=2.0, epsilon=1.0, batch_substrates=1,
        lam=0.1, max_size=3, top_k=200,
    )
    trainer = SetGFlowNetTrainer(gen, rr, cfg, annotated_ik_fn=lambda root: set())

    # epsilon=1.0 forces uniform-random action selection (add keto / add enol / STOP),
    # so across a handful of seeds both tautomers get added to the SAME forest at least
    # once. Regardless of seed, the terminal set can never legitimately exceed size 1,
    # since the two candidate children are tautomers of one metabolite.
    saw_both_added = False
    for seed in range(20):
        torch.manual_seed(seed)
        state, _sum_log_pf, _post_add = trainer.sample_forest("CCO")
        terminal = state.terminal_set()
        assert len(terminal) <= 1, (
            f"seed={seed}: terminal_set={terminal} has >1 member for tautomeric children "
            "-- state identity is not tautomer-invariant"
        )
        if len(terminal) == 1:
            saw_both_added = True
    assert saw_both_added, "no seed ever added a child -- test rollout never exercised the ADD path"


def test_set_coverage_logreward_counts_tautomer_hit_from_other_tautomer_annotation():
    """A forest terminal produced in the KETO form must still count as a TP when the
    annotation set is given in the ENOL form (and vice versa) -- reward-side proof that
    terminal_set() and annotated_ik share the tautomer-invariant key space."""
    gen = _TautomerGen()
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(
        max_depth=1, beta=2.0, epsilon=1.0, batch_substrates=1,
        lam=0.1, max_size=3, top_k=200,
    )
    trainer = SetGFlowNetTrainer(gen, rr, cfg, annotated_ik_fn=lambda root: set())

    annotated_ik = {_tautomer_inchikey(_ENOL)}  # reference given in the OTHER tautomer form
    hit = False
    for seed in range(20):
        torch.manual_seed(seed)
        state, _sum_log_pf, _post_add = trainer.sample_forest("CCO")
        terminal = state.terminal_set()
        if terminal and (terminal & annotated_ik):
            hit = True
            break
    assert hit, "KETO-form terminal never matched an ENOL-form annotation"

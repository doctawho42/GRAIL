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
    # Compare the actual parameter devices to EACH OTHER, not to trainer.device: moving a
    # module to torch.device("cuda") (index None) lands its params on "cuda:0" (index 0),
    # and torch's strict device __eq__ treats cuda != cuda:0. What matters is that all three
    # sit on the SAME device -- they diverged cpu-vs-cuda under the bug this guards.
    rr_dev = next(trainer.reranker.parameters()).device
    sh_dev = next(trainer.stop_head.parameters()).device
    lz_dev = trainer.log_z.device
    assert rr_dev == sh_dev == lz_dev, f"device split: reranker={rr_dev} stop_head={sh_dev} log_z={lz_dev}"


def test_expand_state_matches_candidate_children(monkeypatch):
    """_expand_state is the pure extraction of the per-state child-expansion logic,
    shared by candidate_children and parallel pre-warm workers. It deduplicates by SMILES,
    drops unparseable molecules, and preserves order."""
    from grail_metabolism.model import set_gflownet as sg

    class _StubGen:
        # returns fixed (smiles, gscore, rid, ...) rows; dupes + a bad smiles to exercise filtering
        def generate_scored_with_details(self, s, top_k, compute_sites=False):
            return [("CCO", 0.9, 3), ("CCO", 0.4, 3), ("bad_smiles", 0.5, 7), ("CCCO", 0.2, 9)]

    out = sg._expand_state(_StubGen(), "c1ccccc1", top_k=5)
    assert out == [("CCO", 0.9, 3), ("CCCO", 0.2, 9)]   # dedup + drop unparseable, order preserved
    assert all(isinstance(g, float) and isinstance(r, int) for _, g, r in out)


def test_persistent_child_cache_roundtrip(tmp_path):
    """The (state,top_k)->children cache persists across trainers: a second trainer with the
    same cache path must serve candidate_children FROM DISK without ever calling the generator
    (RDKit rule application is deterministic, so cross-run caching is exact)."""
    child_path = tmp_path / "child.pkl"
    ik_path = tmp_path / "ik.pkl"
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(max_depth=1, beta=2.0, epsilon=0.0, batch_substrates=1,
                         lam=0.1, max_size=3, top_k=200)
    t1 = SetGFlowNetTrainer(_TwoChildrenGen(), rr, cfg, annotated_ik_fn=lambda root: set(),
                            child_cache_path=str(child_path), ik_cache_path=str(ik_path))
    kids = t1.candidate_children(_ROOT)          # populates the in-memory cache
    assert kids and all(len(x) == 3 for x in kids)
    t1.save_caches()
    assert child_path.exists()

    class _BoomGen:
        rule_prior_logits = torch.zeros(8)

        def generate_scored_with_details(self, *a, **k):
            raise AssertionError("generator called -- the persistent cache was not loaded")

    t2 = SetGFlowNetTrainer(_BoomGen(), rr, cfg, annotated_ik_fn=lambda root: set(),
                            child_cache_path=str(child_path), ik_cache_path=str(ik_path))
    assert t2.candidate_children(_ROOT) == kids  # served from the on-disk cache, no generator call


# --------------------------------------------------------------------------- #
# Task 2: prewarm_caches -- two-wave parallel cache build. The pytest here only
# exercises the workers=1 in-method serial map (identical to lazy candidate_children);
# the real spawn>1 path needs a real generator checkpoint and is validated on Modal.
# --------------------------------------------------------------------------- #

def test_prewarm_matches_serial(monkeypatch):
    """prewarm_caches(workers=1) must populate _child_cache identically to lazily calling
    candidate_children on the roots + their depth-1 children (max_depth=2 here), and
    _ik_cache must cover every touched smiles."""
    from grail_metabolism.model import set_gflownet as sg

    # deterministic 1-level expansion: root -> two children, each child -> one grandchild
    KIDS = {
        "ROOT":  [("CCO", 0.9, 1), ("CCCO", 0.5, 2)],
        "CCO":   [("CCOC", 0.7, 3)],
        "CCCO":  [("CCCOC", 0.4, 4)],
    }

    class _StubGen:
        rule_prior_logits = torch.zeros(8)

        def generate_scored_with_details(self, s, top_k, compute_sites=False, max_pool=None):
            return [(c, g, r) for (c, g, r) in KIDS.get(s, [])]

    # plain-identity tautomer-IK so no RDKit dependency in the test
    monkeypatch.setattr(sg, "_tautomer_inchikey", lambda s: f"IK::{s}")

    cfg = GFlowNetConfig(max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1,
                         lam=0.1, max_size=5, top_k=5)

    def _mk():
        rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
        return sg.SetGFlowNetTrainer(_StubGen(), rr, cfg, annotated_ik_fn=lambda s: set())

    # serial reference: lazily expand roots + their depth-1 children
    ser = _mk()
    for root in ("ROOT",):
        for c, _, _ in ser.candidate_children(root):
            ser.candidate_children(c)          # depth-1 children expanded (depth-2 are terminal)

    # prewarm path (workers=1 => in-method serial map over the same _expand_state)
    par = _mk()
    par.prewarm_caches(["ROOT"], workers=1, gen_ckpt=None)

    assert par._child_cache == ser._child_cache      # identical (dict eq ignores insertion order)
    # ik cache covers every touched smiles (root + all children)
    for s in ("ROOT", "CCO", "CCCO", "CCOC", "CCCOC"):
        assert par._ik_cache[s] == f"IK::{s}"


def test_prewarm_skips_already_cached_states(monkeypatch):
    """prewarm_caches is safe to call repeatedly: states already in _child_cache are not
    re-expanded (a generator that raises on repeat calls for a cached state must not blow up)."""
    from grail_metabolism.model import set_gflownet as sg

    calls = {"n": 0}

    class _CountingGen:
        rule_prior_logits = torch.zeros(8)

        def generate_scored_with_details(self, s, top_k, compute_sites=False, max_pool=None):
            calls["n"] += 1
            return {"ROOT": [("CCO", 0.9, 1)]}.get(s, [])

    monkeypatch.setattr(sg, "_tautomer_inchikey", lambda s: f"IK::{s}")
    cfg = GFlowNetConfig(max_depth=1, beta=2.0, epsilon=0.0, batch_substrates=1,
                         lam=0.1, max_size=5, top_k=5)
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    trainer = sg.SetGFlowNetTrainer(_CountingGen(), rr, cfg, annotated_ik_fn=lambda s: set())

    trainer.prewarm_caches(["ROOT"], workers=1, gen_ckpt=None)
    assert calls["n"] == 1
    trainer.prewarm_caches(["ROOT"], workers=1, gen_ckpt=None)  # second call: ROOT already cached
    assert calls["n"] == 1


def test_prewarm_expands_depth1_of_precached_root(monkeypatch):
    """T2a regression: when a root is ALREADY in _child_cache (resume / partial cache) but its
    depth-1 children are not yet expanded, prewarm_caches must STILL expand those depth-1
    children -- wave2 reads all roots' cached children, not just wave1's freshly-expanded ones.
    (Old code harvested depth1 from wave1.values(), which is empty for a pre-cached root, so the
    depth-1 child was left cold -- exactly the no-op observed when reusing a banked cache.)"""
    from grail_metabolism.model import set_gflownet as sg

    KIDS = {"ROOT": [("CCO", 0.9, 1)], "CCO": [("CCOC", 0.7, 3)]}

    class _StubGen:
        rule_prior_logits = torch.zeros(8)

        def generate_scored_with_details(self, s, top_k, compute_sites=False, max_pool=None):
            return [(c, g, r) for (c, g, r) in KIDS.get(s, [])]

    monkeypatch.setattr(sg, "_tautomer_inchikey", lambda s: f"IK::{s}")
    cfg = GFlowNetConfig(max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1,
                         lam=0.1, max_size=5, top_k=5)
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    trainer = sg.SetGFlowNetTrainer(_StubGen(), rr, cfg, annotated_ik_fn=lambda s: set())

    # Simulate a partial/resumed cache: ROOT cached (children list present) but "CCO" NOT expanded.
    trainer._child_cache["ROOT"] = [("CCO", 0.9, 1)]
    assert "CCO" not in trainer._child_cache

    trainer.prewarm_caches(["ROOT"], workers=1, gen_ckpt=None)

    # wave2 must have expanded the pre-cached root's depth-1 child:
    assert "CCO" in trainer._child_cache, "wave2 skipped the pre-cached root's depth-1 child"
    assert trainer._child_cache["CCO"] == [("CCOC", 0.7, 3)]


def test_prewarm_waves1_expands_roots_only(monkeypatch):
    """waves=1 must expand ONLY the roots and leave depth-1 children COLD -- fit/eval expand the
    visited depth-1 subset lazily via candidate_children. This is the over-expansion fix: at scale
    the full depth-1 frontier (all top_k children of every root) dwarfs the subset a policy visits,
    so prewarming it all is a multi-hour waste. Guards that waves=1 skips wave2 entirely."""
    from grail_metabolism.model import set_gflownet as sg

    KIDS = {"ROOT": [("CCO", 0.9, 1)], "CCO": [("CCOC", 0.7, 3)]}

    class _StubGen:
        rule_prior_logits = torch.zeros(8)

        def generate_scored_with_details(self, s, top_k, compute_sites=False, max_pool=None):
            return [(c, g, r) for (c, g, r) in KIDS.get(s, [])]

    monkeypatch.setattr(sg, "_tautomer_inchikey", lambda s: f"IK::{s}")
    cfg = GFlowNetConfig(max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1,
                         lam=0.1, max_size=5, top_k=5)
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    trainer = sg.SetGFlowNetTrainer(_StubGen(), rr, cfg, annotated_ik_fn=lambda s: set())

    trainer.prewarm_caches(["ROOT"], workers=1, gen_ckpt=None, waves=1)

    # Root expanded; its depth-1 child left COLD (waves=1 skips wave2 even at max_depth=2):
    assert "ROOT" in trainer._child_cache, "waves=1 must still expand the roots"
    assert trainer._child_cache["ROOT"] == [("CCO", 0.9, 1)]
    assert "CCO" not in trainer._child_cache, "waves=1 must NOT expand depth-1 children"

    # And lazy candidate_children still expands it on demand (identical result), proving fit/eval
    # recover the depth-1 state when the policy actually visits it.
    assert trainer.candidate_children("CCO") == [("CCOC", 0.7, 3)]
    assert "CCO" in trainer._child_cache


def test_train_checkpoint_resumes_from_last_epoch(tmp_path):
    """Model-checkpointing (preemption/crash recovery): fit(resume_path=p) persists the trainable
    state every epoch, and a fresh trainer given the same path RESUMES from the last completed
    epoch -- it does NOT restart at 0 -- while restoring logZ + weights. This is the fix for a
    multi-hour training being lost when a preemptible worker is reclaimed."""
    ckpt = str(tmp_path / "seed0.ckpt.pt")

    def _mk():
        rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
        cfg = GFlowNetConfig(max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1,
                             lam=0.1, max_size=5, top_k=200)
        return SetGFlowNetTrainer(_MiniGen(), rr, cfg, annotated_ik_fn=lambda root: set())

    torch.manual_seed(0)
    t1 = _mk()
    t1.fit(["CCO"], epochs=2, resume_path=ckpt)
    assert (tmp_path / "seed0.ckpt.pt").exists(), "fit must persist the resume checkpoint"
    assert len(t1.loss_history_) == 2
    logz_at_2 = float(t1.log_z)

    # Direct load restores epoch + logZ exactly. Done BEFORE the resume-fit below, which would
    # advance (overwrite) the checkpoint to epoch 4. The optimizer must be built over the SAME
    # params in the SAME order as fit() (optimizer state_dict maps by index).
    torch.manual_seed(2)
    t3 = _mk()
    params = [p for p in t3.reranker.parameters() if p.requires_grad] + list(t3.stop_head.parameters())
    opt = torch.optim.Adam([{"params": params}, {"params": [t3.log_z]}])
    assert float(t3.log_z) == 0.0                       # fresh trainer starts at logZ=0
    start = t3._load_train_ckpt(ckpt, opt)
    assert start == 2, "checkpoint must report the epoch to resume FROM"
    assert abs(float(t3.log_z) - logz_at_2) < 1e-5, "logZ must be restored from the checkpoint"

    # Fresh trainer, SAME checkpoint, asked for 4 epochs total: it must load epoch=2 and run
    # only epochs 2,3 -> loss_history length 4. A broken resume (restart at 0) would run 4 fresh
    # epochs on top of the 2 loaded entries -> length 6.
    torch.manual_seed(1)
    t2 = _mk()
    t2.fit(["CCO"], epochs=4, resume_path=ckpt)
    assert len(t2.loss_history_) == 4, "resume must continue (2 loaded + 2 new), not restart at 0"


# --------------------------------------------------------------------------- #
# Task 1: single_hit_logreward + SingleTerminalGFlowNetTrainer (max_size=1
# single-terminal ablation baseline, ABL-01/02). See 03-01-PLAN.md.
# --------------------------------------------------------------------------- #

from grail_metabolism.model.set_gflownet import (  # noqa: E402
    single_hit_logreward,
    SingleTerminalGFlowNetTrainer,
)


def test_single_hit_logreward_hit_miss_empty():
    annotated = {"A", "B"}
    # hit: single-member terminal set intersects the annotated set -> beta
    assert single_hit_logreward(frozenset({"A"}), annotated, beta=2.0) == 2.0
    # miss: single-member terminal set does NOT intersect -> 0.0, no size penalty
    assert single_hit_logreward(frozenset({"X"}), annotated, beta=2.0) == 0.0
    # empty set -> 0.0, no size penalty
    assert single_hit_logreward(frozenset(), annotated, beta=2.0) == 0.0


def test_single_terminal_max_size_one_yields_at_most_one_member():
    gen = _MiniGen()
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(
        max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1,
        lam=0.1, max_size=1, top_k=200,
    )
    trainer = SingleTerminalGFlowNetTrainer(gen, rr, cfg, annotated_ik_fn=lambda root: set())
    for seed in range(20):
        torch.manual_seed(seed)
        state, _sum_log_pf, post_add = trainer.sample_forest("CCO")
        assert len(state.terminal_set()) <= 1, (
            f"seed={seed}: terminal_set={state.terminal_set()} exceeds max_size=1"
        )
        if len(post_add) <= 1:
            assert log_pb_trajectory(post_add) == 0.0


def test_single_terminal_tb_loss_finite_and_backprops():
    gen = _MiniGen()
    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(
        max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1,
        lam=0.1, max_size=1, top_k=200,
    )
    trainer = SingleTerminalGFlowNetTrainer(gen, rr, cfg, annotated_ik_fn=lambda root: set())
    torch.manual_seed(0)
    loss = trainer.tb_loss("CCO")
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in rr.parameters()
    )
    backbone_params = list(rr.encoder.parameters())
    assert backbone_params, "rr.encoder has no parameters to check"
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in backbone_params
    )


def test_single_terminal_trainer_checkpoint_roundtrips_at_max_size_one(tmp_path):
    """Mirrors test_train_checkpoint_resumes_from_last_epoch for the
    SingleTerminalGFlowNetTrainer subclass: the inherited _save_train_ckpt/
    _load_train_ckpt/fit machinery must round-trip correctly (resume from the last
    completed epoch, restore logZ) unaffected by max_size=1."""
    ckpt = str(tmp_path / "single_seed0.ckpt.pt")

    def _mk():
        rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
        cfg = GFlowNetConfig(max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1,
                             lam=0.1, max_size=1, top_k=200)
        return SingleTerminalGFlowNetTrainer(_MiniGen(), rr, cfg, annotated_ik_fn=lambda root: set())

    torch.manual_seed(0)
    t1 = _mk()
    t1.fit(["CCO"], epochs=2, resume_path=ckpt)
    assert (tmp_path / "single_seed0.ckpt.pt").exists(), "fit must persist the resume checkpoint"
    assert len(t1.loss_history_) == 2
    logz_at_2 = float(t1.log_z)

    torch.manual_seed(2)
    t3 = _mk()
    params = [p for p in t3.reranker.parameters() if p.requires_grad] + list(t3.stop_head.parameters())
    opt = torch.optim.Adam([{"params": params}, {"params": [t3.log_z]}])
    assert float(t3.log_z) == 0.0
    start = t3._load_train_ckpt(ckpt, opt)
    assert start == 2, "checkpoint must report the epoch to resume FROM"
    assert abs(float(t3.log_z) - logz_at_2) < 1e-5, "logZ must be restored from the checkpoint"

    torch.manual_seed(1)
    t2 = _mk()
    t2.fit(["CCO"], epochs=4, resume_path=ckpt)
    assert len(t2.loss_history_) == 4, "resume must continue (2 loaded + 2 new), not restart at 0"


def test_train_checkpoint_ignores_corrupt_file(tmp_path):
    """A corrupt/unreadable training checkpoint must be IGNORED (train from scratch, start_epoch
    0), never crash the run -- a half-written file from a kill mid-save must not brick recovery."""
    bad = tmp_path / "seed0.ckpt.pt"
    bad.write_bytes(b"not a torch checkpoint")

    rr = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cfg = GFlowNetConfig(max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1,
                         lam=0.1, max_size=5, top_k=200)
    trainer = SetGFlowNetTrainer(_MiniGen(), rr, cfg, annotated_ik_fn=lambda root: set())
    opt = torch.optim.Adam([{"params": list(trainer.reranker.parameters())},
                            {"params": [trainer.log_z]}])
    assert trainer._load_train_ckpt(str(bad), opt) == 0   # ignored, not raised


# --------------------------------------------------------------------------- #
# Plan 03-02 Task 3(a): ABL-02 round-robin ensemble draw-count allocator guard.
# Pure arithmetic over scripts.run_gflownet._round_robin_draw_counts (D-04).
# --------------------------------------------------------------------------- #

import scripts.run_gflownet as _rg  # noqa: E402


def test_round_robin_draw_counts_even_divide():
    # k_max=9, M=3 divides evenly -> every member gets exactly 3.
    counts = _rg._round_robin_draw_counts(k_max=9, m_ensemble=3)
    assert counts == [3, 3, 3]
    assert sum(counts) == 9


def test_round_robin_draw_counts_uneven_ceil_split():
    # k_max=50, M=3 -> ceil(50/3)=17 for the first (50 % 3 = 2) members, 16 for the rest;
    # summing to >= 50, no member off by more than 1 from another.
    counts = _rg._round_robin_draw_counts(k_max=50, m_ensemble=3)
    assert sum(counts) >= 50
    assert max(counts) - min(counts) <= 1
    assert counts == [17, 17, 16]


def test_round_robin_draw_counts_m_ensemble_exceeds_k_max():
    # M_ensemble=10 > k_max=5 -> 5 members get 1 draw, 5 members get 0; still sums to >= 5
    # and no member is off by more than 1 from any other.
    counts = _rg._round_robin_draw_counts(k_max=5, m_ensemble=10)
    assert len(counts) == 10
    assert sum(counts) >= 5
    assert max(counts) - min(counts) <= 1
    assert sorted(counts, reverse=True) == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_round_robin_draw_counts_degenerate_single_member():
    # M_ensemble=1 -> the single member draws the entire k_max budget.
    counts = _rg._round_robin_draw_counts(k_max=12, m_ensemble=1)
    assert counts == [12]


# --------------------------------------------------------------------------- #
# Adversarial-review fix (BLOCKER): every GFlowNet-family trainer must get its OWN
# fresh reranker loaded from one warm-start snapshot -- never share the live `reranker`
# object -- so ablation arms train independent parameter tensors and don't retroactively
# corrupt each other's / the set-GFlowNet reference's eval policy. Dataset-free guard,
# reproducing main()'s construction pattern (warm_state snapshot + fresh
# BiEncoderReranker(in_channels=SINGLE_NODE_DIM).load_state_dict(...) per trainer).
# --------------------------------------------------------------------------- #

import copy  # noqa: E402


def _fresh_warm_reranker(warm_state):
    r = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    r.load_state_dict(copy.deepcopy(warm_state))
    return r


def test_gflownet_family_trainers_get_independent_reranker_objects():
    gen = _MiniGen()
    warm_reranker = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    warm_state = copy.deepcopy(warm_reranker.state_dict())

    cfg = GFlowNetConfig(max_depth=2, beta=2.0, epsilon=0.0, batch_substrates=1,
                         lam=0.1, max_size=5, top_k=200)
    single_cfg = GFlowNetConfig(max_depth=2, beta=1.5, epsilon=0.0, batch_substrates=1,
                                lam=0.1, max_size=1, top_k=200)

    set_trainer = SetGFlowNetTrainer(
        gen, _fresh_warm_reranker(warm_state), cfg, annotated_ik_fn=lambda root: set()
    )

    torch.manual_seed(0)
    abl01 = SingleTerminalGFlowNetTrainer(
        gen, _fresh_warm_reranker(warm_state), single_cfg, annotated_ik_fn=lambda root: {"CCOO"}
    )
    torch.manual_seed(1)
    abl02_member = SingleTerminalGFlowNetTrainer(
        gen, _fresh_warm_reranker(warm_state), single_cfg, annotated_ik_fn=lambda root: {"CCOO"}
    )

    # (a) distinct objects, no shared parameter tensors, across ALL three trainers.
    assert abl01.reranker is not abl02_member.reranker
    assert set_trainer.reranker is not abl01.reranker
    assert set_trainer.reranker is not abl02_member.reranker
    for p1, p2 in zip(abl01.reranker.parameters(), abl02_member.reranker.parameters()):
        assert p1 is not p2
        assert p1.data_ptr() != p2.data_ptr()
    for p1, p2 in zip(set_trainer.reranker.parameters(), abl01.reranker.parameters()):
        assert p1 is not p2
        assert p1.data_ptr() != p2.data_ptr()

    # All three start from the IDENTICAL warm-start (purity).
    for pa, pb in zip(abl01.reranker.parameters(), abl02_member.reranker.parameters()):
        assert torch.allclose(pa, pb)
    for pa, pb in zip(set_trainer.reranker.parameters(), abl01.reranker.parameters()):
        assert torch.allclose(pa, pb)

    # (b) after a brief fit() under two different seeds, the ensemble members' reranker
    # parameters DIVERGE -- proving they are training independently, not sharing tensors.
    torch.manual_seed(0)
    abl01.fit(["CCO"], epochs=1, verbose=False)
    torch.manual_seed(1)
    abl02_member.fit(["CCO"], epochs=1, verbose=False)

    diverged = any(
        not torch.allclose(pa, pb)
        for pa, pb in zip(abl01.reranker.parameters(), abl02_member.reranker.parameters())
    )
    assert diverged, "ensemble members' reranker params must diverge after independent fit()"


# --------------------------------------------------------------------------- #
# Adversarial-review fix (structural purity guard): SingleTerminalGFlowNetTrainer's
# single-variable-ablation contract must hold STRUCTURALLY, not just behaviorally --
# it may override ONLY tb_loss. A future edit silently overriding another inherited
# method (sample_forest, candidate_children, fit, checkpoint helpers, ...) would break
# the ablation's single-variable-change guarantee without any behavioral test catching it.
# --------------------------------------------------------------------------- #

def test_single_terminal_trainer_overrides_only_tb_loss():
    inherited_methods = [
        "sample_forest",
        "candidate_children",
        "policy_logits",
        "_reranker_child_logits",
        "fit",
        "prewarm_caches",
        "_expand_many",
        "_load_caches",
        "save_caches",
        "_save_train_ckpt",
        "_load_train_ckpt",
    ]
    for name in inherited_methods:
        assert hasattr(SetGFlowNetTrainer, name), f"parent has no method {name!r} (name drift?)"
        assert getattr(SingleTerminalGFlowNetTrainer, name) is getattr(SetGFlowNetTrainer, name), (
            f"{name!r} must be inherited UNCHANGED from SetGFlowNetTrainer, not overridden"
        )

    own_members = [
        n for n in SingleTerminalGFlowNetTrainer.__dict__
        if not (n.startswith("__") and n.endswith("__"))
    ]
    assert own_members == ["tb_loss"], (
        f"SingleTerminalGFlowNetTrainer must define ONLY tb_loss; found {own_members}"
    )


def test_child_cache_lru_bounded_prevents_unbounded_growth():
    """The forest-rollout environment caches must be LRU-bounded so top_k~200 rollouts don't
    OOM at scale (the first end-to-end run was jetsam-killed by an unbounded _child_cache)."""
    from collections import OrderedDict
    from grail_metabolism.model.set_gflownet import SetGFlowNetTrainer

    # Eviction primitive: cap keeps the most-recent entries; cap<=0 disables the bound.
    c = OrderedDict((str(i), i) for i in range(5))
    SetGFlowNetTrainer._trim_cache(c, 3)
    assert list(c) == ["2", "3", "4"]
    SetGFlowNetTrainer._trim_cache(c, 0)
    assert len(c) == 3

    # candidate_children honors the cap and LRU-evicts the oldest state.
    trainer = SetGFlowNetTrainer(
        _MiniGen(), BiEncoderReranker(in_channels=SINGLE_NODE_DIM),
        GFlowNetConfig(max_depth=2, top_k=200, child_cache_max=1),
        annotated_ik_fn=lambda root: set(),
    )
    trainer.candidate_children("CCO")
    trainer.candidate_children("CCO O")
    assert len(trainer._child_cache) == 1
    assert "CCO O" in trainer._child_cache and "CCO" not in trainer._child_cache

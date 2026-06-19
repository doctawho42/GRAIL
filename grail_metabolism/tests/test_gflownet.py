"""Tests for GFlowNet (Trajectory Balance) generator training (model/gflownet.py)."""
from __future__ import annotations

import torch

from grail_metabolism.config import GFlowNetConfig
from grail_metabolism.metrics import _inchikey
from grail_metabolism.model.gflownet import GFlowNetTrainer, annotation_reward_fn
from grail_metabolism.model.grail import summon_the_grail
from grail_metabolism.utils.seed import seed_everything

# Two phase-I/phase-II rules: CH2OH -> CHO oxidation, and aromatic O-methylation.
RULES = ["[CH2:1][OH:2]>>[CH:1]=[O:2]", "[c:2][OX2H:1]>>[c:2][O:1]C"]
SUB = "OCc1ccccc1O"
TARGET_IK = _inchikey("COc1ccccc1C=O")  # requires BOTH transformations (a 2-step terminal)


def _new_trainer(cfg):
    model = summon_the_grail(RULES)
    reward = annotation_reward_fn({SUB: {TARGET_IK}}, cfg.beta)
    return GFlowNetTrainer(model.generator, cfg, reward, device=torch.device("cpu"))


def _path_logpf(gen):
    """Differentiable log P_F of the fixed 2-step path SUB -> oxidation -> +methylation = TARGET.

    Deterministic (no sampling), so it isolates the TB learning signal from rollout noise.
    """
    path = [(SUB, "O=Cc1ccccc1O"), ("O=Cc1ccccc1O", "COc1ccccc1C=O")]
    total = torch.zeros(())
    for state, nxt in path:
        children, child_logits, stop_logit = gen.action_distribution(state)
        logits = torch.cat([child_logits, stop_logit]) if children else stop_logit
        logp = torch.log_softmax(logits, dim=0)
        nxt_ik = _inchikey(nxt)
        idx = next(i for i, c in enumerate(children) if _inchikey(c) == nxt_ik)
        total = total + logp[idx]
    return total


def test_action_distribution_is_differentiable_and_has_stop():
    seed_everything(0)
    gen = summon_the_grail(RULES).generator
    children, child_logits, stop_logit = gen.action_distribution(SUB)
    assert children, "expected at least one applicable child"
    assert child_logits.requires_grad and stop_logit.requires_grad  # gradients flow to the policy
    assert child_logits.shape[0] == len(children)
    assert stop_logit.shape[0] == 1  # the STOP action logit


def test_sample_trajectory_respects_max_depth():
    seed_everything(0)
    cfg = GFlowNetConfig(max_depth=1, per_node_top_k=8, epsilon=0.0)
    trainer = _new_trainer(cfg)
    # With max_depth=1 the terminal is the parent or a single-step child, never deeper.
    children = set(trainer.generator.action_distribution(SUB)[0])
    for _ in range(20):
        terminal, _ = trainer.sample_trajectory(SUB)
        assert terminal == SUB or terminal in children


def test_trajectory_balance_loss_is_finite_and_backprops():
    seed_everything(0)
    cfg = GFlowNetConfig(max_depth=2, per_node_top_k=8, beta=6.0, epsilon=0.0)
    trainer = _new_trainer(cfg)
    loss = trainer.trajectory_balance_loss(SUB)
    assert torch.isfinite(loss)
    loss.backward()  # gradient flows to generator params + logZ
    assert trainer.log_z.grad is not None


def test_tb_objective_increases_target_path_probability():
    # Deterministic check that the Trajectory-Balance objective produces the correct
    # learning signal: minimizing TB loss for the high-reward target path raises that
    # path's probability under the forward policy.
    seed_everything(0)
    cfg = GFlowNetConfig(max_depth=2, per_node_top_k=8, beta=6.0)
    trainer = _new_trainer(cfg)
    gen = trainer.generator

    before = float(_path_logpf(gen).item())
    optimizer = torch.optim.Adam(list(gen.parameters()) + [trainer.log_z], lr=1e-2)
    log_reward = cfg.beta  # log(exp(beta)) for a hit
    for _ in range(40):
        loss = (trainer.log_z.squeeze() + _path_logpf(gen) - log_reward) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = float(_path_logpf(gen).item())
    assert after > before  # the rewarded path became more probable


def test_trainer_fit_runs_and_logz_finite():
    seed_everything(0)
    cfg = GFlowNetConfig(epochs=3, batch_substrates=1, max_depth=2, per_node_top_k=8, beta=4.0, epsilon=0.1)
    trainer = _new_trainer(cfg)
    trainer.fit([SUB])
    assert len(trainer.loss_history_) == cfg.epochs
    assert torch.isfinite(trainer.log_z).all()
    assert all(l == l for l in trainer.loss_history_)  # no NaNs

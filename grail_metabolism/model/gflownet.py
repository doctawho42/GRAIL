"""GFlowNet (Trajectory Balance) training of the generator over the metabolic tree.

Phase 2 of multi-step generation. The metabolic-tree environment (state = molecule,
action = apply a rule, terminal = STOP / depth cap) is shared with model.multistep; here
the deterministic beam policy is replaced by a LEARNED forward policy P_F trained so that
terminal sampling probability is proportional to a reward.

- Forward policy P_F(action | molecule) = softmax over the generator's per-rule logits
  (mapped to child products) plus a learned STOP logit (Generator.action_distribution).
- Reward (training): annotation-based by default — exp(beta) if the terminal hits an
  annotated metabolite of the parent, else 1.0. This directly optimizes multi-step recall
  toward KNOWN metabolites and needs no trained filter (the generator trains before the
  filter in the ensemble). A filter-based reward is supported via a custom reward_fn.
- Objective: Trajectory Balance, L = (logZ + sum_t log P_F(a_t|s_t) - log R(x))^2, with a
  learned scalar logZ and a tree-structured backward policy (P_B = 1 along the sampled
  path, so its log term is 0). InChIKey-canonical states make "same molecule" well-defined.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..config import GFlowNetConfig
from ..metrics import _inchikey
from .generator import Generator

RewardFn = Callable[[str, str], float]


def annotation_reward_fn(annotated_ik: Dict[str, set], beta: float) -> RewardFn:
    """exp(beta) when the terminal's InChIKey is an annotated metabolite of the parent, else 1.0."""
    hit_reward = math.exp(float(beta))

    def reward(parent: str, terminal: str) -> float:
        return hit_reward if _inchikey(terminal) in annotated_ik.get(parent, set()) else 1.0

    return reward


class GFlowNetTrainer:
    def __init__(
        self,
        generator: Generator,
        config: GFlowNetConfig,
        reward_fn: RewardFn,
        device: Optional[torch.device] = None,
    ) -> None:
        self.generator = generator
        self.config = config
        self.reward_fn = reward_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Learned log-partition for Trajectory Balance.
        self.log_z = nn.Parameter(torch.zeros(1, device=self.device))
        self.loss_history_: List[float] = []

    @torch.no_grad()
    def _sample_action(self, log_probs: torch.Tensor, n: int) -> int:
        if self.config.epsilon > 0.0 and float(torch.rand(())) < self.config.epsilon:
            return int(torch.randint(0, n, (1,)))
        return int(torch.multinomial(log_probs.exp(), 1))

    def sample_trajectory(self, parent: str) -> Tuple[str, torch.Tensor]:
        """Roll out a trajectory under P_F. Returns (terminal_smiles, sum_log_P_F) where
        sum_log_P_F is differentiable w.r.t. the generator + stop head."""
        state = parent
        log_pf_terms: List[torch.Tensor] = []
        for _ in range(max(1, self.config.max_depth)):
            children, child_logits, stop_logit = self.generator.action_distribution(state)
            logits = torch.cat([child_logits, stop_logit]) if children else stop_logit
            log_probs = F.log_softmax(logits, dim=0)
            n = int(logits.shape[0])
            action = self._sample_action(log_probs.detach(), n)
            log_pf_terms.append(log_probs[action])
            stop_index = len(children)  # STOP is the last action; 0 when there are no children
            if action == stop_index:
                break
            state = children[action]
        if log_pf_terms:
            sum_log_pf = torch.stack(log_pf_terms).sum()
        else:
            sum_log_pf = torch.zeros((), device=self.device)
        return state, sum_log_pf

    def trajectory_balance_loss(self, parent: str) -> torch.Tensor:
        terminal, sum_log_pf = self.sample_trajectory(parent)
        reward = float(self.reward_fn(parent, terminal))
        log_reward = math.log(max(reward, 1e-30))
        # Tree-structured backward policy P_B = 1 along the sampled path -> log P_B term = 0.
        return (self.log_z.squeeze() + sum_log_pf - log_reward) ** 2

    def fit(self, substrates: Sequence[str], epochs: Optional[int] = None, verbose: bool = False) -> "GFlowNetTrainer":
        cfg = self.config
        epochs = epochs if epochs is not None else cfg.epochs
        subs = [s for s in substrates if s]
        if not subs:
            return self
        self.generator.to(self.device)
        self.log_z = self.log_z.to(self.device)
        trainable = [p for p in self.generator.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            [{"params": trainable, "lr": cfg.lr}, {"params": [self.log_z], "lr": cfg.logz_lr}]
        )
        self.loss_history_ = []
        for epoch in range(epochs):
            self.generator.train()
            order = torch.randperm(len(subs)).tolist()
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, len(subs), cfg.batch_substrates):
                batch = [subs[i] for i in order[start : start + cfg.batch_substrates]]
                losses = [self.trajectory_balance_loss(s) for s in batch]
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                num_batches += 1
            avg = epoch_loss / max(num_batches, 1)
            self.loss_history_.append(avg)
            if verbose:
                print(f"gflownet epoch={epoch + 1} tb_loss={avg:.4f} logZ={float(self.log_z):.3f}", flush=True)
        return self

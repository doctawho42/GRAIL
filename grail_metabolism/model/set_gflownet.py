"""Multi-step Set-GFlowNet over the rule forest. Terminal = a set of metabolites;
reward = PU set-coverage; forward policy = the Stage-2a reranker; backward = analytic
1/#leaves. See docs/superpowers/specs/2026-07-01-set-gflownet-stage2b-design.md."""
from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Optional


@dataclass(frozen=True)
class ForestState:
    root: str
    max_depth: int
    max_size: int
    parent: Dict[str, str] = field(default_factory=dict)   # child_ik -> parent_ik

    def add(self, parent_ik: str, child_ik: str) -> "ForestState":
        new_parent = dict(self.parent)
        new_parent[child_ik] = parent_ik
        return replace(self, parent=new_parent)

    def terminal_set(self) -> FrozenSet[str]:
        return frozenset(self.parent.keys())

    def depth_of(self, ik: str) -> int:
        d = 0
        while ik in self.parent:
            ik = self.parent[ik]; d += 1
        return d

    def leaves(self) -> List[str]:
        parents = set(self.parent.values())
        return [ik for ik in self.parent if ik not in parents]

    def frontier(self) -> List[str]:
        nodes = [self.root] + list(self.parent.keys())
        return [n for n in nodes
                if self.depth_of(n) < self.max_depth and len(self.parent) < self.max_size]


def set_coverage_logreward(terminal_set, annotated_ik, beta: float, lam: float) -> float:
    """log R(S) = beta * (TP - lam*|S|). PU-aware: non-annotated members cost only lam
    (size), never a false-negative penalty."""
    tp = len(terminal_set & annotated_ik)
    return float(beta) * (tp - float(lam) * len(terminal_set))


def log_pb_trajectory(post_add_states) -> float:
    """Sum of log(1/#leaves) over the states reached AFTER each ADD action. The last-added
    node of a forest must be a current leaf, so P_B(remove leaf)=1/#leaves is the exact
    analytic backward for forest construction."""
    total = 0.0
    for st in post_add_states:
        n_leaves = max(len(st.leaves()), 1)
        total += math.log(1.0 / n_leaves)
    return total

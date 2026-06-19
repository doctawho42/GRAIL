"""Multi-step (depth>1) metabolite generation over a metabolic tree.

Builds a metabolic tree from a parent molecule by iteratively applying the generator's
per-molecule rule policy and pruning with the filter as a terminal reward. Reuses the
ALREADY-TRAINED generator and filter — no new training. This same environment is the
intended substrate for a future GFlowNet forward policy (the policy swaps, the env stays).

Design (see docs plan):
- state = molecule, action = apply a rule, terminal reward = filter(parent, node).
- node rank = filter(parent, node) corrected by a generator path-prior (aggregate of the
  per-step generator scores along the path to the node).
- expand into the next depth ONLY filter-plausible nodes (filter(parent, node) >= tau).
- output = all visited nodes (each is a candidate metabolite of the parent), InChIKey-
  deduped, ranked by node score, capped by max_output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from ..config import MultiStepConfig
from ..metrics import _inchikey
from ..utils.preparation import _standardize_smiles_cached

if TYPE_CHECKING:  # avoid an import cycle with wrapper (it imports this module lazily)
    from .wrapper import GFilter, GGenerator


@dataclass(frozen=True)
class TreeNode:
    smiles: str                      # standardized SMILES
    inchikey: str                    # dedup / matching key (metrics._inchikey)
    depth: int                       # 0 = parent
    parent_smiles: Optional[str]
    rule_score: float                # generator score of the step that produced THIS node
    path_scores: Tuple[float, ...] = field(default_factory=tuple)  # per-step gen scores root->node


def _path_prior(path_scores: Sequence[float], mode: str) -> float:
    if not path_scores:
        return 1.0
    values = [float(s) for s in path_scores]
    if mode == "min":
        return min(values)
    if mode == "product":
        out = 1.0
        for v in values:
            out *= v
        return out
    if mode == "noisy_or":
        out = 1.0
        for v in values:
            out *= (1.0 - min(max(v, 0.0), 1.0))
        return 1.0 - out
    return sum(values) / len(values)  # mean (default)


class MetabolicTree:
    def __init__(
        self,
        generator: "GGenerator",
        filter: "GFilter",
        config: MultiStepConfig,
        rule_threshold: Optional[float] = None,
    ) -> None:
        self.generator = generator
        self.filter = filter
        self.config = config
        # Per-expansion generator threshold; default to the generator's calibrated one.
        self.rule_threshold = (
            rule_threshold if rule_threshold is not None else getattr(generator, "calibrated_threshold", None)
        )

    def _normalize(self, smiles: str) -> Optional[str]:
        try:
            return _standardize_smiles_cached(smiles)
        except Exception:
            return None

    def _combine(self, reward: float, prior: float) -> float:
        return float(reward) * (max(float(prior), 0.0) ** float(self.config.reward_prior_weight))

    def _score_terminal(self, parent_smiles: str, smiles_list: List[str]) -> List[float]:
        if not smiles_list:
            return []
        if hasattr(self.filter, "score_batch"):
            return [float(s) for s in self.filter.score_batch(parent_smiles, smiles_list)]
        return [float(self.filter.score(parent_smiles, s)) for s in smiles_list]

    def expand(self, node: TreeNode) -> List[TreeNode]:
        """One generation step from `node`: generator-selected rules -> standardized children."""
        scored = self.generator.generate_scored(
            node.smiles, top_k=self.config.per_node_top_k, threshold=self.rule_threshold
        )
        children: List[TreeNode] = []
        for child_smiles, gen_score in scored:
            normalized = self._normalize(child_smiles)
            if normalized is None:
                continue
            children.append(
                TreeNode(
                    smiles=normalized,
                    inchikey=_inchikey(normalized),
                    depth=node.depth + 1,
                    parent_smiles=node.smiles,
                    rule_score=float(gen_score),
                    path_scores=node.path_scores + (float(gen_score),),
                )
            )
        return children

    def beam_search(self, parent_smiles: str, max_output: Optional[int] = None) -> List[Tuple[str, float]]:
        cfg = self.config
        root_smiles = self._normalize(parent_smiles)
        if root_smiles is None:
            return []
        parent_ik = _inchikey(root_smiles)
        root = TreeNode(root_smiles, parent_ik, depth=0, parent_smiles=None, rule_score=1.0, path_scores=())

        visited: Dict[str, Tuple[TreeNode, float]] = {}  # ik -> (node, node_score), best per structure
        frontier: List[TreeNode] = [root]
        expansions = 0

        for _ in range(max(1, cfg.max_depth)):
            # (a) expand current frontier into raw children (bounded by node_budget)
            raw_children: List[TreeNode] = []
            for node in frontier:
                if expansions >= cfg.node_budget:
                    break
                raw_children.extend(self.expand(node))
                expansions += 1

            # (b) collapse by InChIKey, drop the parent structure, keep higher path-prior on dupes
            best_by_ik: Dict[str, TreeNode] = {}
            for child in raw_children:
                if child.inchikey == parent_ik:
                    continue
                prev = best_by_ik.get(child.inchikey)
                if prev is None or _path_prior(child.path_scores, cfg.prior_aggregation) > _path_prior(prev.path_scores, cfg.prior_aggregation):
                    best_by_ik[child.inchikey] = child
            new_children = list(best_by_ik.values())
            if not new_children:
                break

            # (c) terminal reward = filter(ORIGINAL parent, child), batched
            rewards = self._score_terminal(root_smiles, [c.smiles for c in new_children])

            # (d) record every node as a candidate metabolite of the parent (best score per structure)
            scored_children: List[Tuple[TreeNode, float, float]] = []
            for child, reward in zip(new_children, rewards):
                node_score = self._combine(reward, _path_prior(child.path_scores, cfg.prior_aggregation))
                scored_children.append((child, reward, node_score))
                prev = visited.get(child.inchikey)
                if prev is None or node_score > prev[1]:
                    visited[child.inchikey] = (child, node_score)

            # (e) filter-gated expansion: only plausible nodes seed the next depth; keep top-B
            survivors = [(c, ns) for (c, r, ns) in scored_children if r >= cfg.expand_threshold]
            survivors.sort(key=lambda cs: (-cs[1], cs[0].smiles))
            frontier = [c for c, _ in survivors[: max(1, cfg.beam_width)]]
            if not frontier or expansions >= cfg.node_budget:
                break

        ranked = sorted(
            (item for ik, item in visited.items() if ik != parent_ik),
            key=lambda item: (-item[1], item[0].smiles),
        )
        out = [(node.smiles, score) for node, score in ranked]
        if max_output is not None and max_output > 0:
            out = out[:max_output]
        return out

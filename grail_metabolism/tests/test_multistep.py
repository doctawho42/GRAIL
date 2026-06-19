"""Tests for multi-step (depth>1) metabolite generation (model/multistep.py)."""
from __future__ import annotations

from typing import Dict, List, Tuple

from grail_metabolism.config import MultiStepConfig
from grail_metabolism.metrics import _inchikey
from grail_metabolism.model.multistep import MetabolicTree, TreeNode
from grail_metabolism.model.wrapper import ModelWrapper
from grail_metabolism.utils.preparation import _standardize_smiles_cached as norm


class FakeGen:
    """Per-molecule rule policy stub: table maps a SMILES to its scored children."""

    def __init__(self, table: Dict[str, List[Tuple[str, float]]], calibrated_threshold=None):
        self.table = {norm(k): v for k, v in table.items()}
        self.calibrated_threshold = calibrated_threshold
        self.calls: List[str] = []

    def generate_scored(self, sub, top_k=None, threshold=None):
        self.calls.append(sub)
        return list(self.table.get(sub, []))


class FakeFilter:
    """Terminal reward stub: score keyed by (normalized) product SMILES, default 1.0."""

    def __init__(self, scores: Dict[str, float], calibrated_threshold=0.0):
        self.scores = {norm(k): v for k, v in scores.items()}
        self.calibrated_threshold = calibrated_threshold

    def score_batch(self, parent, prods):
        return [self.scores.get(p, 1.0) for p in prods]

    def score(self, parent, prod):
        return self.scores.get(prod, 1.0)


def _tree(table, scores, **cfg):
    return MetabolicTree(FakeGen(table), FakeFilter(scores), MultiStepConfig(**cfg))


def test_depth1_dispatch_equals_single_step():
    # The wrapper must NOT route to the tree at max_depth=1, so output is byte-identical
    # to the single-step path.
    table = {"c1ccccc1": [("Oc1ccccc1", 0.9), ("Nc1ccccc1", 0.8)]}
    model = ModelWrapper(FakeFilter({}, calibrated_threshold=0.0), FakeGen(table))
    single = model.generate("c1ccccc1")
    depth1 = model.generate("c1ccccc1", multistep=MultiStepConfig(enabled=True, max_depth=1))
    assert single == depth1


def test_inchikey_dedup_of_visited_set():
    # Two equivalent representations of acetic acid collapse to one candidate.
    tree = _tree({"CCO": [("OC(=O)C", 0.9), ("CC(O)=O", 0.8)]}, {}, max_depth=1)
    out = tree.beam_search("CCO")
    assert len(out) == 1
    assert _inchikey(out[0][0]) == _inchikey("CC(=O)O")


def test_node_budget_caps_expansions():
    table = {
        "c1ccccc1": [("Oc1ccccc1", 0.9)],
        "Oc1ccccc1": [("Oc1ccccc1O", 0.9)],
        "Oc1ccccc1O": [("Oc1cc(O)cc(O)c1", 0.9)],
    }
    gen = FakeGen(table)
    tree = MetabolicTree(gen, FakeFilter({}), MultiStepConfig(max_depth=3, node_budget=1, beam_width=10))
    tree.beam_search("c1ccccc1")
    assert len(gen.calls) == 1  # node_budget=1 -> only the root is ever expanded


def test_expand_threshold_gates_next_depth():
    table = {
        "c1ccccc1": [("Oc1ccccc1", 0.9), ("Nc1ccccc1", 0.9)],
        "Oc1ccccc1": [("Oc1ccccc1O", 0.9)],   # phenol -> catechol (should appear)
        "Nc1ccccc1": [("Nc1ccccc1N", 0.9)],   # aniline -> diamine (should NOT appear)
    }
    scores = {"Oc1ccccc1": 0.9, "Nc1ccccc1": 0.1}  # aniline below tau -> not expanded
    tree = _tree(table, scores, max_depth=2, expand_threshold=0.5, beam_width=10)
    out_ik = {_inchikey(s) for s, _ in tree.beam_search("c1ccccc1")}
    assert _inchikey("Oc1ccccc1O") in out_ik   # child of the plausible phenol
    assert _inchikey("Nc1ccccc1N") not in out_ik  # child of the gated aniline
    assert _inchikey("Nc1ccccc1") in out_ik    # aniline itself is still a recorded candidate


def test_known_two_step_example():
    table = {
        "CCCCc1ccccc1": [("CCCC(O)c1ccccc1", 0.8)],      # benzylic hydroxylation
        "CCCC(O)c1ccccc1": [("CCCC(=O)c1ccccc1", 0.7)],  # further oxidation to ketone
    }
    # path-level check on the env
    one = _tree(table, {}, max_depth=1).beam_search("CCCCc1ccccc1")
    two = _tree(table, {}, max_depth=2).beam_search("CCCCc1ccccc1")
    one_ik = {_inchikey(s) for s, _ in one}
    two_ik = {_inchikey(s) for s, _ in two}
    assert _inchikey("CCCC(=O)c1ccccc1") not in one_ik  # unreachable single-step
    assert _inchikey("CCCC(=O)c1ccccc1") in two_ik       # reached at depth 2

    # node carries depth and full path of step scores
    env = _tree(table, {}, max_depth=2)
    root = TreeNode(norm("CCCCc1ccccc1"), _inchikey("CCCCc1ccccc1"), 0, None, 1.0, ())
    d1 = env.expand(root)
    assert len(d1) == 1 and d1[0].depth == 1 and d1[0].path_scores == (0.8,)
    d2 = env.expand(d1[0])
    assert d2[0].depth == 2 and d2[0].path_scores == (0.8, 0.7)


def test_max_output_cap_multistep():
    table = {"c1ccccc1": [("Oc1ccccc1", 0.9), ("Nc1ccccc1", 0.8), ("Clc1ccccc1", 0.7)]}
    tree = _tree(table, {}, max_depth=1)
    assert len(tree.beam_search("c1ccccc1", max_output=2)) == 2
    assert len(tree.beam_search("c1ccccc1")) == 3


def test_parent_structure_excluded():
    # A rule that regenerates the parent must not surface the parent in the output.
    table = {"c1ccccc1": [("c1ccccc1", 0.9), ("Oc1ccccc1", 0.8)]}
    out_ik = {_inchikey(s) for s, _ in _tree(table, {}, max_depth=1).beam_search("c1ccccc1")}
    assert _inchikey("c1ccccc1") not in out_ik
    assert _inchikey("Oc1ccccc1") in out_ik

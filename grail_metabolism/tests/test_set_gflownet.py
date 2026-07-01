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

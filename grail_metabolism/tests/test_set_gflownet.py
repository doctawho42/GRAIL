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

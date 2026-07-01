from scripts.census_multistep import census_depth2

class _StubGen:
    """generate_scored_with_details(sub) -> children of `sub` from a fixed graph."""
    def __init__(self, graph):
        self._g = graph  # dict: smiles -> list of (child_smiles, gen_score, rule_id)
    def generate_scored_with_details(self, sub, top_k=200, max_pool=None, compute_sites=False):
        return list(self._g.get(sub, []))

def _ik(s):  # identity "InChIKey" for the test: the smiles itself
    return s

def test_depth2_only_counts_chain_not_reachable_in_one_step(monkeypatch):
    import scripts.census_multistep as m
    monkeypatch.setattr(m, "_tautomer_inchikey", _ik)
    # root R -> A (depth1). A -> B (so B is depth2-only). R does NOT reach B directly.
    gen = _StubGen({"R": [("A", 0.9, 1)], "A": [("B", 0.8, 2)]})
    annotated_ik = {"A", "B"}          # both A and B are annotated metabolites of R
    out = census_depth2("R", annotated_ik, gen, top_k=10, max_pool=10)
    assert out["n_annot"] == 2
    assert out["depth1"] == 1          # A
    assert out["depth2_only"] == 1     # B (reachable only via R->A->B)
    assert out["unreach"] == 0

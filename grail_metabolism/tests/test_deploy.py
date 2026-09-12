"""Deploy-build guards: release ranking parity and CLI behavior added by the way2drug deploy plan."""
from grail_metabolism.model.ranking import release_order


def _pool():
    # filter, generator, key; two candidates share a key to exercise dedup
    return [
        {"smiles": "A", "filter": 0.9, "generator": 0.2, "key": "k1"},
        {"smiles": "B", "filter": 0.1, "generator": 0.9, "key": "k2"},
        {"smiles": "A2", "filter": 0.5, "generator": 0.5, "key": "k1"},
        {"smiles": "S", "filter": 0.8, "generator": 0.8, "key": "parent"},
    ]


def test_release_order_dedups_caps_fuses_and_drops_parent():
    ordered = release_order(_pool(), self_key="parent", cap=100, rrf_k=60)
    keys = [c["key"] for c in ordered]
    assert "parent" not in keys          # parent dropped
    assert keys.count("k1") == 1         # deduped by key
    assert set(keys) == {"k1", "k2"}     # only non-parent keys survive
    # k1 kept the higher product row (0.9*0.2=0.18 vs 0.5*0.5=0.25 -> keeps A2)
    kept_k1 = next(c for c in ordered if c["key"] == "k1")
    assert kept_k1["smiles"] == "A2"


def test_release_order_matches_the_analysis_deployed_order():
    import sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from dehydrogenation_diagnostic import deployed_order  # the analysis order
    from grail_metabolism.model.ranking import release_order
    pool = [
        {"smiles": "A", "filter": 0.9, "generator": 0.2, "key": "k1"},
        {"smiles": "B", "filter": 0.1, "generator": 0.9, "key": "k2"},
        {"smiles": "C", "filter": 0.4, "generator": 0.7, "key": "k3"},
    ]
    lib = [c["key"] for c in release_order(pool, self_key="parent")]
    assert lib == deployed_order(pool, "parent")

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


def test_subset_rule_tensors_selects_rows_for_per_rule_tensors_only():
    import torch
    from scripts.build_released_checkpoint import subset_rule_tensors
    sd = {
        "bias": torch.tensor([10.0, 11.0, 12.0]),
        "rule_prior_logits": torch.tensor([0.0, 1.0, 2.0]),
        "pos_weight": torch.ones(3),
        "propensity_weight": torch.ones(3),
        "parse.id_embedding.weight": torch.tensor([[1.0], [2.0], [3.0]]),
        "encoder.lin.weight": torch.eye(4),   # rule-agnostic, must be untouched
    }
    out = subset_rule_tensors(sd, kept_idx=[0, 2])
    assert out["bias"].tolist() == [10.0, 12.0]
    assert out["rule_prior_logits"].tolist() == [0.0, 2.0]
    assert out["parse.id_embedding.weight"].tolist() == [[1.0], [3.0]]
    assert torch.equal(out["encoder.lin.weight"], sd["encoder.lin.weight"])


import pathlib as _pathlib
import pytest as _pytest

_HAVE_FULL_BANK_AND_CHECKPOINT = (
    (_pathlib.Path(__file__).resolve().parents[2] / "grail_metabolism/resources/extended_smirks.txt").exists()
    and (_pathlib.Path(__file__).resolve().parents[2] / "artifacts/full5000_implicit/checkpoints/generator.pt").exists()
)


@_pytest.mark.skipif(
    not _HAVE_FULL_BANK_AND_CHECKPOINT,
    reason="full bank/checkpoint not shipped; parity checked where present",
)
def test_released_generator_scores_match_full_on_kept_rules():
    import torch, pathlib
    from grail_metabolism.config import GeneratorConfig
    from grail_metabolism.workflows.factory import build_generator
    root = pathlib.Path(__file__).resolve().parents[2]
    def load(ckpt, bank):
        rules = [l.strip() for l in open(bank) if l.strip()]
        p = torch.load(ckpt, map_location="cpu", weights_only=False)
        g = build_generator(GeneratorConfig(**p["arch"]), rules); g.load_state_dict(p["state_dict"], strict=False); g.eval()
        return g, rules
    full_g, full_rules = load(root / "artifacts/full5000_implicit/checkpoints/generator.pt",
                              root / "grail_metabolism/resources/extended_smirks.txt")
    rel_g, rel_rules = load(root / "artifacts/full5000_released/checkpoints/generator.pt",
                            root / "grail_metabolism/resources/extended_smirks_released.txt")
    sub = "CCO"
    with torch.no_grad():
        fs, _ = full_g.score_rules(sub, return_mask=True)
        rs, _ = rel_g.score_rules(sub, return_mask=True)
    pos = {r: i for i, r in enumerate(full_rules)}
    kept = [pos[r] for r in rel_rules]
    assert torch.allclose(torch.as_tensor(rs), torch.as_tensor(fs)[kept], atol=1e-5)


def test_predict_rows_handles_ok_and_unparseable_and_limits_top_k():
    from grail_metabolism.deploy.predict_cli import predict_rows
    class Stub:
        def rank(self, smiles, top_k):
            if smiles == "BAD":
                raise ValueError("unparseable")
            return [("m1", 0.9), ("m2", 0.8), ("m3", 0.7)][:top_k]
    rows = predict_rows(Stub(), [("s1", "CCO"), ("s2", "BAD")], top_k=2, timeout_seconds=1)
    assert [(r["parent_id"], r["rank"], r["metabolite_smiles"], r["status"]) for r in rows] == [
        ("s1", 1, "m1", "ok"), ("s1", 2, "m2", "ok"),
        ("s2", 0, "", "no_parse"),
    ]


import pathlib
import pytest

_HAVE_RELEASED_CHECKPOINT = (
    pathlib.Path(__file__).resolve().parents[2] / "artifacts/full5000_released/checkpoints/generator.pt"
).exists()


@pytest.mark.skipif(not _HAVE_RELEASED_CHECKPOINT, reason="released checkpoint not built in this checkout")
def test_cli_end_to_end_small(tmp_path):
    from grail_metabolism.deploy.predict_cli import main
    inp = tmp_path / "in.smi"
    inp.write_text("s1\tCCO\nBADSMILES\n")
    out = tmp_path / "out.tsv"
    assert main([str(inp), str(out), "--top-k", "5"]) == 0
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == ["parent_id", "rank", "metabolite_smiles", "score", "status"]
    ids = {ln.split("\t")[0] for ln in lines[1:]}
    assert ids == {"s1", "2"}                      # line-2 id defaulted to its line number
    assert any(ln.split("\t")[4] == "no_parse" for ln in lines[1:])  # BADSMILES flagged


import signal as _signal
import time as _time


@pytest.mark.skipif(not hasattr(_signal, "SIGALRM"), reason="SIGALRM not available on this platform")
def test_predict_rows_enforces_per_substrate_timeout():
    from grail_metabolism.deploy.predict_cli import predict_rows

    class SlowStub:
        def rank(self, smiles, top_k):
            if smiles == "SLOW":
                _time.sleep(2)
            return [("m1", 0.9)][:top_k]

    rows = predict_rows(SlowStub(), [("s1", "SLOW"), ("s2", "CCO")], top_k=1, timeout_seconds=1)
    assert [(r["parent_id"], r["status"]) for r in rows] == [("s1", "timeout"), ("s2", "ok")]


_RELEASED_BANK = (_pathlib.Path(__file__).resolve().parents[2]
                  / "grail_metabolism/resources/extended_smirks_released.txt")


@pytest.mark.skipif(not _RELEASED_BANK.exists(), reason="released bank not present")
def test_rule_graph_disk_cache_is_opt_in_and_gives_identical_graphs(tmp_path, monkeypatch):
    """The deploy's start-up cache must be invisible: off by default, and identical when on.

    Building the rule graphs is most of the cost of constructing a generator over the released
    bank, and the deploy pays it per process. The cache is only safe if a graph read from disk is
    the graph that would have been built, so that is asserted tensor by tensor rather than trusted.
    """
    import torch
    from grail_metabolism.model.generator import Generator
    from grail_metabolism.workflows import factory

    rules = [ln.strip() for ln in _RELEASED_BANK.read_text().splitlines() if ln.strip()][:40]
    monkeypatch.setenv(factory.RULE_GRAPH_CACHE_DIR_ENV, str(tmp_path))

    # Off by default: the directory is named but nothing may be written to it.
    monkeypatch.delenv(factory.RULE_GRAPH_CACHE_ENV, raising=False)
    fresh = factory.build_rule_dict(rules)
    assert list(tmp_path.glob("*.pt")) == []

    # On: the first call writes one file, keyed by this rule list.
    monkeypatch.setenv(factory.RULE_GRAPH_CACHE_ENV, "1")
    factory.build_rule_dict(rules)
    assert len(list(tmp_path.glob("*.pt"))) == 1

    # Drop the in-process cache so the next build has to come off disk.
    Generator._rule_graph_cache.clear()
    from_disk = factory.build_rule_dict(rules)

    assert set(from_disk) == set(fresh)
    for rule in rules:
        a, b = fresh[rule], from_disk[rule]
        assert torch.equal(a.x, b.x)
        assert torch.equal(a.edge_index, b.edge_index)
        assert (a.edge_attr is None) == (b.edge_attr is None)
        if a.edge_attr is not None:
            assert torch.equal(a.edge_attr, b.edge_attr)

    # A different rule list must not read this entry: the key is the input.
    other = factory.build_rule_dict(rules[:20])
    assert len(other) == 20
    assert len(list(tmp_path.glob("*.pt"))) == 2

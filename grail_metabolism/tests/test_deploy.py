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

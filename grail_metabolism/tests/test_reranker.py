"""Tests for rule + firing-site provenance from the generator (Stage 2a Task 1)."""
import grail_metabolism.utils.preparation as prep  # noqa: F401
from grail_metabolism.workflows.factory import build_generator
from grail_metabolism.config import GeneratorConfig

RULES = ["[CH2:1][OH:2]>>[CH:1]=[O:2]", "[c:1][H:2]>>[c:1][OH]"]
SUB = "OCc1ccccc1"


def _gen():
    # GeneratorConfig defaults: in_channels=16, rule hidden_dims=[128,128,128]
    return build_generator(GeneratorConfig(), RULES)


def test_generate_scored_with_details_carries_rule_and_site():
    gen = _gen()
    detailed = gen.generate_scored_with_details(SUB, top_k=50)
    assert detailed, "expected at least one candidate"
    for smiles, gscore, rule_id, sites in detailed:
        assert isinstance(smiles, str) and isinstance(gscore, float)
        assert 0 <= rule_id < gen.num_rules
        assert isinstance(sites, tuple) and all(isinstance(a, int) for a in sites)


def test_generate_scored_public_api_unchanged():
    gen = _gen()
    plain = gen.generate_scored(SUB, top_k=50)
    assert isinstance(plain, list) and all(len(t) == 2 for t in plain)
    # same candidate set as the detailed path
    assert {s for s, _ in plain} == {s for s, _, _, _ in gen.generate_scored_with_details(SUB, top_k=50)}


def test_generate_scored_with_details_budget_cap():
    gen = _gen()
    results = gen.generate_scored_with_details(SUB, top_k=100)
    assert len(results) <= 100

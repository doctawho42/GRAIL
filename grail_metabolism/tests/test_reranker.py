"""Tests for rule + firing-site provenance from the generator (Stage 2a Task 1)
and the minimal listwise reranker (Stage 2a Task 2/3)."""
import torch

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
    # full list equality: (smiles, gen_score) must match exactly, same order
    assert plain == [(s, sc) for s, sc, _, _ in gen.generate_scored_with_details(SUB, top_k=50)]


def test_generate_scored_with_details_budget_cap():
    gen = _gen()
    results = gen.generate_scored_with_details(SUB, top_k=100)
    assert len(results) <= 100


# --------------------------------------------------------------------------- #
# Stage 2a: MinimalReranker model + listwise InfoNCE loss + hit labelling.
# --------------------------------------------------------------------------- #

from rdkit import Chem  # noqa: E402

from grail_metabolism.model.reranker import MinimalReranker  # noqa: E402
from grail_metabolism.workflows.reranker import (  # noqa: E402
    label_hits,
    listwise_infonce,
)
from grail_metabolism.utils.transform import PAIR_NODE_DIM, EDGE_DIM, from_pair  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402


def _pair_graphs(sub_smiles, cand_smiles_list):
    sub_mol = Chem.MolFromSmiles(sub_smiles)
    graphs = []
    for cand in cand_smiles_list:
        cand_mol = Chem.MolFromSmiles(cand)
        graph = from_pair(sub_mol, cand_mol)
        assert graph is not None
        graphs.append(graph)
    return graphs


def test_reranker_forward_shape_and_differentiable():
    """forward(pair_batch, rule_id) -> (N,) logits, and .backward() populates grads."""
    n_rules = 5
    model = MinimalReranker(
        in_channels=PAIR_NODE_DIM, edge_dim=EDGE_DIM, n_rules=n_rules
    )
    cands = ["O=Cc1ccccc1", "OCc1ccccc1", "Oc1ccccc1"]
    graphs = _pair_graphs("OCc1ccccc1", cands)
    batch = Batch.from_data_list(graphs)
    rule_id = torch.tensor([0, 1, 2], dtype=torch.long)

    logits = model(batch, rule_id)
    assert logits.shape == (len(cands),), f"expected ({len(cands)},), got {logits.shape}"

    loss = logits.sum()
    loss.backward()
    # at least one parameter received a gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients populated -- model is not differentiable end to end"
    assert any(torch.any(g != 0) for g in grads), "all gradients are zero"


def test_label_hits_tautomer_invariant():
    """A true product (even a tautomer/charge variant) is a hit; a random non-product isn't."""
    # Acetamide and its imidic-acid tautomer share a tautomer-canonical InChIKey.
    true_products = ["CC(N)=O"]  # acetamide (amide form)
    pool = [
        ("CC(O)=N", 0.9, 0),     # imidic-acid tautomer of acetamide -> HIT
        ("c1ccccc1", 0.5, 1),    # benzene, clearly not a product -> non-hit
    ]
    mask = label_hits(pool, true_products)
    assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
    assert mask.shape == (2,)
    assert bool(mask[0]) is True, "tautomer of true product should be a hit"
    assert bool(mask[1]) is False, "non-product should not be a hit"


def test_listwise_infonce_flips_ranking():
    """On a 3-candidate pool with 1 hit, Adam steps on JUST that example raise the hit's
    logit above the non-hits (loss decreases, ranking flips). Guards loss sign/correctness."""
    torch.manual_seed(0)
    n_rules = 3
    model = MinimalReranker(
        in_channels=PAIR_NODE_DIM, edge_dim=EDGE_DIM, n_rules=n_rules
    )
    # 3 distinct candidates; index 2 is the (synthetic) hit.
    cands = ["O=Cc1ccccc1", "Oc1ccccc1", "OCc1ccccc1"]
    graphs = _pair_graphs("OCc1ccccc1", cands)
    batch = Batch.from_data_list(graphs)
    rule_id = torch.tensor([0, 1, 2], dtype=torch.long)
    hit_mask = torch.tensor([False, False, True])

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    with torch.no_grad():
        first = listwise_infonce(model(batch, rule_id), hit_mask).item()
    for _ in range(60):
        opt.zero_grad()
        logits = model(batch, rule_id)
        loss = listwise_infonce(logits, hit_mask)
        loss.backward()
        opt.step()
    with torch.no_grad():
        last = listwise_infonce(model(batch, rule_id), hit_mask).item()
        final_logits = model(batch, rule_id)

    assert last < first, f"loss did not decrease ({first:.4f} -> {last:.4f})"
    hit_logit = final_logits[hit_mask].max().item()
    non_hit_logit = final_logits[~hit_mask].max().item()
    assert hit_logit > non_hit_logit, (
        f"hit logit {hit_logit:.3f} did not rise above non-hits {non_hit_logit:.3f}"
    )


def test_listwise_infonce_zero_hits_is_zero():
    """A pool with no hits contributes nothing (skipped substrate)."""
    logits = torch.tensor([0.2, -0.1, 0.5], requires_grad=True)
    hit_mask = torch.tensor([False, False, False])
    loss = listwise_infonce(logits, hit_mask)
    assert float(loss.item()) == 0.0

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


# --------------------------------------------------------------------------- #
# Stage 2a (fair): no-MCS BiEncoderReranker (siamese single-graph + rule-prior
# scalar feature). No nn.Embedding over rules, no from_pair/rdFMCS in the path.
# --------------------------------------------------------------------------- #

from grail_metabolism.model.reranker import BiEncoderReranker  # noqa: E402
from grail_metabolism.utils.transform import SINGLE_NODE_DIM, from_rdmol  # noqa: E402


def _single_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    graph = from_rdmol(mol)
    assert graph is not None
    return graph


def test_bi_encoder_forward_shape_and_differentiable():
    """forward(sub_graph, prod_batch, rule_prior, gen_score) -> (N,) logits; grads flow.

    The substrate is encoded ONCE and broadcast across the N candidates; the rule signal
    is the scalar ``rule_prior`` feature (no nn.Embedding).
    """
    model = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    # No rule embedding anywhere in this architecture.
    assert not any(
        isinstance(m, torch.nn.Embedding) for m in model.modules()
    ), "bi-encoder must not contain an nn.Embedding over rules"

    cands = ["O=Cc1ccccc1", "OCc1ccccc1", "Oc1ccccc1"]
    sub_graph = _single_graph("OCc1ccccc1")
    prod_batch = Batch.from_data_list([_single_graph(c) for c in cands])
    rule_prior = torch.tensor([0.5, -1.2, 0.0], dtype=torch.float32)
    gen_score = torch.tensor([0.9, 0.4, 0.3], dtype=torch.float32)

    logits = model(sub_graph, prod_batch, rule_prior, gen_score)
    assert logits.shape == (len(cands),), f"expected ({len(cands)},), got {logits.shape}"

    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients populated -- model is not differentiable end to end"
    assert any(torch.any(g != 0) for g in grads), "all gradients are zero"


def test_bi_encoder_path_has_no_mcs():
    """The bi-encoder example path must NOT import/use from_pair or rdFMCS. Guards that we
    removed the MCS confound: build_examples_bi uses from_rdmol only."""
    import inspect

    import grail_metabolism.workflows.reranker as rr

    src = inspect.getsource(rr.build_examples_bi)
    assert "from_pair" not in src, "bi path must not call from_pair (MCS bottleneck)"
    assert "rdFMCS" not in src and "FindMCS" not in src, "bi path must not use rdFMCS"
    assert "from_rdmol" in src, "bi path should featurize single graphs via from_rdmol"


def test_bi_encoder_infonce_flips_ranking():
    """On a 3-candidate pool with 1 hit, Adam steps on JUST that example raise the hit's
    logit above the non-hits (same loss as pair path, bi-encoder model)."""
    torch.manual_seed(0)
    model = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    cands = ["O=Cc1ccccc1", "Oc1ccccc1", "OCc1ccccc1"]
    sub_graph = _single_graph("OCc1ccccc1")
    prod_batch = Batch.from_data_list([_single_graph(c) for c in cands])
    rule_prior = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
    gen_score = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    hit_mask = torch.tensor([False, False, True])

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    with torch.no_grad():
        first = listwise_infonce(model(sub_graph, prod_batch, rule_prior, gen_score), hit_mask).item()
    for _ in range(80):
        opt.zero_grad()
        logits = model(sub_graph, prod_batch, rule_prior, gen_score)
        loss = listwise_infonce(logits, hit_mask)
        loss.backward()
        opt.step()
    with torch.no_grad():
        last = listwise_infonce(model(sub_graph, prod_batch, rule_prior, gen_score), hit_mask).item()
        final_logits = model(sub_graph, prod_batch, rule_prior, gen_score)

    assert last < first, f"loss did not decrease ({first:.4f} -> {last:.4f})"
    hit_logit = final_logits[hit_mask].max().item()
    non_hit_logit = final_logits[~hit_mask].max().item()
    assert hit_logit > non_hit_logit, (
        f"hit logit {hit_logit:.3f} did not rise above non-hits {non_hit_logit:.3f}"
    )


# --------------------------------------------------------------------------- #
# Ablation flags: use_rule_prior and use_gen_score.
# --------------------------------------------------------------------------- #


def _bi_forward(model, cands, sub="OCc1ccccc1"):
    sub_graph = _single_graph(sub)
    prod_batch = Batch.from_data_list([_single_graph(c) for c in cands])
    rule_prior = torch.tensor([0.5, -1.2, 0.0], dtype=torch.float32)
    gen_score = torch.tensor([0.9, 0.4, 0.3], dtype=torch.float32)
    return model(sub_graph, prod_batch, rule_prior, gen_score)


def test_bi_ablation_no_rule_prior():
    """BiEncoderReranker(use_rule_prior=False) still produces (N,) logits and is
    differentiable -- the rule_prior scalar is zeroed but the head shape is unchanged."""
    model = BiEncoderReranker(in_channels=SINGLE_NODE_DIM, use_rule_prior=False)
    assert not model.use_rule_prior
    assert model.use_gen_score

    cands = ["O=Cc1ccccc1", "OCc1ccccc1", "Oc1ccccc1"]
    logits = _bi_forward(model, cands)
    assert logits.shape == (len(cands),), f"expected ({len(cands)},), got {logits.shape}"

    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients populated with use_rule_prior=False"


def test_bi_ablation_no_gen_score():
    """BiEncoderReranker(use_gen_score=False) still produces (N,) logits and is
    differentiable -- the gen_score scalar is zeroed but the head shape is unchanged."""
    model = BiEncoderReranker(in_channels=SINGLE_NODE_DIM, use_gen_score=False)
    assert model.use_rule_prior
    assert not model.use_gen_score

    cands = ["O=Cc1ccccc1", "OCc1ccccc1", "Oc1ccccc1"]
    logits = _bi_forward(model, cands)
    assert logits.shape == (len(cands),), f"expected ({len(cands)},), got {logits.shape}"

    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients populated with use_gen_score=False"


def test_bi_ablation_both_disabled():
    """Both scalars disabled: forward still runs, produces (N,) logits. Guards the
    'both off' degenerate corner (graph-only baseline)."""
    model = BiEncoderReranker(
        in_channels=SINGLE_NODE_DIM, use_rule_prior=False, use_gen_score=False
    )
    cands = ["O=Cc1ccccc1", "OCc1ccccc1", "Oc1ccccc1"]
    logits = _bi_forward(model, cands)
    assert logits.shape == (len(cands),), f"expected ({len(cands)},), got {logits.shape}"


# --------------------------------------------------------------------------- #
# Parallel (spawn-Pool) bi-example builder: must produce IDENTICAL examples to the
# serial path (only speed changes). Built around a tiny in-test generator + checkpoint.
# --------------------------------------------------------------------------- #

import types  # noqa: E402

from grail_metabolism.config import GeneratorConfig as _GenCfg  # noqa: E402
from grail_metabolism.workflows.reranker import (  # noqa: E402
    build_examples_bi,
    build_examples_bi_parallel,
)


def _tiny_gate_checkpoint(tmp_path, rules):
    """Build a tiny real generator (a couple of rules), save it in the gate checkpoint
    format ({state_dict, arch, rules}), and return (generator, ckpt_path)."""
    gen = build_generator(_GenCfg(), rules)
    gen.eval()
    ckpt = tmp_path / "tiny_generator.pt"
    torch.save(
        {
            "state_dict": gen.state_dict(),
            "arch": _GenCfg().__dict__.copy(),
            "rules": list(rules),
        },
        ckpt,
    )
    return gen, ckpt


def test_parallel_bi_builder_matches_serial(tmp_path):
    """build_examples_bi_parallel(workers=2) yields the SAME examples as the serial
    build_examples_bi (same count, smiles, hit_masks, rule_priors) on a tiny MolFrame."""
    rules = ["[CH2:1][OH:2]>>[CH:1]=[O:2]", "[c:1][H:2]>>[c:1][OH]"]
    # 3 small substrates; map values are the "true products" (recall denominator).
    fake_map = {
        "OCc1ccccc1": {"O=Cc1ccccc1"},
        "CCO": {"CC=O"},
        "OCc1ccc(O)cc1": {"O=Cc1ccc(O)cc1"},
    }
    molframe = types.SimpleNamespace(map=fake_map)

    gen, ckpt = _tiny_gate_checkpoint(tmp_path, rules)

    n = len(fake_map)
    serial = build_examples_bi(gen, molframe, n, top_k=50, max_pool=40, verbose=False)
    parallel = build_examples_bi_parallel(
        gen, molframe, n, top_k=50, max_pool=40,
        gen_ckpt=str(ckpt), workers=2,
        prior_strength=float(getattr(gen, "prior_strength", 0.4)),
        verbose=False,
    )

    assert len(serial) >= 1, "expected the tiny pool to yield at least one example"
    assert len(parallel) == len(serial), (
        f"parallel built {len(parallel)} examples, serial built {len(serial)}"
    )
    # parallel is UNORDERED (imap_unordered), so match examples by substrate, not by position.
    by_sub = {b.sub: b for b in parallel}
    for a in serial:
        assert a.sub in by_sub, f"parallel is missing substrate {a.sub!r}"
        b = by_sub[a.sub]
        assert a.smiles == b.smiles, f"smiles differ for {a.sub!r}"
        assert torch.equal(a.hit_mask, b.hit_mask), f"hit_mask differs for {a.sub!r}"
        assert torch.equal(a.rule_priors, b.rule_priors), f"rule_priors differ for {a.sub!r}"
        assert torch.allclose(a.gen_scores, b.gen_scores), f"gen_scores differ for {a.sub!r}"
        assert a.true_products == b.true_products, f"true_products differ for {a.sub!r}"


# --------------------------------------------------------------------------- #
# Task 6: intermediate-node (depth-2) bootstrap pairs for the forest policy.
#
# root -> m1 -> m2 chain built from REAL SMIRKS rules (mirrors the rest of this file's
# fixture style, rather than a hand-rolled stub generator): benzyl alcohol --r0--> benz-
# aldehyde --r1--> benzoic acid. m2 is annotated for `root` but is NOT a depth-1 child of
# `root` (only m1 is) -- i.e. exactly the depth-2-only case build_intermediate_pairs must
# catch.
# --------------------------------------------------------------------------- #

from grail_metabolism.workflows.reranker import build_intermediate_pairs  # noqa: E402

_ROOT = "OCc1ccccc1"       # benzyl alcohol
_M1 = "O=Cc1ccccc1"        # benzaldehyde (depth-1 child of root via rule 0)
_M2 = "O=C(O)c1ccccc1"     # benzoic acid (depth-1 child of m1 via rule 1; depth-2-only from root)


def _tiny_generator_depth2():
    rules = ["[CH2:1][OH:2]>>[CH:1]=[O:2]", "[CH:1]=[O:2]>>[C:1](=[O:2])[OH]"]
    gen = build_generator(GeneratorConfig(), rules)
    gen.eval()
    return gen


def _tiny_molframe_depth2():
    # root's annotated set contains m2 (depth-2-only) but NOT m1.
    fake_map = {_ROOT: {_M2}}
    return types.SimpleNamespace(map=fake_map)


def test_intermediate_pairs_rooted_at_intermediate():
    gen = _tiny_generator_depth2()
    molframe = _tiny_molframe_depth2()
    ex = build_intermediate_pairs(gen, molframe, n_substrates=5, top_k=50)
    assert any(e.sub == _M1 and bool(e.hit_mask.any()) for e in ex), (
        f"expected a _BiExample rooted at {_M1!r} with a hit; got subs={[e.sub for e in ex]}"
    )

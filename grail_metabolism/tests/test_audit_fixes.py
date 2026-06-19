"""Regression tests for the audit fixes (correctness, calibration, reproducibility)."""
from __future__ import annotations

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch

from grail_metabolism.metrics import aggregate_prediction_metrics
from grail_metabolism.model.filter import Filter
from grail_metabolism.model.grail import summon_the_grail
from grail_metabolism.model.train_model import PULoss
from grail_metabolism.utils.preparation import (
    MolFrame,
    iscorrect,
    load_default_rules,
    load_phase2_rules,
    resolve_default_rule_bank,
)
from grail_metabolism.utils.seed import seed_everything
from grail_metabolism.utils.transform import from_pair

RULE = "[CH2:1][OH:2]>>[CH:1]=[O:2]"


def _cross_edges(graph, n_sub):
    edges = set()
    for k in range(graph.edge_index.size(1)):
        a, b = int(graph.edge_index[0, k]), int(graph.edge_index[1, k])
        if (a < n_sub) != (b < n_sub) and float(graph.edge_attr[k].abs().sum()) == 0.0:
            edges.add((a, b) if a < b else (b, a))
    return edges


def test_mcs_cross_edges_connect_corresponding_elements():
    # aniline -> 4-aminophenol: every alignment edge must join same-element atoms.
    sub = Chem.MolFromSmiles("c1ccc(N)cc1")
    prod = Chem.MolFromSmiles("Nc1ccc(O)cc1")
    graph = from_pair(sub, prod)
    n_sub = sub.GetNumAtoms()
    cross = _cross_edges(graph, n_sub)
    assert cross, "expected MCS alignment cross-edges"
    for lo, hi in cross:
        assert sub.GetAtomWithIdx(lo).GetSymbol() == prod.GetAtomWithIdx(hi - n_sub).GetSymbol()


def test_iscorrect_keeps_small_metabolites_drops_lone_atoms():
    assert iscorrect("C=O")  # formaldehyde (2 heavy atoms)
    assert iscorrect("OC=O")  # formate
    assert iscorrect("CCO")  # ethanol
    assert not iscorrect("O")  # water (lone heavy atom)
    assert not iscorrect("[Cl-]")  # chloride leaving group


def test_filter_return_logits_is_logit_domain():
    graph = from_pair(Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC=O"))
    model = Filter(18, 18, [32, 64, 32, 64, 32, 16], mode="pair")
    model.eval()  # disable dropout so the two forwards are comparable
    batch = Batch.from_data_list([graph])
    prob = model(batch)
    logit = model(batch, return_logits=True)
    assert 0.0 <= float(prob) <= 1.0
    assert torch.allclose(torch.sigmoid(logit), prob, atol=1e-5)


def test_puloss_trains_on_logits():
    # Before the fix, probabilities were fed into a logit-domain surrogate (double
    # sigmoid), collapsing the loss range and killing the gradient.
    seed_everything(0)
    model = Filter(18, 18, [32, 64, 32, 64, 32, 16], mode="pair")
    crit = PULoss(0.5)
    pos = from_pair(Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC=O"))
    pos.y = torch.tensor([1.0])
    neg = from_pair(Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCO"))
    neg.y = torch.tensor([0.0])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = last = None
    for _ in range(30):
        batch = Batch.from_data_list([pos, neg])
        out = model(batch, return_logits=True)
        loss = crit(out, batch.y.view(-1, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
        last = float(loss)
    assert last < first  # the loss actually moves now


def test_score_batch_matches_per_item_score():
    seed_everything(0)
    model = Filter(18, 18, [32, 64, 32, 64, 32, 16], mode="pair")
    model.eval()
    prods = ["CC=O", "CCO", "CC(=O)O"]
    batched = model.score_batch("CCO", prods)
    per_item = [model.score("CCO", p) for p in prods]
    assert max(abs(a - b) for a, b in zip(batched, per_item)) < 1e-6


def test_seed_everything_makes_training_reproducible():
    frame = MolFrame(pd.DataFrame([{"sub": "CCO", "prod": "CC=O", "real": 1},
                                   {"sub": "CCO", "prod": "CCO", "real": 0}]))
    frame.full_setup(rules=[RULE], include_pair_graphs=False, include_morgan=False)

    def run():
        seed_everything(123)
        model = summon_the_grail([RULE])
        model.generator.fit(frame, eps=3, verbose=False)
        return list(model.generator.loss_history_)

    assert run() == run()


def test_default_rule_bank_is_consistent_across_entry_points():
    from grail_metabolism.experiments.presets import _default_rules_path

    bank = resolve_default_rule_bank()
    assert bank is not None
    assert _default_rules_path().endswith(bank.name)
    with open(bank) as handle:
        file_rules = [line.strip() for line in handle if line.strip()]
    assert load_default_rules() == file_rules


def test_phase2_rules_all_compile_and_fire():
    from rdkit.Chem import AllChem

    rules = load_phase2_rules()
    assert len(rules) >= 10
    for rule in rules:
        rxn = AllChem.ReactionFromSmarts(rule)
        assert rxn is not None and rxn.GetNumReactantTemplates() >= 1


def test_metrics_inchikey_matching_and_output_size():
    # Two equivalent SMILES for acetic acid: exact string match misses, InChIKey catches.
    preds = [{"predicted": ["OC(=O)C", "c1ccccc1"], "real": ["CC(O)=O"]}]
    exact = aggregate_prediction_metrics(preds, ks=[1], match="exact")
    inchi = aggregate_prediction_metrics(preds, ks=[1], match="inchikey")
    assert exact["recall"] == 0.0
    assert inchi["recall"] == 1.0
    assert inchi["mean_output_size"] == 2.0


def test_generate_respects_max_output_cap():
    seed_everything(0)
    model = summon_the_grail([RULE])
    model.filter.calibrated_threshold = 0.0  # accept everything

    def fake_scored(sub, top_k=None, threshold=None):
        return [("CCO", 0.9), ("CC=O", 0.8), ("CC(=O)O", 0.7)]

    model.generator.generate_scored = fake_scored
    assert len(model.generate("CCO", max_output=2)) == 2
    assert len(model.generate("CCO")) == 3

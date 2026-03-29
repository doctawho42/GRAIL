from __future__ import annotations

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch

import grail_metabolism.utils.preparation as preparation
from grail_metabolism.cli import main
from grail_metabolism.model.filter import Filter
from grail_metabolism.model.grail import summon_the_grail
from grail_metabolism.model.generator import generate_vectors
from grail_metabolism.utils.preparation import MolFrame
from grail_metabolism.utils.transform import from_pair, from_rdmol, from_rule


RULE = "[CH2:1][OH:2]>>[CH:1]=[O:2]"


def make_frame() -> MolFrame:
    return MolFrame(
        pd.DataFrame(
            [
                {"sub": "CCO", "prod": "CC=O", "real": 1},
                {"sub": "CCO", "prod": "CCO", "real": 0},
            ]
        )
    )


def test_molframe_initialization_from_dataframe_and_mapping():
    frame_from_df = make_frame()
    frame_from_map = MolFrame({"CCO": {"CC=O"}}, gen_map={"CCO": {"CCO"}})

    assert isinstance(frame_from_df, MolFrame)
    assert isinstance(frame_from_map, MolFrame)
    assert frame_from_df.map["CCO"] == {"CC=O"}
    assert "CCO" in frame_from_df.negs["CCO"]


def test_generate_vectors():
    reaction_dict = {
        "substrate1": {"product1": [0, 1], "product2": [2]},
        "substrate2": {"product3": [3]},
    }
    real_products_dict = {
        "substrate1": {"product1"},
        "substrate2": {"product3"},
    }

    vectors = generate_vectors(reaction_dict, real_products_dict, 5)

    assert vectors["substrate1"] == [1, 1, 0, 0, 0]
    assert vectors["substrate2"] == [0, 0, 0, 1, 0]


def test_transform_graph_shapes():
    single = from_rdmol(Chem.MolFromSmiles("CCO"))
    pair = from_pair(
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("CC=O"),
    )
    rule = from_rule(RULE)

    assert single is not None
    assert pair is not None
    assert single.x.shape[1] == 16
    assert pair.x.shape[1] == 18
    assert rule.edge_attr.shape[1] == 18


def test_filter_forward_pair():
    graph = from_pair(
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("CC=O"),
    )
    assert graph is not None

    model = Filter(in_channels=18, edge_dim=18, arg_vec=[32, 64, 32, 64, 32, 16], mode="pair")
    batch = Batch.from_data_list([graph])
    output = model(batch)

    assert output.shape == (1, 1)


def test_end_to_end_pipeline_smoke():
    torch.manual_seed(0)
    frame = make_frame()
    frame.full_setup(rules=[RULE])

    model = summon_the_grail([RULE])
    model.filter.fit(frame, eps=10, verbose=False, nnPU=False)

    assert model.generator.generate("CCO", top_k=1) == ["CC=O"]
    assert model.generate("CCO", top_k=1) == ["CC=O"]


def test_generator_fit_stops_cleanly_on_timeout_budget():
    torch.manual_seed(0)
    frame = make_frame()
    frame.full_setup(rules=[RULE], include_pair_graphs=False, include_morgan=False)

    model = summon_the_grail([RULE])
    model.generator.fit(frame, eps=5, verbose=False, val_data=frame, timeout_seconds=0.0)

    assert model.generator.timed_out_ is True
    assert model.generator.stop_reason_ == "timeout"
    assert model.generator.epochs_trained_ == 1
    assert len(model.generator.loss_history_) == 1


def test_filter_fit_stops_cleanly_on_timeout_budget():
    torch.manual_seed(0)
    frame = make_frame()
    frame.full_setup(rules=[RULE])

    model = summon_the_grail([RULE])
    model.filter.fit(frame, eps=5, verbose=False, nnPU=False, val_data=frame, timeout_seconds=0.0)

    assert model.filter.timed_out_ is True
    assert model.filter.stop_reason_ == "timeout"
    assert model.filter.epochs_trained_ == 1
    assert len(model.filter.loss_history_) == 1


def test_full_setup_reuses_persistent_stage_cache(tmp_path, monkeypatch):
    single_cache = tmp_path / "single_graphs.pt"
    reaction_cache = tmp_path / "reaction_labels.pt"

    frame = make_frame()
    frame.full_setup(
        rules=[RULE],
        include_pair_graphs=False,
        include_morgan=False,
        single_cache_path=single_cache,
        reaction_label_cache_path=reaction_cache,
    )
    expected_labels = dict(frame.reaction_labels)

    assert single_cache.exists()
    assert reaction_cache.exists()

    def fail_from_rdmol(*args, **kwargs):
        raise AssertionError("singlegraphs should be restored from cache")

    def fail_apply_rules(*args, **kwargs):
        raise AssertionError("label_reactions should be restored from cache")

    monkeypatch.setattr(preparation, "from_rdmol", fail_from_rdmol)
    monkeypatch.setattr(preparation, "apply_rules_to_molecule", fail_apply_rules)

    restored = make_frame()
    restored.full_setup(
        rules=[RULE],
        include_pair_graphs=False,
        include_morgan=False,
        single_cache_path=single_cache,
        reaction_label_cache_path=reaction_cache,
    )

    assert "CCO" in restored.single
    assert restored.reaction_labels == expected_labels


def test_cli_predict(capsys, tmp_path):
    rules_path = tmp_path / "rules.txt"
    rules_path.write_text(f"{RULE}\n")

    exit_code = main(["predict", "CCO", "--rules", str(rules_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "CC=O" in captured.out

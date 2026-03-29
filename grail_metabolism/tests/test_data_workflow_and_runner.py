from __future__ import annotations

from collections import defaultdict
import warnings

import pandas as pd

from grail_metabolism.artifacts import ArtifactStore
from grail_metabolism.config import DatasetConfig, ExperimentConfig
from grail_metabolism.experiments.runner import ExperimentRunner
from grail_metabolism.tests.test_grail_metabolism import RULE, make_frame
from grail_metabolism.workflows.data import DatasetBundle, _resolve_triples_path, load_dataset_bundle
from grail_metabolism.workflows.pretraining import PretrainingWorkflow


def test_resolve_triples_path_prefers_clean_file(tmp_path):
    original = tmp_path / "train_triples.txt"
    clean = tmp_path / "train_triples_clean.txt"
    original.write_text("1 2 1\n")
    clean.write_text("1 2 1\n")

    assert _resolve_triples_path(str(original), use_clean_splits=True) == str(clean)
    assert _resolve_triples_path(str(original), use_clean_splits=False) == str(original)


def test_resolve_triples_path_warns_when_clean_missing(tmp_path):
    original = tmp_path / "train_triples.txt"
    original.write_text("1 2 1\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve_triples_path(str(original), use_clean_splits=True)

    assert resolved == str(original)
    assert any("Clean triples requested but not found" in str(item.message) for item in caught)


def test_dataset_bundle_prepare_creates_split_cache(tmp_path):
    train = make_frame()
    val = make_frame()
    bundle = DatasetBundle(
        train=train,
        val=val,
        test=make_frame(),
        rules=[RULE],
        config=DatasetConfig(cache_preprocessed=True, cache_dir=str(tmp_path), standardize=False),
        split_sources={
            "train": {"sdf": "missing_train.sdf", "triples": "missing_train.txt", "max_substrates": None, "sampling_seed": 42},
            "val": {"sdf": "missing_val.sdf", "triples": "missing_val.txt", "max_substrates": None, "sampling_seed": 43},
        },
    )

    bundle.prepare(include_val=True, include_test=False, include_pair_graphs=False, include_morgan=False)

    train_meta = list((tmp_path / "train").glob("*/meta.json"))
    val_meta = list((tmp_path / "val").glob("*/meta.json"))
    assert len(train_meta) == 1
    assert len(val_meta) == 1
    assert list((tmp_path / "train").glob("*/single_graphs.pt"))
    assert list((tmp_path / "train").glob("*/reaction_labels.pt"))


def test_load_dataset_bundle_uses_clean_paths_and_rules(monkeypatch, tmp_path):
    calls = []

    def fake_load_split(sdf_path, triples_path, standardize, max_substrates, seed):
        calls.append((sdf_path, triples_path, standardize, max_substrates, seed))
        return make_frame()

    monkeypatch.setattr("grail_metabolism.workflows.data._load_split", fake_load_split)
    monkeypatch.setattr("grail_metabolism.workflows.data._load_rules", lambda config: [RULE])

    train_original = tmp_path / "train_triples.txt"
    train_clean = tmp_path / "train_triples_clean.txt"
    val_original = tmp_path / "val_triples.txt"
    val_clean = tmp_path / "val_triples_clean.txt"
    test_original = tmp_path / "test_triples.txt"
    test_clean = tmp_path / "test_triples_clean.txt"
    for path in [train_original, train_clean, val_original, val_clean, test_original, test_clean]:
        path.write_text("1 2 1\n")

    config = DatasetConfig(
        train_sdf="train.sdf",
        train_triples=str(train_original),
        val_sdf="val.sdf",
        val_triples=str(val_original),
        test_sdf="test.sdf",
        test_triples=str(test_original),
        rules_path=None,
        use_clean_splits=True,
        standardize=False,
    )

    bundle = load_dataset_bundle(config)

    assert bundle.rules == [RULE]
    assert calls[0][1] == str(train_clean)
    assert calls[1][1] == str(val_clean)
    assert calls[2][1] == str(test_clean)


def test_pretraining_workflow_loads_supported_uspto_columns(tmp_path):
    csv_path = tmp_path / "uspto.csv"
    pd.DataFrame(
        {
            "reaction_smiles": ["CCO.CN>>CC=O.CN"],
            "reactions": ["CCC>O>CC=C"],
        }
    ).to_csv(csv_path, index=False)

    config = ExperimentConfig(name="demo")
    config.dataset.uspto_csv = str(csv_path)
    config.dataset.max_uspto_rows = 1
    workflow = PretrainingWorkflow(config, ArtifactStore.create(tmp_path, "artifacts"))

    smiles = workflow._load_uspto_smiles()
    assert smiles == ["CCO", "CN", "CC=O", "CN", "CCC", "CC=C"]


def test_experiment_runner_compare_and_run_config(monkeypatch, tmp_path):
    runner = ExperimentRunner(output_dir=str(tmp_path))

    monkeypatch.setattr(
        "grail_metabolism.experiments.runner.EnsembleWorkflow.run",
        lambda self: {"generator": {"f1": 0.4}, "ensemble": {"top_1_recall": 0.5}},
    )

    config = ExperimentConfig(name="demo")
    config.output_dir = str(tmp_path)
    result = runner.run_config(config)

    assert result.name == "demo"
    assert result.metrics["generator"]["f1"] == 0.4
    assert (result.artifact_dir / "config.yaml").exists()

    comparison = runner.compare([result])
    assert comparison == [{"experiment": "demo", "generator.f1": 0.4, "ensemble.top_1_recall": 0.5}]

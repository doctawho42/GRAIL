from __future__ import annotations

from grail_metabolism.cli import main
from grail_metabolism.config import ExperimentConfig, load_experiment_config
from grail_metabolism.experiments.presets import get_experiment_preset, list_experiment_presets
from grail_metabolism.workflows.pretraining import PretrainingWorkflow


def test_preset_registry_contains_main_variants():
    presets = list_experiment_presets()
    assert "paper_full_ensemble" in presets
    assert "paper_minimal_baseline" in presets
    assert "paper_generator_dot" in presets


def test_config_roundtrip_yaml(tmp_path):
    config = get_experiment_preset("paper_minimal_baseline").with_overrides(
        dataset={"max_train_substrates": 32, "max_val_substrates": 8, "max_test_substrates": 8}
    )
    path = tmp_path / "experiment.yaml"
    config.dump_yaml(path)
    loaded = load_experiment_config(path)
    assert isinstance(loaded, ExperimentConfig)
    assert loaded.name == config.name
    assert loaded.generator.scoring == config.generator.scoring
    assert loaded.filter.model_type == config.filter.model_type
    assert loaded.dataset.max_train_substrates == 32


def test_cli_presets_export(tmp_path, capsys):
    exit_code = main(["presets", "--export-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "paper_full_ensemble" in captured.out
    assert (tmp_path / "paper_full_ensemble.yaml").exists()


def test_uspto_reaction_parser_handles_both_formats():
    patent_style = list(PretrainingWorkflow._split_reaction_smiles("CCO.CN>O>CC=O.CN"))
    arrow_style = list(PretrainingWorkflow._split_reaction_smiles("CCO.CN>>CC=O.CN"))
    assert patent_style == ["CCO", "CN", "CC=O", "CN"]
    assert arrow_style == ["CCO", "CN", "CC=O", "CN"]

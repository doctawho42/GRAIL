from __future__ import annotations

import warnings
from pathlib import Path

from grail_metabolism.config import ExperimentConfig, GeneratorConfig, PretrainConfig
from grail_metabolism.experiments.presets import _default_rules_path, get_experiment_preset


def test_default_rules_path_points_to_existing_resource():
    path = Path(_default_rules_path())
    assert path.exists()
    assert path.name in {"extended_smirks.txt", "notebooks_rules.txt", "merged_smirks.txt"}


def test_default_preset_uses_clean_splits_and_packaged_rules():
    config = get_experiment_preset("paper_full_ensemble")
    assert config.dataset.use_clean_splits is True
    assert Path(config.dataset.rules_path).exists()
    assert "grail_metabolism/resources" in config.dataset.rules_path or "grail_metabolism/data" in config.dataset.rules_path


def test_experiment_config_emits_dead_flag_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ExperimentConfig(name="demo", generator=GeneratorConfig(use_pretraining=False), pretrain=PretrainConfig(enabled=False))

    assert caught == []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ExperimentConfig(
            name="demo-warning",
            generator=GeneratorConfig(use_pretraining=True),
            pretrain=PretrainConfig(enabled=False),
        )

    assert any("deprecated dead flag" in str(item.message) for item in caught)

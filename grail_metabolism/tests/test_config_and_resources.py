from __future__ import annotations

import warnings
from pathlib import Path

from grail_metabolism.config import ExperimentConfig, GeneratorConfig, PretrainConfig
from grail_metabolism.experiments.presets import _default_rules_path, get_experiment_preset


def test_default_rules_path_points_to_existing_resource():
    """The resolved bank must be one the resolver offers, whichever of them a checkout has.

    The allowed names were listed here by hand and the list went stale: it never gained
    extended_smirks_released.txt, which is the bank a CLONE resolves to, because the measured one
    is not redistributed. So this failed in the only tree that matters for a reader -- a fresh
    clone -- while passing in every tree that holds a file the release withholds. The set now comes
    from the resolver, so a bank the resolver offers cannot be one this test rejects.
    """
    from grail_metabolism.utils.preparation import _default_rule_bank_candidates

    path = Path(_default_rules_path())
    assert path.exists(), f"{path} does not exist; no bank in the resolver's list is present"
    offered = {c.name for c in _default_rule_bank_candidates()}
    assert path.name in offered, (
        f"the default bank resolved to {path.name}, which _default_rule_bank_candidates() does "
        f"not offer: {sorted(offered)}")


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

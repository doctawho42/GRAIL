from __future__ import annotations

from importlib import import_module

__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "ExperimentRunner",
    "export_presets",
    "get_experiment_preset",
    "list_experiment_presets",
    "Filter",
    "Generator",
    "ModelWrapper",
    "PretrainedGrail",
    "SimpleGenerator",
    "summon_the_grail",
    "MolFrame",
    "generate_vectors",
    "from_pair",
    "from_rdmol",
    "from_rule",
    "load_default_rules",
    "standardize_mol",
    "EnsembleWorkflow",
    "InferenceService",
    "load_dataset_bundle",
]

__version__ = "0.2.0"

_EXPORTS = {
    "ExperimentConfig": ("grail_metabolism.config", "ExperimentConfig"),
    "load_experiment_config": ("grail_metabolism.config", "load_experiment_config"),
    "ExperimentRunner": ("grail_metabolism.experiments", "ExperimentRunner"),
    "export_presets": ("grail_metabolism.experiments", "export_presets"),
    "get_experiment_preset": ("grail_metabolism.experiments", "get_experiment_preset"),
    "list_experiment_presets": ("grail_metabolism.experiments", "list_experiment_presets"),
    "Filter": ("grail_metabolism.model", "Filter"),
    "Generator": ("grail_metabolism.model", "Generator"),
    "ModelWrapper": ("grail_metabolism.model", "ModelWrapper"),
    "PretrainedGrail": ("grail_metabolism.model", "PretrainedGrail"),
    "SimpleGenerator": ("grail_metabolism.model", "SimpleGenerator"),
    "summon_the_grail": ("grail_metabolism.model", "summon_the_grail"),
    "MolFrame": ("grail_metabolism.utils", "MolFrame"),
    "generate_vectors": ("grail_metabolism.utils", "generate_vectors"),
    "from_pair": ("grail_metabolism.utils", "from_pair"),
    "from_rdmol": ("grail_metabolism.utils", "from_rdmol"),
    "from_rule": ("grail_metabolism.utils", "from_rule"),
    "load_default_rules": ("grail_metabolism.utils", "load_default_rules"),
    "standardize_mol": ("grail_metabolism.utils", "standardize_mol"),
    "EnsembleWorkflow": ("grail_metabolism.workflows", "EnsembleWorkflow"),
    "InferenceService": ("grail_metabolism.workflows", "InferenceService"),
    "load_dataset_bundle": ("grail_metabolism.workflows", "load_dataset_bundle"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

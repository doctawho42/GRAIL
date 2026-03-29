from __future__ import annotations

from importlib import import_module

__all__ = [
    "DatasetBundle",
    "load_dataset_bundle",
    "EnsembleWorkflow",
    "evaluate_ensemble",
    "evaluate_filter",
    "evaluate_generator",
    "InferenceService",
    "PretrainingWorkflow",
    "FilterTrainingWorkflow",
    "GeneratorTrainingWorkflow",
]

_EXPORTS = {
    "DatasetBundle": ("grail_metabolism.workflows.data", "DatasetBundle"),
    "load_dataset_bundle": ("grail_metabolism.workflows.data", "load_dataset_bundle"),
    "EnsembleWorkflow": ("grail_metabolism.workflows.ensemble", "EnsembleWorkflow"),
    "evaluate_ensemble": ("grail_metabolism.workflows.evaluation", "evaluate_ensemble"),
    "evaluate_filter": ("grail_metabolism.workflows.evaluation", "evaluate_filter"),
    "evaluate_generator": ("grail_metabolism.workflows.evaluation", "evaluate_generator"),
    "InferenceService": ("grail_metabolism.workflows.inference", "InferenceService"),
    "PretrainingWorkflow": ("grail_metabolism.workflows.pretraining", "PretrainingWorkflow"),
    "FilterTrainingWorkflow": ("grail_metabolism.workflows.training", "FilterTrainingWorkflow"),
    "GeneratorTrainingWorkflow": ("grail_metabolism.workflows.training", "GeneratorTrainingWorkflow"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

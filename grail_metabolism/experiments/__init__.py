from __future__ import annotations

from importlib import import_module

__all__ = [
    "export_presets",
    "get_experiment_preset",
    "list_experiment_presets",
    "ExperimentResult",
    "ExperimentRunner",
]

_EXPORTS = {
    "export_presets": ("grail_metabolism.experiments.presets", "export_presets"),
    "get_experiment_preset": ("grail_metabolism.experiments.presets", "get_experiment_preset"),
    "list_experiment_presets": ("grail_metabolism.experiments.presets", "list_experiment_presets"),
    "ExperimentResult": ("grail_metabolism.experiments.runner", "ExperimentResult"),
    "ExperimentRunner": ("grail_metabolism.experiments.runner", "ExperimentRunner"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

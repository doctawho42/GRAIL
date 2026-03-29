from __future__ import annotations

from importlib import import_module

__all__ = [
    "Filter",
    "GATv2Filter",
    "GCNFilter",
    "GINFilter",
    "MolPathFilter",
    "MorganOnlyFilter",
    "Generator",
    "RuleParse",
    "generate_vectors",
    "PretrainedGrail",
    "summon_the_grail",
    "GFilter",
    "GGenerator",
    "ModelWrapper",
    "SimpleGenerator",
]

_EXPORTS = {
    "Filter": ("grail_metabolism.model.filter", "Filter"),
    "GATv2Filter": ("grail_metabolism.model.filter", "GATv2Filter"),
    "GCNFilter": ("grail_metabolism.model.filter", "GCNFilter"),
    "GINFilter": ("grail_metabolism.model.filter", "GINFilter"),
    "MolPathFilter": ("grail_metabolism.model.filter", "MolPathFilter"),
    "MorganOnlyFilter": ("grail_metabolism.model.filter", "MorganOnlyFilter"),
    "Generator": ("grail_metabolism.model.generator", "Generator"),
    "RuleParse": ("grail_metabolism.model.generator", "RuleParse"),
    "generate_vectors": ("grail_metabolism.model.generator", "generate_vectors"),
    "PretrainedGrail": ("grail_metabolism.model.grail", "PretrainedGrail"),
    "summon_the_grail": ("grail_metabolism.model.grail", "summon_the_grail"),
    "GFilter": ("grail_metabolism.model.wrapper", "GFilter"),
    "GGenerator": ("grail_metabolism.model.wrapper", "GGenerator"),
    "ModelWrapper": ("grail_metabolism.model.wrapper", "ModelWrapper"),
    "SimpleGenerator": ("grail_metabolism.model.wrapper", "SimpleGenerator"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

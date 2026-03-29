from __future__ import annotations

from importlib import import_module

__all__ = [
    "MolFrame",
    "cpunum",
    "extract",
    "generate_vectors",
    "iscorrect",
    "load_default_rules",
    "metaboliser",
    "standardize_mol",
    "EDGE_DIM",
    "FINGERPRINT_DIM",
    "PAIR_NODE_DIM",
    "SINGLE_NODE_DIM",
    "from_pair",
    "from_rdmol",
    "from_rule",
]

_EXPORTS = {
    "MolFrame": ("grail_metabolism.utils.preparation", "MolFrame"),
    "cpunum": ("grail_metabolism.utils.preparation", "cpunum"),
    "extract": ("grail_metabolism.utils.preparation", "extract"),
    "generate_vectors": ("grail_metabolism.utils.preparation", "generate_vectors"),
    "iscorrect": ("grail_metabolism.utils.preparation", "iscorrect"),
    "load_default_rules": ("grail_metabolism.utils.preparation", "load_default_rules"),
    "metaboliser": ("grail_metabolism.utils.preparation", "metaboliser"),
    "standardize_mol": ("grail_metabolism.utils.preparation", "standardize_mol"),
    "EDGE_DIM": ("grail_metabolism.utils.transform", "EDGE_DIM"),
    "FINGERPRINT_DIM": ("grail_metabolism.utils.transform", "FINGERPRINT_DIM"),
    "PAIR_NODE_DIM": ("grail_metabolism.utils.transform", "PAIR_NODE_DIM"),
    "SINGLE_NODE_DIM": ("grail_metabolism.utils.transform", "SINGLE_NODE_DIM"),
    "from_pair": ("grail_metabolism.utils.transform", "from_pair"),
    "from_rdmol": ("grail_metabolism.utils.transform", "from_rdmol"),
    "from_rule": ("grail_metabolism.utils.transform", "from_rule"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

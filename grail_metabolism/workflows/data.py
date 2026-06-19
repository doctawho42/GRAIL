from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
import warnings

import numpy as np
import pandas as pd

from ..config import DatasetConfig
from ..utils.preparation import MolFrame, load_default_rules


@dataclass
class DatasetBundle:
    train: MolFrame
    val: MolFrame
    test: MolFrame
    rules: List[str]
    config: DatasetConfig
    split_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def prepare(
        self,
        pca: bool | None = None,
        rules: Sequence[str] | None = None,
        include_val: bool = False,
        include_test: bool = False,
        include_pair_graphs: bool = False,
        include_morgan: bool = False,
        single_substrates_only: bool = False,
    ) -> None:
        selected_rules = list(rules or self.rules)
        use_pca = self.config.pca if pca is None else pca
        targets = [("train", self.train)]
        if include_val:
            targets.append(("val", self.val))
        if include_test:
            targets.append(("test", self.test))
        for name, split in targets:
            if self.config.augment_negatives:
                split.augment(selected_rules)
            split_sources = dict(self.split_sources.get(name, {}))
            single_scope = "substrates" if single_substrates_only else "all_molecules"
            cache_dir = _prepare_cache_dir(
                config=self.config,
                split_name=name,
                split=split,
                split_sources=split_sources,
                rules=selected_rules,
                pca=use_pca,
                include_reaction_labels=(name == "train") or (include_val and name == "val"),
                include_pair_graphs=include_pair_graphs,
                include_morgan=include_morgan,
                include_single_graphs=True,
                single_scope=single_scope,
            )
            split.full_setup(
                pca=use_pca,
                rules=selected_rules,
                include_reaction_labels=(name == "train") or (include_val and name == "val"),
                include_pair_graphs=include_pair_graphs,
                include_morgan=include_morgan,
                single_smiles=split.map.keys() if single_substrates_only else None,
                morgan_cache_path=cache_dir / "morgan.pt" if cache_dir is not None and include_morgan else None,
                single_cache_path=cache_dir / "single_graphs.pt" if cache_dir is not None else None,
                reaction_label_cache_path=cache_dir / "reaction_labels.pt"
                if cache_dir is not None and ((name == "train") or (include_val and name == "val"))
                else None,
            )


def _sample_triples(
    triples: list[tuple[int, int, int]],
    max_substrates: int | None,
    seed: int,
) -> list[tuple[int, int, int]]:
    if not max_substrates or max_substrates <= 0:
        return triples
    unique_substrates = sorted({sub_idx for sub_idx, _, _ in triples})
    if max_substrates >= len(unique_substrates):
        return triples
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(np.array(unique_substrates), size=max_substrates, replace=False).tolist())
    return [triple for triple in triples if triple[0] in selected]


def _load_split(
    sdf_path: str | None,
    triples_path: str | None,
    standardize: bool,
    max_substrates: int | None,
    seed: int,
) -> MolFrame:
    if not sdf_path or not triples_path:
        return MolFrame(pd.DataFrame(columns=["sub", "prod", "real"]))
    triples = MolFrame.read_triples(triples_path)
    triples = _sample_triples(triples, max_substrates=max_substrates, seed=seed)
    return MolFrame.from_file(sdf_path, triples, standartize=standardize)


def _resolve_triples_path(triples_path: str | None, use_clean_splits: bool) -> str | None:
    if not triples_path:
        return triples_path
    if not use_clean_splits:
        return triples_path

    path = Path(triples_path)
    if path.name.endswith("_clean.txt") and path.exists():
        return str(path)
    clean_candidate = path.with_name(path.stem + "_clean" + path.suffix)
    if clean_candidate.exists():
        return str(clean_candidate)
    warnings.warn(
        f"Clean triples requested but not found for {triples_path}; falling back to original triples.",
        RuntimeWarning,
        stacklevel=2,
    )
    return triples_path


def _load_rules(config: DatasetConfig) -> List[str]:
    if config.rules_path:
        with open(config.rules_path) as handle:
            rules = [line.strip() for line in handle if line.strip()]
    else:
        rules = load_default_rules()
    if getattr(config, "include_phase2_rules", False):
        from ..utils.preparation import load_phase2_rules

        seen = set(rules)
        for rule in load_phase2_rules():
            if rule not in seen:
                rules.append(rule)
                seen.add(rule)
    return rules


def _load_excluded_substrates(config: DatasetConfig) -> list[str]:
    excluded = list(config.excluded_substrates)
    if config.excluded_substrates_path:
        with open(config.excluded_substrates_path) as handle:
            excluded.extend(line.strip() for line in handle if line.strip())
    return sorted(set(excluded))


def _path_signature(path_str: str | None) -> Dict[str, Any]:
    if not path_str:
        return {"path": None, "exists": False}
    path = Path(path_str)
    if not path.exists():
        return {"path": str(path.resolve()), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _hash_rules(rules: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for rule in rules:
        digest.update(rule.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _cache_key(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _write_cache_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def _prepare_cache_dir(
    config: DatasetConfig,
    split_name: str,
    split: MolFrame,
    split_sources: Dict[str, Any],
    rules: Sequence[str],
    pca: bool,
    include_reaction_labels: bool,
    include_pair_graphs: bool,
    include_morgan: bool,
    include_single_graphs: bool,
    single_scope: str,
) -> Path | None:
    if not config.cache_preprocessed:
        return None

    signature = {
        "schema_version": 1,
        "split": split_name,
        "dataset": {
            "sdf": _path_signature(split_sources.get("sdf")),
            "triples": _path_signature(split_sources.get("triples")),
            "standardize": config.standardize,
            "use_clean_splits": config.use_clean_splits,
            "augment_negatives": config.augment_negatives,
            "pca": pca,
            "max_substrates": split_sources.get("max_substrates"),
            "sampling_seed": split_sources.get("sampling_seed"),
        },
        "preparation": {
            "include_reaction_labels": include_reaction_labels,
            "include_pair_graphs": include_pair_graphs,
            "include_morgan": include_morgan,
            "include_single_graphs": include_single_graphs,
            "single_scope": single_scope,
        },
        "rules": {
            "count": len(rules),
            "sha256": _hash_rules(rules),
        },
        "split_stats": {
            "substrates": len(split.map),
            "products": sum(len(products) for products in split.map.values()),
            "negatives": sum(len(products) for products in split.negs.values()),
        },
    }
    cache_root = Path(config.cache_dir)
    cache_dir = cache_root / split_name / _cache_key(signature)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_cache_metadata(cache_dir / "meta.json", signature)
    return cache_dir


def load_dataset_bundle(config: DatasetConfig) -> DatasetBundle:
    resolved_train_triples = _resolve_triples_path(config.train_triples, config.use_clean_splits)
    resolved_val_triples = _resolve_triples_path(config.val_triples, config.use_clean_splits)
    resolved_test_triples = _resolve_triples_path(config.test_triples, config.use_clean_splits)
    train = _load_split(
        config.train_sdf,
        resolved_train_triples,
        config.standardize,
        max_substrates=config.max_train_substrates,
        seed=config.sampling_seed,
    )
    val = _load_split(
        config.val_sdf,
        resolved_val_triples,
        config.standardize,
        max_substrates=config.max_val_substrates,
        seed=config.sampling_seed + 1,
    )
    test = _load_split(
        config.test_sdf,
        resolved_test_triples,
        config.standardize,
        max_substrates=config.max_test_substrates,
        seed=config.sampling_seed + 2,
    )
    excluded = _load_excluded_substrates(config)
    if excluded:
        train = train.exclude_substrates(excluded)
        val = val.exclude_substrates(excluded)
        test = test.exclude_substrates(excluded)
    rules = _load_rules(config)
    split_sources = {
        "train": {
            "sdf": config.train_sdf,
            "triples": resolved_train_triples,
            "max_substrates": config.max_train_substrates,
            "sampling_seed": config.sampling_seed,
        },
        "val": {
            "sdf": config.val_sdf,
            "triples": resolved_val_triples,
            "max_substrates": config.max_val_substrates,
            "sampling_seed": config.sampling_seed + 1,
        },
        "test": {
            "sdf": config.test_sdf,
            "triples": resolved_test_triples,
            "max_substrates": config.max_test_substrates,
            "sampling_seed": config.sampling_seed + 2,
        },
    }
    return DatasetBundle(train=train, val=val, test=test, rules=rules, config=config, split_sources=split_sources)

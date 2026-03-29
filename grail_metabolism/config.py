from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence
import warnings


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for YAML config support") from exc
    return yaml


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(updated.get(key), dict):
            updated[key] = _deep_update(updated[key], value)
        else:
            updated[key] = value
    return updated


@dataclass
class DatasetConfig:
    train_sdf: Optional[str] = None
    train_triples: Optional[str] = None
    val_sdf: Optional[str] = None
    val_triples: Optional[str] = None
    test_sdf: Optional[str] = None
    test_triples: Optional[str] = None
    rules_path: Optional[str] = None
    uspto_csv: Optional[str] = None
    use_clean_splits: bool = True
    standardize: bool = True
    pca: bool = False
    augment_negatives: bool = False
    cache_preprocessed: bool = True
    cache_dir: str = "artifacts/preprocessed"
    max_uspto_rows: Optional[int] = None
    max_train_substrates: Optional[int] = None
    max_val_substrates: Optional[int] = None
    max_test_substrates: Optional[int] = None
    sampling_seed: int = 42
    excluded_substrates: List[str] = field(default_factory=list)
    excluded_substrates_path: Optional[str] = None


@dataclass
class GeneratorConfig:
    in_channels: int = 16
    edge_dim: int = 18
    hidden_dims: List[int] = field(default_factory=lambda: [128, 256])
    rule_hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
    projection_dim: int = 128
    scoring: Literal["bilinear", "dot", "mlp", "retrieval"] = "retrieval"
    use_pretraining: bool = False
    freeze_pretrained: bool = False
    use_maccs_pretraining: bool = False
    top_k: int = 10
    threshold: Optional[float] = None
    conv_kind: Literal["gatv2", "gcn", "gin"] = "gatv2"
    use_fingerprint: bool = True
    rank_weight: float = 0.2
    ranking_margin: float = 0.4
    prior_strength: float = 0.35
    use_applicability_mask: bool = True
    applicability_penalty: float = 7.5
    candidate_aggregation: Literal["max", "mean", "noisy_or", "hybrid"] = "noisy_or"


@dataclass
class FilterConfig:
    in_channels: int = 18
    edge_dim: int = 18
    hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 128, 128, 64, 32])
    mode: Literal["pair", "single"] = "pair"
    model_type: Literal["Filter", "GATv2Filter", "GCNFilter", "GINFilter", "MolPathFilter", "MorganOnlyFilter"] = "GATv2Filter"
    conv_kind: Literal["gatv2", "gcn", "gin"] = "gatv2"
    use_graph: bool = True
    use_fingerprint: bool = True
    dropout: float = 0.1
    molpath_cutoff: Optional[int] = None
    molpath_y: Optional[float] = None
    molpath_hidden: Optional[int] = None
    train_on_candidates: bool = False
    candidate_generation_top_k: int = 200


@dataclass
class PretrainConfig:
    enabled: bool = False
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-4
    contrastive_ratio: float = 0.4
    maccs_ratio: float = 0.3
    masked_ratio: float = 0.3
    save_path: Optional[str] = None


@dataclass
class OptimConfig:
    lr: float = 1e-4
    epochs: int = 20
    batch_size: int = 64
    weight_decay: float = 1e-6
    prior: float = 0.75
    nnpu: bool = True
    patience: int = 7
    min_delta: float = 1e-4


@dataclass
class EvaluationConfig:
    generator_top_k: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    candidate_top_k: int = 10
    threshold: Optional[float] = None
    export_predictions: bool = True


@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    output_dir: str = "artifacts"
    tags: List[str] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    generator_optim: OptimConfig = field(default_factory=OptimConfig)
    filter_optim: OptimConfig = field(default_factory=OptimConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if self.generator.use_pretraining != self.pretrain.enabled:
            warnings.warn(
                "generator.use_pretraining is a deprecated dead flag. "
                f"pretrain.enabled={self.pretrain.enabled} will be used.",
                DeprecationWarning,
                stacklevel=2,
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> "ExperimentConfig":
        merged = _deep_update(self.to_dict(), overrides)
        return experiment_from_dict(merged)

    def dump_yaml(self, path: str | Path) -> None:
        yaml = _require_yaml()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    def dump_json(self, path: str | Path) -> None:
        import json

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        yaml = _require_yaml()
        with open(path) as handle:
            payload = yaml.safe_load(handle)
        return experiment_from_dict(payload)

    @staticmethod
    def from_json(path: str | Path) -> "ExperimentConfig":
        import json

        with open(path) as handle:
            payload = json.load(handle)
        return experiment_from_dict(payload)


def experiment_from_dict(payload: Dict[str, Any]) -> ExperimentConfig:
    dataset = DatasetConfig(**payload.get("dataset", {}))
    generator = GeneratorConfig(**payload.get("generator", {}))
    filter_config = FilterConfig(**payload.get("filter", {}))
    pretrain = PretrainConfig(**payload.get("pretrain", {}))
    generator_optim = OptimConfig(**payload.get("generator_optim", {}))
    filter_optim = OptimConfig(**payload.get("filter_optim", {}))
    evaluation = EvaluationConfig(**payload.get("evaluation", {}))
    return ExperimentConfig(
        name=payload["name"],
        description=payload.get("description", ""),
        output_dir=payload.get("output_dir", "artifacts"),
        tags=list(payload.get("tags", [])),
        dataset=dataset,
        generator=generator,
        filter=filter_config,
        pretrain=pretrain,
        generator_optim=generator_optim,
        filter_optim=filter_optim,
        evaluation=evaluation,
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return ExperimentConfig.from_yaml(path)
    if suffix == ".json":
        return ExperimentConfig.from_json(path)
    raise ValueError(f"Unsupported config format: {path}")

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
    # Append the curated phase II conjugation rule bank to the active rules. Phase II
    # (glucuronidation, sulfation, etc.) is the main rule-based coverage gap.
    include_phase2_rules: bool = False
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
    # PU-aware down-weighting of applicable-but-unobserved rules (incomplete labels).
    # < 1.0 avoids punishing the generator at full strength for plausible-but-unannotated
    # products, which otherwise suppresses recall.
    unlabeled_weight: float = 0.5
    prior_strength: float = 0.35
    use_applicability_mask: bool = True
    applicability_penalty: float = 7.5
    candidate_aggregation: Literal["max", "mean", "noisy_or", "hybrid"] = "noisy_or"
    # Generator training mode. "supervised" = multi-label rule classification (default).
    # "gflownet" = after supervised warm-start, train the generator as a forward flow
    # policy over the metabolic tree (Trajectory Balance), so terminal sampling is
    # proportional to reward and multi-step paths to annotated metabolites are learned.
    training_mode: Literal["supervised", "gflownet"] = "supervised"


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
class GFlowNetConfig:
    """GFlowNet (Trajectory Balance) generator training over the metabolic tree.

    Active only when GeneratorConfig.training_mode == "gflownet". The forward policy is
    the generator (warm-started from supervised training) + a STOP head; the reward is
    annotation-based at training time (terminal hits an annotated metabolite of the
    parent), which directly optimizes multi-step recall without needing the filter.
    """
    epochs: int = 10
    lr: float = 1e-4               # generator/stop-head learning rate
    logz_lr: float = 1e-2          # larger LR for the scalar logZ (standard for TB)
    batch_substrates: int = 16     # substrates per optimizer step
    max_depth: int = 2             # trajectory length cap
    per_node_top_k: int = 12       # generator rules considered per expansion
    beta: float = 6.0              # reward sharpness: R = exp(beta) for a hit, exp(0)=1 for a miss
    epsilon: float = 0.1           # epsilon-uniform exploration mixed into P_F
    anchor_weight: float = 0.0     # optional supervised imitation bonus on annotated terminals
    node_budget: int = 64          # per-trajectory expansion safety cap
    reward: Literal["annotation", "filter"] = "annotation"


@dataclass
class MultiStepConfig:
    """Multi-step (depth>1) metabolite generation over the metabolic tree.

    Defaults reduce to current single-step behavior (enabled=False / max_depth=1), so an
    unchanged config produces identical metrics. Used both as the runtime config for
    model.multistep.MetabolicTree and as the serializable EvaluationConfig.multistep.
    """
    enabled: bool = False
    max_depth: int = 1                 # 1 == single-step; >=2 engages the beam search
    beam_width: int = 25               # frontier nodes carried to the next depth
    expand_threshold: float = 0.5      # filter(parent, node) >= tau to spawn children
    node_budget: int = 2000            # global per-substrate expansion cap
    per_node_top_k: int = 10           # generator top_k per expansion
    prior_aggregation: Literal["mean", "min", "product", "noisy_or"] = "mean"
    reward_prior_weight: float = 1.0   # node_score = reward * prior**reward_prior_weight


@dataclass
class EvaluationConfig:
    generator_top_k: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    candidate_top_k: int = 10
    # Hard cap on the size of the final ensemble output set. None = uncapped. The
    # headline set-F1/precision collapses under an uncapped candidate flood, so this
    # lets the operating point be bounded to a small k at evaluation time.
    max_output: Optional[int] = None
    # Structure matching for metrics: "exact" canonical-SMILES set equality, or
    # "inchikey" (the literature convention; absorbs tautomer/charge differences).
    match: Literal["exact", "inchikey"] = "exact"
    threshold: Optional[float] = None
    export_predictions: bool = True
    # Multi-step generation at evaluation time (off by default -> single-step).
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    output_dir: str = "artifacts"
    # Global RNG seed for model init / training / shuffling (distinct from
    # dataset.sampling_seed, which only controls data subsampling). Recorded in the
    # metrics report so headline numbers are reproducible.
    seed: int = 42
    tags: List[str] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    generator_optim: OptimConfig = field(default_factory=OptimConfig)
    filter_optim: OptimConfig = field(default_factory=OptimConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    gflownet: GFlowNetConfig = field(default_factory=GFlowNetConfig)

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
    eval_payload = dict(payload.get("evaluation", {}))
    multistep_payload = eval_payload.pop("multistep", {}) or {}
    evaluation = EvaluationConfig(**eval_payload, multistep=MultiStepConfig(**multistep_payload))
    gflownet = GFlowNetConfig(**payload.get("gflownet", {}))
    return ExperimentConfig(
        name=payload["name"],
        description=payload.get("description", ""),
        output_dir=payload.get("output_dir", "artifacts"),
        seed=int(payload.get("seed", 42)),
        tags=list(payload.get("tags", [])),
        dataset=dataset,
        generator=generator,
        filter=filter_config,
        pretrain=pretrain,
        generator_optim=generator_optim,
        filter_optim=filter_optim,
        evaluation=evaluation,
        gflownet=gflownet,
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return ExperimentConfig.from_yaml(path)
    if suffix == ".json":
        return ExperimentConfig.from_json(path)
    raise ValueError(f"Unsupported config format: {path}")

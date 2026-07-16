from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..config import DatasetConfig, EvaluationConfig, ExperimentConfig, FilterConfig, GeneratorConfig, OptimConfig, PretrainConfig


def _default_rules_path() -> str:
    # Resolve through the single shared resolver so presets, load_default_rules() and
    # PretrainedGrail all agree on the default bank (no train/inference label mismatch).
    from ..utils.preparation import _package_root, resolve_default_rule_bank

    bank = resolve_default_rule_bank()
    if bank is None:
        return "grail_metabolism/data/merged_smirks.txt"
    repo_root = _package_root().parent
    try:
        return str(bank.relative_to(repo_root))
    except ValueError:
        return str(bank)


def _default_dataset() -> DatasetConfig:
    return DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf",
        train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf",
        val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf",
        test_triples="grail_metabolism/data/test_triples.txt",
        rules_path=_default_rules_path(),
        uspto_csv="grail_metabolism/data/USPTO_FULL.csv",
        standardize=True,
        pca=False,
        augment_negatives=False,
        max_uspto_rows=10000,
        excluded_substrates_path="grail_metabolism/resources/slow_substrates.txt",
    )


def _base_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="paper_full_ensemble",
        description="Default JCIM-style tri-component ensemble: pretrained generator + rule application + pair filter",
        output_dir="artifacts",
        tags=["paper", "default", "ensemble"],
        dataset=_default_dataset(),
        generator=GeneratorConfig(
            in_channels=16,
            edge_dim=18,
            hidden_dims=[192, 256],
            rule_hidden_dims=[128, 192, 128],
            projection_dim=128,
            scoring="retrieval",
            use_pretraining=True,
            use_maccs_pretraining=True,
            top_k=128,
            threshold=None,
            use_fingerprint=True,
            rank_weight=0.25,
            ranking_margin=0.45,
            prior_strength=0.4,
        ),
        filter=FilterConfig(
            in_channels=18,
            edge_dim=18,
            hidden_dims=[192, 256, 128, 128, 64, 32],
            mode="pair",
            model_type="GATv2Filter",
            use_graph=True,
            use_fingerprint=True,
            dropout=0.1,
            train_on_candidates=False,
            candidate_generation_top_k=200,
        ),
        pretrain=PretrainConfig(
            enabled=True,
            epochs=30,
            batch_size=128,
            lr=1e-4,
            contrastive_ratio=0.4,
            maccs_ratio=0.3,
            masked_ratio=0.3,
        ),
        generator_optim=OptimConfig(lr=1e-4, epochs=20, batch_size=64, weight_decay=1e-6, nnpu=True),
        filter_optim=OptimConfig(lr=1e-4, epochs=20, batch_size=64, weight_decay=1e-6, nnpu=False),
        evaluation=EvaluationConfig(generator_top_k=[1, 3, 5, 10, 20], candidate_top_k=128),
    )


def _preset_map() -> Dict[str, ExperimentConfig]:
    base = _base_experiment()
    presets = {
        "paper_full_ensemble": base,
        "paper_full_ensemble_two_stage_filter": base.with_overrides(
            name="paper_full_ensemble_two_stage_filter",
            description="Default ensemble with a second filter stage trained on generator-produced candidates",
            tags=["paper", "ensemble", "two-stage-filter"],
            filter={"train_on_candidates": True, "candidate_generation_top_k": 200},
        ),
        "paper_full_ensemble_hybrid": base.with_overrides(
            name="paper_full_ensemble_hybrid",
            description=(
                "Default ensemble + factorized hybrid re-ranker (rank by filter*gen*type*site; §10). "
                "Rank-only, never gates. Uses the JOINT-trained heads (rank loss vs frozen gen+filter), "
                "which beat the independently-trained bolt-on by paired +0.0089 [0.003,0.015] and the "
                "filter*gen baseline by +0.0091. Requires a trained FactorizedGenerator checkpoint "
                "(default artifacts/factorized_joint; fall back to artifacts/factorized_v1 for the bolt-on)."
            ),
            tags=["paper", "ensemble", "hybrid-rerank"],
            evaluation={
                "factorized_rerank": True,
                "factorized_rerank_checkpoint": "artifacts/factorized_joint/checkpoints/factorized.pt",
                "factorized_rerank_vocab": "grail_metabolism/resources/coarse_type_vocab.json",
                "factorized_rerank_aggregation": "max",
            },
        ),
        "paper_no_pretrain": base.with_overrides(
            name="paper_no_pretrain",
            description="Ablation: generator without any USPTO-style pretraining",
            tags=["ablation", "no-pretrain"],
            pretrain={"enabled": False},
            generator={"use_pretraining": False, "use_maccs_pretraining": False},
        ),
        "paper_filter_graph_only": base.with_overrides(
            name="paper_filter_graph_only",
            description="Ablation: pair filter without Morgan fingerprints",
            tags=["ablation", "graph-only-filter"],
            filter={"use_graph": True, "use_fingerprint": False},
        ),
        "paper_filter_morgan_only": base.with_overrides(
            name="paper_filter_morgan_only",
            description="Ablation: Morgan-only pair classifier",
            tags=["ablation", "morgan-only-filter"],
            filter={"model_type": "MorganOnlyFilter", "use_graph": False, "use_fingerprint": True},
        ),
        "paper_filter_single": base.with_overrides(
            name="paper_filter_single",
            description="Ablation: independent substrate/product filter instead of pair graph",
            tags=["ablation", "single-filter"],
            filter={"mode": "single", "in_channels": 16, "model_type": "GATv2Filter"},
        ),
        "paper_generator_dot": base.with_overrides(
            name="paper_generator_dot",
            description="Ablation: generator with dot-product scoring instead of bilinear attention",
            tags=["ablation", "generator-dot"],
            generator={"scoring": "dot"},
        ),
        "paper_generator_mlp": base.with_overrides(
            name="paper_generator_mlp",
            description="Ablation: generator with MLP substrate-rule scorer",
            tags=["ablation", "generator-mlp"],
            generator={"scoring": "mlp"},
        ),
        "paper_filter_gcn": base.with_overrides(
            name="paper_filter_gcn",
            description="Ablation: GCN pair filter backbone",
            tags=["ablation", "filter-gcn"],
            filter={"model_type": "GCNFilter", "conv_kind": "gcn"},
        ),
        "paper_filter_gin": base.with_overrides(
            name="paper_filter_gin",
            description="Ablation: GIN pair filter backbone",
            tags=["ablation", "filter-gin"],
            filter={"model_type": "GINFilter", "conv_kind": "gin"},
        ),
        "paper_minimal_baseline": base.with_overrides(
            name="paper_minimal_baseline",
            description="Fast smoke preset for local iteration and CI",
            tags=["baseline", "fast"],
            dataset={"max_uspto_rows": 500},
            pretrain={"enabled": False},
            generator_optim={"epochs": 3, "batch_size": 16},
            filter_optim={"epochs": 3, "batch_size": 16},
            evaluation={"candidate_top_k": 5, "generator_top_k": [1, 3, 5]},
        ),
    }
    return presets


PRESETS = _preset_map()


def list_experiment_presets() -> List[str]:
    return sorted(PRESETS.keys())


def get_experiment_preset(name: str) -> ExperimentConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    return PRESETS[name]


def export_presets(directory: str | Path) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name, config in PRESETS.items():
        config.dump_yaml(destination / f"{name}.yaml")

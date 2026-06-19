from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency fallback
    sns = None

from ..artifacts import ArtifactStore
from ..config import ExperimentConfig
from ..workflows.data import load_dataset_bundle
from ..workflows.ensemble import EnsembleWorkflow
from .presets import get_experiment_preset
from .reports import comparison_markdown


@dataclass(frozen=True)
class BudgetSpec:
    phase_name: str
    train_substrates: int
    val_substrates: int
    test_substrates: int
    max_uspto_rows: int
    pretrain_epochs: int
    generator_epochs: int
    filter_epochs: int
    batch_size: int = 32
    candidate_top_k: int = 15
    sampling_seed: int = 42


@dataclass
class StudyRun:
    phase: str
    preset: str
    config: ExperimentConfig
    artifact_dir: Path
    runtime_sec: float
    metrics: Dict[str, Dict[str, float]]

    def flat_metrics(self) -> Dict[str, float | str]:
        row: Dict[str, float | str] = {
            "phase": self.phase,
            "preset": self.preset,
            "experiment": self.config.name,
            "runtime_sec": round(self.runtime_sec, 3),
        }
        for section, section_metrics in self.metrics.items():
            for metric_name, value in section_metrics.items():
                row[f"{section}.{metric_name}"] = float(value)
        return row


def _make_budgeted_config(preset: str, budget: BudgetSpec, output_dir: str | Path) -> ExperimentConfig:
    return get_experiment_preset(preset).with_overrides(
        name=f"{budget.phase_name}_{preset}",
        output_dir=str(output_dir),
        dataset={
            "max_train_substrates": budget.train_substrates,
            "max_val_substrates": budget.val_substrates,
            "max_test_substrates": budget.test_substrates,
            "max_uspto_rows": budget.max_uspto_rows,
            "sampling_seed": budget.sampling_seed,
            # Standardize identically to the headline pipeline so screening numbers
            # are comparable (the field standardizes both predictions and references).
            "standardize": True,
        },
        pretrain={
            "epochs": budget.pretrain_epochs,
            "batch_size": budget.batch_size,
        },
        generator_optim={
            "epochs": budget.generator_epochs,
            "batch_size": budget.batch_size,
        },
        filter_optim={
            "epochs": budget.filter_epochs,
            "batch_size": budget.batch_size,
        },
        evaluation={
            "candidate_top_k": budget.candidate_top_k,
            "export_predictions": False,
        },
    )


def _load_budget_bundle(reference_config: ExperimentConfig) -> object:
    bundle = load_dataset_bundle(reference_config.dataset)
    bundle.prepare(
        rules=bundle.rules,
        include_val=False,
        include_test=False,
        include_pair_graphs=False,
        include_morgan=False,
        single_substrates_only=reference_config.filter.mode == "pair",
    )
    return bundle


def run_budgeted_suite(
    presets: Sequence[str],
    budget: BudgetSpec,
    artifact_root: str | Path,
) -> List[StudyRun]:
    artifact_root = Path(artifact_root)
    configs = [_make_budgeted_config(preset, budget, artifact_root) for preset in presets]
    bundle = _load_budget_bundle(configs[0])
    runs: List[StudyRun] = []

    for preset, config in zip(presets, configs):
        artifacts = ArtifactStore.create(config.output_dir, config.name)
        config.dump_yaml(artifacts.path("config.yaml"))
        workflow = EnsembleWorkflow(config, artifacts)
        started = time.perf_counter()
        metrics = workflow.run_bundle(bundle)
        runtime_sec = time.perf_counter() - started
        artifacts.save_json("reports/runtime.json", {"runtime_sec": runtime_sec})
        runs.append(
            StudyRun(
                phase=budget.phase_name,
                preset=preset,
                config=config,
                artifact_dir=artifacts.root,
                runtime_sec=runtime_sec,
                metrics=metrics,
            )
        )
    return runs


def runs_to_frame(runs: Iterable[StudyRun]) -> pd.DataFrame:
    rows = [run.flat_metrics() for run in runs]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["phase", "preset"]).reset_index(drop=True)


def _save_table(df: pd.DataFrame, columns: Sequence[str], path: str | Path) -> None:
    view = df.loc[:, columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: round(float(value), 4))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(comparison_markdown(view.to_dict(orient="records")))


def _style_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)


def _plot_heatmap(matrix: pd.DataFrame, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(matrix.columns)), max(3, 0.55 * len(matrix.index))))
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix.iloc[row_idx, col_idx]:.3f}", ha="center", va="center", color="white", fontsize=8)
    _style_axes(ax, title, xlabel="Metric", ylabel="Preset")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ensemble_f1(df: pd.DataFrame, path: str | Path) -> None:
    ordered = df.sort_values("ensemble.f1", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    if sns is not None:
        sns.barplot(data=ordered, x="ensemble.f1", y="preset", palette="crest", ax=ax)
    else:
        ax.barh(ordered["preset"], ordered["ensemble.f1"], color="#2a9d8f")
        ax.invert_yaxis()
    _style_axes(ax, "Ensemble F1 Across Experiments", xlabel="Ensemble F1", ylabel="Preset")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_filter_tradeoff(df: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    if sns is not None:
        sns.scatterplot(data=df, x="filter.roc_auc", y="filter.mcc", hue="preset", s=110, ax=ax)
    else:
        ax.scatter(df["filter.roc_auc"], df["filter.mcc"], s=110, c="#1d3557")
    for _, row in df.iterrows():
        ax.text(float(row["filter.roc_auc"]) + 0.002, float(row["filter.mcc"]) + 0.002, str(row["preset"]), fontsize=8)
    _style_axes(ax, "Filter Quality Trade-off", xlabel="ROC-AUC", ylabel="MCC")
    if sns is not None:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pipeline_gain(df: pd.DataFrame, path: str | Path) -> None:
    frame = df.loc[:, ["preset", "generator.f1", "ensemble.f1"]].melt(id_vars="preset", var_name="stage", value_name="f1")
    frame["stage"] = frame["stage"].map({"generator.f1": "Generator", "ensemble.f1": "Ensemble"})
    fig, ax = plt.subplots(figsize=(11, 5))
    if sns is not None:
        sns.barplot(data=frame, x="preset", y="f1", hue="stage", palette="deep", ax=ax)
    else:
        presets = list(df["preset"])
        x = range(len(presets))
        generator_values = df.set_index("preset").loc[presets, "generator.f1"].tolist()
        ensemble_values = df.set_index("preset").loc[presets, "ensemble.f1"].tolist()
        width = 0.38
        ax.bar([value - width / 2 for value in x], generator_values, width=width, label="Generator", color="#457b9d")
        ax.bar([value + width / 2 for value in x], ensemble_values, width=width, label="Ensemble", color="#e76f51")
        ax.set_xticks(list(x))
        ax.set_xticklabels(presets, rotation=35, ha="right")
    ax.tick_params(axis="x", rotation=35)
    _style_axes(ax, "Generator vs Ensemble F1", xlabel="Preset", ylabel="F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_topk_heatmap(df: pd.DataFrame, path: str | Path, stage: str = "generator") -> None:
    columns = [f"{stage}.top_{k}_recall" for k in (1, 3, 5, 10, 20) if f"{stage}.top_{k}_recall" in df.columns]
    if not columns:
        return
    matrix = df.set_index("preset")[columns].rename(columns=lambda value: value.split(".")[-1])
    if sns is not None:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(matrix))))
        sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
        _style_axes(ax, f"{stage.capitalize()} Top-k Recall", xlabel="Metric", ylabel="Preset")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return
    _plot_heatmap(matrix, path, f"{stage.capitalize()} Top-k Recall")


def plot_runtime(df: pd.DataFrame, path: str | Path) -> None:
    ordered = df.sort_values("runtime_sec", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    if sns is not None:
        sns.barplot(data=ordered, x="preset", y="runtime_sec", palette="mako", ax=ax)
    else:
        ax.bar(ordered["preset"], ordered["runtime_sec"], color="#6d597a")
    ax.tick_params(axis="x", rotation=35)
    _style_axes(ax, "Runtime per Experiment", xlabel="Preset", ylabel="Seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confirmation_heatmap(df: pd.DataFrame, path: str | Path) -> None:
    columns = [column for column in ["ensemble.f1", "ensemble.jaccard", "generator.top_5_recall", "filter.mcc", "filter.roc_auc"] if column in df.columns]
    matrix = df.set_index("preset")[columns]
    if sns is not None:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(matrix))))
        sns.heatmap(matrix, annot=True, fmt=".3f", cmap="rocket_r", ax=ax)
        _style_axes(ax, "Confirmation Metrics", xlabel="Metric", ylabel="Preset")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return
    _plot_heatmap(matrix, path, "Confirmation Metrics")


def create_plots(screening_df: pd.DataFrame, confirmation_df: pd.DataFrame, figure_dir: str | Path) -> List[str]:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []
    if not screening_df.empty:
        plot_ensemble_f1(screening_df, figure_dir / "screening_ensemble_f1.png")
        plot_filter_tradeoff(screening_df, figure_dir / "screening_filter_tradeoff.png")
        plot_pipeline_gain(screening_df, figure_dir / "screening_pipeline_gain.png")
        plot_topk_heatmap(screening_df, figure_dir / "screening_generator_topk_heatmap.png", stage="generator")
        plot_runtime(screening_df, figure_dir / "screening_runtime.png")
        created.extend(
            [
                "screening_ensemble_f1.png",
                "screening_filter_tradeoff.png",
                "screening_pipeline_gain.png",
                "screening_generator_topk_heatmap.png",
                "screening_runtime.png",
            ]
        )
    if not confirmation_df.empty:
        plot_confirmation_heatmap(confirmation_df, figure_dir / "confirmation_heatmap.png")
        created.append("confirmation_heatmap.png")
    return created


def _table_view(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "preset",
        "ensemble.f1",
        "ensemble.jaccard",
        "generator.top_5_recall",
        "filter.mcc",
        "filter.roc_auc",
        "runtime_sec",
    ]
    return df.loc[:, [column for column in columns if column in df.columns]].sort_values("ensemble.f1", ascending=False)


def write_results_markdown(
    screening_df: pd.DataFrame,
    confirmation_df: pd.DataFrame,
    screening_budget: BudgetSpec,
    confirmation_budget: BudgetSpec,
    path: str | Path,
) -> Path:
    best_screening = screening_df.sort_values("ensemble.f1", ascending=False).iloc[0] if not screening_df.empty else None
    best_confirmation = confirmation_df.sort_values("ensemble.f1", ascending=False).iloc[0] if not confirmation_df.empty else None
    best_screening_filter = screening_df.sort_values("filter.roc_auc", ascending=False).iloc[0] if not screening_df.empty else None
    best_confirmation_filter = confirmation_df.sort_values("filter.roc_auc", ascending=False).iloc[0] if not confirmation_df.empty else None
    best_confirmation_generator = confirmation_df.sort_values("generator.top_5_recall", ascending=False).iloc[0] if not confirmation_df.empty else None
    screening_end_to_end_zero = bool(not screening_df.empty and screening_df["ensemble.f1"].max() <= 0.0)
    confirmation_end_to_end_zero = bool(not confirmation_df.empty and confirmation_df["ensemble.f1"].max() <= 0.0)
    confirmation_generator_zero = bool(not confirmation_df.empty and confirmation_df["generator.top_5_recall"].max() <= 0.0)
    pretrain_delta = None
    if {"paper_full_ensemble", "paper_no_pretrain"}.issubset(set(screening_df["preset"])) if not screening_df.empty else False:
        full = screening_df.loc[screening_df["preset"] == "paper_full_ensemble", "ensemble.f1"].iloc[0]
        no_pretrain = screening_df.loc[screening_df["preset"] == "paper_no_pretrain", "ensemble.f1"].iloc[0]
        pretrain_delta = float(full) - float(no_pretrain)

    lines = [
        "# Results",
        "",
        "## Protocol",
        "",
        f"- Screening budget: train/val/test substrates = {screening_budget.train_substrates}/{screening_budget.val_substrates}/{screening_budget.test_substrates}, USPTO rows = {screening_budget.max_uspto_rows}, pretrain/generator/filter epochs = {screening_budget.pretrain_epochs}/{screening_budget.generator_epochs}/{screening_budget.filter_epochs}.",
        f"- Confirmation budget: train/val/test substrates = {confirmation_budget.train_substrates}/{confirmation_budget.val_substrates}/{confirmation_budget.test_substrates}, USPTO rows = {confirmation_budget.max_uspto_rows}, pretrain/generator/filter epochs = {confirmation_budget.pretrain_epochs}/{confirmation_budget.generator_epochs}/{confirmation_budget.filter_epochs}.",
        "- All runs use the manuscript-aligned 3-stage pipeline: multi-label rule scorer, RDKit rule application, pair filter on MCS-aware merged graphs plus Morgan fingerprints.",
        "- These are low-budget autonomy runs meant to validate pipeline behaviour, rank ablations and surface architecture choices, not final manuscript headline numbers.",
        "",
        "## Screening Summary",
        "",
    ]
    if best_screening is not None:
        if screening_end_to_end_zero:
            lines.append("- Screening produced zero end-to-end metabolite retrieval for all presets at this budget, so ranking falls back to component-wise signals rather than ensemble F1.")
        else:
            lines.append(
                f"- Best screening preset by ensemble F1: `{best_screening['preset']}` with F1 = {best_screening['ensemble.f1']:.4f}, Jaccard = {best_screening['ensemble.jaccard']:.4f}, filter MCC = {best_screening['filter.mcc']:.4f}."
            )
    if best_screening_filter is not None:
        lines.append(
            f"- Best screening filter ROC-AUC: `{best_screening_filter['preset']}` with ROC-AUC = {best_screening_filter['filter.roc_auc']:.4f} and MCC = {best_screening_filter['filter.mcc']:.4f}."
        )
    if pretrain_delta is not None:
        lines.append(f"- Pretraining effect on screening ensemble F1 (`paper_full_ensemble` minus `paper_no_pretrain`): {pretrain_delta:+.4f}.")
    if best_confirmation is not None:
        if confirmation_end_to_end_zero:
            lines.append("- Confirmation rerun also kept ensemble F1 at zero across all selected presets, so the meaningful comparison remains at generator recall and filter ranking quality.")
        else:
            lines.append(
                f"- Best confirmation preset by ensemble F1: `{best_confirmation['preset']}` with F1 = {best_confirmation['ensemble.f1']:.4f}."
            )
    if confirmation_generator_zero:
        lines.append("- Confirmation generator top-5 recall also remained zero across the selected presets at this budget.")
    elif best_confirmation_generator is not None:
        lines.append(
            f"- Best confirmation generator top-5 recall: `{best_confirmation_generator['preset']}` with top-5 recall = {best_confirmation_generator['generator.top_5_recall']:.4f}."
        )
    if best_confirmation_filter is not None:
        lines.append(
            f"- Best confirmation filter ROC-AUC: `{best_confirmation_filter['preset']}` with ROC-AUC = {best_confirmation_filter['filter.roc_auc']:.4f}."
        )
    lines.extend(
        [
            "",
            comparison_markdown(_table_view(screening_df).round(4).to_dict(orient="records")) if not screening_df.empty else "_No screening runs._",
            "",
            "## Confirmation Summary",
            "",
            comparison_markdown(_table_view(confirmation_df).round(4).to_dict(orient="records")) if not confirmation_df.empty else "_No confirmation runs._",
            "",
            "## Figures",
            "",
            "![Screening ensemble F1](results/figures/screening_ensemble_f1.png)",
            "",
            "![Screening filter tradeoff](results/figures/screening_filter_tradeoff.png)",
            "",
            "![Generator vs ensemble F1](results/figures/screening_pipeline_gain.png)",
            "",
            "![Screening top-k heatmap](results/figures/screening_generator_topk_heatmap.png)",
            "",
            "![Runtime by experiment](results/figures/screening_runtime.png)",
            "",
            "![Confirmation heatmap](results/figures/confirmation_heatmap.png)",
            "",
            "## Interpretation",
            "",
            "- The best configuration should be treated as the current low-budget winner, not the final paper model.",
            "- On these ultra-low budgets the filter is the most informative component: several variants reach useful ROC-AUC even when end-to-end metabolite retrieval is still zero.",
            "- Generator ranking is now score-aware and the ensemble ranks accepted metabolites by combined generator and filter confidence, so top-k metrics are meaningful.",
            "- USPTO pretraining now reads the actual `reactions` format used in the repository CSV, making the pretrain ablation valid.",
            "- The main failure mode exposed by this study is generator recall under tiny supervision budgets; this is the first axis to scale up for manuscript-grade runs.",
        ]
    )

    destination = Path(path)
    destination.write_text("\n".join(lines))
    return destination

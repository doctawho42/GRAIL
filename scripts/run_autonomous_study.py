from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.experiments.study import (
    BudgetSpec,
    create_plots,
    run_budgeted_suite,
    runs_to_frame,
    write_results_markdown,
)


def main() -> int:
    output_root = ROOT / "results"
    output_root.mkdir(exist_ok=True)
    figure_dir = output_root / "figures"
    table_dir = output_root / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = ROOT / "artifacts" / "autonomous_study"

    screening_budget = BudgetSpec(
        phase_name="screening",
        train_substrates=10,
        val_substrates=8,
        test_substrates=6,
        max_uspto_rows=200,
        pretrain_epochs=1,
        generator_epochs=1,
        filter_epochs=1,
        batch_size=32,
        candidate_top_k=10,
        sampling_seed=42,
    )
    screening_presets = [
        "paper_full_ensemble",
        "paper_no_pretrain",
        "paper_filter_graph_only",
        "paper_filter_morgan_only",
        "paper_filter_single",
        "paper_generator_dot",
        "paper_generator_mlp",
        "paper_filter_gcn",
        "paper_filter_gin",
        "paper_minimal_baseline",
    ]
    screening_runs = run_budgeted_suite(screening_presets, screening_budget, artifact_root)
    screening_df = runs_to_frame(screening_runs)
    screening_df.to_csv(table_dir / "screening_metrics.csv", index=False)

    top_confirmation = (
        screening_df.sort_values(
            [
                "ensemble.f1",
                "ensemble.jaccard",
                "generator.top_5_recall",
                "filter.roc_auc",
                "filter.mcc",
                "runtime_sec",
            ],
            ascending=[False, False, False, False, False, True],
        )
        .head(3)["preset"]
        .tolist()
    )
    confirmation_budget = BudgetSpec(
        phase_name="confirmation",
        train_substrates=20,
        val_substrates=16,
        test_substrates=10,
        max_uspto_rows=400,
        pretrain_epochs=1,
        generator_epochs=2,
        filter_epochs=2,
        batch_size=32,
        candidate_top_k=15,
        sampling_seed=84,
    )
    confirmation_runs = run_budgeted_suite(top_confirmation, confirmation_budget, artifact_root)
    confirmation_df = runs_to_frame(confirmation_runs)
    confirmation_df.to_csv(table_dir / "confirmation_metrics.csv", index=False)

    all_runs = pd.concat([screening_df, confirmation_df], ignore_index=True)
    all_runs.to_csv(table_dir / "all_metrics.csv", index=False)
    create_plots(screening_df, confirmation_df, figure_dir)
    write_results_markdown(screening_df, confirmation_df, screening_budget, confirmation_budget, ROOT / "RESULTS.md")
    print("Study complete")
    print(f"Screening presets: {', '.join(screening_presets)}")
    print(f"Confirmation presets: {', '.join(top_confirmation)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

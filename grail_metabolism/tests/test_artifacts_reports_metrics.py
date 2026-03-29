from __future__ import annotations

import json

import torch

from grail_metabolism.artifacts import ArtifactStore
from grail_metabolism.experiments.reports import comparison_markdown, save_comparison_markdown
from grail_metabolism.metrics import (
    aggregate_prediction_metrics,
    binary_confusion,
    exact_match,
    f1,
    jaccard,
    precision,
    recall,
    top_k_recall,
)


def test_artifact_store_roundtrip(tmp_path):
    store = ArtifactStore.create(tmp_path, "demo")

    json_path = store.save_json("reports/metrics.json", {"f1": 0.5})
    csv_path = store.save_csv("predictions/out.csv", [{"sub": "CCO", "pred": "CC=O"}])
    text_path = store.save_text("reports/note.txt", "ok")
    checkpoint_path = store.save_checkpoint("checkpoints/model_epoch_001.pt", {"weight": torch.tensor([1.0])})

    assert json.loads(json_path.read_text()) == {"f1": 0.5}
    assert "sub,pred" in csv_path.read_text()
    assert text_path.read_text() == "ok"
    assert checkpoint_path.exists()
    assert store.latest_checkpoint("model_epoch_") == checkpoint_path


def test_comparison_report_helpers(tmp_path):
    rows = [
        {"experiment": "a", "ensemble.f1": 0.4},
        {"experiment": "b", "ensemble.f1": 0.5},
    ]
    markdown = comparison_markdown(rows)
    assert "| experiment | ensemble.f1 |" in markdown
    assert "| b | 0.5 |" in markdown

    path = save_comparison_markdown(tmp_path / "comparison.md", rows)
    assert path.read_text() == markdown


def test_binary_metrics_and_zero_division():
    predicted = ["A", "B"]
    real = ["B", "C"]
    assert binary_confusion(predicted, real) == {"tp": 1, "fp": 1, "fn": 1}
    assert precision(predicted, real) == 0.5
    assert recall(predicted, real) == 0.5
    assert round(f1(predicted, real), 6) == 0.5
    assert round(jaccard(predicted, real), 6) == round(1 / 3, 6)
    assert exact_match(predicted, real) == 0.0

    assert precision([], []) == 0.0
    assert recall([], []) == 0.0
    assert f1([], []) == 0.0
    assert jaccard([], []) == 0.0


def test_top_k_and_aggregate_prediction_metrics():
    predictions = [
        {"predicted": ["M1", "M2", "M3"], "real": ["M2", "M4"]},
        {"predicted": ["X1"], "real": ["X1"]},
    ]

    assert top_k_recall(["M1", "M2", "M3"], ["M2", "M4"], k=2) == 0.5

    metrics = aggregate_prediction_metrics(predictions, ks=[1, 2, 3])
    assert set(metrics) == {
        "jaccard",
        "precision",
        "recall",
        "f1",
        "exact_match",
        "top_1_recall",
        "top_2_recall",
        "top_3_recall",
    }
    assert metrics["top_1_recall"] == 0.5
    assert metrics["top_3_recall"] >= metrics["top_1_recall"]

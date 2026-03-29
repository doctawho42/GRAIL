from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..artifacts import ArtifactStore
from ..config import ExperimentConfig
from ..model.filter import Filter
from ..model.generator import Generator
from .data import DatasetBundle


def _frame_summary(data) -> Dict[str, int]:
    return {
        "substrates": len(data.map),
        "positive_pairs": sum(len(products) for products in data.map.values()),
        "negative_pairs": sum(len(products) for products in data.negs.values()),
    }


def _training_report(model, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "best_loss": getattr(model, "best_loss_", None),
        "best_val_loss": getattr(model, "best_val_loss_", None),
        "epochs_trained": getattr(model, "epochs_trained_", None),
        "early_stopped_epoch": getattr(model, "early_stopped_epoch_", None),
        "loss_history": list(getattr(model, "loss_history_", [])),
        "val_loss_history": list(getattr(model, "val_loss_history_", [])),
        "calibrated_threshold": getattr(model, "calibrated_threshold", None),
        "timed_out": getattr(model, "timed_out_", False),
        "timeout_seconds": getattr(model, "timeout_seconds_", None),
        "stop_reason": getattr(model, "stop_reason_", None),
        "last_epoch_seconds": getattr(model, "last_epoch_seconds_", None),
    }
    if extra:
        report.update(extra)
    return report


@dataclass
class GeneratorTrainingWorkflow:
    config: ExperimentConfig
    artifacts: ArtifactStore

    def run(self, generator: Generator, bundle: DatasetBundle, timeout_seconds: Optional[float] = None) -> Generator:
        generator.fit(
            bundle.train,
            lr=self.config.generator_optim.lr,
            eps=self.config.generator_optim.epochs,
            batch_size=self.config.generator_optim.batch_size,
            weight_decay=self.config.generator_optim.weight_decay,
            freeze_pretrained=self.config.generator.freeze_pretrained,
            val_data=bundle.val,
            patience=self.config.generator_optim.patience,
            min_delta=self.config.generator_optim.min_delta,
            timeout_seconds=timeout_seconds,
            verbose=True,
        )
        threshold, metric = generator.calibrate_threshold(bundle.val, bundle.rules, target="recall_at_precision", verbose=True)
        self.artifacts.save_checkpoint(
            "checkpoints/generator.pt",
            {
                "state_dict": generator.state_dict(),
                "calibrated_threshold": threshold,
                "calibration_metric": metric,
            },
        )
        self.artifacts.save_json(
            "reports/generator_calibration.json",
            {"calibrated_threshold": threshold, "metric": metric, "target": "recall_at_precision"},
        )
        self.artifacts.save_json(
            "reports/generator_training.json",
            _training_report(
                generator,
                extra={
                    "train_data": _frame_summary(bundle.train),
                    "val_data": _frame_summary(bundle.val),
                    "calibration_metric": metric,
                    "calibration_target": "recall_at_precision",
                },
            ),
        )
        return generator


@dataclass
class FilterTrainingWorkflow:
    config: ExperimentConfig
    artifacts: ArtifactStore

    def run(self, filter_model: Filter, bundle: DatasetBundle, train_data=None, timeout_seconds: Optional[float] = None):
        selected_train = train_data if train_data is not None else bundle.train
        filter_model.fit(
            selected_train,
            lr=self.config.filter_optim.lr,
            eps=self.config.filter_optim.epochs,
            batch_size=self.config.filter_optim.batch_size,
            weight_decay=self.config.filter_optim.weight_decay,
            prior=self.config.filter_optim.prior,
            nnPU=self.config.filter_optim.nnpu,
            val_data=bundle.val,
            patience=self.config.filter_optim.patience,
            min_delta=self.config.filter_optim.min_delta,
            timeout_seconds=timeout_seconds,
            verbose=True,
        )
        threshold, metric = filter_model.calibrate_threshold(bundle.val, target="f1", verbose=True)
        self.artifacts.save_checkpoint(
            "checkpoints/filter.pt",
            {
                "state_dict": filter_model.state_dict(),
                "calibrated_threshold": threshold,
                "calibration_metric": metric,
            },
        )
        self.artifacts.save_json(
            "reports/filter_calibration.json",
            {"calibrated_threshold": threshold, "metric": metric, "target": "f1"},
        )
        self.artifacts.save_json(
            "reports/filter_training.json",
            _training_report(
                filter_model,
                extra={
                    "train_on_candidates": bool(self.config.filter.train_on_candidates),
                    "train_data": _frame_summary(selected_train),
                    "val_data": _frame_summary(bundle.val),
                    "candidate_stats": dict(getattr(selected_train, "generated_candidate_stats", {})),
                    "calibration_metric": metric,
                    "calibration_target": "f1",
                },
            ),
        )
        return filter_model

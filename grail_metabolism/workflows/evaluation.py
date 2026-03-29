from __future__ import annotations

from typing import Dict, List

from ..config import EvaluationConfig
from ..metrics import aggregate_prediction_metrics
from ..model.filter import Filter
from ..model.generator import Generator
from ..model.wrapper import ModelWrapper
from .data import DatasetBundle


def evaluate_generator(generator: Generator, bundle: DatasetBundle, config: EvaluationConfig) -> Dict[str, float]:
    predictions = []
    threshold = config.threshold if config.threshold is not None else getattr(generator, "calibrated_threshold", None)
    for substrate, products in bundle.test.map.items():
        ranked = generator.generate(
            substrate,
            top_k=config.candidate_top_k,
            threshold=threshold,
        )
        predictions.append({"substrate": substrate, "predicted": ranked, "real": sorted(products)})
    return aggregate_prediction_metrics(predictions, config.generator_top_k)


def evaluate_filter(filter_model: Filter, bundle: DatasetBundle) -> Dict[str, float]:
    mcc, roc_auc = bundle.test.test(filter_model, mode=filter_model.mode)
    return {"mcc": float(mcc), "roc_auc": float(roc_auc)}


def evaluate_ensemble(model: ModelWrapper, bundle: DatasetBundle, config: EvaluationConfig) -> Dict[str, float]:
    predictions = []
    threshold = config.threshold if config.threshold is not None else getattr(model.generator, "calibrated_threshold", None)
    for substrate, products in bundle.test.map.items():
        ranked = model.generate(
            substrate,
            top_k=config.candidate_top_k,
            threshold=threshold,
        )
        predictions.append({"substrate": substrate, "predicted": ranked, "real": sorted(products)})
    return aggregate_prediction_metrics(predictions, config.generator_top_k)


def collect_ensemble_predictions(model: ModelWrapper, bundle: DatasetBundle, config: EvaluationConfig) -> List[Dict[str, object]]:
    rows = []
    threshold = config.threshold if config.threshold is not None else getattr(model.generator, "calibrated_threshold", None)
    for substrate, products in bundle.test.map.items():
        ranked = model.generate(
            substrate,
            top_k=config.candidate_top_k,
            threshold=threshold,
        )
        rows.append(
            {
                "substrate": substrate,
                "predicted": ranked,
                "real": sorted(products),
            }
        )
    return rows

from __future__ import annotations

from typing import Dict, List

from ..config import EvaluationConfig
from ..metrics import aggregate_prediction_metrics
from ..model.filter import Filter
from ..model.generator import Generator
from ..model.wrapper import ModelWrapper
from .data import DatasetBundle


def _generator_predictions(generator: Generator, frame, config: EvaluationConfig) -> List[Dict[str, object]]:
    threshold = config.threshold if config.threshold is not None else getattr(generator, "calibrated_threshold", None)
    predictions = []
    for substrate, products in frame.map.items():
        ranked = generator.generate(substrate, top_k=config.candidate_top_k, threshold=threshold)
        predictions.append({"substrate": substrate, "predicted": ranked, "real": sorted(products)})
    return predictions


def _ensemble_predictions(model: ModelWrapper, frame, config: EvaluationConfig) -> List[Dict[str, object]]:
    threshold = config.threshold if config.threshold is not None else getattr(model.generator, "calibrated_threshold", None)
    ms = getattr(config, "multistep", None)
    multistep = ms if (ms is not None and ms.enabled and ms.max_depth > 1) else None
    # Headline metric is recall@k, so default to rank-only (no hard filter gate). The
    # gate is precision-oriented and measurably lowers recall@k.
    gate_by_filter = getattr(config, "ranking_policy", "rank") == "gate"
    filter_candidate_cap = getattr(config, "filter_candidate_cap", None)
    rows = []
    for substrate, products in frame.map.items():
        ranked = model.generate(
            substrate,
            top_k=config.candidate_top_k,
            threshold=threshold,
            max_output=config.max_output,
            multistep=multistep,
            gate_by_filter=gate_by_filter,
            filter_candidate_cap=filter_candidate_cap,
        )
        rows.append({"substrate": substrate, "predicted": ranked, "real": sorted(products)})
    return rows


def evaluate_generator(generator: Generator, bundle: DatasetBundle, config: EvaluationConfig) -> Dict[str, float]:
    return aggregate_prediction_metrics(_generator_predictions(generator, bundle.test, config), config.generator_top_k, match=config.match)


def evaluate_filter(filter_model: Filter, bundle: DatasetBundle) -> Dict[str, float]:
    mcc, roc_auc = bundle.test.test(filter_model, mode=filter_model.mode)
    return {"mcc": float(mcc), "roc_auc": float(roc_auc)}


def evaluate_ensemble(model: ModelWrapper, bundle: DatasetBundle, config: EvaluationConfig) -> Dict[str, float]:
    return aggregate_prediction_metrics(_ensemble_predictions(model, bundle.test, config), config.generator_top_k, match=config.match)


def evaluate_ensemble_val(model: ModelWrapper, bundle: DatasetBundle, config: EvaluationConfig) -> Dict[str, float]:
    """Ensemble metrics on the VALIDATION split, for model/preset selection.

    Selecting the best preset/hyperparameters by test metrics is selection-on-test
    leakage. Use this for ranking; reserve evaluate_ensemble (test) for the single
    final report of the chosen configuration.
    """
    return aggregate_prediction_metrics(_ensemble_predictions(model, bundle.val, config), config.generator_top_k, match=config.match)


def collect_ensemble_predictions(model: ModelWrapper, bundle: DatasetBundle, config: EvaluationConfig) -> List[Dict[str, object]]:
    return _ensemble_predictions(model, bundle.test, config)

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_confusion(predicted: Iterable[str], real: Iterable[str]) -> Dict[str, int]:
    pred = set(predicted)
    truth = set(real)
    return {
        "tp": len(pred & truth),
        "fp": len(pred - truth),
        "fn": len(truth - pred),
    }


def precision(predicted: Iterable[str], real: Iterable[str]) -> float:
    conf = binary_confusion(predicted, real)
    return _safe_div(conf["tp"], conf["tp"] + conf["fp"])


def recall(predicted: Iterable[str], real: Iterable[str]) -> float:
    conf = binary_confusion(predicted, real)
    return _safe_div(conf["tp"], conf["tp"] + conf["fn"])


def f1(predicted: Iterable[str], real: Iterable[str]) -> float:
    p = precision(predicted, real)
    r = recall(predicted, real)
    return _safe_div(2 * p * r, p + r)


def jaccard(predicted: Iterable[str], real: Iterable[str]) -> float:
    pred = set(predicted)
    truth = set(real)
    union = pred | truth
    return _safe_div(len(pred & truth), len(union))


def top_k_recall(ranked: Sequence[str], real: Iterable[str], k: int) -> float:
    truth = set(real)
    return _safe_div(len(set(ranked[:k]) & truth), len(truth))


def exact_match(predicted: Iterable[str], real: Iterable[str]) -> float:
    return float(set(predicted) == set(real))


def aggregate_prediction_metrics(predictions: List[Dict[str, object]], ks: Sequence[int]) -> Dict[str, float]:
    if not predictions:
        return {}
    metrics = {
        "jaccard": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "exact_match": 0.0,
    }
    for k in ks:
        metrics[f"top_{k}_recall"] = 0.0

    for row in predictions:
        ranked = list(row["predicted"])  # type: ignore[index]
        real = set(row["real"])  # type: ignore[index]
        metrics["jaccard"] += jaccard(ranked, real)
        metrics["precision"] += precision(ranked, real)
        metrics["recall"] += recall(ranked, real)
        metrics["f1"] += f1(ranked, real)
        metrics["exact_match"] += exact_match(ranked, real)
        for k in ks:
            metrics[f"top_{k}_recall"] += top_k_recall(ranked, real, k)

    total = float(len(predictions))
    return {key: value / total for key, value in metrics.items()}

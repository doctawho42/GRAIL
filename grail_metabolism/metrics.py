from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Literal, Sequence, Set


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@lru_cache(maxsize=131072)
def _inchikey(smiles: str) -> str:
    """InChIKey for a SMILES, falling back to the raw string if RDKit can't parse it.

    The metabolite-prediction literature (MetaTrans, GLORYx, LAGOM) matches predicted
    vs reference structures by InChIKey / Tanimoto=1 rather than raw SMILES equality,
    because that absorbs tautomer/charge/canonicalization discrepancies that would
    otherwise miss true matches. This enables an InChIKey matching mode for
    apples-to-apples comparison with those papers.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToInchiKey(mol) or smiles
    except Exception:
        return smiles


def _match_keys(items: Iterable[str], match: str) -> Set[str]:
    if match == "inchikey":
        return {_inchikey(str(item)) for item in items}
    return {str(item) for item in items}


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


def aggregate_prediction_metrics(
    predictions: List[Dict[str, object]],
    ks: Sequence[int],
    match: Literal["exact", "inchikey"] = "exact",
) -> Dict[str, float]:
    """Macro-averaged set metrics over per-substrate predictions.

    match="inchikey" compares structures by InChIKey (literature convention) instead of
    raw SMILES equality. ``mean_output_size`` reports the average number of predicted
    structures per substrate, which the field always reports next to precision because
    precision is a pessimistic lower bound under incomplete annotation (an unannotated
    prediction is counted as a false positive but may be a real, unrecorded metabolite).
    """
    if not predictions:
        return {}
    metrics = {
        "jaccard": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "exact_match": 0.0,
        "mean_output_size": 0.0,
    }
    for k in ks:
        metrics[f"top_{k}_recall"] = 0.0

    for row in predictions:
        raw_ranked = list(row["predicted"])  # type: ignore[index]
        metrics["mean_output_size"] += float(len(raw_ranked))
        # Map to comparison keys but preserve rank order for top-k.
        ranked = [next(iter(_match_keys([item], match))) for item in raw_ranked]
        real = _match_keys(row["real"], match)  # type: ignore[index]
        metrics["jaccard"] += jaccard(ranked, real)
        metrics["precision"] += precision(ranked, real)
        metrics["recall"] += recall(ranked, real)
        metrics["f1"] += f1(ranked, real)
        metrics["exact_match"] += exact_match(ranked, real)
        for k in ks:
            metrics[f"top_{k}_recall"] += top_k_recall(ranked, real, k)

    total = float(len(predictions))
    return {key: value / total for key, value in metrics.items()}

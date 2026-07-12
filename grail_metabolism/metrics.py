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


def _taut_key_raw(smiles: str) -> str:
    """Tautomer-canonical InChIKey with NO fallback: standardize (Cleanup -> FragmentParent ->
    uncharge -> TautomerEnumerator.Canonicalize) then InChIKey. Raises on any failure — the caller
    decides whether that failure is per-molecule (fall back) or systemic (fail loud)."""
    from rdkit import Chem

    from grail_metabolism.utils.preparation import _standardize_smiles_cached

    standardized = _standardize_smiles_cached(smiles)
    mol = Chem.MolFromSmiles(standardized)
    if mol is None:
        raise ValueError(f"could not parse standardized SMILES for {smiles!r}")
    return Chem.MolToInchiKey(mol) or standardized


_TAUTOMER_PATH_OK: "bool | None" = None


def _ensure_tautomer_path() -> None:
    """One-time self-check that the tautomer path is actually functional. ``_tautomer_inchikey``
    falls back to plain ``_inchikey`` on failure — correct for a single un-standardizable molecule,
    but catastrophic if the WHOLE standardize stack is broken (e.g. a missing dependency): every key
    would silently collapse to the plain InChIKey, degrading every tautomer number (the 0.735 ceiling
    -> the plain 0.718, etc.) with no error. So on first use we verify a known keto/enol pair really
    merges; if it does not, we fail LOUDLY rather than let the headline numbers silently degrade."""
    global _TAUTOMER_PATH_OK
    if _TAUTOMER_PATH_OK:
        return
    try:
        merged = _taut_key_raw("CC(=O)CC(C)=O") == _taut_key_raw("CC(=O)C=C(O)C")
    except Exception as exc:  # standardization itself is broken -> systemic, not a bad input
        raise RuntimeError(
            "tautomer-InChIKey standardization is broken (cannot standardize a known keto/enol "
            "pair); refusing to silently fall back to plain InChIKey and corrupt every tautomer "
            f"number. Original error: {exc!r}"
        ) from exc
    if not merged:
        raise RuntimeError(
            "tautomer-InChIKey path is non-functional: a known keto/enol pair did NOT merge, so "
            "tautomer canonicalization is inactive (likely a broken standardize dependency). "
            "Refusing to silently degrade every tautomer number to plain InChIKey."
        )
    _TAUTOMER_PATH_OK = True


@lru_cache(maxsize=131072)
def _tautomer_inchikey(smiles: str) -> str:
    """Tautomer-canonical InChIKey: full standardization (incl. tautomer canonicalization) then InChIKey.

    Standard InChI only normalizes a *subset* of tautomers, so two SMILES that are the
    same molecule up to a proton shift (e.g. keto/enol, amide/imidic-acid, several N-H
    heterocycles) can get different InChIKeys. The rule engine routinely emits a
    different tautomer of the annotated metabolite than the reference SMILES, so plain
    ``match="inchikey"`` silently misses those true hits (measured ~4.5x recall loss on
    the (в) subset run). Running ``standardize_mol`` (Cleanup -> FragmentParent ->
    uncharge -> TautomerEnumerator.Canonicalize) on BOTH sides collapses them onto the
    same representative before the InChIKey, so the match becomes tautomer-invariant.

    This is heavier than ``_inchikey`` (tautomer enumeration), so it is reserved for the
    small final output set + reference set at scoring time, never the generation loop.
    Both this and the underlying standardize are cached, so repeats are free.

    A one-time canary (``_ensure_tautomer_path``) fails loudly if the whole standardize path is
    broken, so a degraded environment cannot silently turn this into plain ``_inchikey``. A single
    un-standardizable molecule still falls back gracefully (per-input, not systemic).
    """
    _ensure_tautomer_path()
    try:
        return _taut_key_raw(smiles)
    except Exception:
        return _inchikey(smiles)  # this ONE molecule couldn't be standardized -> per-input fallback


@lru_cache(maxsize=131072)
def _inchikey_skeleton(smiles: str) -> str:
    """First InChIKey block (the connectivity skeleton) -- stereo/charge/isotope-blind.

    Approximates GLORYx's matching, which generates InChI *without stereochemistry*. Two
    stereoisomers (or charge variants) share this key but differ under full `inchikey`.
    """
    key = _inchikey(smiles)
    return key.split("-")[0] if "-" in key else key


@lru_cache(maxsize=131072)
def _morgan_key(smiles: str) -> str:
    """Morgan/ECFP(r=2, 2048) bit-string. Two molecules share it iff their fingerprints are
    identical, i.e. Tanimoto = 1 -- the matching MetaTrans uses (deliberately stereo/charge
    blind, and lenient to fingerprint collisions)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()
    except Exception:
        return smiles


@lru_cache(maxsize=131072)
def _canonical_key(smiles: str) -> str:
    """Stereo-free canonical SMILES (RDKit, isomericSmiles=False) -- the matching LAGOM uses
    (canonical-SMILES top-k). Stricter than InChIKey (sensitive to charge/protonation form)."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else smiles
    except Exception:
        return smiles


def _match_keys(items: Iterable[str], match: str) -> Set[str]:
    # Each metabolite-prediction paper matches structures differently; exposing them all as
    # set keys lets one prediction set be re-scored under every protocol (the rank-flip /
    # match-sensitivity experiment). `exact` is raw-SMILES equality; `canonical` is RDKit
    # stereo-free canonical SMILES (LAGOM).
    if match == "canonical":
        return {_canonical_key(str(item)) for item in items}
    if match == "inchikey":
        return {_inchikey(str(item)) for item in items}
    if match == "inchikey_tautomer":
        return {_tautomer_inchikey(str(item)) for item in items}
    if match == "inchi_no_stereo":
        return {_inchikey_skeleton(str(item)) for item in items}
    if match == "tanimoto1":
        return {_morgan_key(str(item)) for item in items}
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
    match: Literal["exact", "canonical", "inchikey", "inchikey_tautomer", "inchi_no_stereo", "tanimoto1"] = "exact",
) -> Dict[str, float]:
    """Macro-averaged set metrics over per-substrate predictions.

    match="inchikey" compares structures by InChIKey (literature convention) instead of
    raw SMILES equality; match="inchikey_tautomer" first tautomer-canonicalizes both
    sides (see ``_tautomer_inchikey``) so a rule-emitted tautomer of the reference still
    matches -- the recall-correct mode for this rule engine. ``mean_output_size`` reports the average number of predicted
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

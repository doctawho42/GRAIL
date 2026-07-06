"""Diverse-subset-selection baselines over a reranker-scored candidate pool.

Three selectors -- ``temperature_topp_select`` (BASE-01), ``dpp_greedy_select``
(BASE-02), ``mmr_select`` (BASE-03) -- all implement one shared contract:

    select(pool, k, method, **knob) -> List[str]

``pool`` is ``List[Tuple[str, float]]`` (SMILES, raw reranker LOGIT score) in
any order; the return is a RANKED list of SMILES (rank 0 = most preferred by
the given method), of length >= k whenever the deduped pool supports it.

Every selector's fingerprint/similarity computation routes through the ONE
shared ``_pool_fingerprints``/``_tanimoto_kernel_matrix`` pair in this module,
which reuse ``eval/diversity.py``'s exact Morgan r=2/2048 fingerprint
convention (``AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)``)
-- no second fingerprint path. Dedup/identity routes through
``grail_metabolism.metrics._tautomer_inchikey``, the SAME canonicalization
source ``eval/diversity.py`` uses; this module does NOT define a second
InChIKey/canonicalization function.

Kept dependency-light like ``eval/diversity.py`` (RDKit only, no torch), with
ONE deliberate, approved deviation: ``numpy`` is imported (already a pinned
hard project dependency) for the vectorized softmax/argmax/incremental-Cholesky
math the selection rules need. ``eval/diversity.py`` itself stays pure-Python +
RDKit; this is a scoped exception for this module only.

All scores in ``pool`` are expected to be RAW, unnormalized LOGITS, exactly as
returned by ``BiEncoderReranker.forward`` (a bare ``nn.Linear(64, 1)`` head,
no sigmoid). Temperature scaling divides the logit directly
(``p_i \\propto exp(logit_i / T)``); it never inverse-sigmoids an
assumed-probability input.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from grail_metabolism.metrics import _tautomer_inchikey

Pool = Sequence[Tuple[str, float]]


def _pool_fingerprints(smiles_list: Sequence[str]) -> List:
    """Morgan/ECFP4 (radius=2, nBits=2048) fingerprints for ``smiles_list``.

    THE single fingerprint-construction path shared by ``dpp_greedy_select``
    and ``mmr_select`` (and available to any future selector) -- never
    redefine this per-selector. Verbatim call signature copied from
    ``eval/diversity.py``'s internal fingerprinting (Pitfall 3 guard: a guard
    test asserts this matches ``diversity.mean_pairwise_tanimoto`` on a fixed
    pair). Unparseable SMILES are parsed-and-skipped
    (``Chem.MolFromSmiles(s) is None``), mirroring ``diversity.py``'s
    existing convention (T-02-01).
    """
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    return fps


def _tanimoto_kernel_matrix(fps: Sequence) -> np.ndarray:
    """NxN Tanimoto similarity matrix over ``fps`` (diagonal 1.0).

    Built via ``DataStructs.TanimotoSimilarity`` -- the same similarity
    primitive ``eval/diversity.py``'s ``mean_pairwise_tanimoto``/
    ``circles_count`` use. This, together with ``_pool_fingerprints``, is the
    SINGLE fingerprint/similarity path shared by DPP and MMR.
    """
    n = len(fps)
    S = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            S[i, j] = sim
            S[j, i] = sim
    return S


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _nucleus_truncate(probs: np.ndarray, p: float) -> List[int]:
    """Indices (into ``probs``) surviving top-p nucleus truncation, sorted desc by prob."""
    order = list(np.argsort(-probs))
    cumulative = 0.0
    support: List[int] = []
    for idx in order:
        support.append(int(idx))
        cumulative += float(probs[idx])
        if cumulative >= p:
            break
    return support


def select(pool: Pool, k: int, method: str, **knob) -> List[str]:
    """Shared dispatcher: ``select(pool, k, method, **knob) -> List[str]``.

    ``pool`` is ``[(smiles, raw_logit_score), ...]`` in any order -- expects
    RAW logits (unnormalized), as returned by ``BiEncoderReranker.forward``.
    Returns a RANKED ``List[str]`` (rank 0 = most preferred), of length >= k
    whenever the deduped pool supports it. Degenerate inputs (empty pool,
    ``k <= 0``) return ``[]``; ``k`` at or beyond the distinct-pool size
    returns the full deduped ranking.

    ``method`` dispatches to one of ``temperature_topp_select`` /
    ``dpp_greedy_select`` / ``mmr_select``; the corresponding **knob kwargs
    are the method's own tunables (``T``/``p``/``rng`` for temperature/top-p,
    ``theta``/``fps``/``eps``/``tol`` for DPP, ``lam``/``fps`` for MMR).
    """
    if not pool or k <= 0:
        return []

    # NOTE: dispatch is resolved by name against this module's globals (not a
    # dict built from direct references) so `select()` stays usable for
    # already-implemented methods even while a later-added selector (DPP/MMR)
    # is still under construction earlier in this module's rollout.
    dispatch_names = {
        "temperature_topp": "temperature_topp_select",
        "dpp": "dpp_greedy_select",
        "mmr": "mmr_select",
    }
    if method not in dispatch_names:
        raise ValueError(
            f"select: unknown method {method!r} (expected one of {sorted(dispatch_names)})"
        )
    fn = globals()[dispatch_names[method]]
    return fn(pool, k, **knob)


def temperature_topp_select(
    pool: Pool,
    k: int,
    T: float = 1.0,
    p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Temperature/top-p (nucleus) stochastic selection over raw reranker logits (BASE-01).

    Per D-BASE01-TEMPGRID: divides the raw logit by ``max(T, 1e-8)``,
    softmaxes, nucleus-truncates the resulting distribution to the smallest
    prefix (sorted descending by probability) whose cumulative mass >= ``p``,
    then samples WITHOUT replacement from the surviving support, deduping
    each pick by ``_tautomer_inchikey`` mid-loop, until ``k`` distinct
    candidates are collected or the nucleus support is exhausted. Does NOT
    inverse-sigmoid -- the input scores are already logit-space.

    ``rng`` defaults to ``np.random.default_rng()`` when not supplied; pass a
    seeded ``Generator`` for deterministic/reproducible tests.
    """
    if not pool or k <= 0:
        return []

    smiles = [s for s, _ in pool]
    scores = np.asarray([sc for _, sc in pool], dtype=np.float64)

    logits = scores / max(T, 1e-8)
    probs = _softmax(logits)
    support = _nucleus_truncate(probs, p)

    if rng is None:
        rng = np.random.default_rng()

    remaining = list(support)
    ranked: List[str] = []
    seen_keys = set()
    while remaining and len(ranked) < k:
        remaining_probs = np.asarray([probs[i] for i in remaining], dtype=np.float64)
        remaining_probs = remaining_probs / remaining_probs.sum()
        choice_pos = int(rng.choice(len(remaining), p=remaining_probs))
        idx = remaining.pop(choice_pos)
        smi = smiles[idx]
        key = _tautomer_inchikey(smi)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        ranked.append(smi)
    return ranked


def dpp_greedy_select(
    pool: Pool,
    k: int,
    theta: float = 1.0,
    fps: Optional[Sequence] = None,
    eps: float = 1e-8,
    tol: float = 1e-6,
) -> List[str]:
    """DPP greedy MAP selection via incremental Cholesky (BASE-02).

    Per D-BASE02-DPPKERNEL: ``L = diag(q).S.diag(q)`` with ``q_i = exp(theta *
    relevance_logit_i)`` (the quality term) and ``S`` the Tanimoto similarity
    kernel over the shared Morgan r=2/2048 fingerprint (``_pool_fingerprints``/
    ``_tanimoto_kernel_matrix``). Greedy MAP inference follows Chen, Zhang and
    Zhou (NeurIPS 2018, arXiv:1709.05135)'s incremental-Cholesky algorithm --
    ``O(K^2 * N_pool)``, not the naive ``O(M^4)`` recompute-determinant
    version.

    Near-duplicate-exclusion + numerical-stability scheme (the load-bearing
    correctness spec -- see D-BASE02-DPPKERNEL's rationale for why this is
    NOT the naive fixed-epsilon-floor-plus-absolute-break sketch):

    1. The SIMILARITY-kernel diagonal is regularized AT CONSTRUCTION to
       ``S_ii = 1 + eps`` (NOT a uniform epsilon added to the whole marginal-
       gain vector), keeping ``L`` positive-definite.
    2. Each per-step marginal gain is CLIPPED to ``>= 0`` before the sqrt
       (``d2j = max(d2[j], 0.0)``) -- this only touches items already at
       ~0, preserving the TRUE relative ordering of genuine diversity.
    3. The loop stops on a RELATIVE tolerance: ``d2_init_max`` is the max of
       the initial marginal-gain vector, and the loop breaks when the argmax
       remaining gain ``d2[j] <= tol * d2_init_max``. Redundant near-duplicate
       items (gain at the jitter-noise floor) trip this stop; genuinely
       diverse items (gain O(1)) do not -- so the selector actually EXCLUDES
       near-duplicate regioisomers (returns FEWER than k on a redundant pool)
       instead of merely avoiding NaN.

    ``fps`` may be a precomputed fingerprint list (reused by an orchestrator
    across multiple selectors on the same pool); when omitted, fingerprints
    are computed internally via ``_pool_fingerprints``.
    """
    if not pool or k <= 0:
        return []

    smiles = [s for s, _ in pool]
    scores = np.asarray([sc for _, sc in pool], dtype=np.float64)
    n = len(smiles)

    if fps is None:
        fps = _pool_fingerprints(smiles)
    # _pool_fingerprints may have skipped unparseable SMILES; guard the shape
    # mismatch defensively (degenerate/malformed pool) rather than crash.
    if len(fps) != n:
        # Fall back to only the parseable subset, keeping smiles/scores aligned.
        parseable_smiles = []
        parseable_idx = []
        for i, s in enumerate(smiles):
            if Chem.MolFromSmiles(s) is not None:
                parseable_smiles.append(s)
                parseable_idx.append(i)
        smiles = parseable_smiles
        scores = scores[parseable_idx]
        n = len(smiles)
        if n == 0:
            return []

    q = np.exp(theta * scores)
    S = _tanimoto_kernel_matrix(fps)
    # Regularize the SIMILARITY-kernel diagonal AT CONSTRUCTION (S_ii = 1+eps),
    # NOT a uniform epsilon added to the whole d2 vector later.
    np.fill_diagonal(S, 1.0 + eps)

    selected: List[int] = []
    ranked: List[str] = []
    c = np.zeros((0, n), dtype=np.float64)  # accumulated Cholesky columns

    # diag(L) under the regularized S: d2_i = q_i^2 * S_ii = q_i^2 * (1+eps)
    d2 = (q * q) * (1.0 + eps)
    d2_init_max = float(np.max(d2)) if n else 0.0

    for _ in range(min(k, n)):
        j = int(np.argmax(d2))
        if d2_init_max <= 0.0 or d2[j] <= tol * d2_init_max:
            break  # relative-tolerance stop: remaining gain is at the noise floor
        d2j = max(float(d2[j]), 0.0)  # per-step gain clip (touches only ~0 items)
        if c.shape[0]:
            e = (q[j] * q * S[j, :] - c[:, j] @ c) / np.sqrt(d2j)
        else:
            e = (q[j] * q * S[j, :]) / np.sqrt(d2j)
        c = np.vstack([c, e])
        d2 = d2 - e ** 2
        d2[j] = -np.inf  # never re-pick
        selected.append(j)
        ranked.append(smiles[j])

    return ranked

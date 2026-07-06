"""Diversity, coverage, and mode-discovery metrics for set-valued generation.

These metrics evaluate a *set generator* (e.g. the Stage-2b Set-GFlowNet) on
how well it covers the space of true annotated metabolites and how diverse/
calibrated its produced sets are, independent of the pair-wise precision/
recall/F1 in ``grail_metabolism/metrics.py``.

Kept dependency-light and pure (RDKit only, no torch) so it can be imported
from evaluation scripts and notebooks without pulling in the model stack.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, Iterable, List, Literal, Sequence, Set

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers

from grail_metabolism.metrics import _match_keys, _tautomer_inchikey


def annotated_coverage_count(sampled_sets: Iterable[frozenset], annotated_ik: Set[str]) -> int:
    """Count distinct annotated InChIKeys found across ALL sampled sets.

    Takes the union of every set in ``sampled_sets`` (each a ``frozenset[str]``
    of InChIKeys) and intersects it with ``annotated_ik``, the ground-truth
    annotated InChIKeys for the substrate. This is a coverage/"mode discovery"
    metric: how many of the true metabolites did the generator find in *any*
    of its sampled sets, not just its best/first one.

    NOTE: this function is a pure rename of GRAIL's original ``modes_discovered``
    (byte-identical behavior). It is gated on the same incomplete PU annotation
    set as ``modes_discovered_canonical`` below -- it can only "find" a
    metabolite that happens to be annotated, so it is a lower bound on true
    coverage, not an exact count (the PU precision-as-lower-bound caveat).
    """
    union: Set[str] = set()
    for s in sampled_sets:
        union.update(s)
    return len(union & set(annotated_ik))


def dedup_to_budget(
    smiles_ranked: Sequence[str], k: int, match: str = "inchikey_tautomer"
) -> List[str]:
    """Truncate a ranked candidate list to ``k`` DISTINCT post-canonicalization molecules.

    Iterates ``smiles_ranked`` in rank order, computes each candidate's match
    key via ``grail_metabolism.metrics._match_keys`` (the SAME canonicalization
    used for recall matching), and keeps the first-seen SMILES per distinct key
    until ``k`` distinct keys are collected. This is THE single dedup+truncate
    utility every method's output (GFlowNet, reranker, DPP, MMR, SyGMa, ...)
    should pass through before any metric (recall OR diversity) is computed on
    it, so a method that happens to emit more tautomer-duplicate pairs cannot
    silently look parsimonious by exhausting its budget on duplicates.

    ``match`` is an explicit parameter (default ``"inchikey_tautomer"``), not a
    hardcoded convention, so callers can re-score under a different matching
    protocol via the same ``_match_keys`` dispatch recall matching uses.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for s in smiles_ranked:
        key = next(iter(_match_keys([s], match)))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= k:
            break
    return out


def union_at_k_curve(
    smiles_ranked: Sequence[str],
    annotated_ik: Set[str],
    ks: Sequence[int] = (5, 10, 15, 20, 30, 50),
) -> Dict[int, float]:
    """union@K for every K in ``ks``, computed over ONE ranked SMILES stream.

    ``smiles_ranked`` is a single ranked list of SMILES in caller-supplied
    order -- the SAME shape ``dedup_to_budget`` consumes, so the GFlowNet path
    (reconstructing SMILES via its ``smiles_of`` map) and Phase-2 baselines
    (which natively produce ranked SMILES) can both call this one function
    without a frozenset<->SMILES bridge (FIX 5: one shared curve primitive).

    For each ``k``, asserts ``len(smiles_ranked) >= k`` (flagging when a
    method under-produces at a requested K -- e.g. SyGMa's internal pruning
    capping output below K -- rather than silently truncating to whatever was
    sampled) and computes the coverage numerator as
    ``|{_tautomer_inchikey(s) for s in smiles_ranked[:k]} & annotated_ik| / |annotated_ik|``
    over the first-``k`` slice of the SAME underlying stream -- routing
    SMILES->identity through the SAME ``_tautomer_inchikey`` as recall/dedup
    (EVAL-04 consistency), which guarantees the curve is monotone in K by
    construction (a K=20 slice is a strict superset of a K=10 slice of the
    same stream).

    Returns an empty-``annotated_ik`` slot as 0.0 (rather than raising) to
    keep the curve total-function over any ``ks`` grid.
    """
    curve: Dict[int, float] = {}
    annotated = set(annotated_ik)
    denom = len(annotated)
    for k in ks:
        assert len(smiles_ranked) >= k, (
            f"union_at_k_curve: candidate stream under-produces at k={k} "
            f"(got {len(smiles_ranked)} candidates, requested k={k}) -- the "
            "method emitted fewer than K distinct-ranked candidates for this K"
        )
        stream_ik = {_tautomer_inchikey(s) for s in smiles_ranked[:k]}
        curve[k] = len(stream_ik & annotated) / denom if denom else 0.0
    return curve


def auc_of_curve(curve: Dict[int, float], k_min: int, k_max: int) -> float:
    """Trapezoidal AUC of a metric-vs-K curve, normalized by ``(k_max - k_min)``.

    ``curve`` maps K -> metric value (as produced by ``union_at_k_curve``,
    possibly over a non-uniform K-grid, e.g. ``(5, 10, 15, 20, 30, 50)``).
    Integrates trapezoidally over the SORTED ``(k, value)`` points between
    ``k_min`` and ``k_max`` inclusive, then divides by ``(k_max - k_min)`` so
    the result stays in the metric's own units (recall-like), not raw area.
    Uses plain-Python arithmetic (no numpy/scipy), keeping the module
    dependency-light per its RDKit-only header.
    """
    points = sorted((k, v) for k, v in curve.items() if k_min <= k <= k_max)
    area = 0.0
    for (k_a, v_a), (k_b, v_b) in zip(points, points[1:]):
        area += 0.5 * (v_a + v_b) * (k_b - k_a)
    span = k_max - k_min
    return area / span if span else 0.0


def compute_ablation_verdict(
    gflownet_auc: float, abl01_auc: float, abl02_auc: float, margin: float
) -> Literal["confirmed", "null", "partial"]:
    """Pre-registered ABL-03 verdict rule over three validation-selected union@K AUCs.

    ``gflownet_auc`` is the Set-GFlowNet's own union@K AUC; ``abl01_auc`` is the
    independent single-terminal baseline's; ``abl02_auc`` is the ensemble
    single-terminal baseline's. All three must be computed by the SAME harness
    (``auc_of_curve`` over ``union_at_k_curve``), on VAL, at the same K-grid and total
    output budget -- this function is pure post-hoc arithmetic over those numbers, no
    I/O, no globals.

    ``margin`` (the pre-specified threshold Delta) is supplied by the caller; its
    numeric value/convention is a Wave 3 decision, not baked in here. A difference
    exactly equal to ``margin`` does NOT count as a beat (strict ``>`` comparison).

    Three outcomes, all reportable:
    - "confirmed": the set reward beats BOTH ablations by more than margin -- the
      set-level reward is confirmed to drive the coverage/diversity gain.
    - "partial": the set reward beats the independent single-terminal baseline by
      more than margin but does NOT beat the ensemble baseline by more than margin --
      the honest framing is a compute-cost tradeoff (set-reward training vs paying
      for M independently-trained policies).
    - "null": neither of the above -- independent sampling matches or nearly matches
      the set reward, i.e. the set reward does not clear even the weaker (single)
      baseline by the pre-specified margin.
    """
    beats_single = gflownet_auc - abl01_auc > margin
    beats_ensemble = gflownet_auc - abl02_auc > margin
    if beats_single and beats_ensemble:
        return "confirmed"
    if beats_single and not beats_ensemble:
        return "partial"
    return "null"


def modes_discovered_canonical(
    sampled_smiles: Sequence[str],
    reward_fn: Callable[[str], float],
    tau: float,
    delta: float = 0.7,
) -> int:
    """GFlowNet-literature-canonical "modes discovered": reward-gated + Tanimoto-excluded.

    A candidate counts as a new mode iff (a) ``reward_fn(x) >= tau`` (the
    reward-threshold gate) AND (b) it is NOT within Tanimoto similarity
    ``delta`` of any other surviving candidate already counted (the sphere-
    exclusion gate) -- both gates are applied, matching the RGFN/QGFN-lineage
    convention. Reuses the SAME ``_greedy_sphere_exclusion`` machinery
    ``circles_count`` uses (no duplicated LeaderPicker plumbing), called at
    distance ``1.0 - delta`` over the surviving candidates' fingerprints.

    Default ``reward_fn``/``tau`` per D-EVAL05-REWARDFN is the binary
    "is annotated true metabolite" gate (``reward_fn(x) = 1 if annotated else 0``,
    ``tau = 1``) -- this makes ``modes_discovered_canonical`` differ from
    ``annotated_coverage_count`` ONLY by the added Tanimoto-exclusion gate. A
    continuous generator-score-based reward proxy is a Phase 4/5 enhancement,
    not implemented here. Like ``annotated_coverage_count``, this is gated on
    an incomplete PU annotation set when a ground-truth-based ``reward_fn`` is
    used -- the same precision-as-lower-bound caveat applies.

    FIX C (EVAL-04, adversarial review): runs the SAME ``_dedup_smiles_by_tautomer_ik``
    pre-pass on the reward-gate survivors that ``circles_count``/``mean_pairwise_tanimoto``/
    ``n_unique_scaffolds`` all run before fingerprinting -- without it, two tautomers of the
    same molecule would each independently pass the reward gate and count as two distinct
    "modes" instead of collapsing to one, forking this function's canonicalization away from
    the rest of the module.
    """
    survivors = [x for x in sampled_smiles if reward_fn(x) >= tau]
    survivors = _dedup_smiles_by_tautomer_ik(survivors)
    mols = [Chem.MolFromSmiles(s) for s in survivors]
    mols = [m for m in mols if m is not None]

    if len(mols) < 2:
        return len(mols)

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    return _greedy_sphere_exclusion(fps, 1.0 - delta)


def _dedup_smiles_by_tautomer_ik(smiles_list: Sequence[str]) -> List[str]:
    """First-seen-wins dedup of ``smiles_list`` keyed by ``_tautomer_inchikey``.

    Every public function in this module runs its input through this pre-pass
    before fingerprinting/scaffolding, so diversity metrics collapse tautomer
    duplicates using the SAME structure-identity path recall matching uses
    (``grail_metabolism.metrics._tautomer_inchikey``) -- defense-in-depth so the
    module is self-consistent even when a caller (e.g. a future baseline) hands
    it un-deduped output, rather than depending on every upstream caller to
    have already deduped.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for s in smiles_list:
        key = _tautomer_inchikey(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _greedy_sphere_exclusion(fps: Sequence, distance_threshold: float) -> int:
    """#Circles-style greedy sphere-exclusion count via RDKit's LeaderPicker.

    ``distance_threshold`` is a Tanimoto DISTANCE (i.e. ``1 - similarity``),
    matching ``LazyBitVectorPick``'s own convention. Returns 0 without calling
    into RDKit when ``fps`` is empty (guards the degenerate-input case).
    """
    if len(fps) == 0:
        return 0
    picker = rdSimDivPickers.LeaderPicker()
    picks = picker.LazyBitVectorPick(list(fps), len(fps), distance_threshold, pickSize=0)
    return len(picks)


def circles_count(smiles_list: Sequence[str], threshold: float) -> int:
    """#Circles: greedy sphere-exclusion diversity count (RDKit LeaderPicker).

    ``threshold`` is a TANIMOTO SIMILARITY radius (e.g. t=0.4 for tight/near-
    duplicate exclusion, t=0.7 for broad scaffold-level exclusion) -- it is a
    REQUIRED, explicit parameter (no silently baked-in default), and is
    internally converted to LeaderPicker's DISTANCE convention as
    ``1.0 - threshold`` before calling ``LazyBitVectorPick``.

    Runs the tautomer-InChIKey dedup pre-pass first (so a tautomer pair counts
    as ONE molecule, matching recall matching's identity function), then
    parses-and-skips unparseable SMILES (mirroring ``mean_pairwise_tanimoto``),
    then builds Morgan/ECFP4 (radius=2, nBits=2048) fingerprints -- the same
    fingerprint convention used everywhere else in this module; no second
    fingerprint convention is introduced.

    For fewer than 2 parseable molecules, returns ``len(mols)`` (0 or 1)
    rather than ``mean_pairwise_tanimoto``'s ``1.0`` convention, matching
    ``LazyBitVectorPick``'s own degenerate behavior on a 0/1-length pool.

    Note: fingerprints are computed on the raw (non-standardized) RDKit
    ``Mol``, so -- per the shared ``standardize_mol``/``isomericSmiles=False``
    convention used project-wide for canonicalization -- stereoisomers are not
    distinguished as diverse by this metric. This is a stated limitation
    inherited from the shared canonicalization, not a bug.
    """
    deduped = _dedup_smiles_by_tautomer_ik(smiles_list)
    mols = [Chem.MolFromSmiles(s) for s in deduped]
    mols = [m for m in mols if m is not None]

    if len(mols) < 2:
        return len(mols)

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    return _greedy_sphere_exclusion(fps, 1.0 - threshold)


def mean_pairwise_tanimoto(smiles_list: Sequence[str]) -> float:
    """Mean Tanimoto similarity (Morgan/ECFP4 fingerprints) over unordered pairs.

    Runs the tautomer-InChIKey dedup pre-pass first (EVAL-04: same identity
    function as recall matching), then parses-and-skips unparseable SMILES
    (``Chem.MolFromSmiles(s) is None``) rather than raising. With fewer than 2
    parseable molecules there are no pairs to average over; by convention
    (matching the "identical molecules -> 1.0" intuition for a degenerate/
    singleton set) this returns 1.0.
    """
    deduped = _dedup_smiles_by_tautomer_ik(smiles_list)
    mols = [Chem.MolFromSmiles(s) for s in deduped]
    mols = [m for m in mols if m is not None]

    if len(mols) < 2:
        return 1.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]

    sims: List[float] = []
    for fp_a, fp_b in combinations(fps, 2):
        sims.append(DataStructs.TanimotoSimilarity(fp_a, fp_b))

    return sum(sims) / len(sims)


def n_unique_scaffolds(smiles_list: Sequence[str]) -> int:
    """Number of distinct Bemis-Murcko scaffolds among the given SMILES.

    Runs the tautomer-InChIKey dedup pre-pass first (EVAL-04: same identity
    function as recall matching). Unparseable SMILES are skipped. An empty
    (or all-unparseable) input returns 0.
    """
    deduped = _dedup_smiles_by_tautomer_ik(smiles_list)
    scaffolds: Set[str] = set()
    for s in deduped:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffolds.add(scaffold)
    return len(scaffolds)


def set_size_calibration(sampled_sets: Iterable[frozenset], annotated_ik: Set[str]) -> float:
    """Mean sampled set size minus the true annotated-set size.

    Positive values mean the generator over-produces relative to the true
    number of annotated metabolites; negative values mean it under-produces.
    An empty ``sampled_sets`` iterable is treated as a mean size of 0.
    """
    sets = list(sampled_sets)
    mean_size = sum(len(s) for s in sets) / len(sets) if sets else 0.0
    return mean_size - len(annotated_ik)

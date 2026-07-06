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
from typing import Iterable, List, Sequence, Set

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers

from grail_metabolism.metrics import _tautomer_inchikey


def modes_discovered(sampled_sets: Iterable[frozenset], annotated_ik: Set[str]) -> int:
    """Count distinct annotated InChIKeys found across ALL sampled sets.

    Takes the union of every set in ``sampled_sets`` (each a ``frozenset[str]``
    of InChIKeys) and intersects it with ``annotated_ik``, the ground-truth
    annotated InChIKeys for the substrate. This is a coverage/"mode discovery"
    metric: how many of the true metabolites did the generator find in *any*
    of its sampled sets, not just its best/first one.
    """
    union: Set[str] = set()
    for s in sampled_sets:
        union.update(s)
    return len(union & set(annotated_ik))


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

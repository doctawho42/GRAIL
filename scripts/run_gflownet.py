#!/usr/bin/env python3
"""Stage 2b Set-GFlowNet: train + dual-eval matrix.

Trains a Set-GFlowNet forest sampler (``model.set_gflownet.SetGFlowNetTrainer``) whose
forward policy is a Stage-2a bi-encoder reranker (``BiEncoderReranker``), warm-started here
in-process (no reranker checkpoint is persisted to disk by ``run_reranker_gate.py``) and
optionally fine-tuned on a depth-2 bootstrap (``build_intermediate_pairs``). Evaluates on a
dual matrix at a matched output budget K = ``max_size``:

  - ``gflownet``: recall@K from the SINGLE highest-log-reward sampled forest (of
    ``--n-samples`` sampled forests), truncated to K, plus diversity metrics
    (``modes_discovered``, ``mean_pairwise_tanimoto``, ``n_unique_scaffolds``,
    ``set_size_calibration``) computed across ALL sampled forests;
  - ``reranker``: top-K of the same trained reranker scoring the root's candidate pool
    (Stage 2a baseline, no forest rollout);
  - ``beam``: ``model.multistep.MetabolicTree.beam_search`` truncated to K (the existing
    filter+generator multi-step baseline).

Recall counts a set member as a hit by tautomer-InChIKey against the substrate's annotated
metabolites (``metrics._tautomer_inchikey``), mirroring ``workflows/reranker.py``.

M0 gate: the external multi-gen (depth>=2 chain) recall claim is only pursued once the
depth-2 census (Task 1 / ``scripts/census_multistep.py``) has confirmed such chains exist
in the annotated data; the diversity metrics here are reported regardless, since they do
not depend on that finding.

Mirrors ``scripts/run_reranker_gate.py``'s structure (checkpoint loading, DatasetConfig,
pool caching, results JSON tagged by seed/split) so ``scripts/aggregate_seeds.py`` can read
the output unchanged.

NOTE: ``eval/diversity.py:modes_discovered_canonical`` (reward-gated + Tanimoto-exclusion-
gated GFlowNet-canonical mode count) is available but intentionally NOT emitted by
``evaluate_matrix`` here -- wiring it in requires a reward-gate methodology decision
(``reward_fn``/``tau``/``delta``) deferred to a future user decision; tracked as a follow-up.

Usage:
  python scripts/run_gflownet.py --train-substrates 300 --eval-split val \
      --max-depth 2 --max-size 15 --epochs 10 --n-samples 32
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from grail_metabolism.config import (
    DatasetConfig,
    FilterConfig,
    GeneratorConfig,
    GFlowNetConfig,
    MultiStepConfig,
)
# NOTE: `modes_discovered_canonical` is available in eval/diversity.py (reward-gated
# GFlowNet-canonical mode count) but is intentionally NOT imported/emitted here -- wiring it
# into evaluate_matrix's output requires a reward-gate methodology decision (reward_fn/tau/
# delta) deferred to a future user decision; tracked as a follow-up (FIX D, adversarial review).
from grail_metabolism.eval.diversity import (
    annotated_coverage_count,
    auc_of_curve,
    circles_count,
    dedup_to_budget,
    mean_pairwise_tanimoto,
    n_unique_scaffolds,
    set_size_calibration,
    union_at_k_curve,
)
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.model.grail import _read_checkpoint
from grail_metabolism.model.multistep import MetabolicTree
from grail_metabolism.model.reranker import BiEncoderReranker
from grail_metabolism.model.set_gflownet import (
    SetGFlowNetTrainer,
    SingleTerminalGFlowNetTrainer,
    set_coverage_logreward,
)
from grail_metabolism.utils.seed import seed_everything
from grail_metabolism.utils.transform import SINGLE_NODE_DIM
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.workflows.factory import build_filter, build_generator
from grail_metabolism.workflows.reranker import (
    BiRerankerTrainer,
    build_intermediate_pairs,
    build_pool,
    load_or_build_examples_bi,
)

GEN_CKPT = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
FILTER_CKPT = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "filter.pt"
CACHE_DIR = ROOT / "artifacts" / "reranker_gate_cache"
RESULTS_PATH = ROOT / "results" / "gflownet.json"


def _load_generator():
    state = _read_checkpoint(GEN_CKPT)
    if state is None or "arch" not in state or "rules" not in state:
        raise SystemExit(f"Generator checkpoint missing arch/rules: {GEN_CKPT}")
    generator = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    generator.load_state_dict(state["state_dict"], strict=False)
    if state.get("calibrated_threshold") is not None:
        generator.calibrated_threshold = state["calibrated_threshold"]
    generator.eval()
    return generator, state["rules"]


def _load_filter():
    """Load the trained filter for the ``beam`` baseline (``MetabolicTree`` needs one).

    Not persisted per-run like the reranker -- the filter IS checkpointed to disk (unlike
    the reranker), so this loads it exactly like ``_load_generator`` loads the generator.
    """
    state = _read_checkpoint(FILTER_CKPT)
    if state is None or "arch" not in state:
        print(f"[gflownet] WARNING: filter checkpoint missing/invalid ({FILTER_CKPT}) -- "
              "the beam baseline needs it, so it is SKIPPED. The gflownet and reranker "
              "baselines (the core comparison) still run.", flush=True)
        return None
    filt = build_filter(FilterConfig(**state["arch"]))
    filt.load_state_dict(state["state_dict"], strict=False)
    filt.calibrated_threshold = state.get("calibrated_threshold")
    filt.eval()
    return filt


def _make_annotated_ik_fn(molframe):
    def annotated_ik_fn(root: str):
        return {_tautomer_inchikey(p) for p in molframe.map.get(root, [])} - {None}

    return annotated_ik_fn


def _reranker_topk_smiles(reranker, generator, root: str, k: int, top_k: int, max_pool: int, device) -> List[str]:
    """Stage-2a baseline: score the root's candidate pool with the trained reranker,
    return the top-K smiles by predicted logit (matches ``workflows.reranker.evaluate_bi``'s
    ranking, minus the recall bookkeeping)."""
    from torch_geometric.data import Batch

    from grail_metabolism.utils.transform import from_rdmol

    pool = build_pool(generator, root, top_k=top_k, max_pool=max_pool)
    if not pool:
        return []
    sub_mol = Chem.MolFromSmiles(root)
    if sub_mol is None:
        return []
    sub_graph = from_rdmol(sub_mol)
    if sub_graph is None:
        return []
    prior = generator.rule_prior_logits.detach().cpu()
    num_rules = int(prior.numel())
    prod_graphs, rule_priors, gen_scores, smiles = [], [], [], []
    for cand_smiles, gen_score, rule_id in pool:
        cand_mol = Chem.MolFromSmiles(cand_smiles)
        if cand_mol is None:
            continue
        graph = from_rdmol(cand_mol)
        if graph is None:
            continue
        prod_graphs.append(graph)
        rid = int(rule_id) if 0 <= int(rule_id) < num_rules else 0
        rule_priors.append(float(prior[rid]) if num_rules else 0.0)
        gen_scores.append(float(gen_score))
        smiles.append(cand_smiles)
    if not prod_graphs:
        return []
    with torch.no_grad():
        prod_batch = Batch.from_data_list(prod_graphs).to(device)
        scores = reranker(
            sub_graph.to(device), prod_batch,
            torch.tensor(rule_priors, device=device), torch.tensor(gen_scores, device=device),
        ).detach().cpu()
    order = sorted(range(len(smiles)), key=lambda i: (-float(scores[i]), i))
    ranked = [smiles[i] for i in order]
    return ranked[:k]


def _recall_at_k(smiles_list: Sequence[str], annotated_ik: set) -> float:
    if not annotated_ik:
        return 0.0
    hit_iks = {_tautomer_inchikey(s) for s in smiles_list}
    return len(hit_iks & annotated_ik) / len(annotated_ik)


def _diversity_block(sampled_sets: List[frozenset], smiles_of: Dict[str, str], annotated_ik: set) -> Dict[str, float]:
    """Diversity metrics over ALL sampled forests for one substrate (Task 8's eval matrix
    reports these regardless of the M0 multi-gen recall gate)."""
    all_smiles = [smiles_of[ik] for s in sampled_sets for ik in s if ik in smiles_of]
    return {
        # NOTE (D-EVAL05-JSONKEY): the value now comes from the renamed
        # `annotated_coverage_count` (was `modes_discovered`); the results-dict KEY is kept
        # literally "modes_discovered" so `aggregate_seeds.py`'s DIVERSITY_KEYS contract stays
        # unbroken. The JSON-key rename is deferred to whichever later phase next regenerates
        # M2-style results.
        "modes_discovered": float(annotated_coverage_count(sampled_sets, annotated_ik)),
        "mean_pairwise_tanimoto": float(mean_pairwise_tanimoto(all_smiles)),
        "n_unique_scaffolds": float(n_unique_scaffolds(all_smiles)),
        "set_size_calibration": float(set_size_calibration(sampled_sets, annotated_ik)),
        # #Circles (D-EVAL03-CIRCLESKEYS): additive co-primary diversity keys, explicit
        # thresholds (t=0.4 headline/tight, t=0.7 broad), no magic default.
        "circles@t0.4": float(circles_count(all_smiles, threshold=0.4)),
        "circles@t0.7": float(circles_count(all_smiles, threshold=0.7)),
    }


def _substrate_set_fingerprint(substrates: Sequence[str]) -> str:
    """Stable sha256 over the sorted root SMILES of ``substrates`` -- a compact, order-
    independent identifier of WHICH substrates are being evaluated (not just how many),
    so a checkpoint written against one substrate set (e.g. a different split, or the
    same split subsampled differently) is never silently accepted against another."""
    return hashlib.sha256("\n".join(sorted(substrates)).encode("utf-8")).hexdigest()


def _eval_config_fingerprint(
    max_size: int, ks: Sequence[int], n_samples: int, top_k: int, max_pool: int,
    circles_thresholds: Sequence[float] = (0.4, 0.7),
    eval_split: str = "val", substrates: Sequence[str] = (), eval_beam: bool = True,
    ablation_mode: str = "off", m_ensemble: int = 0,
) -> str:
    """Stable hash of the eval-config fields that would make blended checkpoint rows WRONG
    if changed between the checkpoint's write and the current run (FIX 6 / D-EVAL06-SCHEMA):
    ``max_size``, the K-grid ``ks``, the #Circles thresholds, ``n_samples``, ``top_k``,
    ``max_pool``.

    Also covers (FIX B, adversarial review): ``eval_split`` (val vs test -- without this a
    ``--resume-eval-ckpt`` written on VAL could be silently adopted on TEST when the other
    params happen to match, since ``next_idx >= len(test_substrates)`` would skip the whole
    TEST loop and report VAL rows as TEST), a stable identifier of the evaluated-substrate
    SET (``_substrate_set_fingerprint(substrates)`` -- a sha256 of the sorted root SMILES,
    not just ``len(substrates)``, so two same-sized-but-different substrate samples are
    distinguished too), and ``eval_beam`` (whether the beam baseline is on -- a checkpoint's
    rows do or don't carry ``beam_recall``, so flipping ``--eval-beam``/``--no-eval-beam``
    must not silently blend rows computed under a different beam-baseline setting).

    Also covers (T-03-04, Plan 03-02): ``ablation_mode`` and ``m_ensemble`` -- a checkpoint
    written under one ablation mode (or a different ensemble size) must be discarded, not
    silently reused, under another (mirrors the FIX-B split guard one level up).

    A mismatch on resume means the checkpoint is stale and must be discarded (see
    ``_load_eval_ckpt``), never silently blended with rows computed under a different
    config."""
    payload = {
        "max_size": int(max_size),
        "ks": [int(k) for k in ks],
        "circles_thresholds": [float(t) for t in circles_thresholds],
        "n_samples": int(n_samples),
        "top_k": int(top_k),
        "max_pool": int(max_pool),
        "eval_split": str(eval_split),
        "substrate_set_fingerprint": _substrate_set_fingerprint(substrates),
        "n_substrates": len(substrates),
        "eval_beam": bool(eval_beam),
        "ablation_mode": str(ablation_mode),
        "m_ensemble": int(m_ensemble),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_eval_ckpt(path: Optional[str], config_fingerprint: str):
    """Load a per-substrate eval checkpoint, mirroring ``set_gflownet.py``'s
    ``_load_train_ckpt`` ignore-and-restart behavior one level down (JSON instead of
    ``torch.load``, per-substrate rows instead of per-epoch model state).

    Returns ``(completed_rows, next_idx)``. Starts fresh (``{}``, ``0``) when:
      - ``path`` is falsy or the file does not exist;
      - the file is corrupt/unreadable (caught, WARNING printed, never fatal);
      - the checkpoint's own ``config_fingerprint`` is absent or differs from
        ``config_fingerprint`` (FIX 6 stale-config-resume guard -- discarded exactly like a
        corrupt file, never blended).
    """
    if not path or not os.path.exists(path):
        return {}, 0
    try:
        with open(path) as fh:
            saved = json.load(fh)
        saved_fp = saved.get("config_fingerprint")
        if saved_fp != config_fingerprint:
            print(
                f"[gflownet] WARNING: eval checkpoint config fingerprint mismatch at {path} -- "
                "discarding stale rows, starting fresh",
                flush=True,
            )
            return {}, 0
        return dict(saved.get("rows", {})), int(saved.get("next_idx", 0))
    except Exception as exc:  # never let a bad checkpoint abort the run
        print(f"[gflownet] WARNING: ignoring unreadable eval checkpoint {path}: {exc}", flush=True)
        return {}, 0


def _save_eval_ckpt(path: str, config_fingerprint: str, rows: Dict[str, dict], next_idx: int) -> None:
    """Atomically persist the per-substrate eval checkpoint: write to a ``.tmp`` sibling then
    ``os.replace`` (the exact atomic pattern ``set_gflownet.py:_save_train_ckpt`` uses for
    training checkpoints, one level down -- JSON rows, not ``torch.save`` model state), so a
    kill mid-write never leaves a half-written file that would fail to load."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"config_fingerprint": config_fingerprint, "rows": rows, "next_idx": next_idx}, fh)
    os.replace(tmp, path)


def _round_robin_draw_counts(k_max: int, m_ensemble: int) -> List[int]:
    """Per-member draw-count allocation for ABL-02's round-robin ensemble (D-04).

    Returns a list of length ``m_ensemble`` where each entry is the number of
    single-terminal draws that member should contribute, summing to >= ``k_max``
    total raw draws, with every member within 1 of every other (the standard
    "as-even-as-possible" ``ceil(k_max/m_ensemble)`` split: the first
    ``k_max % m_ensemble`` members get one extra draw when the division isn't
    exact). Pure arithmetic, no side effects -- unit-testable in isolation
    (Task 3 guard test), and the single source of truth the ABL-02 stream
    builder below calls into (no separate/duplicated allocation logic).

    Handles ``m_ensemble > k_max`` (some members get 0 draws) and
    ``m_ensemble`` evenly dividing ``k_max`` (every member gets exactly the
    same count) without special-casing either.
    """
    if m_ensemble <= 0:
        return []
    base, extra = divmod(int(k_max), int(m_ensemble))
    return [base + 1 if i < extra else base for i in range(m_ensemble)]


def _draw_until_budget(draw_fn, k_max: int, cap: int) -> "tuple[List[str], int]":
    """Shared adaptive raw-draw discipline for ALL GFlowNet-family arms in ablation
    mode (D-03): repeatedly call ``draw_fn()`` (each call performs ONE raw draw --
    one sampled forest, or one single-terminal rollout -- and returns a list of
    produced SMILES, possibly empty) and accumulate the raw stream, until EITHER
    ``dedup_to_budget(raw_stream, k=k_max)`` yields ``k_max`` distinct entries, OR
    ``cap`` raw draws have been made (whichever first). Returns ``(raw_stream,
    n_raw_draws)`` -- the caller re-runs its own ``dedup_to_budget``/
    ``union_at_k_curve`` on ``raw_stream`` exactly like the existing fixed-loop
    streams did (no parallel truncation/coverage logic here, per D-05).

    This is the ONE cap definition every GFlowNet-family arm (gflownet /
    ablation01 / ablation02) shares in ablation mode -- a mode-collapsed policy
    that never reaches ``k_max`` distinct candidates hits ``cap`` and stops
    (bounded loop; the caller's under-production guard then skips the row),
    rather than looping unboundedly (T-03-03).
    """
    raw_stream: List[str] = []
    n_draws = 0
    while n_draws < cap:
        raw_stream.extend(draw_fn())
        n_draws += 1
        if len(dedup_to_budget(raw_stream, k=k_max)) >= k_max:
            break
    return raw_stream, n_draws


def _shared_substrate_survivors(
    arm_curve_maps: Dict[str, Dict[str, Dict[int, float]]],
) -> "set[str]":
    """Intersection of substrate keys for which EVERY GFlowNet-family arm in
    ``arm_curve_maps`` produced a curve (D-04b differential-missingness fix).

    ``arm_curve_maps`` maps arm name -> {substrate_root: curve_dict}. Returns the
    substrate-key intersection across all arms passed in (arms that did not run
    are simply absent from ``arm_curve_maps`` and do not constrain the
    intersection). An empty ``arm_curve_maps`` returns an empty set.
    """
    if not arm_curve_maps:
        return set()
    substrate_sets = [set(curves.keys()) for curves in arm_curve_maps.values()]
    return set.intersection(*substrate_sets)


def evaluate_matrix(
    trainer: SetGFlowNetTrainer,
    generator,
    reranker,
    beam_tree: "MetabolicTree",
    eval_bundle,
    n_eval: int,
    n_samples: int,
    max_size: int,
    top_k: int,
    max_pool: int,
    device,
    ks: Sequence[int] = (5, 10, 15, 20, 30, 50),
    resume_path: Optional[str] = None,
    eval_ckpt_every: int = 10,
    eval_split: str = "val",
    ablation_mode: str = "off",
    single_trainer: Optional["SingleTerminalGFlowNetTrainer"] = None,
    ensemble_trainers: Optional[List["SingleTerminalGFlowNetTrainer"]] = None,
    raw_draw_cap_mult: int = 10,
) -> Dict[str, float]:
    """Dual eval matrix at matched output budget K = max_size, PLUS a budget-matched
    union@K curve over ``ks`` (D-EVAL01-SEMANTICS / D-EVAL02-KGRID). See module docstring.

    Two DISTINCT aggregation semantics are reported, never conflated (D-EVAL01-SEMANTICS):
      - ``gflownet_recall@{max_size}`` / ``reranker_recall@{max_size}`` / ``beam_recall@
        {max_size}`` (unchanged, best-of-N-forest / single-ranked-list rows, preserved for
        historical M1/M2 comparability).
      - ``gflownet_union@{k}`` / ``reranker_union@{k}`` for every ``k`` in ``ks`` (the
        union-of-all-N-forests / full-reranked-pool operating point, matched to how
        Phase 2's baselines will emit their own full K-budget set), plus a
        ``*_union_at_k_auc`` scalar summary per series via ``auc_of_curve``.

    Both the gflownet-union stream and the reranker stream reaching ``dedup_to_budget`` are
    built/requested at ``k=max(ks)`` -- NEITHER is pre-truncated to ``max_size`` before that
    dedup pass (EVAL-01 budget-fairness fix) -- so ``union@30``/``union@50`` are computed
    over the FULL available pool whenever it supports that many candidates, not silently
    capped by the legacy (smaller) ``max_size`` budget.

    EVAL-06 resumability: when ``resume_path`` is set, per-substrate result rows are
    accumulated in ``completed_rows`` (keyed by root SMILES) and persisted atomically
    (``.tmp`` + ``os.replace``, mirroring ``set_gflownet.py:_save_train_ckpt``) every
    ``eval_ckpt_every`` substrates and once more after the loop. The checkpoint embeds a
    ``config_fingerprint`` over the fields that would make blended rows wrong if changed
    (``max_size``, ``ks``, the #Circles thresholds, ``n_samples``, ``top_k``, ``max_pool``,
    plus ``eval_split``, the evaluated-substrate-set fingerprint, and ``eval_beam`` per FIX B);
    a fingerprint mismatch on load discards the checkpoint (WARNING, start fresh) rather than
    blending old-config and new-config rows (FIX 6). Final aggregation runs over
    ``completed_rows.values()`` so a resumed run's metrics are identical to an uninterrupted
    run's, independent of the order rows were (re)computed in. This is a SEPARATE file with a
    separate lifecycle from the environment-cache checkpoint (``trainer.save_caches()``) --
    the two are never conflated.

    Phase 3 ablation (ABL-01/ABL-02, D-03/D-04/D-04b/D-05): when ``ablation_mode`` is
    ``"single"`` or ``"ensemble"``, this ALSO builds budget-matched ``ablation01_union@{k}``
    / ``ablation02_union@{k}`` streams (independent single-terminal / round-robin ensemble
    of single-terminal policies), reusing the SAME ``dedup_to_budget`` / ``union_at_k_curve``
    / ``auc_of_curve`` primitives as the gflownet/reranker streams -- no parallel truncation
    or coverage implementation. Critically, in this mode the gflownet arm's OWN raw-draw
    loop is ALSO switched from the fixed ``for _ in range(n_samples)`` to the SAME adaptive
    "draw until dedup hits k_max OR the shared ``raw_draw_cap_mult*k_max`` cap" discipline
    the ablation arms use (D-03), so the reference arm is never budget-starved relative to
    the baselines. When ``ablation_mode == "off"`` (the default), the gflownet loop is the
    ORIGINAL fixed ``for _ in range(n_samples)`` loop, byte-unchanged, so no Phase 1/2
    headline number moves. The GFlowNet-family arms' (gflownet/ablation01/ablation02)
    ``*_union_at_k_auc`` are means over the INTERSECTED shared-surviving-substrate set
    (D-04b) -- not each arm's own private survivor population -- with per-arm skip counts
    (``{arm}_n_skipped``) surfaced in the returned metrics dict.
    """
    substrates = list(eval_bundle.map.keys())[:n_eval]
    annotated_ik_fn = _make_annotated_ik_fn(eval_bundle)
    k_max = max(ks)

    m_ensemble = len(ensemble_trainers) if ensemble_trainers else 0
    config_fingerprint = _eval_config_fingerprint(
        max_size, ks, n_samples, top_k, max_pool,
        eval_split=eval_split, substrates=substrates, eval_beam=(beam_tree is not None),
        ablation_mode=ablation_mode, m_ensemble=m_ensemble,
    )
    completed_rows, next_idx = _load_eval_ckpt(resume_path, config_fingerprint)
    raw_draw_cap = int(raw_draw_cap_mult) * k_max

    # ik->SMILES map, built ONCE and grown INCREMENTALLY across substrates (each child-cache
    # parent folded in exactly once) using the WARM trainer._ik_cache. The prior code rebuilt
    # this per-substrate by scanning the ENTIRE ~1M-entry child cache and calling the COLD
    # module-level _tautomer_inchikey on every child -- re-tautomer-canonicalizing the whole
    # cache once PER substrate. That O(cache x n_subs) cold-canon flood was the eval's
    # multi-hour bottleneck; this makes it O(cache) total with warm lookups. Deterministic:
    # same ik->smiles content, just not recomputed.
    global_smiles_of: Dict[str, str] = {}
    _folded_parents: set = set()

    # Every GFlowNet-family trainer whose child-cache may need folding into
    # global_smiles_of this eval run -- the reference gflownet trainer ALWAYS, plus
    # any ablation trainer(s) (they each hold their OWN in-memory _child_cache/_ik_cache,
    # loaded from the SAME shared cache files at construction but mutated independently
    # during their own rollouts -- so a child a single-terminal/ensemble trainer
    # discovers must be folded in too, or its SMILES would be missing from smiles_of).
    _cache_trainers = [trainer]
    if single_trainer is not None:
        _cache_trainers.append(single_trainer)
    if ensemble_trainers:
        _cache_trainers.extend(ensemble_trainers)

    def _ik_warm(s: str):
        ik = trainer._ik_cache.get(s)
        if ik is None:
            ik = _tautomer_inchikey(s)
            if ik is not None:
                trainer._ik_cache[s] = ik
        return ik

    def _fold_new_cache_entries():
        # Fold any child-cache parents not seen yet (incl. states this substrate's forests just
        # lazily expanded, across ALL GFlowNet-family trainers) into the global ik->smiles map,
        # exactly once each (per (trainer identity, parent smiles) pair).
        for cache_trainer in _cache_trainers:
            for p_smiles in list(cache_trainer._child_cache.keys()):
                key = (id(cache_trainer), p_smiles)
                if key in _folded_parents:
                    continue
                _folded_parents.add(key)
                for c_smiles, _g, _rid in cache_trainer._child_cache[p_smiles]:
                    ik = _ik_warm(c_smiles)
                    if ik is not None:
                        global_smiles_of.setdefault(ik, c_smiles)

    for i, root in enumerate(substrates):
        if i < next_idx:
            continue  # already completed in a prior (checkpointed) run -- resume index

        annotated_ik = set(annotated_ik_fn(root))
        if not annotated_ik:
            continue
        root_mol = Chem.MolFromSmiles(root)
        if root_mol is None:
            continue

        # Sample forests; keep the SMILES for every produced InChIKey (needed for
        # diversity/tanimoto/scaffold metrics and for materializing the gflownet@K set).
        #
        # FIX A / D-03: in ablation mode ALL GFlowNet-family arms share ONE adaptive
        # raw-draw discipline (draw until dedup_to_budget hits k_max distinct OR the
        # SHARED raw_draw_cap_mult*k_max cap is hit) -- the gflownet arm's fixed
        # `for _ in range(n_samples)` loop (<= n_samples raw draws, no retry/cap) is
        # REPLACED for ablation mode ONLY, so the reference arm is not budget-starved
        # vs the ablation baselines. `ablation_mode == "off"` keeps the ORIGINAL fixed
        # loop byte-unchanged so the Phase 1/2 headline path never moves.
        sampled_sets: List[frozenset] = []
        best_log_r, best_state = None, None
        gflownet_n_raw_draws = n_samples
        with torch.no_grad():
            trainer.reranker.eval()
            if ablation_mode == "off":
                for _ in range(n_samples):
                    state, _sum_log_pf, _post_add = trainer.sample_forest(root)
                    terminal = state.terminal_set()
                    sampled_sets.append(terminal)
                    log_r = set_coverage_logreward(
                        terminal, annotated_ik, trainer.config.beta, getattr(trainer.config, "lam", 0.1)
                    )
                    if best_log_r is None or log_r > best_log_r:
                        best_log_r, best_state = log_r, state
            else:
                # Adaptive draw-to-k_max-or-shared-cap loop (D-03). The stopping check
                # uses the COUNT of distinct terminal-set InChIKeys seen so far as a
                # cheap, exact proxy for "distinct dedup_to_budget-eligible candidates":
                # ``dedup_to_budget``'s match key for the default "inchikey_tautomer"
                # protocol IS the tautomer InChIKey (the same identity space
                # ``ForestState.terminal_set()`` already uses), so the eventual deduped-
                # SMILES stream can have AT MOST this many distinct entries -- reaching
                # k_max distinct InChIKeys here is both necessary and sufficient for the
                # downstream dedup_to_budget(..., k=k_max) call to also reach k_max.
                n_draws = 0
                union_iks: set = set()
                while n_draws < raw_draw_cap:
                    state, _sum_log_pf, _post_add = trainer.sample_forest(root)
                    terminal = state.terminal_set()
                    sampled_sets.append(terminal)
                    union_iks.update(terminal)
                    log_r = set_coverage_logreward(
                        terminal, annotated_ik, trainer.config.beta, getattr(trainer.config, "lam", 0.1)
                    )
                    if best_log_r is None or log_r > best_log_r:
                        best_log_r, best_state = log_r, state
                    n_draws += 1
                    if len(union_iks) >= k_max:
                        break
                gflownet_n_raw_draws = n_draws

        # Reconstruct smiles for every InChIKey touched by any sampled forest via the
        # trainer's candidate cache (populated during sample_forest for root + all
        # frontier nodes visited); fall back to the root's own candidate pool.
        _fold_new_cache_entries()          # fold in states this substrate's forests just expanded
        smiles_of = global_smiles_of       # shared, incrementally-grown map (warm-cache lookups)
        rik = _ik_warm(root)
        if rik is not None:
            smiles_of.setdefault(rik, root)

        # gflownet_recall@{max_size}: unchanged best-of-N-forest row (D-EVAL01-SEMANTICS
        # "if you can only ship one forest" operating point) -- preserved for historical
        # M1/M2 comparability, dedup-truncated to max_size only (not the K-grid).
        best_terminal = best_state.terminal_set() if best_state is not None else frozenset()
        best_smiles_raw = [smiles_of[ik] for ik in best_terminal if ik in smiles_of]
        best_smiles = dedup_to_budget(best_smiles_raw, k=max_size)
        gflownet_recall_val = _recall_at_k(best_smiles, annotated_ik)

        # reranker_recall@{max_size}: legacy single-ranked-list row. FIX (budget-fairness,
        # EVAL-01): request the reranker stream at k=max(ks), NOT max_size -- top_k/max_pool
        # are the pool-construction knobs (distinct from this output-budget k), and must be
        # large enough for the reranked pool to actually supply k_max candidates. Slice
        # locally to max_size for this legacy row so its semantics are unchanged; the FULL
        # (up to k_max) stream is what reaches dedup_to_budget/union_at_k_curve below.
        reranker_stream_raw = _reranker_topk_smiles(
            reranker, generator, root, k=max(ks), top_k=top_k, max_pool=max_pool, device=device
        )
        reranker_smiles = dedup_to_budget(reranker_stream_raw, k=max_size)
        reranker_recall_val = _recall_at_k(reranker_smiles, annotated_ik)

        beam_recall_val = None
        if beam_tree is not None:  # beam baseline only when a filter checkpoint was available
            beam_out = beam_tree.beam_search(root, max_output=max_size)
            beam_smiles = [s for s, _score in beam_out]
            beam_recall_val = _recall_at_k(beam_smiles, annotated_ik)

        # gflownet_union@{k} / reranker_union@{k}: budget-matched union@K curve
        # (D-EVAL01-SEMANTICS union-of-N-forests / full-reranked-pool operating point).
        # Each stream is deduped ONCE at k=k_max (NOT re-truncated per k), then
        # union_at_k_curve slices that single deduped stream at every k in ks -- monotone
        # by construction, no re-inlined slice+recall loop (FIX 5).
        gflownet_union_stream_raw = [smiles_of[ik] for s in sampled_sets for ik in s if ik in smiles_of]
        gflownet_union_stream = dedup_to_budget(gflownet_union_stream_raw, k=k_max)
        gflownet_union_curve = None
        if len(gflownet_union_stream) >= k_max:
            gflownet_union_curve = union_at_k_curve(gflownet_union_stream, annotated_ik, ks=ks)
        else:
            print(
                f"[gflownet] WARNING: root={root!r} gflownet union stream under-produces at "
                f"k_max={k_max} (got {len(gflownet_union_stream)} distinct candidates) -- "
                "skipping this substrate's gflownet_union@K row (EVAL-02 under-production guard)",
                flush=True,
            )

        reranker_union_stream = dedup_to_budget(reranker_stream_raw, k=k_max)
        reranker_union_curve = None
        if len(reranker_union_stream) >= k_max:
            reranker_union_curve = union_at_k_curve(reranker_union_stream, annotated_ik, ks=ks)
        else:
            print(
                f"[gflownet] WARNING: root={root!r} reranker union stream under-produces at "
                f"k_max={k_max} (got {len(reranker_union_stream)} distinct candidates) -- "
                "skipping this substrate's reranker_union@K row (EVAL-02 under-production guard)",
                flush=True,
            )

        # ablation01_union@{k} / ablation02_union@{k}: budget-matched union@K streams for
        # the independent single-terminal (ABL-01) and ensemble single-terminal (ABL-02)
        # baselines (D-03, D-04). Both call the SAME dedup_to_budget/union_at_k_curve as
        # every other arm (D-05) -- no parallel truncation/coverage implementation -- and
        # share the SAME raw_draw_cap as the gflownet arm above (FIX A / D-03).
        ablation01_union_curve = None
        ablation01_n_raw_draws = None
        if ablation_mode == "single" and single_trainer is not None:
            def _single_draw_fn():
                with torch.no_grad():
                    single_trainer.reranker.eval()
                    state, _sum_log_pf, _post_add = single_trainer.sample_forest(root)
                terminal = state.terminal_set()
                _fold_new_cache_entries()  # fold this draw's newly-expanded states first
                return [smiles_of[ik] for ik in terminal if ik in smiles_of]

            ablation01_raw, ablation01_n_raw_draws = _draw_until_budget(
                _single_draw_fn, k_max=k_max, cap=raw_draw_cap
            )
            ablation01_union_stream = dedup_to_budget(ablation01_raw, k=k_max)
            if len(ablation01_union_stream) >= k_max:
                ablation01_union_curve = union_at_k_curve(ablation01_union_stream, annotated_ik, ks=ks)
            else:
                print(
                    f"[gflownet] WARNING: root={root!r} ablation01 union stream under-produces at "
                    f"k_max={k_max} (got {len(ablation01_union_stream)} distinct candidates) -- "
                    "skipping this substrate's ablation01_union@K row (shared raw-draw cap hit)",
                    flush=True,
                )

        ablation02_union_curve = None
        ablation02_n_raw_draws = None
        if ablation_mode == "ensemble" and ensemble_trainers:
            # D-04's ceil(k_max/M_ensemble)-per-member allocation (_round_robin_draw_counts)
            # schedules the FIRST pass -- member 0 draws draw_counts[0] times, member 1
            # draws draw_counts[1] times, etc, summing to >= k_max total raw draws, each
            # member within 1 draw of every other. If the shared raw-draw cap allows further
            # draws after that first pass (e.g. some members under-produce and dedup hasn't
            # hit k_max distinct yet), the schedule CYCLES back through the SAME per-member
            # allocation again rather than stopping -- so a mode-collapsed member does not
            # stall the whole draw loop; it just contributes fewer distinct candidates per
            # pass while still getting drawn from at the SAME even cadence.
            draw_counts = _round_robin_draw_counts(k_max, len(ensemble_trainers))
            first_pass_schedule = [
                m for m, cnt in enumerate(draw_counts) for _ in range(cnt)
            ] or list(range(len(ensemble_trainers)))

            def _member_cycle():
                while True:
                    for m in first_pass_schedule:
                        yield m

            _member_iter = _member_cycle()

            def _ensemble_draw_fn():
                member_idx = next(_member_iter)
                trainer_i = ensemble_trainers[member_idx]
                with torch.no_grad():
                    trainer_i.reranker.eval()  # FIX F: restated for EVERY member
                    state, _sum_log_pf, _post_add = trainer_i.sample_forest(root)
                terminal = state.terminal_set()
                _fold_new_cache_entries()  # fold this draw's newly-expanded states first
                return [smiles_of[ik] for ik in terminal if ik in smiles_of]

            ablation02_raw, ablation02_n_raw_draws = _draw_until_budget(
                _ensemble_draw_fn, k_max=k_max, cap=raw_draw_cap
            )
            ablation02_union_stream = dedup_to_budget(ablation02_raw, k=k_max)
            if len(ablation02_union_stream) >= k_max:
                ablation02_union_curve = union_at_k_curve(ablation02_union_stream, annotated_ik, ks=ks)
            else:
                print(
                    f"[gflownet] WARNING: root={root!r} ablation02 union stream under-produces at "
                    f"k_max={k_max} (got {len(ablation02_union_stream)} distinct candidates) -- "
                    "skipping this substrate's ablation02_union@K row (shared raw-draw cap hit)",
                    flush=True,
                )

        row: Dict[str, object] = {
            "gflownet_recall": gflownet_recall_val,
            "reranker_recall": reranker_recall_val,
            "diversity": _diversity_block(sampled_sets, smiles_of, annotated_ik),
        }
        if beam_recall_val is not None:
            row["beam_recall"] = beam_recall_val
        if gflownet_union_curve is not None:
            row["gflownet_union_curve"] = gflownet_union_curve
        if reranker_union_curve is not None:
            row["reranker_union_curve"] = reranker_union_curve
        if ablation_mode != "off":
            row["gflownet_n_raw_draws"] = gflownet_n_raw_draws
        if ablation01_union_curve is not None:
            row["ablation01_union_curve"] = ablation01_union_curve
            row["ablation01_n_raw_draws"] = ablation01_n_raw_draws
        if ablation02_union_curve is not None:
            row["ablation02_union_curve"] = ablation02_union_curve
            row["ablation02_n_raw_draws"] = ablation02_n_raw_draws
        completed_rows[root] = row

        if resume_path and (i + 1) % eval_ckpt_every == 0:
            _save_eval_ckpt(resume_path, config_fingerprint, completed_rows, i + 1)

    if resume_path:
        _save_eval_ckpt(resume_path, config_fingerprint, completed_rows, len(substrates))

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    rows = list(completed_rows.values())
    n_evaluated = len(rows)
    gflownet_recall = [r["gflownet_recall"] for r in rows]
    reranker_recall = [r["reranker_recall"] for r in rows]
    beam_recall = [r["beam_recall"] for r in rows if "beam_recall" in r]
    diversity_rows = [r["diversity"] for r in rows]

    # Per-substrate curve maps (root -> {k: value}), keyed by substrate so the shared-
    # substrate-set restriction below (D-04b) can intersect them by key, not just zip by
    # position (rows may have skipped DIFFERENT substrates for different arms).
    gflownet_curves_by_root: Dict[str, Dict[int, float]] = {
        root: {int(k): v for k, v in row["gflownet_union_curve"].items()}
        for root, row in completed_rows.items() if "gflownet_union_curve" in row
    }
    reranker_curves_by_root: Dict[str, Dict[int, float]] = {
        root: {int(k): v for k, v in row["reranker_union_curve"].items()}
        for root, row in completed_rows.items() if "reranker_union_curve" in row
    }
    ablation01_curves_by_root: Dict[str, Dict[int, float]] = {
        root: {int(k): v for k, v in row["ablation01_union_curve"].items()}
        for root, row in completed_rows.items() if "ablation01_union_curve" in row
    }
    ablation02_curves_by_root: Dict[str, Dict[int, float]] = {
        root: {int(k): v for k, v in row["ablation02_union_curve"].items()}
        for root, row in completed_rows.items() if "ablation02_union_curve" in row
    }

    metrics: Dict[str, float] = {
        "n_substrates": float(n_evaluated),
        f"gflownet_recall@{max_size}": _mean(gflownet_recall),
        f"reranker_recall@{max_size}": _mean(reranker_recall),
    }
    if beam_recall:  # only present when the beam baseline ran (filter checkpoint available)
        metrics[f"beam_recall@{max_size}"] = _mean(beam_recall)
    for key in (
        "modes_discovered", "mean_pairwise_tanimoto", "n_unique_scaffolds", "set_size_calibration",
        "circles@t0.4", "circles@t0.7",
    ):
        metrics[key] = _mean([row[key] for row in diversity_rows])

    # FIX A (D-04b/D-05): the GFlowNet-family arms (gflownet/ablation01/ablation02, whichever
    # ran) must have their *_union@{k}/*_union_at_k_auc computed over the IDENTICAL
    # (intersected) surviving-substrate set -- NOT each arm's own private survivor population
    # -- so a differential-missingness confound (arms mode-collapsing on DIFFERENT
    # substrates) cannot silently invalidate the head-to-head verdict. The reranker series is
    # NOT part of the GFlowNet-family comparison (not part of the ABL-03 verdict) and keeps
    # its existing full-survivor mean, unrestricted.
    gflownet_family_curves: Dict[str, Dict[str, Dict[int, float]]] = {"gflownet": gflownet_curves_by_root}
    if ablation01_curves_by_root:
        gflownet_family_curves["ablation01"] = ablation01_curves_by_root
    if ablation02_curves_by_root:
        gflownet_family_curves["ablation02"] = ablation02_curves_by_root

    if len(gflownet_family_curves) > 1:
        # Multiple GFlowNet-family arms ran (ablation mode) -- restrict ALL of them to the
        # shared surviving-substrate intersection before averaging.
        shared_substrates = _shared_substrate_survivors(gflownet_family_curves)
    else:
        # Only the plain gflownet arm ran (ablation_mode == "off") -- no restriction needed;
        # this is the byte-unchanged Phase 1/2 headline path's own full-survivor set.
        shared_substrates = set(gflownet_curves_by_root.keys())

    for series_name, curves_by_root in gflownet_family_curves.items():
        restricted = {
            root: curve for root, curve in curves_by_root.items() if root in shared_substrates
        }
        curves = list(restricted.values())
        for k in ks:
            metrics[f"{series_name}_union@{k}"] = _mean([c[k] for c in curves if k in c])
        auc_values = [auc_of_curve(c, k_min=min(ks), k_max=k_max) for c in curves]
        metrics[f"{series_name}_union_at_k_auc"] = _mean(auc_values)
        # Per-arm skip count: substrates where this arm produced a curve but was excluded
        # from the shared intersection (under-produced for a DIFFERENT arm), surfaced
        # rather than hidden, per D-04b.
        metrics[f"{series_name}_n_skipped"] = float(len(curves_by_root) - len(restricted))

    # reranker_union@{k} (D-EVAL01-SEMANTICS): mean-over-substrates of the reranker's own
    # per-substrate union@K curve over its OWN full survivor set (not restricted to the
    # GFlowNet-family intersection -- it is not part of the ablation verdict).
    reranker_curves = list(reranker_curves_by_root.values())
    for k in ks:
        metrics[f"reranker_union@{k}"] = _mean([c[k] for c in reranker_curves if k in c])
    reranker_auc_values = [auc_of_curve(c, k_min=min(ks), k_max=k_max) for c in reranker_curves]
    metrics["reranker_union_at_k_auc"] = _mean(reranker_auc_values)

    # Single unqualified `union_at_k_auc` summary (mean over gflownet + reranker) for a simple
    # top-line aggregate_seeds.py DIVERSITY_KEYS entry, alongside the per-series variants.
    # Deliberately excludes ablation01/ablation02 (an ablation-only concept) so this combined
    # scalar's meaning is unchanged from the Phase 1/2 headline path.
    metrics["union_at_k_auc"] = _mean(
        [metrics[f"{s}_union_at_k_auc"] for s in ("gflownet", "reranker") if f"{s}_union_at_k_auc" in metrics]
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2b Set-GFlowNet: train + dual-eval matrix")
    parser.add_argument("--train-substrates", type=int, default=300)
    parser.add_argument(
        "--test-substrates", type=int, default=2000,
        help="Test substrates for --eval-split test. Default 2000 exceeds the full clean "
             "test split, so the touch-once eval uses the ENTIRE test set (no subsampling).",
    )
    parser.add_argument(
        "--eval-split", choices=["val", "test"], default="val",
        help="Which split to evaluate on. 'val' for selection; 'test' ONCE for the final report.",
    )
    parser.add_argument(
        "--no-eval-beam", dest="eval_beam", action="store_false",
        help="Skip the optional filter multistep-beam baseline in eval (a per-substrate depth-D "
             "beam_search). It is NOT part of the core gflownet-vs-reranker comparison; dropping "
             "it keeps the (uncheckpointed) eval short enough to finish inside a preemptible "
             "worker's window.",
    )
    parser.set_defaults(eval_beam=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=6.0, help="Reward sharpness in set_coverage_logreward.")
    parser.add_argument("--lam", type=float, default=0.1, help="Size penalty in set_coverage_logreward.")
    parser.add_argument("--max-depth", type=int, default=2, help="Forest rollout depth cap.")
    parser.add_argument("--max-size", type=int, default=15, help="Forest size cap == matched output budget K.")
    parser.add_argument("--top-k", type=int, default=200, help="Generator candidates enumerated per frontier node.")
    parser.add_argument("--epochs", type=int, default=10, help="Set-GFlowNet TB training epochs.")
    parser.add_argument(
        "--logz-lr", type=float, default=1e-2,
        help="Learning rate for the scalar logZ. logZ's target scales with beta (~O(beta*max_TP)), "
             "so with the default it moves only ~(steps/epoch)*logz_lr per epoch and needs many "
             "epochs to converge. Raise it (e.g. 0.3) for a decisive few-epoch sanity run.",
    )
    parser.add_argument("--n-samples", type=int, default=32, help="Forests sampled per eval substrate (M).")
    parser.add_argument(
        "--eval-substrates", type=int, default=100,
        help="Cap on VAL substrates evaluated (eval is expensive: n_samples rollouts each, on a "
             "COLD cache -- val states weren't warmed by training). Ignored for --eval-split test "
             "(that uses --test-substrates).",
    )
    parser.add_argument(
        "--bootstrap", action="store_true", default=True,
        help="Fine-tune the reranker on the depth-2 bootstrap (build_intermediate_pairs) "
             "after the depth-1 InfoNCE fit.",
    )
    parser.add_argument("--no-bootstrap", dest="bootstrap", action="store_false")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel pool-generation workers for the reranker pool cache (spawn Pool). 1 = serial.",
    )
    parser.add_argument(
        "--prewarm-waves", type=int, default=2, choices=(1, 2),
        help="Prewarm depth: 1 = expand roots only (fit/eval lazily expand the visited depth-1 "
             "subset -- avoids depth-1 over-expansion at scale); 2 = also expand ALL depth-1 up "
             "front (fully parallel but over-expands unvisited children).",
    )
    parser.add_argument("--out", type=str, default=None, help="Override the results JSON path.")
    parser.add_argument(
        "--resume-ckpt", type=str, default=None,
        help="Path to a Set-GFlowNet training checkpoint (saved every epoch). If it exists, "
             "training RESUMES from the last completed epoch instead of restarting at 0 -- "
             "makes a multi-hour run survive preemption/crash on preemptible workers.",
    )
    parser.add_argument(
        "--resume-eval-ckpt", type=str, default=None,
        help="Path to a per-substrate EVAL checkpoint (saved every --eval-ckpt-every "
             "substrates inside evaluate_matrix). If it exists (and its embedded config "
             "fingerprint matches this run's config), eval RESUMES mid-loop from the last "
             "completed substrate instead of restarting at 0 -- mirrors --resume-ckpt's UX "
             "but for the eval loop, not training. A corrupt or config-mismatched checkpoint "
             "is ignored (WARNING, start fresh), never fatal. Distinct from the environment-"
             "cache checkpoint (save_caches) -- a separate file, separate lifecycle.",
    )
    parser.add_argument(
        "--eval-ckpt-every", type=int, default=10,
        help="Persist the eval checkpoint every N substrates (D-EVAL06-INTERVAL). Each "
             "substrate does n_samples full forest rollouts + a reranker pool build, so this "
             "interval is tighter than the env-cache prewarm interval; tune against measured "
             "per-substrate wall-clock at scale.",
    )
    # Reranker warm-start knobs, mirroring run_reranker_gate.py's --arch bi defaults.
    parser.add_argument("--rerank-epochs", type=int, default=15, help="Depth-1 reranker InfoNCE epochs.")
    parser.add_argument("--bootstrap-epochs", type=int, default=5, help="Depth-2 bootstrap fine-tune epochs.")
    parser.add_argument(
        "--bootstrap-substrates", type=int, default=150,
        help="How many train roots build_intermediate_pairs SCANS for depth-2 pairs (the depth-2 "
             "expansion is expensive; a sample is enough to fine-tune). Not the reranker's train size.",
    )
    parser.add_argument("--max-pool", type=int, default=100)
    # Phase 3 (Set-Reward Novelty Ablation) knobs -- ABL-01/ABL-02/ABL-03. Default
    # ablation_mode="off" keeps every existing run byte-unchanged (D-03).
    parser.add_argument(
        "--ablation-mode", choices=["off", "single", "ensemble"], default="off",
        help="off (default, byte-unchanged headline path) | single (ABL-01 independent "
             "single-terminal baseline) | ensemble (ABL-02 round-robin ensemble of "
             "--m-ensemble single-terminal policies). See 03-02-PLAN.md D-03/D-04.",
    )
    parser.add_argument(
        "--beta-prime", type=float, default=None,
        help="Reward sharpness for single_hit_logreward (the single-terminal ablation "
             "reward), selected on VAL INDEPENDENTLY of --beta (D-07) -- the single-"
             "terminal reward's natural scale differs from the set reward's, so silently "
             "inheriting --beta risks an unfairly weak/strong baseline. Defaults to --beta "
             "so a run can opt into inheritance explicitly.",
    )
    parser.add_argument(
        "--m-ensemble", type=int, default=3,
        help="Ensemble size M for --ablation-mode ensemble (ABL-02): the number of "
             "independently-seeded SingleTerminalGFlowNetTrainer policies round-robin-"
             "drawn from. Distinct from k_max (the total output budget).",
    )
    parser.add_argument(
        "--raw-draw-cap-mult", type=int, default=10,
        help="Shared raw-draw cap multiplier for ALL GFlowNet-family arms in ablation "
             "mode (gflownet + ablation01 + ablation02): each arm draws more raw forests/"
             "single-terminal samples until dedup_to_budget reaches k_max distinct "
             "candidates OR raw_draw_cap_mult*k_max raw draws are made, whichever first "
             "(D-03). A mode-collapsed policy hits the cap and that substrate's row is "
             "skipped for that arm (WARNING), rather than looping unboundedly.",
    )
    parser.add_argument(
        "--verdict-margin", type=float, default=0.02,
        help="Strict-> margin Delta for compute_ablation_verdict's confirmed/partial/null "
             "call over the gflownet/ablation01/ablation02 union_at_k_auc values. Only "
             "used (printed, non-gating) when --ablation-mode != off and both ablation "
             "AUCs are present.",
    )
    args = parser.parse_args()
    if args.beta_prime is None:
        args.beta_prime = args.beta

    t_start = time.time()
    seed_everything(args.seed)
    print(
        f"[gflownet] seed={args.seed} beta={args.beta} lam={args.lam} max_depth={args.max_depth} "
        f"max_size={args.max_size} top_k={args.top_k} bootstrap={args.bootstrap}",
        flush=True,
    )

    print("[gflownet] loading trained generator ...", flush=True)
    t0 = time.time()
    generator, rules = _load_generator()
    print(f"[gflownet] generator loaded in {time.time()-t0:.1f}s; num_rules={generator.num_rules}", flush=True)

    if args.eval_beam:
        print("[gflownet] loading trained filter (beam baseline) ...", flush=True)
        t0 = time.time()
        filt = _load_filter()
        print(f"[gflownet] filter loaded in {time.time()-t0:.1f}s", flush=True)
    else:
        # --no-eval-beam: skip the OPTIONAL filter multistep-beam baseline. It's a per-substrate
        # depth-D beam_search (expensive) and is NOT part of the core gflownet-vs-reranker
        # comparison; dropping it keeps the eval short enough to finish inside a preemptible
        # worker's window (the eval is not checkpointed, so a >window eval never completes).
        print("[gflownet] --no-eval-beam: skipping the filter beam baseline (faster eval)", flush=True)
        filt = None

    eval_is_test = args.eval_split == "test"
    cfg = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf",
        train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf",
        val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf",
        test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True,
        standardize=False,
        cache_preprocessed=False,
        max_train_substrates=args.train_substrates + 60,
        max_val_substrates=(1 if eval_is_test else args.eval_substrates + 30),
        max_test_substrates=(args.test_substrates + 60 if eval_is_test else 1),
        sampling_seed=args.seed,
    )
    print("[gflownet] loading dataset bundle (SDF standardization is the slow load) ...", flush=True)
    t0 = time.time()
    bundle = load_dataset_bundle(cfg)
    eval_bundle = bundle.test if eval_is_test else bundle.val
    eval_count = args.test_substrates if eval_is_test else min(args.eval_substrates, len(eval_bundle.map))
    eval_prefix = "test" if eval_is_test else "val"
    print(
        f"[gflownet] bundle loaded in {time.time()-t0:.1f}s; "
        f"train={len(bundle.train.map)} {eval_prefix}={len(eval_bundle.map)}",
        flush=True,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_cache = CACHE_DIR / f"train_bi_s{args.train_substrates}_seed{args.seed}_k{args.top_k}.pt"

    print("[gflownet] assembling TRAIN pools for reranker warm-start (cached, bi-encoder path) ...", flush=True)
    t0 = time.time()
    train_examples = load_or_build_examples_bi(
        generator, bundle.train, args.train_substrates, train_cache,
        top_k=args.top_k, max_pool=args.max_pool,
        workers=args.workers, gen_ckpt=str(GEN_CKPT),
    )
    print(f"[gflownet] train examples={len(train_examples)} in {time.time()-t0:.1f}s", flush=True)

    print("[gflownet] warm-starting reranker (P_F init) via depth-1 listwise InfoNCE ...", flush=True)
    t0 = time.time()
    reranker = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    rr_trainer = BiRerankerTrainer(reranker, lr=1e-3, seed=args.seed)
    rr_trainer.fit(train_examples, epochs=args.rerank_epochs)
    # Snapshot the WARM-STARTED weights ONCE, before any GFlowNet TB training touches
    # `reranker`. Every GFlowNet-family trainer below (the set-GFlowNet reference AND each
    # single-terminal ablation trainer) gets its OWN fresh BiEncoderReranker loaded from this
    # snapshot -- never the same live `reranker` object -- so their TB `fit()` calls train
    # disjoint parameter tensors. Without this, `SetGFlowNetTrainer.__init__`'s
    # `self.reranker = reranker.to(self.device)` is a no-copy alias: sharing one `reranker`
    # object across trainers means later `fit()` calls silently overwrite the weights an
    # earlier trainer's eval policy depends on (breaking single-variable purity for the
    # reference arm) and would make ABL-02's "ensemble" members just sequential snapshots of
    # one continuous trajectory instead of independently trained policies.
    warm_state = copy.deepcopy(reranker.state_dict())
    print(f"[gflownet] reranker warm-start done in {time.time()-t0:.1f}s", flush=True)

    if args.bootstrap:
        print("[gflownet] building depth-2 bootstrap examples (build_intermediate_pairs) ...", flush=True)
        t0 = time.time()
        # NOTE: --bootstrap-substrates bounds ROOTS SCANNED, not examples produced (unlike the
        # depth-1 builders) -- the per-root depth-2 expansion is expensive, so a small sample of
        # roots is enough to fine-tune. See build_intermediate_pairs's docstring.
        bootstrap_examples = build_intermediate_pairs(
            generator, bundle.train, args.bootstrap_substrates, top_k=args.top_k, max_pool=args.max_pool,
        )
        print(
            f"[gflownet] bootstrap examples={len(bootstrap_examples)} "
            f"(scanned up to {args.bootstrap_substrates} roots) in {time.time()-t0:.1f}s",
            flush=True,
        )
        if bootstrap_examples:
            combined = train_examples + bootstrap_examples
            print(
                f"[gflownet] fine-tuning reranker on depth-1 + depth-2-bootstrap "
                f"({len(train_examples)} + {len(bootstrap_examples)} = {len(combined)} examples) ...",
                flush=True,
            )
            t0 = time.time()
            rr_trainer.fit(combined, epochs=args.bootstrap_epochs)
            print(f"[gflownet] bootstrap fine-tune done in {time.time()-t0:.1f}s", flush=True)
        else:
            print("[gflownet] no depth-2 bootstrap examples found; skipping fine-tune.", flush=True)

    print("[gflownet] training Set-GFlowNet (TB loss over forest rollouts) ...", flush=True)
    t0 = time.time()
    gfn_config = GFlowNetConfig(
        max_depth=args.max_depth,
        beta=args.beta,
        lam=args.lam,
        max_size=args.max_size,
        top_k=args.top_k,
        epochs=args.epochs,
        logz_lr=args.logz_lr,
    )
    train_substrates_list = list(bundle.train.map.keys())[: args.train_substrates]
    # Persistent, cross-run environment caches (exact -- RDKit rule application and tautomer
    # canonicalization are deterministic). Built once, reused by every later M1/M2/seed run,
    # so the slow "epoch 1 pool-gen" is paid only the FIRST time. child cache is (generator,
    # top_k)-specific (keyed by top_k in the filename); ik cache is universal.
    child_cache_path = CACHE_DIR / f"gfn_child_cache_k{args.top_k}.pkl"
    ik_cache_path = CACHE_DIR / "gfn_ik_cache.pkl"

    def _fresh_warm_reranker() -> BiEncoderReranker:
        """A fresh ``BiEncoderReranker`` loaded from the warm-start snapshot, sharing no
        parameter tensors with `reranker` or any other trainer's reranker (see the
        `warm_state` snapshot comment above -- this is what makes each GFlowNet-family
        trainer's TB training independent)."""
        r = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
        r.load_state_dict(copy.deepcopy(warm_state))
        return r

    trainer = SetGFlowNetTrainer(
        generator, _fresh_warm_reranker(), gfn_config, _make_annotated_ik_fn(bundle.train),
        device=rr_trainer.device,
        child_cache_path=str(child_cache_path), ik_cache_path=str(ik_cache_path),
    )
    if args.workers > 1:
        print(f"[gflownet] parallel pre-warming train caches ({args.workers} workers) ...", flush=True)
        t0 = time.time()
        trainer.prewarm_caches(train_substrates_list, args.workers, gen_ckpt=str(GEN_CKPT),
                               verbose=True, waves=args.prewarm_waves)
        print(f"[gflownet] train prewarm done in {time.time()-t0:.1f}s", flush=True)
    trainer.fit(train_substrates_list, epochs=args.epochs, verbose=True,
                resume_path=args.resume_ckpt)
    trainer.save_caches()  # persist env caches populated during training
    print(f"[gflownet] Set-GFlowNet training done in {time.time()-t0:.1f}s", flush=True)

    # Phase 3 ablation (ABL-01/ABL-02): single-terminal trainer(s), single-variable-ablation
    # contract (per D-02/D-03/D-07) -- ONLY max_size (1, not args.max_size) and the reward
    # (single_hit_logreward via SingleTerminalGFlowNetTrainer, beta=args.beta_prime NOT
    # args.beta) differ from the set-GFlowNet trained above. Same generator, same
    # annotated_ik_fn, same device, and the SAME child/ik env-cache files (the environment is
    # deterministic and seed-independent, so sharing the cache across the set-GFlowNet and the
    # ablation trainer(s) is exact, not an approximation). Each trainer (the set-GFlowNet
    # reference AND every ablation trainer) gets its OWN fresh reranker loaded from the SAME
    # `warm_state` snapshot (see snapshot comment above) -- identical warm-start (purity),
    # independent parameter tensors (no cross-arm corruption, genuine ensemble). The ensemble
    # members' per-seed training/checkpoint orchestration across multiple processes is Wave 3's
    # concern (modal_m2.py); here we construct/train in-process for eval given this run's own
    # train split.
    single_trainer = None
    ensemble_trainers = None
    if args.ablation_mode != "off":
        single_gfn_config = GFlowNetConfig(
            max_depth=args.max_depth,
            beta=args.beta_prime,
            max_size=1,
            top_k=args.top_k,
            epochs=args.epochs,
            logz_lr=args.logz_lr,
        )

        def _build_single_trainer(seed: int) -> SingleTerminalGFlowNetTrainer:
            seed_everything(seed)
            t = SingleTerminalGFlowNetTrainer(
                generator, _fresh_warm_reranker(), single_gfn_config,
                _make_annotated_ik_fn(bundle.train),
                device=rr_trainer.device,
                child_cache_path=str(child_cache_path), ik_cache_path=str(ik_cache_path),
            )
            t.fit(train_substrates_list, epochs=args.epochs, verbose=False)
            t.save_caches()
            return t

        if args.ablation_mode == "single":
            print("[gflownet] training ABL-01 independent single-terminal trainer ...", flush=True)
            t0 = time.time()
            single_trainer = _build_single_trainer(args.seed)
            print(f"[gflownet] ABL-01 trainer done in {time.time()-t0:.1f}s", flush=True)
        elif args.ablation_mode == "ensemble":
            print(
                f"[gflownet] training ABL-02 ensemble of {args.m_ensemble} single-terminal "
                "trainers ...", flush=True,
            )
            t0 = time.time()
            ensemble_trainers = [
                _build_single_trainer(args.seed * 1000 + m) for m in range(args.m_ensemble)
            ]
            print(f"[gflownet] ABL-02 ensemble done in {time.time()-t0:.1f}s", flush=True)
        seed_everything(args.seed)  # restore this run's own seed after ensemble-member seeding

    print(f"[gflownet] evaluating dual matrix on {eval_prefix.upper()} (touch-once for test) ...", flush=True)
    t0 = time.time()
    multistep_cfg = MultiStepConfig(enabled=True, max_depth=args.max_depth, per_node_top_k=10)
    beam_tree = MetabolicTree(generator, filt, multistep_cfg) if filt is not None else None
    if args.workers > 1:
        eval_roots = list(eval_bundle.map.keys())[:eval_count]
        print(f"[gflownet] parallel pre-warming eval caches ({len(eval_roots)} roots) ...", flush=True)
        t0_prewarm = time.time()
        trainer.prewarm_caches(eval_roots, args.workers, gen_ckpt=str(GEN_CKPT),
                               verbose=True, waves=args.prewarm_waves)
        print(f"[gflownet] eval prewarm done in {time.time()-t0_prewarm:.1f}s", flush=True)
    metrics = evaluate_matrix(
        trainer, generator, reranker, beam_tree, eval_bundle,
        n_eval=eval_count, n_samples=args.n_samples, max_size=args.max_size,
        top_k=args.top_k, max_pool=args.max_pool, device=rr_trainer.device,
        ks=(5, 10, 15, 20, 30, 50),
        resume_path=args.resume_eval_ckpt, eval_ckpt_every=args.eval_ckpt_every,
        eval_split=args.eval_split,
        ablation_mode=args.ablation_mode, single_trainer=single_trainer,
        ensemble_trainers=ensemble_trainers, raw_draw_cap_mult=args.raw_draw_cap_mult,
    )
    trainer.save_caches()  # persist env caches populated by eval (test states are reused across seeds)
    if single_trainer is not None:
        single_trainer.save_caches()
    if ensemble_trainers:
        for t in ensemble_trainers:
            t.save_caches()
    print(f"[gflownet] eval done in {time.time()-t0:.1f}s", flush=True)

    if args.ablation_mode != "off" and "ablation01_union_at_k_auc" in metrics and \
            "ablation02_union_at_k_auc" in metrics:
        from grail_metabolism.eval.diversity import compute_ablation_verdict

        verdict = compute_ablation_verdict(
            gflownet_auc=metrics["gflownet_union_at_k_auc"],
            abl01_auc=metrics["ablation01_union_at_k_auc"],
            abl02_auc=metrics["ablation02_union_at_k_auc"],
            margin=args.verdict_margin,
        )
        metrics["ablation_verdict"] = verdict
        print(
            f"[gflownet] ABL-03 verdict (margin={args.verdict_margin}): {verdict}  "
            f"(gflownet_auc={metrics['gflownet_union_at_k_auc']:.4f}  "
            f"ablation01_auc={metrics['ablation01_union_at_k_auc']:.4f}  "
            f"ablation02_auc={metrics['ablation02_union_at_k_auc']:.4f})",
            flush=True,
        )

    result = {
        "seed": args.seed,
        "config": {
            "train_substrates_requested": args.train_substrates,
            "eval_split": args.eval_split,
            "eval_substrates_requested": eval_count,
            "beta": args.beta,
            "lam": args.lam,
            "max_depth": args.max_depth,
            "max_size": args.max_size,
            "top_k": args.top_k,
            "epochs": args.epochs,
            "n_samples": args.n_samples,
            "bootstrap": args.bootstrap,
            "rerank_epochs": args.rerank_epochs,
            "bootstrap_epochs": args.bootstrap_epochs,
            "ablation_mode": args.ablation_mode,
            "beta_prime": args.beta_prime,
            "m_ensemble": args.m_ensemble,
            "raw_draw_cap_mult": args.raw_draw_cap_mult,
            "verdict_margin": args.verdict_margin,
        },
        "counts": {
            "train_examples": len(train_examples),
            "eval_substrates_evaluated": int(metrics["n_substrates"]),
        },
        "metrics": metrics,
        "wall_seconds": time.time() - t_start,
    }

    if args.out:
        results_path = Path(args.out)
    else:
        suffix = ""
        if eval_is_test:
            suffix += "_test"
        if args.seed != 0:
            suffix += f"_seed{args.seed}"
        results_path = RESULTS_PATH if suffix == "" else RESULTS_PATH.with_name(f"gflownet{suffix}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as handle:
        json.dump(result, handle, indent=2)

    print("\n========== STAGE 2b SET-GFLOWNET ==========", flush=True)
    print(f"  eval split: {args.eval_split.upper()}", flush=True)
    print(f"  {eval_prefix} substrates evaluated: {int(metrics['n_substrates'])}", flush=True)
    _beam_key = f"beam_recall@{args.max_size}"
    _beam_str = f"{metrics[_beam_key]:.4f}" if _beam_key in metrics else "n/a (no filter)"
    print(
        f"  recall@{args.max_size}  gflownet={metrics[f'gflownet_recall@{args.max_size}']:.4f}  "
        f"reranker={metrics[f'reranker_recall@{args.max_size}']:.4f}  "
        f"beam={_beam_str}",
        flush=True,
    )
    print(
        f"  diversity: modes_discovered={metrics['modes_discovered']:.2f}  "
        f"mean_pairwise_tanimoto={metrics['mean_pairwise_tanimoto']:.4f}  "
        f"n_unique_scaffolds={metrics['n_unique_scaffolds']:.2f}  "
        f"set_size_calibration={metrics['set_size_calibration']:.2f}  "
        f"circles@t0.4={metrics['circles@t0.4']:.2f}  "
        f"circles@t0.7={metrics['circles@t0.7']:.2f}",
        flush=True,
    )
    print(f"  results -> {results_path}", flush=True)
    print(f"  total wall: {result['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

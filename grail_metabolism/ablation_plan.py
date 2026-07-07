"""Shared, dataset-free ablation planning/selection/verdict module for Phase 3's
set-reward novelty ablation (03-03, Modal un-defer).

This module factors OUT the pure-Python planning/selection/verdict logic that
``scripts/run_ablation_local.py`` implemented inline, so BOTH the local sequential
runner and ``scripts/modal_ablation.py`` (the parallel Modal orchestrator) share ONE
source of truth. The only thing that differs between the two callers is HOW one
config is executed (local ``subprocess.run`` vs a Modal ``@app.function`` call) --
that stays in each caller's own module; this module never shells out and never
imports Modal.

Functions here mirror (not replace) the science already implemented in
``scripts/run_gflownet.py`` (the ``--ablation-mode`` single-variable-pure per-config
unit) and ``grail_metabolism/eval/diversity.py`` (``compute_ablation_verdict``,
``paired_bootstrap_delta_ci``, ``assert_config_match``, ``auc_of_curve``). This file
does not reimplement any of that; it only:

  1. ``plan_configs``: enumerates the independent config list for one ablation run
     (beta-prime VAL sweep + ensemble/reference seed runs + the single test-touch),
     as plain dicts -- no subprocess, no Modal, no I/O.
  2. ``select_beta_prime`` / ``pick_beta_prime``: VAL-side beta-prime selection
     (identical selection rule to ``run_ablation_local.py``'s ``pick_beta_prime``).
  3. ``aggregate_and_verdict``: reads a set of already-materialized per-config result
     JSONs (plain dicts, however they got read -- local disk or downloaded off a
     Modal Volume) and calls the existing verdict primitives to produce the final
     three-way report, byte-identical in shape to ``run_ablation_local.py``'s
     ``report`` dict.

Config dict shape (the ONE contract both callers must honor):
    {
        "tag": str,                 # unique, filesystem/Volume-safe identifier
        "mode": "single" | "ensemble",
        "seed": int,
        "beta_prime": float,
        "eval_split": "val" | "test",
        "m_ensemble": int,
        "eval_substrates": Optional[int],
    }

``plan_configs`` returns configs in the ORDER they should conceptually run (sweep
first, then VAL seed runs, then the test touch) but the whole point of this
refactor is that everything in one "wave" (e.g. all beta-prime sweep configs, or
all VAL seed-run configs) is INDEPENDENT of every other config in that wave --
safe to fan out in parallel. Waves themselves are sequential (the sweep must
finish before beta_prime is chosen; VAL must finish before the test touch runs).
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence

from grail_metabolism.eval.diversity import (
    assert_config_match,
    auc_of_curve,
    compute_ablation_verdict,
    paired_bootstrap_delta_ci,
)

KS = (5, 10, 15, 20, 30, 50)
K_MAX = max(KS)

DEFAULT_BETA_PRIME_GRID: Sequence[float] = (2.0, 4.0, 6.0, 8.0, 10.0)
DEFAULT_SEEDS: Sequence[int] = (0, 1, 2)
DEFAULT_M_ENSEMBLE = 3


def _sweep_tag(seed: int, beta_prime: float) -> str:
    return f"sweep_seed{seed}_bp{beta_prime}"


def _val_tag(mode: str, seed: int) -> str:
    return f"val_{mode}_seed{seed}"


def _test_tag(mode: str, seed: int) -> str:
    return f"test_{mode}_seed{seed}"


def plan_configs(
    beta_prime_grid: Sequence[float] = DEFAULT_BETA_PRIME_GRID,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    m_ensemble: int = DEFAULT_M_ENSEMBLE,
    eval_substrates: Optional[int] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """Enumerate the full independent config set for one ablation run, split into
    the three sequential WAVES (each wave's configs are independent of one another
    and safe to run in parallel; waves themselves are sequential -- see module
    docstring).

    Returns a dict with three keys:
      - "sweep": beta-prime VAL sweep configs (mode="single", one config per grid
        point, all at ``seeds[0]`` -- mirrors ``run_ablation_local.py``'s
        ``sweep_beta_prime`` using only the first seed, since the sweep is cheap
        ABL-01-only and its purpose is picking one scalar, not a seed-level stat).
      - "val": VAL seed-run configs (mode="single" AND mode="ensemble", one pair
        per seed in ``seeds``) -- these still need ``beta_prime`` filled in by the
        caller AFTER the sweep resolves it (each config's ``beta_prime`` field is
        left as ``None`` here; the caller fills it via ``select_beta_prime``'s
        result before dispatching this wave).
      - "test": the single test-touch pair (mode="single" + mode="ensemble", at
        ``seeds[0]``, ``beta_prime=None`` to be filled the same way).

    Every config is a plain, JSON-serializable dict (the CROSS-PROCESS-safe
    contract for both local subprocess argv building and Modal ``.map()`` argument
    passing).
    """
    sweep: List[Dict[str, object]] = []
    sweep_seed = seeds[0]
    for bp in beta_prime_grid:
        sweep.append({
            "tag": _sweep_tag(sweep_seed, bp),
            "mode": "single",
            "seed": sweep_seed,
            "beta_prime": float(bp),
            "eval_split": "val",
            "m_ensemble": m_ensemble,
            "eval_substrates": eval_substrates,
        })

    val: List[Dict[str, object]] = []
    for seed in seeds:
        val.append({
            "tag": _val_tag("single", seed),
            "mode": "single",
            "seed": seed,
            "beta_prime": None,   # filled in by the caller after select_beta_prime
            "eval_split": "val",
            "m_ensemble": m_ensemble,
            "eval_substrates": eval_substrates,
        })
        val.append({
            "tag": _val_tag("ensemble", seed),
            "mode": "ensemble",
            "seed": seed,
            "beta_prime": None,
            "eval_split": "val",
            "m_ensemble": m_ensemble,
            "eval_substrates": eval_substrates,
        })

    test_seed = seeds[0]
    test: List[Dict[str, object]] = [
        {
            "tag": _test_tag("single", test_seed),
            "mode": "single",
            "seed": test_seed,
            "beta_prime": None,
            "eval_split": "test",
            "m_ensemble": m_ensemble,
            "eval_substrates": None,   # test touch always uses the full --test-substrates
        },
        {
            "tag": _test_tag("ensemble", test_seed),
            "mode": "ensemble",
            "seed": test_seed,
            "beta_prime": None,
            "eval_split": "test",
            "m_ensemble": m_ensemble,
            "eval_substrates": None,
        },
    ]

    return {"sweep": sweep, "val": val, "test": test}


def fill_beta_prime(configs: Sequence[Dict[str, object]], beta_prime: float) -> List[Dict[str, object]]:
    """Return a COPY of ``configs`` with ``beta_prime`` filled in (never mutates the
    input list/dicts -- both callers may still hold references to the planned
    wave for logging/bookkeeping)."""
    out = []
    for cfg in configs:
        new_cfg = dict(cfg)
        new_cfg["beta_prime"] = float(beta_prime)
        out.append(new_cfg)
    return out


def sweep_scores_from_results(
    sweep_configs: Sequence[Dict[str, object]], results_by_tag: Dict[str, Optional[dict]],
) -> Dict[float, Optional[float]]:
    """Map each sweep config's ``beta_prime`` to its ``ablation01_union_at_k_auc``
    score, reading from an already-materialized ``{tag: result_dict_or_None}`` map
    (``result_dict`` is the JSON ``run_gflownet.py --out`` writes, however it was
    obtained -- local disk read or Modal Volume download). A missing/None result
    (e.g. a config that failed or was never run) maps to ``None``, mirroring
    ``run_ablation_local.py``'s ``sweep_beta_prime``'s dry-run-skip behavior."""
    scores: Dict[float, Optional[float]] = {}
    for cfg in sweep_configs:
        raw_bp = cfg.get("beta_prime")
        if raw_bp is None:
            raise ValueError(
                f"sweep_scores_from_results: config {cfg.get('tag')!r} has no "
                "beta_prime set (sweep configs must be fully materialized by "
                "plan_configs before being passed here)"
            )
        bp = float(raw_bp)  # type: ignore[arg-type]
        result = results_by_tag.get(str(cfg["tag"]))
        if result is None:
            scores[bp] = None
            continue
        scores[bp] = result["metrics"].get("ablation01_union_at_k_auc")
    return scores


def select_beta_prime(scores: Dict[float, Optional[float]]) -> float:
    """Pick the VAL-better beta-prime (max ablation01 union@K AUC); warns (prints)
    on the D-10 endpoint-widen criterion exactly like ``run_ablation_local.py``'s
    ``pick_beta_prime`` (kept as a re-exported alias below for drop-in compatibility
    with existing call sites/tests)."""
    valid = {bp: v for bp, v in scores.items() if v is not None}
    if not valid:
        raise RuntimeError("select_beta_prime: no valid beta-prime scores (all runs failed/missing)")
    best = max(valid, key=lambda bp: valid[bp])
    grid_sorted = sorted(valid)
    if len(grid_sorted) >= 2 and best in (grid_sorted[0], grid_sorted[-1]):
        print(
            f"[ablation_plan] WARNING (D-10 endpoint-widen criterion): best beta_prime="
            f"{best} is at a GRID ENDPOINT ({grid_sorted[0]}..{grid_sorted[-1]}) -- the true "
            "optimum may lie outside the sampled range. Widen the grid before trusting the "
            "verdict if this is the full sweep.",
            flush=True,
        )
    return best


# Backward-compatible alias (run_ablation_local.py's original name).
pick_beta_prime = select_beta_prime


def degeneracy_guarded_margin(std: float, mean_auc: float, floor: float = 0.005, cv_bound: float = 1.0) -> float:
    """D-11 pre-registered degeneracy fallback: use the fixed Delta=0.02 when the
    3-seed std is below an absolute floor OR its coefficient of variation exceeds
    ``cv_bound``. Verbatim port of ``run_ablation_local.py``'s function of the same
    name (now the single shared source)."""
    cv = (std / mean_auc) if mean_auc else float("inf")
    if std < floor or cv > cv_bound:
        return 0.02
    return 1.0 * std


def compute_delta_sensitivity_grid(
    gflownet_auc: float, abl01_auc: float, abl02_auc: float, std: float,
) -> Dict[str, str]:
    """SENSITIVITY grid (D-11): the seed-level ``compute_ablation_verdict``'s outcome
    across {0.5x, 1.0x, 1.5x}xstd plus the fixed degeneracy-fallback margin=0.02.
    Verbatim port of ``run_ablation_local.py``'s function of the same name."""
    grid: Dict[str, str] = {}
    for mult in (0.5, 1.0, 1.5):
        margin = mult * std
        grid[f"{mult}x_std"] = compute_ablation_verdict(gflownet_auc, abl01_auc, abl02_auc, margin=margin)
    grid["fixed_0.02"] = compute_ablation_verdict(gflownet_auc, abl01_auc, abl02_auc, margin=0.02)
    return grid


def mean_std(xs: Sequence[float]) -> "tuple[float, float]":
    if not xs:
        return 0.0, 0.0
    mean = statistics.fmean(xs)
    std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return mean, std


def per_substrate_aucs(eval_ckpt: dict, series: str) -> Dict[str, float]:
    """Compute ``{root: union_at_k_auc}`` for one series (e.g. "gflownet",
    "ablation01", "ablation02") from an ALREADY-LOADED ``--resume-eval-ckpt`` JSON
    dict's per-substrate union curves (this is the only place per-substrate,
    non-mean-aggregated numbers are available -- ``result["metrics"]`` only
    carries the already-averaged scalar).

    Unlike ``run_ablation_local.py``'s ``_per_substrate_aucs`` (which takes a file
    PATH and does its own ``open()``), this takes the already-parsed dict so it
    stays I/O-free and equally usable whether the eval checkpoint was read off
    local disk or downloaded from a Modal Volume."""
    rows = eval_ckpt.get("rows", {})
    out: Dict[str, float] = {}
    curve_key = f"{series}_union_curve"
    for root, row in rows.items():
        if curve_key not in row:
            continue
        curve = {int(k): v for k, v in row[curve_key].items()}
        out[root] = auc_of_curve(curve, k_min=min(KS), k_max=K_MAX)
    return out


def paired_arrays_from_ckpts(
    gflownet_eval_ckpt: dict, abl_eval_ckpt: dict, abl_series: str,
) -> "tuple[List[float], List[float]]":
    """Build matched (gflownet_i, abl_i) per-substrate AUC arrays over the shared
    root intersection between two ALREADY-LOADED eval checkpoint dicts. I/O-free
    counterpart of ``run_ablation_local.py``'s ``paired_arrays`` (which reads paths)."""
    gflownet_aucs = per_substrate_aucs(gflownet_eval_ckpt, "gflownet")
    abl_aucs = per_substrate_aucs(abl_eval_ckpt, abl_series)
    shared_roots = sorted(set(gflownet_aucs) & set(abl_aucs))
    return [gflownet_aucs[r] for r in shared_roots], [abl_aucs[r] for r in shared_roots]


def aggregate_and_verdict(
    val_single_results: Sequence[dict],
    val_ensemble_results: Sequence[dict],
    test_single_result: dict,
    test_ensemble_result: dict,
    test_single_eval_ckpt: dict,
    test_ensemble_eval_ckpt: dict,
    chosen_beta_prime: float,
    sweep_scores: Dict[float, Optional[float]],
    m_ensemble: int,
) -> Dict[str, object]:
    """Compute the final three-way ABL-03 verdict report, byte-shape-identical to
    ``run_ablation_local.py``'s ``report`` dict.

    Every argument is an already-materialized plain dict (parsed JSON) -- this
    function does no subprocess execution, no file I/O, no Modal calls. Both the
    local runner and the Modal orchestrator read their config-keyed result/eval-
    checkpoint JSONs (from local disk or a downloaded Volume snapshot,
    respectively) and pass the parsed dicts in here.

    Raises ``ValueError`` (via ``assert_config_match``) if the three test-touch
    arms' ``result["config"]`` dicts differ outside the allowed-drift fields
    (FIX C gate, applied BEFORE any test-table value is read).
    """
    assert_config_match({
        "gflownet": test_single_result["config"],
        "ablation01": test_single_result["config"],
        "ablation02": test_ensemble_result["config"],
    })

    # `.get(..., 0.0)`, not `[...]`: run_gflownet.py only POPULATES
    # ablation01_union_at_k_auc/ablation02_union_at_k_auc when at least one eval
    # substrate's (weaker) single-terminal/ensemble arm reaches k_max=50 DISTINCT
    # candidates (see its own defensive `"ablation01_union_at_k_auc" in metrics`
    # guard before computing its CLI-only verdict print) -- an under-provisioned
    # eval window (few/small substrates) can legitimately leave the key absent for
    # EVERY substrate. Absent means that arm produced no valid union@k curve at all
    # on this split, i.e. its floor performance is 0.0 -- the same "no data -> 0.0"
    # convention `_mean([])` already uses elsewhere in this module. Falling back to
    # 0.0 here (rather than a bare `[...]` KeyError) keeps aggregate_and_verdict
    # robust to this legitimate, substrate-structure-dependent degeneracy without
    # changing the verdict arithmetic for the (expected, common) case where the key
    # IS present.
    val_gflownet_aucs = [r["metrics"]["gflownet_union_at_k_auc"] for r in val_single_results]
    val_abl01_aucs = [r["metrics"].get("ablation01_union_at_k_auc", 0.0) for r in val_single_results]
    val_abl02_aucs = [r["metrics"].get("ablation02_union_at_k_auc", 0.0) for r in val_ensemble_results]

    test_gflownet_auc = test_single_result["metrics"]["gflownet_union_at_k_auc"]
    test_abl01_auc = test_single_result["metrics"].get("ablation01_union_at_k_auc", 0.0)
    test_abl02_auc = test_ensemble_result["metrics"].get("ablation02_union_at_k_auc", 0.0)

    gflownet_paired, abl01_paired = paired_arrays_from_ckpts(
        test_single_eval_ckpt, test_single_eval_ckpt, "ablation01"
    )
    gflownet_paired_e, abl02_paired = paired_arrays_from_ckpts(
        test_ensemble_eval_ckpt, test_ensemble_eval_ckpt, "ablation02"
    )

    # An EMPTY shared-substrate intersection means the weaker ablation arm never
    # produced a *_union_curve row on ANY test substrate (the same k_max=50
    # under-production degeneracy the metrics-level `.get(..., 0.0)` fallback above
    # guards against, but surfacing one layer deeper -- at the per-substrate
    # eval-checkpoint level rather than the already-averaged scalar). Unlike the
    # scalar fallback, fabricating a bootstrap CI from zero pairs would misrepresent
    # statistical confidence that does not exist, so report "no data" explicitly
    # instead of calling paired_bootstrap_delta_ci (which correctly raises
    # ValueError on empty input -- that guard stays intact for its other callers).
    if abl01_paired:
        primary_ci_abl01 = paired_bootstrap_delta_ci(gflownet_paired, abl01_paired, n_boot=10000, ci=0.95)
    else:
        primary_ci_abl01 = {"confirmed": False, "n_pairs": 0, "note": "no shared substrates with ablation01 curves"}
    if abl02_paired:
        primary_ci_abl02 = paired_bootstrap_delta_ci(gflownet_paired_e, abl02_paired, n_boot=10000, ci=0.95)
    else:
        primary_ci_abl02 = {"confirmed": False, "n_pairs": 0, "note": "no shared substrates with ablation02 curves"}

    mean_gflownet, std_gflownet = mean_std(val_gflownet_aucs)
    margin = degeneracy_guarded_margin(std_gflownet, mean_gflownet)
    secondary_verdict = compute_ablation_verdict(
        test_gflownet_auc, test_abl01_auc, test_abl02_auc, margin=margin,
    )
    sensitivity_grid = compute_delta_sensitivity_grid(
        test_gflownet_auc, test_abl01_auc, test_abl02_auc, std=std_gflownet,
    )

    return {
        "beta_prime_sweep": sweep_scores,
        "chosen_beta_prime": chosen_beta_prime,
        "val_seed_aucs": {
            "gflownet": val_gflownet_aucs, "ablation01": val_abl01_aucs, "ablation02": val_abl02_aucs,
        },
        "test_table": {
            "gflownet_union_at_k_auc": test_gflownet_auc,
            "ablation01_union_at_k_auc": test_abl01_auc,
            "ablation02_union_at_k_auc": test_abl02_auc,
        },
        "primary_paired_bootstrap": {
            "vs_ablation01": primary_ci_abl01,
            "vs_ablation02": primary_ci_abl02,
        },
        "secondary_seed_level_verdict": secondary_verdict,
        "secondary_margin_used": margin,
        "sensitivity_grid": sensitivity_grid,
        "m_ensemble": m_ensemble,
        "note_fix_e": (
            "Phase 3 reports the generous-ensemble variant only; the compute-matched "
            "ensemble is deferred to Phase 4."
        ),
    }

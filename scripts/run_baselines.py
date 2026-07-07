#!/usr/bin/env python3
"""Plan 04-01 Task 4: val-knob-sweep baseline orchestrator (Phase-2 deferred item).

For each diverse-selection baseline (temperature/top-p, DPP, MMR) this sweeps the
method's OWN diversity knob on the VAL split, picks the knob maximizing the val
objective, then evaluates that ONE knob on TEST exactly once -- never selecting a knob
on test (SCALE-03). Pools come from ``eval/pool_cache`` (built once per substrate,
shared across methods and knob-points); each pool's lambda/theta-INDEPENDENT Tanimoto
kernel is built once via ``eval/baselines.dedup_and_fingerprint`` and reused across every
knob-point (Plan 04-01 Task 3 / D-40-02). All candidates route through the Phase-1 shared
Morgan fingerprint + ``metrics._tautomer_inchikey`` -- no forked path (BASE-05/EVAL-04).

This module exposes a dataset-free, testable CORE (``sweep_knob_on_val`` /
``evaluate_on_test`` / ``run_baseline``) plus a ``main()`` that wires real reranker-scored
pools. It does NOT launch a full-scale (~1246) run -- the headline/Pareto is Plan 04-02
(scope fence D-40-05).

The knob registry (D-40-03) names each method's OWN diversity knob:
  - temperature_topp -> ``T`` (temperature over raw logits)
  - dpp              -> ``theta`` (quality/diversity trade in the DPP kernel)
  - mmr              -> ``lam`` (relevance-vs-diversity dial)
The GFlowNet's own knob for the Pareto (SCALE-02) is ``beta`` -- swept in Plan 04-02, not
here (this module covers the post-hoc baselines only).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from grail_metabolism.eval.baselines import dedup_and_fingerprint, select
from grail_metabolism.metrics import _tautomer_inchikey

# method -> list of (knob kwarg name, default val grid) AXES. A method's knob-space is the
# cartesian product of its axes, so a knob-point is a dict {name: value}. temperature_topp is
# genuinely TWO-dimensional (T AND top-p p); sweeping T alone with p pinned at 1.0 leaves the
# nucleus axis dead and under-explores the (T,p) frontier -- so both axes are swept (the
# research spec's T x p grid), not just T.
KNOB_REGISTRY: Dict[str, List[Tuple[str, List[float]]]] = {
    "temperature_topp": [("T", [0.5, 1.0, 2.0]), ("p", [0.9, 0.95, 1.0])],
    "dpp": [("theta", [0.5, 1.0, 2.0, 4.0])],
    "mmr": [("lam", [0.0, 0.25, 0.5, 0.75, 1.0])],
}

# Documented per-method NEUTRAL knob (no strong diversity/relevance bias). Used ONLY as the
# tie-break target so a val objective that saturates/ties across knobs resolves to the neutral
# point deterministically, never to whichever knob happens to appear first in the grid.
KNOB_NEUTRAL: Dict[str, Dict[str, float]] = {
    "temperature_topp": {"T": 1.0, "p": 1.0},
    "dpp": {"theta": 1.0},
    "mmr": {"lam": 0.5},
}


def knob_points(method: str, grid_override: Optional[Sequence[float]] = None) -> List[Dict[str, float]]:
    """Enumerate every knob-point (a dict of the method's axis kwargs) as the cartesian product
    of its axis grids. ``grid_override`` (if given) replaces the FIRST axis's grid -- convenient
    for single-axis methods (dpp/mmr)."""
    axes = KNOB_REGISTRY[method]
    if grid_override is not None:
        axes = [(axes[0][0], list(grid_override))] + list(axes[1:])
    names = [a[0] for a in axes]
    grids = [a[1] for a in axes]
    return [dict(zip(names, combo)) for combo in itertools.product(*grids)]


def _dist_to_neutral(method: str, knob: Dict[str, float]) -> float:
    neutral = KNOB_NEUTRAL[method]
    return sum((knob[k] - neutral.get(k, 0.0)) ** 2 for k in knob)


def _neg_values(knob: Dict[str, float]) -> Tuple[float, ...]:
    return tuple(-v for _, v in sorted(knob.items()))


@dataclass
class ScoredPool:
    """A reranker-scored candidate pool for one substrate, with its precomputed
    (deduped) fingerprints + Tanimoto kernel so a whole knob-sweep reuses ONE kernel."""

    substrate: str
    pool: List[Tuple[str, float]]          # (smiles, reranker_logit), any order
    annotated_ik: Set[str] = field(default_factory=set)
    fps: list = field(default_factory=list)
    kernel: Optional[np.ndarray] = None


def prepare_scored_pool(substrate: str, pool_scored: Sequence[Tuple[str, float]],
                        annotated_ik: Set[str]) -> ScoredPool:
    """Build the per-pool fingerprints + kernel ONCE (score-aware dedup) so the knob-sweep
    over this pool never rebuilds the O(N^2) kernel (Task 3 reuse)."""
    _smiles, _scores, fps, S = dedup_and_fingerprint(pool_scored)
    return ScoredPool(substrate=substrate, pool=list(pool_scored), annotated_ik=set(annotated_ik),
                      fps=fps, kernel=S)


def recall_objective(selected: Sequence[str], annotated_ik: Set[str]) -> float:
    """Fraction of annotated true metabolites recovered by ``selected`` (tautomer-aware)."""
    if not annotated_ik:
        return 0.0
    hit = {_tautomer_inchikey(s) for s in selected} & annotated_ik
    return len(hit) / len(annotated_ik)


def _default_objective(selected: Sequence[str], sp: ScoredPool) -> float:
    return recall_objective(selected, sp.annotated_ik)


def _select_with_knob(method: str, knob: Dict[str, float], sp: ScoredPool, k: int) -> List[str]:
    """One selection at a single knob-point (a dict of the method's axis kwargs), reusing the
    pool's cached fps+kernel for DPP/MMR (kernel built once per pool, not once per knob-point)."""
    kwargs: Dict[str, object] = dict(knob)
    if method in ("dpp", "mmr"):
        kwargs["fps"] = sp.fps
        kwargs["kernel"] = sp.kernel
    if method == "temperature_topp":
        kwargs["rng"] = np.random.default_rng(0)  # deterministic sampling for reproducibility
    return select(sp.pool, k, method=method, **kwargs)


def sweep_knob_on_val(
    method: str,
    val_pools: Sequence[ScoredPool],
    k: int,
    objective_fn: Callable[[Sequence[str], ScoredPool], float] = _default_objective,
    grid: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Sweep ``method``'s knob-space (cartesian product of its axes) on the VAL pools; return
    the argmax knob (a dict) and the full ``[{knob, score}, ...]`` curve. The knob is selected
    HERE (val), never on test. Ties are broken DETERMINISTICALLY -- highest val score, then
    nearest the method's documented neutral knob, then a stable value order -- so the selected
    knob never depends on grid ordering."""
    points = knob_points(method, grid)
    scored: List[Tuple[Dict[str, float], float]] = []
    for knob in points:
        vals = [objective_fn(_select_with_knob(method, knob, sp, k), sp) for sp in val_pools]
        scored.append((knob, float(np.mean(vals)) if vals else 0.0))
    best_knob, _best_score = max(
        scored,
        key=lambda ks: (ks[1], -_dist_to_neutral(method, ks[0]), _neg_values(ks[0])),
    )
    curve: List[Dict[str, object]] = [{"knob": kn, "score": sc} for kn, sc in scored]
    return best_knob, curve


def evaluate_on_test(
    method: str,
    knob: Dict[str, float],
    test_pools: Sequence[ScoredPool],
    k: int,
    metric_fns: Dict[str, Callable[[Sequence[str], ScoredPool], float]],
) -> Dict[str, float]:
    """Evaluate the SINGLE (val-selected) ``knob`` on the TEST pools -- one pass, the only time
    test is touched (SCALE-03)."""
    acc: Dict[str, List[float]] = {name: [] for name in metric_fns}
    for sp in test_pools:
        sel = _select_with_knob(method, knob, sp, k)
        for name, fn in metric_fns.items():
            acc[name].append(fn(sel, sp))
    return {name: (float(np.mean(vals)) if vals else 0.0) for name, vals in acc.items()}


def run_baseline(
    method: str,
    val_pools: Sequence[ScoredPool],
    test_pools: Sequence[ScoredPool],
    k: int,
    objective_fn: Callable[[Sequence[str], ScoredPool], float] = _default_objective,
    metric_fns: Optional[Dict[str, Callable[[Sequence[str], ScoredPool], float]]] = None,
    grid: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Full val-select -> test-once cycle for one baseline. Returns the val-selected knob (a
    dict), the val curve, and the test metrics computed with THAT knob (never a test-optimal one)."""
    best, curve = sweep_knob_on_val(method, val_pools, k, objective_fn, grid)
    if metric_fns is None:
        metric_fns = {"recall": lambda sel, sp: recall_objective(sel, sp.annotated_ik)}
    test_metrics = evaluate_on_test(method, best, test_pools, k, metric_fns)
    return {
        "method": method,
        "knob_names": [axis[0] for axis in KNOB_REGISTRY[method]],
        "val_selected_knob": best,
        "val_curve": curve,
        "test_metrics": test_metrics,
    }


# --------------------------------------------------------------------------- #
# main(): wires real reranker-scored pools. Intentionally NOT a full-scale run
# (scope fence D-40-05) -- it is the entry point Plan 04-02 will scale up.
# --------------------------------------------------------------------------- #

def main() -> None:  # pragma: no cover - real-data entry point, not exercised in tests
    import argparse

    ap = argparse.ArgumentParser(description="Val-knob-sweep baseline orchestrator (Plan 04-01 Task 4).")
    ap.add_argument("--methods", nargs="+", default=list(KNOB_REGISTRY.keys()))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", type=str, default="results/baselines_report.json")
    ap.add_argument("--limit-substrates", type=int, default=0,
                    help="Cap substrates for a smoke run (0 = all; this module never launches "
                         "the full-scale ~1246 headline itself -- that is Plan 04-02).")
    args = ap.parse_args()

    raise SystemExit(
        "run_baselines.main() wires real pools via eval/pool_cache + a reranker; the full-scale "
        "headline run is Plan 04-02 (gated on the ablation verdict + the budget go/no-go). This "
        "module ships the tested val-select/test-once CORE (sweep_knob_on_val / evaluate_on_test / "
        "run_baseline); wire the real pool loader in 04-02. Args parsed: "
        f"methods={args.methods} k={args.k} out={args.out} limit={args.limit_substrates}."
    )


if __name__ == "__main__":
    main()

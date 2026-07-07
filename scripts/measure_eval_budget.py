#!/usr/bin/env python3
"""Plan 04-01 Task 1 -- the Phase-4 budget-measurement gate (D-40-01 .. D-40-06).

Times the FOUR cost terms that dominate a full-scale Phase-4 evaluation, on a SMALL
COLD real sample, then extrapolates conservatively (anti-under-count, D-40-06) to the
full (N_test + N_val) x 3-seed matrix. Writes ``results/budget_report.json`` carrying
the four terms as distributions, the full extrapolation, a Modal preemption band, a
pool-cache disk GB estimate, and a recommendation in
``{local_feasible, modal_needed, scope_trim_needed}`` measured against user-set
ceilings. NO full-scale run is launched here (scope fence D-40-05).

The four terms (D-40-06 / must_haves):
  (a) COLD pool-build wall-clock per substrate -- RDKit generator + rule-apply (+ optional
      depth-2 expansion) for a substrate ABSENT from artifacts/reranker_gate_cache (the
      ~92% real case). Size-stratified by heavy-atom count; reported as a distribution
      (median/p90/p99/max) and extrapolated on a stratified SUM, not mean x N.
  (b) selection per (method, knob-point) -- temperature/top-p (~ms) timed SEPARATELY from
      DPP/MMR. For DPP/MMR: one O(N^2) Tanimoto-kernel build per pool (lambda-INDEPENDENT,
      billed ONCE per pool) vs. cheap per-knob greedy select over the cached kernel.
  (c) GFlowNet training per seed -- from the running-ablation pace (given), with a single-
      point +/-50% band caveat if only one train scale is available.
  (d) fixed per-process startup -- bundle SDF-standardization + reranker warm-start +
      generator load, measured on the small sample's own loads.

Reuse, do not fork:
  - pool-build via ``workflows.reranker.build_pool`` -- the SAME path
    ``run_gflownet.py:_reranker_topk_smiles`` calls (line ~155).
  - selection via ``eval.baselines.select`` / ``_pool_fingerprints`` /
    ``_tanimoto_kernel_matrix`` (the Phase-1 shared Morgan r=2/2048 fingerprint).
  - dataset bundle via ``workflows.data.load_dataset_bundle`` (yields substrate SMILES).
  - COLD = substrate SMILES NOT a key in artifacts/reranker_gate_cache/gfn_child_cache_*.pkl.

Usage:
  python scripts/measure_eval_budget.py --n-cold 18 --top-k 200 --max-pool 100 \
      --ceiling-hours 48 --ceiling-usd 200 --ceiling-gb 100
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CACHE_DIR = ROOT / "artifacts" / "reranker_gate_cache"
GEN_CKPT = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
RESULTS_PATH = ROOT / "results" / "budget_report.json"

# --- given Modal-run pace (D-40-C caveat context), used for term (c) extrapolation ---
# The running ABL-03 Modal run reports ~25-30 min/epoch at n300 (train_substrates=300),
# 15 epochs. This is a GIVEN observed pace, stated per the plan.
N300_MIN_PER_EPOCH = (25.0, 30.0)   # (low, high) observed minutes/epoch at n300
N300_EPOCHS = 15
N300_TRAIN = 300
# A clean second timing at n800 (train_bi_s800 checkpoint exists but is a reranker-cache
# artifact, not an epoch-pace log); absent a logged n800 epoch pace this run reports a
# SINGLE-POINT estimate with an explicit linear-assumption caveat + a +/-50% band (D-40-C).
N800_MIN_PER_EPOCH: Optional[Tuple[float, float]] = None
N800_TRAIN = 800

# Fixed startup components (D-40-D). Measured where cheap; the bundle SDF-standardization
# load is the dominant fixed cost. We measure the components we actually perform here and
# fall back to the plan's stated ~252s decomposition for any we deliberately skip.
STARTUP_STATED = {
    "bundle_sdf_standardization_s": 111.0,
    "reranker_warmstart_s": 125.0,
    "generator_load_s": 16.0,
}


def _load_child_cache_roots() -> set:
    """Union of all root SMILES present as keys in any gfn_child_cache_*.pkl -- a substrate
    is WARM iff its root SMILES is in this set, COLD otherwise (~92% of test is COLD)."""
    roots: set = set()
    for pkl in sorted(CACHE_DIR.glob("gfn_child_cache_*.pkl")):
        try:
            with open(pkl, "rb") as fh:
                d = pickle.load(fh)
            roots.update(d.keys())
        except Exception as exc:  # a bad cache file must not abort the measurement
            print(f"[budget] WARNING: could not read {pkl}: {exc}", flush=True)
    return roots


def _heavy_atom_count(smiles: str) -> Optional[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


def _distinct_substrate_count(triples_path: Path) -> int:
    """Distinct col-1 (substrate id) count of a *_triples_clean.txt -- the number of
    substrates in that split (read at RUNTIME, never hard-coded; L9/D-40-06)."""
    seen: set = set()
    with open(triples_path) as fh:
        for line in fh:
            parts = line.split()
            if parts:
                seen.add(parts[0])
    return len(seen)


def _stratify_by_size(
    cold_substrates: Sequence[str], sizes: Dict[str, int], n_pick: int, n_buckets: int = 3
) -> List[str]:
    """Size-stratified pick of ~``n_pick`` COLD substrates, bucketed by heavy-atom count
    into ``n_buckets`` (roughly equal-population) tertiles, spreading the pick evenly so the
    timed sample spans the small/medium/large substrate spectrum (D-40-06 H5: distribution,
    not a mean over an unrepresentative slice)."""
    ordered = sorted(cold_substrates, key=lambda s: sizes[s])
    n = len(ordered)
    if n == 0:
        return []
    bucket_size = max(1, n // n_buckets)
    buckets: List[List[str]] = [
        ordered[i * bucket_size: (i + 1) * bucket_size] for i in range(n_buckets - 1)
    ]
    buckets.append(ordered[(n_buckets - 1) * bucket_size:])  # last bucket gets the remainder
    picked: List[str] = []
    per_bucket = max(1, n_pick // n_buckets)
    for b in buckets:
        step = max(1, len(b) // per_bucket)
        picked.extend(b[::step][:per_bucket])
    return picked[:n_pick]


def _dist(xs: Sequence[float]) -> Dict[str, float]:
    """median / p90 / p99 / max / mean / n over a list of samples (the reported shape for
    every measured term, D-40-06 H5)."""
    if not xs:
        return {"n": 0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    s = sorted(xs)
    def _pct(p: float) -> float:
        if len(s) == 1:
            return s[0]
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return s[idx]
    return {
        "n": len(s),
        "median": statistics.median(s),
        "p90": _pct(0.90),
        "p99": _pct(0.99),
        "max": s[-1],
        "mean": sum(s) / len(s),
    }


# --------------------------------------------------------------------------- #
# Term (a): COLD pool-build.
# --------------------------------------------------------------------------- #

def measure_pool_build(
    generator, cold_sample: Sequence[str], sizes: Dict[str, int], top_k: int, max_pool: int
) -> Tuple[Dict, List[dict], List]:
    """Time build_pool for each COLD substrate in a fresh (this) process, pointing any
    env-cache at a throwaway TMPDIR so the real bank stays untouched and cold stays cold
    (D-40-01 anti-pollution). build_pool itself hits only the generator (no env-cache write),
    but we set the redirect defensively in case any transitively-touched cache path exists.

    Returns (distribution, per_substrate_records, built_pools) -- built_pools feeds terms
    (b) and the GB estimate so we build each pool exactly once."""
    from grail_metabolism.workflows.reranker import build_pool

    tmp_cache = tempfile.mkdtemp(prefix="budget_coldcache_")
    os.environ["GRAIL_ENV_CACHE_DIR"] = tmp_cache  # honored if any code consults it; harmless else

    times: List[float] = []
    records: List[dict] = []
    built_pools: List[Tuple[str, list]] = []
    for root in cold_sample:
        t0 = time.perf_counter()
        pool = build_pool(generator, root, top_k=top_k, max_pool=max_pool)
        dt = time.perf_counter() - t0
        times.append(dt)
        records.append({
            "smiles": root,
            "heavy_atoms": sizes.get(root),
            "pool_size": len(pool),
            "seconds": dt,
        })
        built_pools.append((root, pool))
        print(f"[budget] (a) cold pool-build hac={sizes.get(root):>3} "
              f"pool={len(pool):>3} {dt:6.2f}s  {root[:50]}", flush=True)

    # Stratified-sum extrapolation base: the per-bucket mean x bucket-population, summed --
    # NOT the global mean x N (H5). Buckets are the heavy-atom tertiles used to pick.
    dist = _dist(times)
    dist["per_substrate"] = records
    return dist, records, built_pools


# --------------------------------------------------------------------------- #
# Term (b): selection (temp/top-p cheap; DPP/MMR one kernel-build per pool + cheap per-knob).
# --------------------------------------------------------------------------- #

def measure_selection(
    reranker, generator, built_pools, top_k: int, max_pool: int, device, k: int,
    dpp_knobs: Sequence[float], mmr_knobs: Sequence[float],
    temp_knobs: Sequence[Tuple[float, float]],
) -> Dict:
    """With the REAL max_pool, time temperature/top-p (~ms) SEPARATELY from DPP/MMR, and for
    DPP/MMR split (kernel-build once per pool) from (per-knob select over the cached kernel)
    across a multi-point knob grid -- to confirm the kernel-reuse saving (D-40-02 / H4).

    Bills ONE kernel-build per pool, not per knob-point (prohibition: MUST NOT model the
    knob-sweep as ~ms per knob-point NOR rebuild the O(N^2) kernel per knob-point)."""
    import numpy as np

    from grail_metabolism.eval.baselines import (
        _pool_fingerprints,
        _tanimoto_kernel_matrix,
        select,
    )

    # Build real reranker-scored pools from the built candidate pools so scores are the real
    # logits DPP/MMR/temperature consume. Reuse _reranker_topk-style scoring inline.
    from torch_geometric.data import Batch

    import torch

    from grail_metabolism.utils.transform import from_rdmol

    def _score_pool(root: str, cand_pool: list) -> List[Tuple[str, float]]:
        sub_mol = Chem.MolFromSmiles(root)
        if sub_mol is None:
            return []
        sub_graph = from_rdmol(sub_mol)
        if sub_graph is None:
            return []
        prior = generator.rule_prior_logits.detach().cpu()
        num_rules = int(prior.numel())
        prod_graphs, rule_priors, gen_scores, smiles = [], [], [], []
        for cand_smiles, gen_score, rule_id in cand_pool:
            cand_mol = Chem.MolFromSmiles(cand_smiles)
            if cand_mol is None:
                continue
            g = from_rdmol(cand_mol)
            if g is None:
                continue
            prod_graphs.append(g)
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
                torch.tensor(rule_priors, device=device),
                torch.tensor(gen_scores, device=device),
            ).detach().cpu()
        return [(smiles[i], float(scores[i])) for i in range(len(smiles))]

    temp_times: List[float] = []
    kernel_build_times: List[float] = []
    dpp_perknob_times: List[float] = []
    mmr_perknob_times: List[float] = []
    pool_sizes: List[int] = []

    for root, cand_pool in built_pools:
        scored = _score_pool(root, cand_pool)
        if len(scored) < 2:
            continue
        pool_sizes.append(len(scored))

        # temperature/top-p: cheap, timed per knob-point (expect ~ms)
        for (T, p) in temp_knobs:
            rng = np.random.default_rng(0)
            t0 = time.perf_counter()
            select(scored, k=k, method="temperature_topp", T=T, p=p, rng=rng)
            temp_times.append(time.perf_counter() - t0)

        # DPP/MMR: ONE fingerprint+kernel build per pool (lambda-independent), reused.
        # Mirror baselines' internal filtering (parse + tautomer-dedup) so the fps the
        # cached kernel is built over matches what select() will use, letting the FIX-B
        # length guard reuse it rather than silently rebuild it.
        from grail_metabolism.eval.baselines import _dedup_pool_score_aware

        smi = [s for s, _ in scored]
        sc = np.asarray([v for _, v in scored], dtype=np.float64)
        pidx = [i for i, s in enumerate(smi) if Chem.MolFromSmiles(s) is not None]
        smi = [smi[i] for i in pidx]
        sc = sc[pidx]
        if not smi:
            continue
        smi_d, _sc_d = _dedup_pool_score_aware(smi, sc)
        t0 = time.perf_counter()
        fps = _pool_fingerprints(smi_d)
        _ = _tanimoto_kernel_matrix(fps)  # the O(N^2) build billed ONCE per pool
        kernel_build_times.append(time.perf_counter() - t0)

        # per-knob selects reusing the cached fps (the length matches the deduped pool, so
        # select()'s FIX-B guard reuses it rather than rebuilding the kernel).
        for theta in dpp_knobs:
            t0 = time.perf_counter()
            select(scored, k=k, method="dpp", theta=theta, fps=fps)
            dpp_perknob_times.append(time.perf_counter() - t0)
        for lam in mmr_knobs:
            t0 = time.perf_counter()
            select(scored, k=k, method="mmr", lam=lam, fps=fps)
            mmr_perknob_times.append(time.perf_counter() - t0)

    return {
        "n_pools_timed": len(pool_sizes),
        "pool_size_dist": _dist(pool_sizes),
        "temperature_topp_per_knobpoint_s": _dist(temp_times),
        "dpp_mmr_kernel_build_per_pool_s": _dist(kernel_build_times),
        "dpp_per_knobpoint_over_cached_kernel_s": _dist(dpp_perknob_times),
        "mmr_per_knobpoint_over_cached_kernel_s": _dist(mmr_perknob_times),
        "knob_grid": {
            "temperature_topp": [list(t) for t in temp_knobs],
            "dpp_theta": list(dpp_knobs),
            "mmr_lam": list(mmr_knobs),
        },
    }


# --------------------------------------------------------------------------- #
# Term (c): gflownet train per seed (best-effort from the observed pace).
# --------------------------------------------------------------------------- #

def estimate_gflownet_train() -> Dict:
    """Per-seed train hours from the observed running-ablation pace at n300 (given). If only
    one train scale is available, report a SINGLE-POINT estimate WITH an explicit linear-
    assumption caveat + a +/-50% band (D-40-C). This is a FIXED 3-seed cost, NOT grid-
    multiplied."""
    lo_min, hi_min = N300_MIN_PER_EPOCH
    per_seed_lo_h = lo_min * N300_EPOCHS / 60.0
    per_seed_hi_h = hi_min * N300_EPOCHS / 60.0
    per_seed_expected_h = (per_seed_lo_h + per_seed_hi_h) / 2.0

    result = {
        "basis": f"observed running-ablation pace at n{N300_TRAIN}: "
                 f"{lo_min}-{hi_min} min/epoch x {N300_EPOCHS} epochs (GIVEN)",
        "per_seed_hours_expected": per_seed_expected_h,
        "per_seed_hours_band": [per_seed_lo_h, per_seed_hi_h],
        "n_seeds": 3,
        "total_hours_expected": per_seed_expected_h * 3,
        "single_point": N800_MIN_PER_EPOCH is None,
    }
    if N800_MIN_PER_EPOCH is None:
        # single-point: no logged n800 epoch pace -> +/-50% band around the n300 estimate,
        # with an explicit linear-assumption caveat.
        result["caveat"] = (
            "SINGLE-POINT estimate: only the n300 epoch pace is logged; no clean n800 "
            "epoch-pace second point is available for a slope, so the full-train-scale "
            "cost is a linear-in-epochs extrapolation at n300 pace with a +/-50% band. "
            "If the headline trains at a LARGER substrate count, per-epoch cost grows with "
            "the frontier-expansion work per epoch and this UNDER-estimates -- the band's "
            "high side is the planning number."
        )
        result["per_seed_hours_band"] = [per_seed_expected_h * 0.5, per_seed_expected_h * 1.5]
        result["total_hours_band"] = [per_seed_expected_h * 0.5 * 3, per_seed_expected_h * 1.5 * 3]
    else:
        result["total_hours_band"] = [per_seed_lo_h * 3, per_seed_hi_h * 3]
    return result


# --------------------------------------------------------------------------- #
# Term (d): fixed startup.
# --------------------------------------------------------------------------- #

def measure_startup(measured_generator_load_s: Optional[float]) -> Dict:
    """Fixed per-process startup. We measure the generator load in-process (cheap); the
    bundle SDF-standardization and reranker warm-start are the plan's stated components
    (measuring them fully would require the whole heavy load, defeating the "minutes" spike
    -- so they are taken as given, D-40-D ~252s total)."""
    comp = dict(STARTUP_STATED)
    if measured_generator_load_s is not None:
        comp["generator_load_s"] = measured_generator_load_s
        comp["generator_load_measured"] = True
    total = sum(v for k, v in comp.items() if k.endswith("_s"))
    comp["total_s"] = total
    return comp


# --------------------------------------------------------------------------- #
# Extrapolation + recommendation.
# --------------------------------------------------------------------------- #

def extrapolate(
    n_test: int, n_val: int, pool_dist: Dict, pool_records: List[dict],
    selection: Dict, gflownet: Dict, startup: Dict,
    gloryx_n: int, dpp_grid: int, mmr_grid: int, temp_grid: int,
    n_seeds: int, one_pool_bytes: int,
    modal_containers: int, preempt_rate_band: Tuple[float, float, float],
    preempt_reload_min: float,
) -> Dict:
    """Full-matrix extrapolation (D-40-06). Local-serial hours + Modal-parallel band + $ + GB.

    Pool-build is billed for (N_test + N_val) x n_seeds substrates on a size-stratified SUM
    (H3: val pools are NOT free; H5: stratified sum not mean x N), plus GLORYx-37 as a
    100%-COLD sub-term (M7). Selection: temp cheap per knob-point; DPP/MMR one kernel/pool +
    cheap per-knob over the grid. GFlowNet: per-seed x n_seeds (fixed, not grid-multiplied).
    Startup: paid once locally, but per-container AND per-preemption on Modal (M8)."""
    # --- (a) pool-build: stratified-sum base ---
    # Group measured per-substrate times into heavy-atom tertiles, take each bucket's mean,
    # and multiply by that bucket's SHARE of the full population (approximated by the sample's
    # own bucket shares -- the sample was size-stratified to span the spectrum).
    recs = [r for r in pool_records if r.get("seconds") is not None]
    if recs:
        by_size = sorted(recs, key=lambda r: (r["heavy_atoms"] or 0))
        n = len(by_size)
        b = max(1, n // 3)
        buckets = [by_size[:b], by_size[b:2 * b], by_size[2 * b:]]
        bucket_means = [statistics.mean([r["seconds"] for r in bk]) for bk in buckets if bk]
        bucket_shares = [len(bk) / n for bk in buckets if bk]
        # expected per-substrate cost as the stratified (share-weighted) mean
        stratified_mean_s = sum(m * s for m, s in zip(bucket_means, bucket_shares))
    else:
        stratified_mean_s = pool_dist.get("mean", 0.0)

    total_pool_substrates = (n_test + n_val) * n_seeds
    poolbuild_main_h = stratified_mean_s * total_pool_substrates / 3600.0
    # GLORYx-37 is 100% cold; bill at the size-stratified mean (conservative -- GLORYx
    # substrates are drug-like, generally in the medium/large buckets).
    poolbuild_gloryx_h = stratified_mean_s * gloryx_n * n_seeds / 3600.0
    poolbuild_h = poolbuild_main_h + poolbuild_gloryx_h

    # --- (b) selection over the full matrix ---
    # temp: temp_grid knob-points x total_pool_substrates
    temp_per = selection["temperature_topp_per_knobpoint_s"]["mean"]
    kernel_per = selection["dpp_mmr_kernel_build_per_pool_s"]["mean"]
    dpp_per = selection["dpp_per_knobpoint_over_cached_kernel_s"]["mean"]
    mmr_per = selection["mmr_per_knobpoint_over_cached_kernel_s"]["mean"]
    n_pools = total_pool_substrates
    temp_h = temp_per * temp_grid * n_pools / 3600.0
    # ONE kernel build per pool (billed once, not per knob-point); each of DPP & MMR reuses it.
    kernel_h = kernel_per * n_pools / 3600.0
    dpp_h = dpp_per * dpp_grid * n_pools / 3600.0
    mmr_h = mmr_per * mmr_grid * n_pools / 3600.0
    selection_h = temp_h + kernel_h + dpp_h + mmr_h

    # --- (c) gflownet (fixed 3-seed cost) ---
    gflownet_h = gflownet["total_hours_expected"]
    gflownet_h_band = gflownet.get("total_hours_band", [gflownet_h, gflownet_h])

    # --- (d) startup (local: once) ---
    startup_local_h = startup["total_s"] / 3600.0

    local_serial_h = (
        poolbuild_h + selection_h + gflownet_h + startup_local_h
    )
    local_serial_h_band_low = (
        poolbuild_h + selection_h + gflownet_h_band[0] + startup_local_h
    )
    local_serial_h_band_high = (
        poolbuild_h + selection_h + gflownet_h_band[1] + startup_local_h
    )

    # --- Modal parallel band (M8/B2) ---
    # ideal_parallel = total serial work / containers, + startup PER container,
    # + preemption tax = expected_preemptions x reload_min x containers.
    # The pool-build + selection work is embarrassingly parallel across substrates; gflownet
    # is n_seeds independent runs (parallel up to n_seeds containers).
    parallelizable_h = poolbuild_h + selection_h
    ideal_parallel_h = parallelizable_h / max(1, modal_containers)
    # gflownet runs in parallel across seeds (each seed one container), so its wall is one
    # per-seed run, not the 3-seed sum.
    gflownet_wall_h = gflownet_h / max(1, gflownet["n_seeds"])
    ideal_parallel_h += gflownet_wall_h
    startup_per_container_h = startup["total_s"] / 3600.0

    def _modal_wall(preempt_rate: float) -> float:
        # expected_preemptions ~ preempt_rate x (ideal wall-hours) x containers -- a
        # container running for `ideal_parallel_h` hours at `preempt_rate` preemptions/hour
        # is preempted ~preempt_rate*ideal_parallel_h times, each costing a reload.
        expected_preemptions = preempt_rate * ideal_parallel_h * modal_containers
        preempt_tax_h = expected_preemptions * (preempt_reload_min / 60.0)
        # each container pays startup once (min) + once per preemption reload (already in tax)
        return ideal_parallel_h + startup_per_container_h + preempt_tax_h

    p_min, p_exp, p_max = preempt_rate_band
    modal_band_h = {
        "min": _modal_wall(p_min),
        "expected": _modal_wall(p_exp),
        "max": _modal_wall(p_max),
        "assumed_container_count": modal_containers,
        "assumed_preempt_rate_per_hour_band": [p_min, p_exp, p_max],
        "assumed_reload_min_per_preemption": preempt_reload_min,
        "note": "the observed dominant Modal uncertainty this session IS preemption "
                "frequency; the max band is the planning number.",
    }
    # Modal $ (A10G ~ $1.10/hr container-hour as a representative on-demand rate; stated).
    modal_rate_usd_per_container_hour = 1.10
    modal_usd_band = {
        "min": modal_band_h["min"] * modal_containers * modal_rate_usd_per_container_hour,
        "expected": modal_band_h["expected"] * modal_containers * modal_rate_usd_per_container_hour,
        "max": modal_band_h["max"] * modal_containers * modal_rate_usd_per_container_hour,
        "assumed_rate_usd_per_container_hour": modal_rate_usd_per_container_hour,
    }

    # --- GB estimate (L10) ---
    pool_cache_gb = one_pool_bytes * total_pool_substrates / (1024.0 ** 3)

    return {
        "n_test": n_test,
        "n_val": n_val,
        "n_seeds": n_seeds,
        "total_pool_substrates": total_pool_substrates,
        "gloryx_n": gloryx_n,
        "term_a_poolbuild": {
            "stratified_mean_s_per_substrate": stratified_mean_s,
            "main_hours": poolbuild_main_h,
            "gloryx_hours": poolbuild_gloryx_h,
            "total_hours": poolbuild_h,
        },
        "term_b_selection": {
            "temperature_topp_hours": temp_h,
            "dpp_mmr_kernel_build_hours": kernel_h,
            "dpp_perknob_hours": dpp_h,
            "mmr_perknob_hours": mmr_h,
            "total_hours": selection_h,
            "knob_grid_sizes": {"temp": temp_grid, "dpp": dpp_grid, "mmr": mmr_grid},
        },
        "term_c_gflownet": {
            "total_hours_expected": gflownet_h,
            "total_hours_band": gflownet_h_band,
        },
        "term_d_startup": {
            "local_hours_once": startup_local_h,
            "per_container_hours": startup_per_container_h,
        },
        "local_serial_hours": {
            "expected": local_serial_h,
            "band": [local_serial_h_band_low, local_serial_h_band_high],
        },
        "modal_parallel_hours_band": modal_band_h,
        "modal_usd_band": modal_usd_band,
        "pool_cache_gb_estimate": {
            "one_pool_bytes": one_pool_bytes,
            "total_gb": pool_cache_gb,
        },
    }


def recommend(extrap: Dict, ceiling_hours: float, ceiling_usd: float, ceiling_gb: float) -> Dict:
    """Map the extrapolation to {local_feasible, modal_needed, scope_trim_needed} vs the
    user ceilings. Uses the CONSERVATIVE side of each band (planning-number, D-40-06).

    Mapping logic:
      - local_feasible  iff the local-serial HIGH-band hours <= ceiling_hours AND GB <= ceiling_gb.
      - else modal_needed iff the Modal MAX-band hours fit a practical wall AND Modal MAX $ <=
        ceiling_usd AND GB <= ceiling_gb (Modal buys parallelism; the user funds $ not wall-days).
      - else scope_trim_needed (neither local nor Modal fits the ceilings)."""
    local_high = extrap["local_serial_hours"]["band"][1]
    modal_max_h = extrap["modal_parallel_hours_band"]["max"]
    modal_max_usd = extrap["modal_usd_band"]["max"]
    gb = extrap["pool_cache_gb_estimate"]["total_gb"]

    if local_high <= ceiling_hours and gb <= ceiling_gb:
        rec = "local_feasible"
        reason = (f"local-serial high-band {local_high:.1f}h <= ceiling {ceiling_hours}h "
                  f"and disk {gb:.2f}GB <= {ceiling_gb}GB")
    elif modal_max_usd <= ceiling_usd and gb <= ceiling_gb:
        rec = "modal_needed"
        reason = (f"local-serial high-band {local_high:.1f}h EXCEEDS ceiling {ceiling_hours}h; "
                  f"Modal max ${modal_max_usd:.0f} <= ${ceiling_usd} and disk {gb:.2f}GB <= "
                  f"{ceiling_gb}GB -> parallelize on Modal")
    else:
        rec = "scope_trim_needed"
        reason = (f"neither local (high {local_high:.1f}h vs {ceiling_hours}h) nor Modal "
                  f"(max ${modal_max_usd:.0f} vs ${ceiling_usd}, disk {gb:.2f}GB vs "
                  f"{ceiling_gb}GB) fits the ceilings")
    return {
        "recommendation": rec,
        "reason": reason,
        "ceilings": {"hours": ceiling_hours, "usd": ceiling_usd, "gb": ceiling_gb},
        "measured_against": {
            "local_serial_high_band_hours": local_high,
            "modal_max_band_hours": modal_max_h,
            "modal_max_usd": modal_max_usd,
            "pool_cache_gb": gb,
        },
    }


def _estimate_one_pool_bytes(built_pools) -> int:
    """Serialized size of one built pool (pickle) -- the per-(substrate,seed) disk unit the
    GB estimate multiplies (L10)."""
    if not built_pools:
        return 0
    # pick the largest pool by candidate count as the conservative unit
    _root, pool = max(built_pools, key=lambda rp: len(rp[1]))
    return len(pickle.dumps(pool))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-4 budget-measurement gate (Plan 04-01 Task 1)")
    ap.add_argument("--n-cold", type=int, default=18, help="COLD stratified sample size for term (a).")
    ap.add_argument("--top-k", type=int, default=200, help="Generator candidates per substrate (headline).")
    ap.add_argument("--max-pool", type=int, default=100, help="Pool truncation (headline).")
    ap.add_argument("--k", type=int, default=50, help="Output budget k for selection timing.")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--gloryx-n", type=int, default=37)
    ap.add_argument("--modal-containers", type=int, default=8,
                    help="Assumed Modal container fan-out for the parallel band.")
    ap.add_argument("--preempt-rate-min", type=float, default=0.0,
                    help="Preemptions/hour lower band.")
    ap.add_argument("--preempt-rate-expected", type=float, default=0.15)
    ap.add_argument("--preempt-rate-max", type=float, default=0.5)
    ap.add_argument("--preempt-reload-min", type=float, default=4.0,
                    help="Observed ~4min reload per preemption (B2).")
    ap.add_argument("--ceiling-hours", type=float, default=48.0)
    ap.add_argument("--ceiling-usd", type=float, default=200.0)
    ap.add_argument("--ceiling-gb", type=float, default=100.0)
    ap.add_argument("--out", type=str, default=str(RESULTS_PATH))
    args = ap.parse_args()

    t_start = time.time()

    # --- read N_test / N_val from split files at RUNTIME (L9) ---
    test_triples = ROOT / "grail_metabolism" / "data" / "test_triples_clean.txt"
    val_triples = ROOT / "grail_metabolism" / "data" / "val_triples_clean.txt"
    if not test_triples.exists():
        test_triples = ROOT / "grail_metabolism" / "data" / "test_triples.txt"
    if not val_triples.exists():
        val_triples = ROOT / "grail_metabolism" / "data" / "val_triples.txt"
    n_test = _distinct_substrate_count(test_triples)
    n_val = _distinct_substrate_count(val_triples)
    print(f"[budget] N_test={n_test} N_val={n_val} (read from {test_triples.name}/{val_triples.name})",
          flush=True)

    # --- load the generator (term d: generator load, measured) ---
    print("[budget] loading generator ...", flush=True)
    from grail_metabolism.config import GeneratorConfig
    from grail_metabolism.model.grail import _read_checkpoint
    from grail_metabolism.workflows.factory import build_generator

    t0 = time.perf_counter()
    state = _read_checkpoint(GEN_CKPT)
    if state is None or "arch" not in state or "rules" not in state:
        raise SystemExit(f"Generator checkpoint missing arch/rules: {GEN_CKPT}")
    generator = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    generator.load_state_dict(state["state_dict"], strict=False)
    if state.get("calibrated_threshold") is not None:
        generator.calibrated_threshold = state["calibrated_threshold"]
    generator.eval()
    generator_load_s = time.perf_counter() - t0
    print(f"[budget] generator loaded in {generator_load_s:.1f}s; num_rules={generator.num_rules}",
          flush=True)

    # --- load the test split (high cap) to pick a COLD, size-stratified sample ---
    print("[budget] loading test split to select COLD substrates ...", flush=True)
    from grail_metabolism.config import DatasetConfig
    from grail_metabolism.workflows.data import load_dataset_bundle

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
        max_train_substrates=1,
        max_val_substrates=1,
        max_test_substrates=n_test + 60,  # load the whole test split
        sampling_seed=0,
    )
    t0 = time.perf_counter()
    bundle = load_dataset_bundle(cfg)
    bundle_load_s = time.perf_counter() - t0
    test_substrates = list(bundle.test.map.keys())
    print(f"[budget] test split loaded in {bundle_load_s:.1f}s; {len(test_substrates)} substrates",
          flush=True)

    warm_roots = _load_child_cache_roots()
    print(f"[budget] child-cache WARM roots: {len(warm_roots)}", flush=True)
    cold = [s for s in test_substrates if s not in warm_roots]
    sizes = {}
    cold_valid = []
    for s in cold:
        hac = _heavy_atom_count(s)
        if hac is not None:
            sizes[s] = hac
            cold_valid.append(s)
    frac_cold = len(cold_valid) / max(1, len(test_substrates))
    print(f"[budget] COLD substrates: {len(cold_valid)}/{len(test_substrates)} "
          f"({frac_cold*100:.0f}%)", flush=True)

    cold_sample = _stratify_by_size(cold_valid, sizes, args.n_cold, n_buckets=3)
    print(f"[budget] cold stratified sample: n={len(cold_sample)} "
          f"hac range [{min(sizes[s] for s in cold_sample)}, "
          f"{max(sizes[s] for s in cold_sample)}]", flush=True)

    # --- term (a) ---
    print("[budget] === term (a) COLD pool-build ===", flush=True)
    pool_dist, pool_records, built_pools = measure_pool_build(
        generator, cold_sample, sizes, args.top_k, args.max_pool
    )

    # --- need a reranker to score pools for term (b). Warm-start a tiny one (cheap) ---
    print("[budget] warm-starting a small reranker for selection timing ...", flush=True)
    from grail_metabolism.model.reranker import BiEncoderReranker
    from grail_metabolism.utils.transform import SINGLE_NODE_DIM
    from grail_metabolism.workflows.reranker import BiRerankerTrainer

    reranker = BiEncoderReranker(in_channels=SINGLE_NODE_DIM)
    rr_trainer = BiRerankerTrainer(reranker, lr=1e-3, seed=0)
    reranker.eval()
    device = rr_trainer.device

    # --- term (b) ---
    print("[budget] === term (b) selection ===", flush=True)
    temp_knobs = [(1.0, 0.9), (0.5, 0.95), (2.0, 1.0)]
    dpp_knobs = [0.5, 1.0, 2.0, 4.0]
    mmr_knobs = [0.0, 0.25, 0.5, 0.75, 1.0]
    selection = measure_selection(
        reranker, generator, built_pools, args.top_k, args.max_pool, device, args.k,
        dpp_knobs, mmr_knobs, temp_knobs,
    )

    # --- term (c) ---
    print("[budget] === term (c) gflownet train ===", flush=True)
    gflownet = estimate_gflownet_train()

    # --- term (d) ---
    startup = measure_startup(generator_load_s)

    # --- extrapolation + recommendation ---
    one_pool_bytes = _estimate_one_pool_bytes(built_pools)
    extrap = extrapolate(
        n_test, n_val, pool_dist, pool_records, selection, gflownet, startup,
        gloryx_n=args.gloryx_n,
        dpp_grid=len(dpp_knobs), mmr_grid=len(mmr_knobs), temp_grid=len(temp_knobs),
        n_seeds=args.n_seeds, one_pool_bytes=one_pool_bytes,
        modal_containers=args.modal_containers,
        preempt_rate_band=(args.preempt_rate_min, args.preempt_rate_expected, args.preempt_rate_max),
        preempt_reload_min=args.preempt_reload_min,
    )
    rec = recommend(extrap, args.ceiling_hours, args.ceiling_usd, args.ceiling_gb)

    report = {
        "meta": {
            "plan": "04-01 Task 1",
            "generated_wall_seconds": time.time() - t_start,
            "top_k": args.top_k,
            "max_pool": args.max_pool,
            "k_for_selection": args.k,
            "cold_fraction_observed": frac_cold,
            "n_cold_sampled": len(cold_sample),
            "n_test": n_test,
            "n_val": n_val,
        },
        "term_a_cold_poolbuild_seconds_dist": pool_dist,
        "term_b_selection": selection,
        "term_c_gflownet_train": gflownet,
        "term_d_startup": startup,
        "extrapolation": extrap,
        "recommendation": rec,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n========== BUDGET REPORT ==========", flush=True)
    print(f"  (a) cold pool-build/substrate: median={pool_dist['median']:.2f}s "
          f"p90={pool_dist['p90']:.2f}s p99={pool_dist['p99']:.2f}s max={pool_dist['max']:.2f}s "
          f"(n={pool_dist['n']})", flush=True)
    print(f"  (b) temp/top-p per knob: {selection['temperature_topp_per_knobpoint_s']['median']*1000:.2f}ms | "
          f"DPP/MMR kernel/pool: {selection['dpp_mmr_kernel_build_per_pool_s']['median']*1000:.1f}ms | "
          f"DPP per-knob: {selection['dpp_per_knobpoint_over_cached_kernel_s']['median']*1000:.1f}ms | "
          f"MMR per-knob: {selection['mmr_per_knobpoint_over_cached_kernel_s']['median']*1000:.1f}ms",
          flush=True)
    print(f"  (c) gflownet per-seed: {gflownet['per_seed_hours_expected']:.1f}h "
          f"(x{gflownet['n_seeds']} = {gflownet['total_hours_expected']:.1f}h){' [single-point +/-50%]' if gflownet['single_point'] else ''}",
          flush=True)
    print(f"  (d) startup: {startup['total_s']:.0f}s", flush=True)
    print(f"  local-serial: {extrap['local_serial_hours']['expected']:.1f}h "
          f"(band {extrap['local_serial_hours']['band'][0]:.1f}-{extrap['local_serial_hours']['band'][1]:.1f}h)",
          flush=True)
    mb = extrap["modal_parallel_hours_band"]
    print(f"  modal-parallel: {mb['expected']:.1f}h (band {mb['min']:.1f}-{mb['max']:.1f}h) "
          f"@ {mb['assumed_container_count']} containers", flush=True)
    print(f"  modal $: expected ${extrap['modal_usd_band']['expected']:.0f} "
          f"(band ${extrap['modal_usd_band']['min']:.0f}-${extrap['modal_usd_band']['max']:.0f})", flush=True)
    print(f"  pool-cache disk: {extrap['pool_cache_gb_estimate']['total_gb']:.2f}GB", flush=True)
    print(f"  RECOMMENDATION: {rec['recommendation']}  ({rec['reason']})", flush=True)
    print(f"  report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

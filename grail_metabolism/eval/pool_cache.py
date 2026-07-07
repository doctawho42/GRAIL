"""Plan 04-01 Task 2: disk cache for the expensive per-substrate candidate Pool.

``build_pool`` (generator forward + rule application + tautomer-dedup) is the dominant
Phase-4 eval cost -- measured at ~5s/substrate COLD (median), heavy-tailed to ~20s
(``scripts/measure_eval_budget.py``). Every post-hoc baseline (temperature/top-p, DPP,
MMR) and every knob-point of their Pareto sweep re-selects from this SAME pool, so it
MUST be built once per (substrate, seed, config) and shared -- or the full-scale budget
collapses (D-40-02 prohibition).

This module caches ``build_pool``'s output to disk keyed by a config fingerprint over
``(substrate, seed, top_k, max_pool, generator-checkpoint)``, mirroring
``run_gflownet.py``'s ``--resume-eval-ckpt`` fingerprint discipline: a rebuild with a
changed config gets a different key and never silently reuses a stale pool.

NOTE on the kernel (deliberate deviation from the plan's "persist fps+kernel to disk"):
the lambda/theta-INDEPENDENT Tanimoto kernel that DPP/MMR reuse across a knob-sweep is
NOT disk-cached here, because it is score-DEPENDENT. ``eval/baselines.select`` tautomer-
dedups SCORE-AWARE (keeps the highest-*reranker*-scored representative), and tautomers
have DIFFERENT Morgan fingerprints, so which representative survives -- and hence the
kernel -- depends on the reranker checkpoint's scores, not on the substrate alone.
Disk-caching a kernel built under one score vector would corrupt selection under another.
The kernel is therefore built once per pool PER RUN and reused across knob-points by
``eval/baselines.dedup_and_fingerprint`` (Plan 04-01 Task 3) -- that is the load-bearing
kernel reuse. This module caches the far more expensive generator pool-build.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "pool_cache"

# (smiles, generator_score, rule_id) -- the candidate-pool entry shape build_pool emits.
PoolEntry = Tuple[str, float, int]


@dataclass
class CachedPool:
    """A built candidate pool plus the config fingerprint it was built under."""

    substrate: str
    pool: List[PoolEntry]
    config_key: str


def config_key(substrate: str, seed: int, top_k: int, max_pool: int, gen_ckpt: str) -> str:
    """Deterministic 16-hex config fingerprint. Any change to the substrate, seed, top_k,
    max_pool, or generator checkpoint identity yields a different key (cache MISS), so a
    stale pool from a different config is never silently reused (D-40 fingerprint discipline)."""
    payload = "|".join(
        [substrate, f"seed={seed}", f"top_k={top_k}", f"max_pool={max_pool}", f"gen={gen_ckpt}"]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_or_load_pool(
    substrate: str,
    seed: int,
    generator,
    *,
    top_k: int = 200,
    max_pool: int = 100,
    cache_dir: Optional[Path] = None,
    gen_ckpt: str = "",
    force: bool = False,
    build_pool_fn: Optional[Callable] = None,
) -> CachedPool:
    """Return the candidate pool for ``substrate``, building it once and caching to disk.

    On a cache HIT (a pickle keyed by the config fingerprint exists) the pool is loaded
    without invoking the generator. On a MISS (or ``force=True``, or a corrupt/key-
    mismatched file) it calls ``build_pool_fn(generator, substrate, top_k, max_pool)``
    (defaulting to ``workflows.reranker.build_pool``), persists the result, and returns it.

    ``build_pool_fn`` is injectable so dataset-free guard tests can supply a fake builder
    without importing the torch-heavy reranker stack.
    """
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    key = config_key(substrate, seed, top_k, max_pool, gen_ckpt)
    path = cdir / f"pool_{key}.pkl"

    if path.exists() and not force:
        try:
            with open(path, "rb") as fh:
                cached = pickle.load(fh)
            if isinstance(cached, CachedPool) and cached.config_key == key:
                return cached
        except Exception:
            pass  # corrupt cache file -> fall through and rebuild

    if build_pool_fn is None:  # lazy import: keeps the module light for tests
        from grail_metabolism.workflows.reranker import build_pool as build_pool_fn

    pool = list(build_pool_fn(generator, substrate, top_k=top_k, max_pool=max_pool))
    cp = CachedPool(substrate=substrate, pool=pool, config_key=key)
    cdir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(cp, fh)
    return cp

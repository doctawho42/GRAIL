"""Modal app: GRAIL Stage-2b Set-Reward Novelty Ablation (Phase 3, 03-03) -- PARALLEL.

Un-defers the Modal orchestration task that ``03-03-PLAN.md`` originally specified but
could not build (Modal was unavailable at the time; ``scripts/run_ablation_local.py``
shipped as a LOCAL sequential fallback instead). Modal is now available, so this fans
the ablation's ~8 INDEPENDENT training configs (the beta-prime VAL sweep + the VAL
seed runs + the single test touch) out across Modal containers IN PARALLEL, instead of
``run_ablation_local.py``'s one-container ``for config in configs:`` loop.

Reuses, unchanged:
  - ``grail_metabolism/ablation_plan.py`` for ALL planning/selection/verdict logic
    (``plan_configs``, ``select_beta_prime``, ``aggregate_and_verdict``) -- the exact
    same module ``run_ablation_local.py`` uses, so both callers agree on config shape,
    beta-prime selection rule, and verdict computation. This file adds ZERO new
    science; it only adds a parallel execution strategy over an existing config list.
  - ``scripts/run_gflownet.py --ablation-mode {single,ensemble}`` as the per-config
    unit (unchanged CLI, unchanged single-variable-pure construction, unchanged
    checkpoint contract).
  - ``scripts/modal_m2.py``'s image, Volumes (``grail-data``/``grail-artifacts``), and
    ``_link_data`` symlink helper (imported by path, not duplicated).

Why safe to parallelize (unlike substrate-level fan-out, which D-08 forbids): every
config in a WAVE (the beta-prime sweep configs; the VAL seed-run configs) is trained
and evaluated INDEPENDENTLY -- different seed and/or different beta_prime, but each
config does its OWN full train+eval in its OWN container, writing ONLY its own
config-keyed result JSON. The environment cache (``gfn_child_cache_k{top_k}.pkl`` /
``gfn_ik_cache.pkl``, deterministic in substrate+rule+top_k, NOT seed/beta_prime -- see
``run_gflownet.py``'s comment above ``child_cache_path``) is built ONCE by a PREWARM
step that runs BEFORE any parallel config, and is only ever READ (not written new
entries needing merge) by the parallel configs after that -- so there is no
Volume-write race between concurrent containers. (``SetGFlowNetTrainer.save_caches()``
inside each parallel config's own subprocess call may still ADD entries lazily
discovered during that config's own training/eval, same as ``run_m2``'s sequential
seeds already do; the risk here is bounded to lost cache-growth from one racing
writer, never corrupted structure or a wrong scientific result, and is priced in as
"some cache growth may not persist" the same way it already is for the existing
Modal M2 app's between-seed behavior).

Waves (sequential; each wave's OWN configs run in PARALLEL via ``.map()``):
  0. PREWARM (barrier): one single tiny/no-op config-independent call that ensures the
     shared env cache exists before any real config starts. Reuses the SAME "run one
     config" function at trivial scale so no separate code path is needed.
  1. beta-prime VAL sweep -- ``run_one_config.map(sweep_configs)`` (parallel).
  2. ``select_beta_prime`` on the sweep results (LOCAL, cheap, from the local
     entrypoint -- no Modal call).
  3. VAL seed runs (ABL-01 + ABL-02) at the chosen beta-prime --
     ``run_one_config.map(val_configs)`` (parallel).
  4. The single test-touch pair (ABL-01 + ABL-02 test) -- ``run_one_config.map(...)``
     (parallel; only 2 configs, but kept on the same fan-out path for uniformity).
  5. Download the per-config result + eval-checkpoint JSONs from the Volume, then
     ``aggregate_and_verdict`` (LOCAL, from the local entrypoint) -- the exact function
     ``run_ablation_local.py`` also calls.

Each config is resumable + idempotent exactly like ``modal_m2.py::run_m2``'s per-seed
loop: config-keyed ``--out``/``--resume-ckpt``/``--resume-eval-ckpt`` paths on the
``grail-artifacts`` Volume, ``if os.path.exists(out): skip`` before running, and an
``art_vol.commit()`` after every unit so a preempted/killed container's completed
configs are never lost on retry.

------------------------------------------------------------------------------------
ONE-TIME SETUP: already done (Modal authed; ``grail-data``/``grail-artifacts`` exist
and hold the staged SDFs/triples + generator/filter checkpoints + the env cache).

TINY SMOKE (proves the parallel path end-to-end on real data, minutes not hours):

    modal run scripts/modal_ablation.py::smoke

FULL RUN (fully unattended, detached; survives local disconnect) -- launched by the
orchestrator, NOT by this executor:

    modal run --detach scripts/modal_ablation.py::run_ablation \\
        --train-substrates 300 --test-substrates 100 --epochs 15 --m-ensemble 3 \\
        --beta-prime-grid 2 4 6 8 10 --seeds 0 1 2

FETCH the verdict report when done:

    modal volume get grail-artifacts /ablation_modal/verdict_report.json results/
------------------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reused, unchanged, from the existing M2 Modal app -- same base image (git clone of
# the pushed `metabench-reranker` branch + pins), same Volumes, same data-symlink
# helper. Imported (not copy-pasted) so a change to the base image/pins only has to
# happen in one place.
from scripts.modal_m2 import DATA_FILES, DATA_MOUNT
from scripts.modal_m2 import image as _base_image  # noqa: E402

# LAYERED OVERLAY (build-time copy, not a runtime mount): the base image's git clone
# only has whatever was last PUSHED to `metabench-reranker` on GitHub -- it does not
# see this worktree's uncommitted-to-that-branch code (ablation_plan.py,
# scripts/modal_ablation.py itself, or any other commit still local to this branch).
# `add_local_dir(..., copy=True)` bakes the CURRENT local repo tree on top of the
# cloned checkout at build time, so the container always runs the code actually being
# edited/tested here -- no push to a shared branch required for iteration or smoke
# runs. copy=True (not the default mount-at-startup) because subsequent `run_commands`
# would not be reachable from a startup-time mount overlay in the same way, and this
# way `pip install -e .` (already baked into `_base_image`) plus the overlaid source
# stay consistent inside one built image layer.
#
# Overlay ONLY the two source directories the ablation path actually needs
# (grail_metabolism/, scripts/) -- NOT the whole worktree root. The worktree root
# also holds large/irrelevant/volatile local-only directories (artifacts/, results/,
# .git, .idea, .pytest_cache, editor/tool state dirs) that are unnecessary to ship
# and, worse, can be mutated by other local processes DURING the Modal build
# (Modal snapshots the tree and errors if a file changes mid-build) -- scoping the
# overlay to just the source dirs sidesteps both problems.
image = (
    _base_image
    .add_local_dir(
        str(ROOT / "grail_metabolism"), "/root/GRAIL/grail_metabolism",
        copy=True,
        ignore=["**/__pycache__/**", "*.pyc", "data/**"],
    )
    .add_local_dir(
        str(ROOT / "scripts"), "/root/GRAIL/scripts",
        copy=True,
        ignore=["**/__pycache__/**", "*.pyc"],
    )
)

from grail_metabolism.ablation_plan import (  # noqa: E402
    DEFAULT_BETA_PRIME_GRID,
    DEFAULT_M_ENSEMBLE,
    DEFAULT_SEEDS,
    aggregate_and_verdict,
    fill_beta_prime,
    plan_configs,
    select_beta_prime,
    sweep_scores_from_results,
)

app = modal.App("grail-ablation")
data_vol = modal.Volume.from_name("grail-data", create_if_missing=True)
art_vol = modal.Volume.from_name("grail-artifacts", create_if_missing=True)

# Config-keyed artifacts live under their own subdirectory on the Volume so they never
# collide with modal_m2.py's gflownet_m2_* files (same Volume, different namespace).
ARTIFACTS_SUBDIR = "ablation_modal"


def _link_data():
    """Symlink the Volume's SDFs + triples into grail_metabolism/data/ (idempotent).
    Byte-identical to ``modal_m2.py``'s ``_link_data`` -- duplicated (not imported)
    ONLY because it must run inside the container after ``os.chdir``; the import
    above already gives us the constants it needs (``DATA_MOUNT``/``DATA_FILES``)."""
    import os

    dst_dir = "/root/GRAIL/grail_metabolism/data"
    for name in DATA_FILES:
        src, dst = f"{DATA_MOUNT}/{name}", f"{dst_dir}/{name}"
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)


def _fixed_args(
    train_substrates: int, test_substrates: int, epochs: int, prewarm_waves: int,
    eval_beam: bool, top_k: int, max_size: int, max_depth: int, workers: int,
    logz_lr: float, n_samples: int,
) -> List[str]:
    """The fixed (non-varying) CLI args shared by EVERY config in this orchestrator.
    Verbatim shape of ``run_ablation_local.py``'s ``_fixed_args`` (kept local, not
    imported, since it is pure CLI-arg-list construction, not scientific logic --
    importing across the local/Modal boundary here would gain nothing and would
    couple this file's argv shape to the local script's internal helper name)."""
    args = [
        "--train-substrates", str(train_substrates),
        "--test-substrates", str(test_substrates),
        "--max-depth", str(max_depth),
        "--max-size", str(max_size),
        "--epochs", str(epochs),
        "--top-k", str(top_k),
        "--logz-lr", str(logz_lr),
        "--n-samples", str(n_samples),
        "--workers", str(workers),
        "--prewarm-waves", str(prewarm_waves),
        "--no-bootstrap",
    ]
    if not eval_beam:
        args.append("--no-eval-beam")
    return args


def _out_path(tag: str) -> str:
    return f"artifacts/{ARTIFACTS_SUBDIR}/{tag}.json"


def _ckpt_paths(tag: str) -> "tuple[str, str]":
    base = f"artifacts/{ARTIFACTS_SUBDIR}/{tag}"
    return f"{base}.ckpt.pt", f"{base}.eval_ckpt.json"


@app.function(
    image=image,
    gpu=None,     # CPU-bound RDKit rule-application + tautomer canonicalization (see modal_m2.py)
    cpu=8.0,
    memory=32768,
    volumes={
        DATA_MOUNT: data_vol,
        "/root/GRAIL/artifacts": art_vol,
    },
    timeout=86400,   # 24h per config; per-epoch/per-substrate checkpoints make a preemption cheap
)
def run_one_config(config: Dict[str, object], fixed_args: List[str]) -> Dict[str, object]:
    """Run ONE independent ablation config end-to-end in its own container: train +
    dual-eval via ``scripts/run_gflownet.py --ablation-mode {single,ensemble}``,
    writing its config-keyed result JSON (+ training/eval checkpoints) to the shared
    ``grail-artifacts`` Volume. Idempotent (skips if the result JSON already exists,
    mirroring ``modal_m2.py::run_m2``) and resumable (per-epoch ``--resume-ckpt`` +
    per-substrate ``--resume-eval-ckpt``, unchanged from ``run_gflownet.py``).

    Returns ``{"tag": ..., "result": <parsed result JSON>}`` so the caller (the local
    entrypoint, reading ``.map()``'s return values) never has to separately download
    the result JSON just to read scalar metrics -- though it DOES still need to
    download the eval-checkpoint JSON separately for the per-substrate paired arrays
    (see ``run_ablation`` below).
    """
    import os
    import subprocess

    os.chdir("/root/GRAIL")
    _link_data()

    tag = str(config["tag"])
    out = _out_path(tag)
    ckpt, eval_ckpt = _ckpt_paths(tag)

    if os.path.exists(out):
        print(f"[modal_ablation] {tag}: already complete ({out}) -- skipping", flush=True)
        with open(out) as fh:
            return {"tag": tag, "result": json.load(fh)}

    cmd = [
        sys.executable, "-u", "scripts/run_gflownet.py", *fixed_args,
        "--seed", str(config["seed"]),
        "--eval-split", str(config["eval_split"]),
        "--ablation-mode", str(config["mode"]),
        "--beta-prime", str(config["beta_prime"]),
        "--m-ensemble", str(config["m_ensemble"]),
        "--out", out,
        "--resume-ckpt", ckpt,
        "--resume-eval-ckpt", eval_ckpt,
    ]
    if config.get("eval_substrates") is not None:
        cmd += ["--eval-substrates", str(config["eval_substrates"])]

    print(f"\n===== modal_ablation: {tag} =====\n{' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)
    art_vol.commit()   # persist this config's result + checkpoints before returning
    print(f"===== {tag} done, artifacts committed =====", flush=True)

    with open(out) as fh:
        return {"tag": tag, "result": json.load(fh)}


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,
    volumes={DATA_MOUNT: data_vol, "/root/GRAIL/artifacts": art_vol},
    timeout=3600,
)
def prewarm() -> str:
    """Barrier step: build/reuse the shared environment cache (child-cache + ik-cache,
    keyed by substrate+rule-bank+top_k, NOT seed/beta_prime) ONCE before any parallel
    config runs, so every subsequent parallel config reads a warm, complete cache
    instead of racing to build it. Runs a trivial ``--ablation-mode off`` single-epoch
    pass at the SAME ``top_k``/``max_depth``/``max_size`` the real configs use (so the
    cache keys match) but at a tiny substrate count -- cheap, and its own env-cache
    writes are exactly the ones the real configs will extend, not replace.

    This reuses ``run_gflownet.py`` unchanged (no separate prewarm-only code path in
    the trained pipeline) -- it is simply a small, throwaway, ablation_mode=off run
    whose OWN result JSON is discarded (written to a scratch path, never read by
    ``run_ablation``); only its cache-file side effects on the Volume matter.
    """
    import os
    import subprocess

    os.chdir("/root/GRAIL")
    _link_data()
    scratch_out = f"artifacts/{ARTIFACTS_SUBDIR}/_prewarm_scratch.json"
    if os.path.exists(scratch_out):
        print("[modal_ablation] prewarm: cache already warmed (scratch marker present) -- skipping", flush=True)
        return "prewarm: skipped (already warm)"
    cmd = [
        sys.executable, "-u", "scripts/run_gflownet.py",
        "--train-substrates", "20", "--eval-substrates", "5",
        "--max-depth", "2", "--max-size", "10", "--epochs", "1", "--top-k", "50",
        "--n-samples", "2", "--workers", "8", "--prewarm-waves", "1",
        "--no-bootstrap", "--no-eval-beam", "--eval-split", "val",
        "--out", scratch_out,
    ]
    print(f"[modal_ablation] prewarm: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    art_vol.commit()
    print("[modal_ablation] prewarm: cache warmed + committed", flush=True)
    return "prewarm: done"


def _download_json(remote_path: str) -> Optional[dict]:
    """Read a JSON file already materialized on the LOCAL filesystem via
    ``art_vol`` reload + a local-entrypoint-side ``modal volume get``-equivalent.

    The local entrypoint runs OUTSIDE any container, so it cannot open Volume paths
    directly -- it must call a tiny Modal function that reads the file INSIDE a
    container (where the Volume is mounted) and returns the parsed JSON over the
    wire. Returns ``None`` if the file does not exist (e.g. a config's result JSON
    when running a small/partial smoke)."""
    return _read_artifact_json.remote(remote_path)


@app.function(
    image=image,
    volumes={"/root/GRAIL/artifacts": art_vol},
    timeout=300,
)
def _read_artifact_json(path: str) -> Optional[dict]:
    import os

    full = f"/root/GRAIL/{path}"
    if not os.path.exists(full):
        return None
    with open(full) as fh:
        return json.load(fh)


def _run_wave(configs: Sequence[Dict[str, object]], fixed_args: List[str]) -> Dict[str, dict]:
    """Fan a wave of INDEPENDENT configs out across Modal containers in PARALLEL via
    ``.map()``, collecting each config's parsed result JSON keyed by its ``tag``.
    This is the ONE place ``.map()`` is used across configs (never across substrates
    within one eval -- that stays inside ``run_gflownet.py``'s own eval loop,
    unchanged, per D-08)."""
    results: Dict[str, dict] = {}
    for out in run_one_config.map(configs, kwargs={"fixed_args": fixed_args}):
        results[out["tag"]] = out["result"]
    return results


@app.local_entrypoint()
def run_ablation(
    train_substrates: int = 300,
    test_substrates: int = 100,
    epochs: int = 15,
    m_ensemble: int = DEFAULT_M_ENSEMBLE,
    beta_prime_grid: str = "2,4,6,8,10",
    seeds: str = "0,1,2",
    top_k: int = 50,
    max_size: int = 10,
    max_depth: int = 2,
    workers: int = 8,
    logz_lr: float = 0.16,
    n_samples: int = 4,
    prewarm_waves: int = 1,
    eval_beam: bool = False,
    eval_substrates: Optional[int] = None,
):
    """PARALLEL ablation orchestration: prewarm (barrier) -> beta-prime VAL sweep
    (parallel) -> select beta_prime (local) -> VAL seed runs (parallel) -> test touch
    (parallel) -> download + aggregate_and_verdict (local). See module docstring for
    the full wave breakdown and the safety argument for why cross-config fan-out
    (unlike cross-substrate fan-out) is safe here.

    ``beta_prime_grid``/``seeds`` are comma-separated strings (Modal local-entrypoint
    args are simple scalars; parsed here into the ``Sequence[float]``/``Sequence[int]``
    ``ablation_plan.plan_configs`` expects).
    """
    grid = [float(x) for x in beta_prime_grid.split(",")]
    seed_list = [int(x) for x in seeds.split(",")]

    fixed_args = _fixed_args(
        train_substrates=train_substrates, test_substrates=test_substrates, epochs=epochs,
        prewarm_waves=prewarm_waves, eval_beam=eval_beam, top_k=top_k, max_size=max_size,
        max_depth=max_depth, workers=workers, logz_lr=logz_lr, n_samples=n_samples,
    )

    plan = plan_configs(
        beta_prime_grid=grid, seeds=seed_list, m_ensemble=m_ensemble, eval_substrates=eval_substrates,
    )

    print("[modal_ablation] === Wave 0: prewarm (barrier) ===", flush=True)
    print(prewarm.remote(), flush=True)

    print(f"[modal_ablation] === Wave 1: beta-prime VAL sweep ({len(plan['sweep'])} configs, parallel) ===", flush=True)
    sweep_results = _run_wave(plan["sweep"], fixed_args)
    sweep_scores = sweep_scores_from_results(plan["sweep"], sweep_results)
    print(f"[modal_ablation] sweep scores: {sweep_scores}", flush=True)
    chosen_beta_prime = select_beta_prime(sweep_scores)
    print(f"[modal_ablation] chosen beta_prime={chosen_beta_prime}", flush=True)

    val_configs = fill_beta_prime(plan["val"], chosen_beta_prime)
    print(f"[modal_ablation] === Wave 2: VAL seed runs ({len(val_configs)} configs, parallel) ===", flush=True)
    val_results = _run_wave(val_configs, fixed_args)

    val_single_results = [val_results[c["tag"]] for c in val_configs if c["mode"] == "single"]
    val_ensemble_results = [val_results[c["tag"]] for c in val_configs if c["mode"] == "ensemble"]

    test_configs = fill_beta_prime(plan["test"], chosen_beta_prime)
    print(f"[modal_ablation] === Wave 3: TEST touch ({len(test_configs)} configs, parallel) ===", flush=True)
    test_results = _run_wave(test_configs, fixed_args)

    test_single_tag = next(c["tag"] for c in test_configs if c["mode"] == "single")
    test_ensemble_tag = next(c["tag"] for c in test_configs if c["mode"] == "ensemble")
    test_single_result = test_results[test_single_tag]
    test_ensemble_result = test_results[test_ensemble_tag]

    print("[modal_ablation] === downloading eval checkpoints for paired-bootstrap arrays ===", flush=True)
    _, test_single_eval_ckpt_path = _ckpt_paths(test_single_tag)
    _, test_ensemble_eval_ckpt_path = _ckpt_paths(test_ensemble_tag)
    test_single_eval_ckpt = _download_json(test_single_eval_ckpt_path)
    test_ensemble_eval_ckpt = _download_json(test_ensemble_eval_ckpt_path)
    if test_single_eval_ckpt is None or test_ensemble_eval_ckpt is None:
        raise RuntimeError(
            "modal_ablation: test-touch eval checkpoints missing on the Volume -- "
            f"single={test_single_eval_ckpt_path} ensemble={test_ensemble_eval_ckpt_path}"
        )

    print("[modal_ablation] === aggregate_and_verdict ===", flush=True)
    report = aggregate_and_verdict(
        val_single_results=val_single_results,
        val_ensemble_results=val_ensemble_results,
        test_single_result=test_single_result,
        test_ensemble_result=test_ensemble_result,
        test_single_eval_ckpt=test_single_eval_ckpt,
        test_ensemble_eval_ckpt=test_ensemble_eval_ckpt,
        chosen_beta_prime=chosen_beta_prime,
        sweep_scores=sweep_scores,
        m_ensemble=m_ensemble,
    )

    print("\n========== ABL-03 VERDICT (Modal parallel) ==========", flush=True)
    print(json.dumps(report, indent=2), flush=True)

    report_path = f"artifacts/{ARTIFACTS_SUBDIR}/verdict_report.json"
    _write_verdict.remote(report_path, report)
    print(f"\n[modal_ablation] verdict report -> Volume:/{report_path}", flush=True)


@app.function(
    image=image,
    volumes={"/root/GRAIL/artifacts": art_vol},
    timeout=300,
)
def _write_verdict(path: str, report: dict) -> None:
    import os

    full = f"/root/GRAIL/{path}"
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as fh:
        json.dump(report, fh, indent=2)
    art_vol.commit()


@app.local_entrypoint()
def smoke():
    """TINY end-to-end smoke on REAL data: 2 configs (1 beta-prime sweep point on VAL
    at trivial scale) run in PARALLEL via ``.map()``, proving the parallel-fan-out path
    works before committing to the full ~8-config run. Cost-conscious: n_train~5,
    n_eval~3, 1 epoch, 1 beta-prime, seed 0 only, no prewarm barrier needed at this
    trivial scale (each config builds its own tiny cache; the point of this smoke is
    the PARALLEL PATH, not cache-sharing, which the full run's prewarm barrier covers).
    """
    fixed_args = _fixed_args(
        train_substrates=5, test_substrates=3, epochs=1, prewarm_waves=1,
        eval_beam=False, top_k=20, max_size=6, max_depth=2, workers=4,
        logz_lr=0.1, n_samples=2,
    )
    plan = plan_configs(beta_prime_grid=[6.0], seeds=[0], m_ensemble=2, eval_substrates=3)
    configs = fill_beta_prime(plan["val"], 6.0)   # both single + ensemble VAL configs at seed 0
    print(f"[modal_ablation] SMOKE: running {len(configs)} configs in parallel ...", flush=True)
    results = _run_wave(configs, fixed_args)
    for tag, result in results.items():
        print(f"[modal_ablation] SMOKE {tag}: metrics keys={sorted(result['metrics'].keys())}", flush=True)
    print("SMOKE OK -- parallel fan-out produced a result JSON per config on the Volume.", flush=True)

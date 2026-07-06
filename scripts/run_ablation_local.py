#!/usr/bin/env python3
"""LOCAL sequential ablation runner (Phase 3, set-reward novelty ablation) -- NO Modal.

Modal is unavailable, so this mirrors ``scripts/modal_m2.py::run_m2``'s per-seed loop
SHAPE (sequential subprocess invocations of ``scripts/run_gflownet.py``, seed-keyed
``--out``/``--resume-ckpt``/``--resume-eval-ckpt`` filenames, idempotent already-complete
skip) but drives it on the local CPU with plain ``subprocess.run`` instead of a Modal
``@app.function``. No ``.map()``/``.starmap()`` fan-out; everything runs sequentially in
this one process, exactly like ``run_m2``'s ``for seed in seeds:`` loop.

What it does, in order:
  1. beta-prime VAL sweep (D-10/FIX D): runs ``--ablation-mode single`` at each beta-prime
     in ``--beta-prime-grid`` on VAL, picks the value maximizing ablation01's own
     ``ablation01_union_at_k_auc``. Cheap -- ABL-01 is "free beyond eval" (RESEARCH:351-356).
  2. ABL-01 (independent single-terminal) + the set-GFlowNet reference: for each seed in
     ``--seeds``, ONE ``run_gflownet.py --ablation-mode single --beta-prime <chosen>``
     invocation gives BOTH ``gflownet_union_at_k_auc`` (the shared-adaptive-loop reference
     arm) AND ``ablation01_union_at_k_auc`` in the SAME results JSON (evaluate_matrix
     computes the gflownet arm's own union stream via the SAME shared adaptive loop
     whenever ablation_mode != "off" -- see run_gflownet.py:435-449).
  3. ABL-02 (ensemble): for each seed in ``--seeds``, ONE
     ``run_gflownet.py --ablation-mode ensemble --beta-prime <chosen> --m-ensemble <M>``
     invocation (run_gflownet.py internally trains M members seeded ``seed*1000+m`` and
     round-robins draws across them -- no separate per-member subprocess is needed here).
  4. Selects on VAL, then touches TEST exactly once (one more triple of invocations at
     ``--eval-split test``), gated by ``assert_config_match`` (FIX C) before accepting the
     three-way test table.
  5. Computes BOTH verdict views: the PRIMARY paired-per-substrate bootstrap CI
     (``paired_bootstrap_delta_ci``, reading the per-substrate curves out of the
     ``--resume-eval-ckpt`` JSON's ``rows`` -- the mean-only ``result["metrics"]`` scalars
     do not carry per-substrate arrays) and the SECONDARY seed-level
     ``compute_ablation_verdict`` + a Delta sensitivity grid.

Resume-safety: every invocation reuses ``run_gflownet.py``'s own two checkpoint
mechanisms unchanged -- ``--resume-ckpt`` (per-epoch training resume) and
``--resume-eval-ckpt`` (per-substrate eval resume) -- plus this script's own idempotent
"skip if the seed-keyed results JSON already exists" check (mirroring
``modal_m2.py::run_m2``'s ``if os.path.exists(out): skip``), so an interrupted local run
resumes without re-doing completed work.

The planning/selection/verdict logic (beta-prime scoring, margin/sensitivity-grid
computation, per-substrate paired-array extraction, final report assembly) now lives in
``grail_metabolism/ablation_plan.py`` -- a shared, dataset-free, Modal-import-free
module also used by ``scripts/modal_ablation.py`` (the PARALLEL Modal orchestrator that
un-defers the Modal task this script's docstring used to say was unavailable). This
script keeps its own path/subprocess/CLI plumbing (the LOCAL sequential execution
strategy); only the pure arithmetic was factored out. Behavior is unchanged.

Usage (FULL scale -- see 03-03-SUMMARY.md for the exact copy-pasteable command):
  python scripts/run_ablation_local.py --train-substrates 300 --test-substrates 100 \\
      --epochs 15 --m-ensemble 3 --beta-prime-grid 2 4 6 8 10 --seeds 0 1 2

Usage (TINY smoke, real data, minutes not hours):
  python scripts/run_ablation_local.py --train-substrates 5 --test-substrates 3 \\
      --epochs 1 --m-ensemble 2 --beta-prime-grid 6 --seeds 0 --prewarm-waves 1 \\
      --no-eval-beam --eval-split val
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
RUN_GFLOWNET = ROOT / "scripts" / "run_gflownet.py"

sys.path.insert(0, str(ROOT))

from grail_metabolism.ablation_plan import (  # noqa: E402
    K_MAX,
    KS,
    compute_delta_sensitivity_grid,
    degeneracy_guarded_margin,
    mean_std,
    per_substrate_aucs as _per_substrate_aucs_from_dict,
    pick_beta_prime,
)
from grail_metabolism.eval.diversity import (  # noqa: E402
    assert_config_match,
    paired_bootstrap_delta_ci,
)

DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "ablation_local"


def _fixed_args(
    train_substrates: int, test_substrates: int, epochs: int, prewarm_waves: int,
    eval_beam: bool, top_k: int, max_size: int, max_depth: int, workers: int,
    logz_lr: float, n_samples: int,
) -> List[str]:
    """The fixed (non-varying) CLI args shared by EVERY invocation in this runner. Built as
    a plain Python list -- never shell-interpolated -- so no untrusted string ever reaches
    a shell (subprocess.run is called with a list, never with a shell wrapper, anywhere
    in this file).
    """
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


def _invoke(
    cmd: List[str], out_path: Path, label: str, dry_run: bool = False,
) -> Optional[dict]:
    """Run ONE run_gflownet.py subprocess, idempotent-skip if ``out_path`` already exists
    (mirrors ``modal_m2.py::run_m2``'s ``if os.path.exists(out): skip``). Returns the
    parsed results JSON (loaded from ``out_path`` either way -- freshly written or
    pre-existing)."""
    if out_path.exists():
        print(f"[run_ablation_local] {label}: already complete ({out_path}) -- skipping", flush=True)
    else:
        if dry_run:
            print(f"[run_ablation_local] DRY-RUN {label}: {' '.join(cmd)}", flush=True)
            return None
        print(f"\n===== run_ablation_local: {label} =====\n{' '.join(cmd)}\n", flush=True)
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        print(f"[run_ablation_local] {label} done in {time.time() - t0:.1f}s", flush=True)
    if dry_run and not out_path.exists():
        return None
    with open(out_path) as fh:
        return json.load(fh)


def _out_path(artifacts_dir: Path, tag: str) -> Path:
    return artifacts_dir / f"{tag}.json"


def _ckpt_paths(artifacts_dir: Path, tag: str) -> tuple:
    ckpt = artifacts_dir / f"{tag}.ckpt.pt"
    eval_ckpt = artifacts_dir / f"{tag}.eval_ckpt.json"
    return ckpt, eval_ckpt


def _run_ablation_mode(
    mode: str, seed: int, beta_prime: float, m_ensemble: int, eval_split: str,
    artifacts_dir: Path, fixed: Sequence[str], eval_substrates: Optional[int] = None,
    dry_run: bool = False,
) -> Optional[dict]:
    """One ``run_gflownet.py --ablation-mode {single,ensemble}`` invocation, seed-keyed
    out/ckpt/eval-ckpt filenames (mirrors ``modal_m2.py``'s ablation02_seed{seed} naming,
    generalized to both ablation modes and both splits)."""
    tag = f"ablation_{mode}_{eval_split}_seed{seed}"
    out = _out_path(artifacts_dir, tag)
    ckpt, eval_ckpt = _ckpt_paths(artifacts_dir, tag)
    cmd = [
        sys.executable, "-u", str(RUN_GFLOWNET), *fixed,
        "--seed", str(seed),
        "--eval-split", eval_split,
        "--ablation-mode", mode,
        "--beta-prime", str(beta_prime),
        "--m-ensemble", str(m_ensemble),
        "--out", str(out),
        "--resume-ckpt", str(ckpt),
        "--resume-eval-ckpt", str(eval_ckpt),
    ]
    if eval_substrates is not None:
        cmd += ["--eval-substrates", str(eval_substrates)]
    return _invoke(cmd, out, f"{mode} seed={seed} beta_prime={beta_prime} split={eval_split}", dry_run=dry_run)


def sweep_beta_prime(
    grid: Sequence[float], seed: int, m_ensemble: int, artifacts_dir: Path,
    fixed: Sequence[str], eval_substrates: Optional[int], dry_run: bool = False,
) -> Dict[float, Optional[float]]:
    """D-10/FIX D: VAL sweep over the full beta-prime grid using the CHEAP ablation01
    (single) mode -- reuses the shared env caches, only the eval-side forward cost varies
    per RESEARCH:351-356. Returns {beta_prime: ablation01_union_at_k_auc or None}."""
    scores: Dict[float, Optional[float]] = {}
    for bp in grid:
        result = _run_ablation_mode(
            "single", seed=seed, beta_prime=bp, m_ensemble=m_ensemble, eval_split="val",
            artifacts_dir=artifacts_dir, fixed=fixed, eval_substrates=eval_substrates,
            dry_run=dry_run,
        )
        if result is None:
            scores[bp] = None
            continue
        scores[bp] = result["metrics"].get("ablation01_union_at_k_auc")
    return scores


# pick_beta_prime, compute_delta_sensitivity_grid, degeneracy_guarded_margin are now
# imported from grail_metabolism.ablation_plan (shared with scripts/modal_ablation.py).
# _mean_std is replaced by the imported mean_std below.
_mean_std = mean_std


def paired_arrays(
    gflownet_eval_ckpt: Path, abl_eval_ckpt: Path, abl_series: str,
) -> "tuple[List[float], List[float]]":
    """Build matched (gflownet_i, abl_i) per-substrate AUC arrays over the shared root
    intersection between two eval checkpoints (both eval checkpoints come from the SAME
    seed's test-touch run in this runner's design, so the shared-substrate-set
    restriction from D-04b already applies within one run; when the two arms come from
    DIFFERENT seed invocations, this additionally intersects by root SMILES so pairing
    stays valid).

    Reads the two JSON files from disk then delegates the pure per-substrate-AUC
    extraction to ``grail_metabolism.ablation_plan.per_substrate_aucs`` (the shared,
    I/O-free function also used by ``scripts/modal_ablation.py``, which reads its
    eval checkpoints off a downloaded Modal Volume snapshot instead of local disk)."""
    with open(gflownet_eval_ckpt) as fh:
        gflownet_ckpt = json.load(fh)
    with open(abl_eval_ckpt) as fh:
        abl_ckpt = json.load(fh) if abl_eval_ckpt != gflownet_eval_ckpt else gflownet_ckpt
    gflownet_aucs = _per_substrate_aucs_from_dict(gflownet_ckpt, "gflownet")
    abl_aucs = _per_substrate_aucs_from_dict(abl_ckpt, abl_series)
    shared_roots = sorted(set(gflownet_aucs) & set(abl_aucs))
    return [gflownet_aucs[r] for r in shared_roots], [abl_aucs[r] for r in shared_roots]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LOCAL sequential set-reward novelty ablation runner (no Modal)."
    )
    parser.add_argument("--train-substrates", type=int, default=300)
    parser.add_argument("--test-substrates", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--m-ensemble", type=int, default=3)
    parser.add_argument("--beta-prime-grid", type=float, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-split", choices=["val", "test"], default="val",
                         help="Split used for the beta-prime sweep + the seed runs BEFORE the "
                              "single final test-touch (which this script always runs once, "
                              "separately, at the end, regardless of this flag).")
    parser.add_argument("--eval-substrates", type=int, default=None,
                         help="Cap on VAL substrates evaluated per run (ignored for test).")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-size", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--logz-lr", type=float, default=0.16)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--prewarm-waves", type=int, default=1, choices=(1, 2))
    parser.add_argument("--no-eval-beam", dest="eval_beam", action="store_false")
    parser.set_defaults(eval_beam=True)
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--skip-test-touch", action="store_true",
                         help="Stop after VAL selection (do not touch test). For smoke runs "
                              "where --eval-split is already val, this is redundant.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the commands that would run without executing them.")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fixed = _fixed_args(
        train_substrates=args.train_substrates, test_substrates=args.test_substrates,
        epochs=args.epochs, prewarm_waves=args.prewarm_waves, eval_beam=args.eval_beam,
        top_k=args.top_k, max_size=args.max_size, max_depth=args.max_depth,
        workers=args.workers, logz_lr=args.logz_lr, n_samples=args.n_samples,
    )

    # 1. beta-prime VAL sweep (D-10/FIX D) -- uses seed 0 only (cheap, VAL-side selection).
    print("[run_ablation_local] === Step 1: beta-prime VAL sweep ===", flush=True)
    sweep_seed = args.seeds[0]
    scores = sweep_beta_prime(
        args.beta_prime_grid, seed=sweep_seed, m_ensemble=args.m_ensemble,
        artifacts_dir=artifacts_dir, fixed=fixed, eval_substrates=args.eval_substrates,
        dry_run=args.dry_run,
    )
    print(f"[run_ablation_local] beta-prime sweep scores: {scores}", flush=True)
    if args.dry_run:
        print("[run_ablation_local] DRY-RUN: stopping after printing the planned commands.", flush=True)
        return
    chosen_beta_prime = pick_beta_prime(scores)
    print(f"[run_ablation_local] chosen beta_prime={chosen_beta_prime}", flush=True)

    # 2. VAL seed runs: ABL-01 (single) + ABL-02 (ensemble) at the chosen beta-prime.
    print("[run_ablation_local] === Step 2: VAL seed runs (ablation01 + ablation02) ===", flush=True)
    val_single_results = []
    val_ensemble_results = []
    for seed in args.seeds:
        r1 = _run_ablation_mode(
            "single", seed=seed, beta_prime=chosen_beta_prime, m_ensemble=args.m_ensemble,
            eval_split="val", artifacts_dir=artifacts_dir, fixed=fixed,
            eval_substrates=args.eval_substrates,
        )
        val_single_results.append(r1)
        r2 = _run_ablation_mode(
            "ensemble", seed=seed, beta_prime=chosen_beta_prime, m_ensemble=args.m_ensemble,
            eval_split="val", artifacts_dir=artifacts_dir, fixed=fixed,
            eval_substrates=args.eval_substrates,
        )
        val_ensemble_results.append(r2)

    val_gflownet_aucs = [r["metrics"]["gflownet_union_at_k_auc"] for r in val_single_results]
    val_abl01_aucs = [r["metrics"]["ablation01_union_at_k_auc"] for r in val_single_results]
    val_abl02_aucs = [r["metrics"]["ablation02_union_at_k_auc"] for r in val_ensemble_results]
    print(
        f"[run_ablation_local] VAL seed AUCs: gflownet={val_gflownet_aucs} "
        f"ablation01={val_abl01_aucs} ablation02={val_abl02_aucs}",
        flush=True,
    )

    if args.skip_test_touch:
        print("[run_ablation_local] --skip-test-touch set; stopping before the test split.", flush=True)
        return

    # 3. Test touched ONCE.
    print("[run_ablation_local] === Step 3: TEST touch (once) ===", flush=True)
    test_seed = args.seeds[0]
    test_single = _run_ablation_mode(
        "single", seed=test_seed, beta_prime=chosen_beta_prime, m_ensemble=args.m_ensemble,
        eval_split="test", artifacts_dir=artifacts_dir, fixed=fixed,
    )
    test_ensemble = _run_ablation_mode(
        "ensemble", seed=test_seed, beta_prime=chosen_beta_prime, m_ensemble=args.m_ensemble,
        eval_split="test", artifacts_dir=artifacts_dir, fixed=fixed,
    )

    # FIX C: automated config-match gate BEFORE accepting the test-split table.
    assert_config_match({
        "gflownet": test_single["config"],
        "ablation01": test_single["config"],
        "ablation02": test_ensemble["config"],
    })
    print("[run_ablation_local] config-match gate PASSED (FIX C).", flush=True)

    test_gflownet_auc = test_single["metrics"]["gflownet_union_at_k_auc"]
    test_abl01_auc = test_single["metrics"]["ablation01_union_at_k_auc"]
    test_abl02_auc = test_ensemble["metrics"]["ablation02_union_at_k_auc"]

    # 4a. PRIMARY: paired per-substrate bootstrap CI, reading per-substrate curves out of
    # the test-touch eval checkpoints.
    single_ckpt = _ckpt_paths(artifacts_dir, f"ablation_single_test_seed{test_seed}")[1]
    ensemble_ckpt = _ckpt_paths(artifacts_dir, f"ablation_ensemble_test_seed{test_seed}")[1]
    gflownet_paired, abl01_paired = paired_arrays(single_ckpt, single_ckpt, "ablation01")
    gflownet_paired_e, abl02_paired = paired_arrays(ensemble_ckpt, ensemble_ckpt, "ablation02")

    primary_ci_abl01 = paired_bootstrap_delta_ci(gflownet_paired, abl01_paired, n_boot=10000, ci=0.95)
    primary_ci_abl02 = paired_bootstrap_delta_ci(gflownet_paired_e, abl02_paired, n_boot=10000, ci=0.95)

    # 4b. SECONDARY: seed-level verdict + degeneracy-guarded margin + sensitivity grid.
    mean_gflownet, std_gflownet = _mean_std(val_gflownet_aucs)
    margin = degeneracy_guarded_margin(std_gflownet, mean_gflownet)
    secondary_verdict = compute_ablation_verdict(
        test_gflownet_auc, test_abl01_auc, test_abl02_auc, margin=margin,
    )
    sensitivity_grid = compute_delta_sensitivity_grid(
        test_gflownet_auc, test_abl01_auc, test_abl02_auc, std=std_gflownet,
    )

    report = {
        "beta_prime_sweep": scores,
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
        "m_ensemble": args.m_ensemble,
        "note_fix_e": (
            "Phase 3 reports the generous-ensemble variant only; the compute-matched "
            "ensemble is deferred to Phase 4."
        ),
    }
    report_path = artifacts_dir / "verdict_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n========== ABL-03 VERDICT ==========", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"\n[run_ablation_local] verdict report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()

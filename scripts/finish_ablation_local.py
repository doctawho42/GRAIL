#!/usr/bin/env python3
"""Local finisher for the Phase-3 ablation after the Modal 24h-timeout crash.

The Modal orchestrator hit its 24h function timeout mid-run. Everything it produced is
checkpointed on the ``grail-artifacts`` Volume; this script downloads that snapshot into
``artifacts/ablation_modal/`` (same subdir + tag names the Modal run used, via
``ablation_plan.plan_configs`` -> ``val_{mode}_seed{s}`` / ``test_{mode}_seed{s}``) and
FINISHES the run LOCALLY -- no Modal, no preemption, no 24h wall.

Two things make this cheap:
  1. run_gflownet.py's own ``--resume-ckpt`` (per-epoch) + ``--resume-eval-ckpt``
     (per-substrate) mean the 6 already-trained VAL models load instantly (epoch 15 ->
     training is a no-op) and the 3 partial VAL evals resume from where Modal left off.
     (run_gflownet.py MUST be the ablation-era version so its eval-config fingerprint
     matches the downloaded eval checkpoints -- it was reverted for exactly this.)
  2. The test-touch model for a seed is IDENTICAL to that seed's VAL model (same
     train_substrates/seed/beta_prime/epochs; only eval_split differs), so we COPY the
     val_{mode}_seed0 train checkpoint to the test_{mode}_seed0 path -> the test config
     resumes it and only EVALS. This skips the ~two full re-trainings the Modal
     orchestrator wasted on the test wave.

Then it runs the exact same ``ablation_plan.aggregate_and_verdict`` the Modal orchestrator
would have, writing ``results/ablation_local_verdict.json``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grail_metabolism.ablation_plan import (  # noqa: E402
    aggregate_and_verdict,
    fill_beta_prime,
    plan_configs,
    select_beta_prime,
    sweep_scores_from_results,
)

ART = ROOT / "artifacts" / "ablation_modal"      # downloaded Modal snapshot (same layout)
RUN = ROOT / "scripts" / "run_gflownet.py"

# MUST match the Modal run's args EXACTLY -- the eval-config fingerprint (and thus
# --resume-eval-ckpt compatibility) depends on them.
TRAIN_SUBSTRATES, TEST_SUBSTRATES, EPOCHS, M_ENSEMBLE = 300, 100, 15, 3
GRID = [2.0, 4.0, 6.0, 8.0, 10.0]
SEEDS = [0, 1, 2]

FIXED = [
    "--train-substrates", str(TRAIN_SUBSTRATES), "--test-substrates", str(TEST_SUBSTRATES),
    "--max-depth", "2", "--max-size", "10", "--epochs", str(EPOCHS), "--top-k", "50",
    "--logz-lr", "0.16", "--n-samples", "4", "--workers", "8", "--prewarm-waves", "1",
    "--no-bootstrap", "--no-eval-beam",
]


def _out(tag: str) -> Path:
    return ART / f"{tag}.json"


def _ckpt(tag: str) -> Path:
    return ART / f"{tag}.ckpt.pt"


def _eval_ckpt(tag: str) -> Path:
    return ART / f"{tag}.eval_ckpt.json"


def _load(tag: str) -> dict:
    with open(_out(tag)) as fh:
        return json.load(fh)


def _run_config(cfg: dict) -> dict:
    tag = str(cfg["tag"])
    if _out(tag).exists():
        print(f"[finish] {tag}: already complete -- skip", flush=True)
        return _load(tag)
    cmd = [
        sys.executable, "-u", str(RUN), *FIXED,
        "--seed", str(cfg["seed"]),
        "--eval-split", str(cfg["eval_split"]),
        "--ablation-mode", str(cfg["mode"]),
        "--beta-prime", str(cfg["beta_prime"]),
        "--m-ensemble", str(cfg["m_ensemble"]),
        "--out", str(_out(tag)),
        "--resume-ckpt", str(_ckpt(tag)),
        "--resume-eval-ckpt", str(_eval_ckpt(tag)),
    ]
    if cfg.get("eval_substrates") is not None:
        cmd += ["--eval-substrates", str(cfg["eval_substrates"])]
    print(f"\n===== finish: {tag} =====\n{' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return _load(tag)


def main() -> None:
    if not ART.exists():
        raise SystemExit(f"[finish] {ART} not found -- download the Modal snapshot first: "
                         "modal volume get grail-artifacts /ablation_modal artifacts/")

    plan = plan_configs(GRID, SEEDS, M_ENSEMBLE)

    # beta-prime from the already-done VAL sweep JSONs (Wave 1, complete on the Volume)
    sweep_results = {c["tag"]: (_load(c["tag"]) if _out(c["tag"]).exists() else None)
                     for c in plan["sweep"]}
    sweep_scores = sweep_scores_from_results(plan["sweep"], sweep_results)
    chosen_bp = select_beta_prime(sweep_scores)
    print(f"[finish] beta_prime sweep scores: {sweep_scores}", flush=True)
    print(f"[finish] chosen beta_prime = {chosen_bp}", flush=True)

    # Wave 2: VAL seed runs (resume; skip the 3 already-complete)
    val_cfgs = fill_beta_prime(plan["val"], chosen_bp)
    val_results = {c["tag"]: _run_config(c) for c in val_cfgs}
    val_single = [val_results[c["tag"]] for c in val_cfgs if c["mode"] == "single"]
    val_ensemble = [val_results[c["tag"]] for c in val_cfgs if c["mode"] == "ensemble"]

    # Wave 3: TEST touch -- copy each seed0 VAL train ckpt to the test ckpt path so the
    # test config resumes a COMPLETE (epoch-15) model and only EVALS (no re-training).
    test_cfgs = fill_beta_prime(plan["test"], chosen_bp)
    for c in test_cfgs:
        t_ckpt = _ckpt(str(c["tag"]))
        v_ckpt = _ckpt(f"val_{c['mode']}_seed{c['seed']}")
        if not t_ckpt.exists() and v_ckpt.exists():
            shutil.copy2(v_ckpt, t_ckpt)
            print(f"[finish] copied {v_ckpt.name} -> {t_ckpt.name} (test evals, no re-train)",
                  flush=True)
    test_results = {c["tag"]: _run_config(c) for c in test_cfgs}
    ts = next(str(c["tag"]) for c in test_cfgs if c["mode"] == "single")
    te = next(str(c["tag"]) for c in test_cfgs if c["mode"] == "ensemble")

    with open(_eval_ckpt(ts)) as fh:
        ts_eval = json.load(fh)
    with open(_eval_ckpt(te)) as fh:
        te_eval = json.load(fh)

    report = aggregate_and_verdict(
        val_single_results=val_single,
        val_ensemble_results=val_ensemble,
        test_single_result=test_results[ts],
        test_ensemble_result=test_results[te],
        test_single_eval_ckpt=ts_eval,
        test_ensemble_eval_ckpt=te_eval,
        chosen_beta_prime=chosen_bp,
        sweep_scores=sweep_scores,
        m_ensemble=M_ENSEMBLE,
    )

    print("\n========== ABL-03 VERDICT (local finish) ==========", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    out = ROOT / "results" / "ablation_local_verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[finish] verdict written -> {out}", flush=True)


if __name__ == "__main__":
    main()

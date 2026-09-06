#!/usr/bin/env python3
"""One seed of the full-split retraining, to be run in a Kaggle notebook cell.

The released checkpoints saw 5,000 of 9,011 training substrates for eight epochs and their early
stopping never engaged, which the manuscript reports as an underfitting signature rather than a
converged run. This trains the same configuration on the whole split with an epoch budget early
stopping can reach, one seed per session, so five sessions give the spread the register's effects
should be read against.

Run it as:

    !python kaggle_b1.py --seed 0

with the two datasets attached. It expects the code archive unpacked at the working directory and
the corpus in grail_metabolism/data/, which the setup cell below arranges.

Three things about this environment are load-bearing and are checked rather than hoped for.

RDKit's version. The pin is rdkit==2022.9.5 because tautomer canonicalisation is not stable
across releases and the matching key every recall figure is scored under depends on it. That
release does not install on Python 3.12, so refusing on it would mean never running this. What the
difference costs is measured instead, in results/rdkit_version_drift.json: over 3,000 training
substrates the standardised structure differs on 3 and the matching key on 1. The version that
actually built the graphs is recorded in the session report, so what a checkpoint was trained under
is a fact about the artifact rather than a recollection.

NumPy's major version. The stack pins numpy<2 and the images ship 2.x.

The session limit. Kaggle stops a session at twelve hours. A seed is expected to take one to three
hours on a P100 once the graph cache exists, but the cache is built once and takes longer, so the
cache is written to the working directory and can be attached to the next session as a dataset
rather than rebuilt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PINS = {"rdkit": "2022.09.5", "numpy_major": 1}
# What running under a different RDKit costs, measured rather than assumed:
# results/rdkit_version_drift.json, 3,000 training substrates against 2026.03.6.
DRIFT = {"artifact": "results/rdkit_version_drift.json", "measured_against": "2026.03.6",
         "substrates": 3000, "structure_differs": 3, "key_differs": 1}


def check_environment() -> dict:
    """Refuse what cannot be repaired, record what can, and return what the session ran under.

    The pin exists because tautomer canonicalisation moves between releases and the matching key
    depends on it. It is also unsatisfiable on a platform whose Python is too new for that release,
    so refusing on it outright would mean never running this at all. What the difference costs is
    measured instead: on 3,000 training substrates the standardised structure differs on 3 and the
    matching key on 1. That is small enough to declare and too small to argue about, so a different
    version is recorded rather than rejected, and the artifact this run produces says which one
    built its graphs.

    NumPy is different. The stack does not work under 2.x at all, so that stays a refusal.
    """
    import numpy
    import rdkit

    if int(numpy.__version__.split(".")[0]) != PINS["numpy_major"]:
        raise SystemExit(f"REFUSING: numpy is {numpy.__version__}; this stack pins numpy<2 and "
                         f"does not run under 2.x.")

    env = {"rdkit": rdkit.__version__, "rdkit_pin": PINS["rdkit"], "numpy": numpy.__version__}
    if rdkit.__version__ != PINS["rdkit"]:
        env["rdkit_drift"] = DRIFT
        print(f"  rdkit {rdkit.__version__}, not the pinned {PINS['rdkit']}. The pinned release "
              f"does not install on this Python. What the difference costs is measured in "
              f"{DRIFT['artifact']}: of {DRIFT['substrates']} training substrates the standardised "
              f"structure differs on {DRIFT['structure_differs']} and the matching key on "
              f"{DRIFT['key_differs']}. The version is recorded with the run.")
    else:
        print(f"  rdkit {rdkit.__version__}, the pinned version")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env["torch"] = torch.__version__
    env["device"] = dev
    if dev == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
        print(f"  torch {torch.__version__} on {env['gpu']}")
    else:
        print(f"  torch {torch.__version__} on cpu")
        print("  NOTE: no GPU visible. This will take a day rather than an hour; turn the "
              "accelerator on before spending a session on it.")
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/paper_full_converged.yaml")
    # Kaggle keeps outputs under /kaggle/working; anywhere else the run writes beside itself,
    # so the same script can be smoke-tested before a session is spent on it.
    work = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    ap.add_argument("--out", default=str(work / "artifacts"))
    ap.add_argument("--dry-run", action="store_true",
                    help="check the environment and write the config, then stop")
    args = ap.parse_args()

    env = check_environment()

    # The seed is the only thing that varies between sessions, and it is written into the config
    # rather than passed, so the run records what it was trained under.
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["seed"] = args.seed
    cfg["name"] = f"paper_full_converged_seed{args.seed}"
    cfg["output_dir"] = args.out
    run_cfg = work / f"config_seed{args.seed}.yaml"
    run_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"  seed {args.seed}, config at {run_cfg}")

    if args.dry_run:
        print("  dry run: environment checked and config written, nothing trained")
        return 0

    t0 = time.perf_counter()
    rc = subprocess.call([sys.executable, "-m", "grail_metabolism", "run-config", str(run_cfg)],
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    took = time.perf_counter() - t0
    print(f"\n  exit {rc} after {took / 3600:.2f} h")

    # What the next session needs to know, written where Kaggle keeps outputs.
    report = Path(args.out) / f"seed{args.seed}_session.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"seed": args.seed, "exit_code": rc,
                                  "seconds": round(float(took), 1),
                                  "config": str(run_cfg), "environment": env}, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

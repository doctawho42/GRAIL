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

RDKit's version. The pin is exact, rdkit==2022.9.5, because tautomer canonicalisation is not
stable across releases and the matching key every recall figure in this paper is scored under is a
tautomer-canonical InChIKey. A model trained on graphs a different RDKit built is not the model this
paper reports, even when the training code is identical. Kaggle's image ships a newer one, so the
pin is installed and then verified; a mismatch stops the run rather than producing a checkpoint
nobody can compare.

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


def check_environment() -> None:
    """Refuse to train under a stack that would not produce a comparable checkpoint."""
    import numpy
    import rdkit

    problems = []
    if rdkit.__version__ != PINS["rdkit"]:
        problems.append(
            f"rdkit is {rdkit.__version__}, not {PINS['rdkit']}. Tautomer canonicalisation is not "
            f"stable across releases and the matching key depends on it, so a checkpoint trained "
            f"here would not be comparable with the paper's.")
    if int(numpy.__version__.split(".")[0]) != PINS["numpy_major"]:
        problems.append(f"numpy is {numpy.__version__}; this stack pins numpy<2.")
    if problems:
        raise SystemExit("REFUSING:\n  " + "\n  ".join(problems))

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  torch {torch.__version__} on {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    if dev == "cpu":
        print("  NOTE: no GPU visible. This will take a day rather than an hour; turn the "
              "accelerator on before spending a session on it.")


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

    check_environment()

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
    report.write_text(json.dumps({"seed": args.seed, "exit_code": rc, "seconds": round(float(took), 1),
                                  "config": str(run_cfg)}, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

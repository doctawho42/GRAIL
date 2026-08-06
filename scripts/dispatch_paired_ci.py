#!/usr/bin/env python3
"""Is the one residual that is not zero distinguishable from zero?

results/hydrogen_dispatch.json reports a residual of $+0.016$ for our own bank: dispatching the
hydrogen convention per template reaches further than the better of the two global settings. Its
marginal interval overlaps that setting's heavily, and a marginal comparison is the wrong test
anyway, so the run that produced it cannot say whether the residual is real.

This runs the two arms in one pass over the same substrates and keeps the per-substrate counts, so
the difference can be resampled as a paired quantity -- the estimator the rest of the paper uses for
a contrast, micro, a difference of ratio-of-sums.

Only our bank is worth the pass. SyGMa dispatches nothing, so its residual is zero by construction,
and BioTransformer's dispatch reproduced its global arm exactly, so there is no difference to test.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem, RDLogger

from bank_engine_replication import load_bank
from engine_knobs import DEFAULT, apply_with
from hydrogen_dispatch import MAJORITY_CONVENTION, _apply, classify
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
BANK = "grail_full"
# results/hydrogen_dispatch.json and results/bank_engine_replication.json
COMMITTED = {"dispatch": 0.8148, "implicit": 0.7989}
_CTX: dict = {}


def _code_version() -> dict:
    import subprocess
    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def _init(rules, wants):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX["rules"], _CTX["wants"] = rules, wants


def _worker(item):
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return sub, 0, 0, 0
    substrates = {True: Chem.AddHs(Chem.Mol(mol)), False: Chem.Mol(mol)}
    disp = _apply(substrates, _CTX["rules"], _CTX["wants"])
    impl = apply_with(mol, _CTX["rules"], False, DEFAULT["norm"], DEFAULT["drop_invalid"])
    u, h_d, _ = _tautomer_recovered(trues, disp, audit=False)
    u2, h_i, _ = _tautomer_recovered(trues, impl, audit=False)
    assert u == u2, "the reference denominator must not depend on the arm"
    return sub, int(u), int(h_d), int(h_i)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "dispatch_paired_ci.json"))
    args = ap.parse_args()

    rules = load_bank(BANK)
    wants = classify(rules, MAJORITY_CONVENTION[BANK])
    subs = [r["sub"] for r in
            json.loads((ROOT / "results/filter_vs_prior_ci.json").read_text())["per_substrate"]]
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    items = [(s, truth[s]) for s in subs if truth.get(s)]
    print(f"{BANK}: {len(rules)} rules, {sum(wants)} dispatched, {len(items)} substrates", flush=True)

    workers = max(1, (os.cpu_count() or 4) - 2)
    with multiprocessing.get_context("spawn").Pool(workers, _init, (rules, wants)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 2), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(items):
                print(f"  {n}/{len(items)}", flush=True)
    rows.sort(key=lambda r: r[0])

    U = np.array([r[1] for r in rows])
    D = np.array([r[2] for r in rows])
    I = np.array([r[3] for r in rows])
    reach_d = round(float(D.sum() / max(U.sum(), 1)), 4)
    reach_i = round(float(I.sum() / max(U.sum(), 1)), 4)
    print(f"\ngate: dispatch {reach_d} against committed {COMMITTED['dispatch']}, "
          f"implicit {reach_i} against {COMMITTED['implicit']}")
    for got, want, name in ((reach_d, COMMITTED["dispatch"], "dispatch"),
                            (reach_i, COMMITTED["implicit"], "implicit")):
        if abs(got - want) > 1e-4:
            raise SystemExit(f"the {name} arm does not reproduce its committed reach")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))
    diff = D - I
    micro = float(diff.sum() / max(U.sum(), 1))
    bt = np.array([diff[j].sum() / max(U[j].sum(), 1) for j in idx])
    lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
    macro = (diff / np.maximum(U, 1))
    abt = macro[idx].mean(axis=1)
    alo, ahi = float(np.quantile(abt, .025)), float(np.quantile(abt, .975))

    rep = {"config": {**_code_version(), "bank": BANK, "n_rules": len(rules),
                      "dispatched": int(sum(wants)), "n_substrates": len(rows),
                      "match": "inchikey_tautomer", "n_boot": N_BOOT, "seed": SEED,
                      "policy": "docs/DISPATCH_PREREGISTRATION.md",
                      "gate": "both arms reproduce their committed reach"},
           "references": int(U.sum()),
           "recovered": {"dispatch": int(D.sum()), "implicit": int(I.sum())},
           "reach": {"dispatch": reach_d, "implicit": reach_i},
           "paired_residual": {"delta": round(micro, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "excludes_zero": bool(lo > 0 or hi < 0), "estimator": "micro",
                               "macro": {"delta": round(float(macro.mean()), 4),
                                         "ci95": [round(alo, 4), round(ahi, 4)],
                                         "excludes_zero": bool(alo > 0 or ahi < 0)}}}
    pr = rep["paired_residual"]
    print(f"\nreferences {int(U.sum())}: dispatch recovers {int(D.sum())}, implicit {int(I.sum())}")
    print(f"paired residual {pr['delta']:+.4f} {pr['ci95']} micro "
          f"({'certified' if pr['excludes_zero'] else 'spans zero'}), "
          f"{pr['macro']['delta']:+.4f} {pr['macro']['ci95']} macro")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

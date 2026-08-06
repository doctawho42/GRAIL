#!/usr/bin/env python3
"""The decomposition's three factors, measured through one hydrogen convention instead of two.

The identity in results/recall_factorization.json compares a ceiling with a pool that were not
built the same way. The ceiling pass applies the bank through
`apply_rules_to_molecule`, whose first act is `Chem.AddHs`
(grail_metabolism/utils/preparation.py:371). The deployed generator does not: its substrate comes
from `_graph_for_substrate`, which is `Chem.MolFromSmiles` and nothing else
(grail_metabolism/model/generator.py), and it fires the selected rules on that molecule.

Appendix app:enginereach measures what that difference costs -- expanding a substrate destroys the
products of templates written against implicit hydrogens, worth $0.069$ of reach on a 245-substrate
subsample -- so the two factors are not on the same footing. The ceiling is measured in the
convention that loses and the pool in the convention that does not, which understates coverage,
overstates selection retention against it, and leaves twelve substrates on the clean test split
where the deployed pipeline recovers a reference the ceiling denies. Two clamps in
factorize_recall.compute_records then carry that denial into the pool and the output.

This recomputes the ceiling in the convention the deployed pipeline actually uses and reports the
factors both ways. The point is not that one number is nicer. It is that a decomposition whose
factors are measured through different procedures charges the difference between the procedures to
whichever stage sits between them, and here that stage is rule selection -- the paper's headline
diagnosis.

Nothing is clamped here. If the nesting holds once the convention is shared, it holds because the
measurement is consistent rather than because a min() was applied.
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

from engine_knobs import apply_with
from grail_metabolism.utils.preparation import load_default_rules
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
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


def _init(rules):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX["rules"] = rules


def _worker(item):
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return sub, 0, 0, 0
    expanded = apply_with(mol, _CTX["rules"], True, "canonical", True)
    implicit = apply_with(mol, _CTX["rules"], False, "canonical", True)
    u, c_exp, _ = _tautomer_recovered(trues, expanded, audit=False)
    _, c_imp, _ = _tautomer_recovered(trues, implicit, audit=False)
    return sub, int(u), int(c_exp), int(c_imp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "ceiling_convention_matched.json"))
    args = ap.parse_args()

    committed = json.loads((ROOT / "results/recall_factorization.json").read_text())
    stored = {r["sub"]: r for r in committed["per_substrate"]}
    rules = load_default_rules()
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    items = [(s, truth[s]) for s in stored if truth.get(s)]
    print(f"{len(rules)} rules over {len(items)} substrates, both conventions", flush=True)

    workers = max(1, (os.cpu_count() or 4) - 2)
    with multiprocessing.get_context("spawn").Pool(workers, _init, (rules,)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 2), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(items):
                print(f"  {n}/{len(items)}", flush=True)
    rows.sort(key=lambda r: r[0])

    U = np.array([r[1] for r in rows])
    Cexp = np.array([r[2] for r in rows])
    Cimp = np.array([r[3] for r in rows])
    # the deployed pool and the deployed output, as the committed record stores them, with the
    # output's hits recomputed from the stored list rather than read from the clamped field
    Cbud = np.array([stored[r[0]]["Cbud"] for r in rows])
    H = np.array([_tautomer_recovered(truth.get(r[0], []),
                                      stored[r[0]].get("deployed_top15") or [], audit=False)[1]
                  for r in rows])

    gate = round(float(Cexp.sum() / max(U.sum(), 1)), 4)
    print(f"\ngate: the expanded-convention ceiling {gate} against the committed "
          f"{committed['factors']['coverage_bank']['point']:.4f}")
    if abs(gate - committed["factors"]["coverage_bank"]["point"]) > 1e-3:
        raise SystemExit("this pass does not reproduce the committed ceiling; it is not the same one")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))

    def factors(Cfull):
        cov = Cfull.sum() / U.sum()
        sel = Cbud.sum() / max(Cfull.sum(), 1)
        rank = H.sum() / max(Cbud.sum(), 1)
        bt = np.array([Cfull[j].sum() / max(U[j].sum(), 1) for j in idx])
        return {"coverage": round(float(cov), 4),
                "coverage_ci95": [round(float(np.quantile(bt, .025)), 4),
                                  round(float(np.quantile(bt, .975)), 4)],
                "selection": round(float(sel), 4), "ranking": round(float(rank), 4),
                "product": round(float(cov * sel * rank), 4),
                "nesting_violations": int((H > Cbud).sum() + (Cbud > Cfull).sum())}

    exp, imp = factors(Cexp), factors(Cimp)
    micro = round(float(H.sum() / U.sum()), 4)
    print(f"\n  {'convention':28} {'coverage':>9} {'selection':>10} {'ranking':>8} "
          f"{'product':>8} {'violations':>11}")
    for name, f in (("expanded (as published)", exp), ("implicit (as deployed)", imp)):
        print(f"  {name:28} {f['coverage']:>9} {f['selection']:>10} {f['ranking']:>8} "
              f"{f['product']:>8} {f['nesting_violations']:>11}")
    print(f"\n  micro recall from the deployed output: {micro}")

    rep = {"config": {**_code_version(), "n_substrates": len(rows), "n_rules": len(rules),
                      "match": "inchikey_tautomer", "n_boot": N_BOOT, "seed": SEED,
                      "deployed_convention": "implicit; generator._graph_for_substrate is "
                                             "MolFromSmiles with no AddHs",
                      "published_convention": "expanded; apply_rules_to_molecule calls AddHs",
                      "clamps": "none applied"},
           "sums": {"U": int(U.sum()), "Cfull_expanded": int(Cexp.sum()),
                    "Cfull_implicit": int(Cimp.sum()), "Cbud": int(Cbud.sum()), "H": int(H.sum())},
           "micro_recall": micro,
           "as_published_expanded_ceiling": exp,
           "convention_matched_implicit_ceiling": imp}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

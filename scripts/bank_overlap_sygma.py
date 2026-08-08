#!/usr/bin/env python3
"""How much of GRAIL's bank is SyGMa's bank, and how much of the ceiling do those rules carry?

Section 4 compares two coverage ceilings, 0.735 against 0.542, and reads the difference as bank
breadth. That reading is only interpretable if the two banks are separate objects. They are not
fully separate: GRAIL's bank is the deduplicated union of four earlier curated banks with the mined
templates (Appendix A.5), and SyGMa's published rule set is one of the things those curated banks
contain. This measures the containment exactly -- verbatim string membership, so nothing is
inferred -- and then applies the contained subset on its own, through the same primitives, the same
matcher and the same substrates as the ceiling itself.

Two numbers come out. The share of SyGMa's rules that sit inside GRAIL's bank says whether the
comparison is between two banks or between a bank and a near-subset of itself. The reach of that
subset alone says how much of GRAIL's ceiling is chemistry SyGMa also has.
"""
from __future__ import annotations

import json
import pathlib
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import sygma
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grail_metabolism.utils.preparation import apply_rules_to_molecule
from _population import POPULATIONS, ceiling_target, population_items, tagged_out
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
# Resolved from the installed package rather than hard-coded: a path under a home directory both
# breaks on any other machine and names the author in an anonymised archive.
SYGMA_RULES = Path(os.environ.get("SYGMA_RULES") or (Path(sygma.__file__).parent / "rules"))
OUT = ROOT / "results" / "bank_overlap_sygma.json"
_SUB: list = []


def sygma_smirks() -> list[str]:
    out = []
    for f in ("phase1.txt", "phase2.txt"):
        for line in (SYGMA_RULES / f).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.split("\t")[0].strip())
    return out


def _init(rules):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    global _SUB
    _SUB = rules


def _worker(item):
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return (sub, 0, 0)
    p = list(apply_rules_to_molecule(mol, _SUB, normalization_mode="canonical").keys())
    u, c, _ = _tautomer_recovered(trues, p, audit=False)
    return (sub, int(u), int(c))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="clean_test", choices=POPULATIONS,
                    help="subsample245 reproduces the committed artifact; clean_test is the split")
    args = ap.parse_args()
    bank = [l.strip() for l in open(ROOT / "grail_metabolism/resources/extended_smirks.txt") if l.strip()]
    mined = {l.strip() for l in open(ROOT / "grail_metabolism/resources/mined_only.txt") if l.strip()}
    curated = [r for r in bank if r not in mined]
    sy = sorted(set(sygma_smirks()))
    bs, cs = set(bank), set(curated)
    inside = [r for r in sy if r in bs]
    print(f"SyGMa rules {len(sy)}; in bank {len(inside)}; in the curated subset "
          f"{sum(1 for r in sy if r in cs)}", flush=True)

    items = population_items(args.population)
    print(f"substrates: {len(items)}", flush=True)

    ctx = multiprocessing.get_context("spawn")
    t = time.perf_counter()
    with ctx.Pool(max(1, (os.cpu_count() or 4) - 2), initializer=_init, initargs=(inside,)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 4), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(items):
                print(f"  {n}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)

    # imap_unordered returns in completion order, so the row order varies between runs.
    # A sum over rows does not care; a bootstrap that resamples row indices does, and the
    # published interval moved in the fourth decimal on re-run because of it.
    rows.sort(key=lambda r: r[0])
    U = np.array([r[1] for r in rows])
    H = np.array([r[2] for r in rows])
    cov = H.sum() / U.sum()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(U), (N_BOOT, len(U)))
    bt = np.array([H[j].sum() / U[j].sum() for j in idx])
    rep = {
        "match": "inchikey_tautomer", "n": len(items), "n_boot": N_BOOT, "seed": SEED,
        "population": args.population,
        "containment": {
            "sygma_rules": len(sy),
            "in_grail_bank": len(inside),
            "in_grail_curated_subset": sum(1 for r in sy if r in cs),
            "in_grail_mined_subset": sum(1 for r in sy if r in mined),
            "share_of_sygma_inside": round(len(inside) / len(sy), 4),
            "share_of_grail_curated_that_is_sygma": round(len(inside) / len(curated), 4),
            "test": "verbatim string membership in extended_smirks.txt",
        },
        "reach_of_contained_subset": {
            "n_rules": len(inside),
            "coverage": round(float(cov), 4),
            "ci95": [round(float(np.quantile(bt, .025)), 4), round(float(np.quantile(bt, .975)), 4)],
            "note": "GRAIL's own depth-1 primitives and matcher, on the same substrates as "
                    "results/ceiling_by_provenance.json for this population",
            "full_bank_ceiling_here": round(ceiling_target([sub for sub, _ in items]), 4),
        },
    }
    out = pathlib.Path(tagged_out(OUT, args.population))
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

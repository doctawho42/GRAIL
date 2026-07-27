#!/usr/bin/env python3
"""Precompute structure keys for every criterion, in parallel, with no torch in the process.

Tautomer canonicalisation dominates every downstream measurement and the prediction sets barely
repeat, so a per-process cache does not help. This keys each distinct structure once and writes a
lookup per criterion, reusable by the decomposition and the match-sensitivity analysis.

Runs standalone on purpose: a process pool inside a torch-loaded interpreter stalls here.
"""
from __future__ import annotations
import json, sys, time
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODES = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
OUTDIR = ROOT / "results" / "key_tables"

SOURCES = [
    ROOT / "results" / "test_references.json",
    ROOT / "results" / "sygma_fulltest_predictions.json",
    ROOT / "artifacts" / "tier2_1170" / "metapredictor_preds.json",
]


def _key(arg):
    from grail_metabolism.metrics import _match_keys
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    s, mode = arg
    return s, next(iter(_match_keys([s], mode)), None)


def main() -> int:
    smiles = set()
    for f in SOURCES:
        if not f.exists():
            print(f"  missing, skipped: {f.name}", flush=True)
            continue
        for v in json.loads(f.read_text()).values():
            smiles.update(v)
    grail = ROOT / "results" / "recall_factorization.json"
    if grail.exists():
        for r in json.loads(grail.read_text())["per_substrate"]:
            smiles.update(r["deployed_top15"])
    smiles = sorted(s for s in smiles if s)
    n_proc = max(1, cpu_count() - 2)
    print(f"distinct structures: {len(smiles):,}; workers: {n_proc}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        out = OUTDIR / f"{mode}.json"
        if out.exists():
            print(f"  {mode:20} already present, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        with Pool(n_proc) as pool:
            pairs = pool.map(_key, [(s, mode) for s in smiles], chunksize=128)
        table = {s: k for s, k in pairs if k}
        out.write_text(json.dumps(table))
        print(f"  {mode:20} {len(table):,} keys in {time.perf_counter()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

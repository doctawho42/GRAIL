#!/usr/bin/env python3
"""Generate and persist SyGMa's predictions on the full clean test split.

Only aggregates were ever kept (results/sygma_robust.json), so SyGMa could not be re-scored under
criteria other than the two already computed, and could not be decomposed on the same population
as GRAIL. This writes the ranked predictions once so both become possible without another run.

Substrates are independent, so the work is spread across processes; each worker builds its own
SyGMa scenario.
"""
from __future__ import annotations
import json, sys, time
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "sygma_fulltest_predictions.json"
_SC = None


def _worker(smiles):
    """One substrate -> its ranked SyGMa predictions. Scenario is built once per process."""
    global _SC
    import sygma
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    if _SC is None:
        _SC = sygma.Scenario([[sygma.ruleset["phase1"], 1], [sygma.ruleset["phase2"], 1]])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, []
    try:
        tree = _SC.run(mol)
        tree.calc_scores()
        return smiles, [e[0] for e in tree.to_smiles()]
    except Exception:
        return smiles, []


def main() -> int:
    from scripts.factorize_recall import build_dataset_config
    from grail_metabolism.workflows.data import load_dataset_bundle

    bundle = load_dataset_bundle(build_dataset_config(100000))
    subs = sorted(s for s, p in bundle.test.map.items() if p)
    n_proc = max(1, cpu_count() - 2)
    print(f"substrates with references: {len(subs)}; workers: {n_proc}", flush=True)

    preds, t0 = {}, time.perf_counter()
    with Pool(n_proc) as pool:
        for i, (s, r) in enumerate(pool.imap_unordered(_worker, subs, chunksize=4), 1):
            preds[s] = r
            if i % 100 == 0 or i == len(subs):
                print(f"  {i}/{len(subs)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    empty = sum(1 for s in subs if not preds[s])
    print(f"substrates with no SyGMa output: {empty}/{len(subs)}", flush=True)
    if empty > 0.02 * len(subs):
        raise SystemExit(f"ERROR: {empty} empty prediction sets -- a broken adapter, not a result.")

    OUT.write_text(json.dumps(preds))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

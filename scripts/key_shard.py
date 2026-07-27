#!/usr/bin/env python3
"""Key one shard of the distinct structures under one criterion.

The tautomer key imports grail_metabolism.utils.preparation, which imports torch. Six torch
imports inside one multiprocessing pool deadlock on this platform, so the work is split across
independent interpreters instead: no pool, no shared start-method machinery, one torch per
process. Shards are merged by build_key_tables.py.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "results" / "key_tables"


def distinct_structures() -> list[str]:
    smiles = set()
    for f in (ROOT / "results" / "test_references.json",
              ROOT / "results" / "sygma_fulltest_predictions.json",
              ROOT / "artifacts" / "tier2_1170" / "metapredictor_preds.json"):
        if f.exists():
            for v in json.loads(f.read_text()).values():
                smiles.update(v)
    g = ROOT / "results" / "recall_factorization.json"
    if g.exists():
        for r in json.loads(g.read_text())["per_substrate"]:
            smiles.update(r["deployed_top15"])
    return sorted(s for s in smiles if s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    a = ap.parse_args()

    from grail_metabolism.metrics import _match_keys
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    allsm = distinct_structures()
    mine = allsm[a.shard::a.of]
    print(f"shard {a.shard}/{a.of}: {len(mine):,} of {len(allsm):,}", flush=True)

    out, t0 = {}, time.perf_counter()
    for i, s in enumerate(mine, 1):
        k = next(iter(_match_keys([s], a.mode)), None)
        if k:
            out[s] = k
        if i % 2000 == 0:
            print(f"  {i}/{len(mine)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"{a.mode}.shard{a.shard}of{a.of}.json"
    p.write_text(json.dumps(out))
    print(f"wrote {p} ({len(out):,} keys, {time.perf_counter()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Key one shard of the MOSES generated molecules under one criterion.

Separate interpreters rather than a pool: the tautomer key imports grail_metabolism.utils.
preparation, which imports torch, and concurrent torch imports inside one pool deadlock here.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "results" / "moses_keys"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    a = ap.parse_args()

    from grail_metabolism.metrics import _match_keys
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    samples = json.loads((ROOT / "results" / "moses_samples.json").read_text())
    allsm = sorted({s for v in samples.values() for s in v})
    mine = allsm[a.shard::a.of]
    print(f"shard {a.shard}/{a.of}: {len(mine):,} of {len(allsm):,}", flush=True)

    out, t0 = {}, time.perf_counter()
    for i, s in enumerate(mine, 1):
        k = next(iter(_match_keys([s], a.mode)), None)
        if k:
            out[s] = k
        if i % 1000 == 0:
            print(f"  {i}/{len(mine)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"{a.mode}.shard{a.shard}of{a.of}.json"
    p.write_text(json.dumps(out))
    print(f"wrote {p} ({len(out):,} keys, {time.perf_counter()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

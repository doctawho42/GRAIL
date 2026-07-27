#!/usr/bin/env python3
"""Key the tier-2 subset predictions (BioTransformer, MetaTrans, MetaPredictor-150).

Separate interpreters rather than a pool: the tautomer key imports a module that imports torch,
and concurrent torch imports inside one pool deadlock on this platform.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTDIR = ROOT / "results" / "key_tables"
SRC = [ROOT / "artifacts" / "tier2" / f for f in
       ("biotransformer_preds.json", "metatrans_preds.json", "metapredictor_preds.json")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    a = ap.parse_args()

    from grail_metabolism.metrics import _match_keys
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    have = set(json.loads((OUTDIR / f"{a.mode}.json").read_text())) if (OUTDIR / f"{a.mode}.json").exists() else set()
    sm = set()
    for f in SRC:
        for v in json.loads(f.read_text()).values():
            sm.update(v)
        sm.update(json.loads(f.read_text()).keys())
    todo = sorted(s for s in sm if s and s not in have)
    mine = todo[a.shard::a.of]
    print(f"shard {a.shard}/{a.of}: {len(mine):,} new of {len(todo):,}", flush=True)

    out, t0 = {}, time.perf_counter()
    for i, s in enumerate(mine, 1):
        k = next(iter(_match_keys([s], a.mode)), None)
        if k:
            out[s] = k
        if i % 500 == 0:
            print(f"  {i}/{len(mine)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    (OUTDIR / f"{a.mode}.tier2shard{a.shard}of{a.of}.json").write_text(json.dumps(out))
    print(f"wrote {len(out):,} keys in {time.perf_counter()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

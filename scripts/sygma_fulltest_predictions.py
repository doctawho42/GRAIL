#!/usr/bin/env python3
"""Generate and persist SyGMa's predictions on the full clean test split.

Only aggregates were ever kept (results/sygma_robust.json), so SyGMa could not be re-scored under
criteria other than the two already computed, and could not be decomposed on the same population
as GRAIL. This writes the ranked predictions once so both become possible without another run.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_on_gloryx import sygma_predictions
from scripts.factorize_recall import build_dataset_config
from grail_metabolism.workflows.data import load_dataset_bundle

OUT = ROOT / "results" / "sygma_fulltest_predictions.json"


def main() -> int:
    bundle = load_dataset_bundle(build_dataset_config(100000))
    subs = sorted(s for s, p in bundle.test.map.items() if p)
    print(f"substrates with references: {len(subs)}", flush=True)

    preds, t0 = {}, time.perf_counter()
    for i, s in enumerate(subs, 1):
        preds[s] = sygma_predictions([s]).get(s, [])
        if i % 50 == 0 or i == len(subs):
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

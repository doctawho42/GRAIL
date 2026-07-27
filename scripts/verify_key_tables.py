#!/usr/bin/env python3
"""Check the precomputed key tables against the paper's own serial scorer on a sample.

Serial, single-process, torch allowed: this is the guard that the parallel keying did not change
any number. Any disagreement is a bug, not a speed-up, and fails the run.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gloryx_rank_flip_ci import per_substrate_recall

K, SAMPLE, SEED = 15, 40, 0
MODES = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]


def recall_vector(preds, truth, subs, table, k=K):
    v = np.empty(len(subs))
    for i, s in enumerate(subs):
        ranked, seen = [], set()
        for item in preds.get(s, []):
            key = table.get(item)
            if key and key not in seen:
                seen.add(key)
                ranked.append(key)
        real = {table[r] for r in truth[s] if table.get(r)}
        v[i] = (len(set(ranked[:k]) & real) / len(real)) if real else 0.0
    return v


def main() -> int:
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    preds = json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text())
    subs = sorted(truth)
    rng = np.random.default_rng(SEED)
    sample = [subs[i] for i in rng.choice(len(subs), size=SAMPLE, replace=False)]
    for mode in MODES:
        table = json.loads((ROOT / "results" / "key_tables" / f"{mode}.json").read_text())
        mine = recall_vector(preds, truth, sample, table)
        theirs = per_substrate_recall(preds, truth, sample, mode, K)
        if not np.allclose(mine, theirs, atol=1e-12):
            j = int(np.argmax(np.abs(mine - theirs)))
            raise SystemExit(f"ERROR: table disagrees with serial scorer on '{mode}' "
                             f"({sample[j]!r}: {mine[j]} vs {theirs[j]})")
        print(f"  {mode:20} matches serial scorer on {SAMPLE} substrates", flush=True)
    print("key tables verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

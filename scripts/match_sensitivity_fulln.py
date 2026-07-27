#!/usr/bin/env python3
"""Match sensitivity at full n, from stored predictions -- no model re-run.

The five-method table is scored on the 150-substrate shared subset, which is the first thing a
reviewer objects to. GRAIL's deployed top-15 for all 1,170 clean-test substrates is already
persisted in results/recall_factorization.json, so its five-criterion row costs nothing but a
re-score. MetaPredictor's five-criterion row on the full split already exists. This script emits
GRAIL's row and per-substrate vectors so the two can be compared at full n.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gloryx_rank_flip_ci import per_substrate_recall  # same matcher as the paper's harness
from scripts.factorize_recall import build_dataset_config
from grail_metabolism.workflows.data import load_dataset_bundle

MODES = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
K = 15
OUT = ROOT / "results" / "match_sensitivity_fulln.json"


def main() -> int:
    recs = json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]
    preds = {r["sub"]: r["deployed_top15"] for r in recs}
    print(f"stored GRAIL predictions: {len(preds)} substrates", flush=True)

    bundle = load_dataset_bundle(build_dataset_config(100000))
    truth = {s: list(p) for s, p in bundle.test.map.items()}
    subs = sorted(s for s in truth if truth[s] and s in preds)
    print(f"scored on {len(subs)} substrates with references", flush=True)

    rep = {"n_substrates": len(subs), "k": K, "modes": MODES, "method": "GRAIL",
           "recall_at_15": {}, "mean_output": float(np.mean([len(preds[s]) for s in subs]))}
    vecs = {}
    for m in MODES:
        v = per_substrate_recall(preds, truth, subs, m, K)
        vecs[m] = v
        rep["recall_at_15"][m] = round(float(v.mean()), 4)
        print(f"  {m:20} {v.mean():.4f}", flush=True)
    rep["per_substrate"] = {m: [round(float(x), 6) for x in vecs[m]] for m in MODES}
    rep["substrates"] = subs
    OUT.write_text(json.dumps(rep))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

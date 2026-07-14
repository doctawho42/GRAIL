"""Score a tier-2 tool's full-1170 predictions with the paper's exact match-sensitivity scorer.

Reuses run_match_sensitivity.score (same uniform canonical dedup + recall@k under every match mode)
so a tier-2 tool re-run on the full clean test scores identically to the n=150 table. References
(substrate -> true products) come from a GRAIL deployed prediction CSV (the full-1170 eval).

Usage: python scripts/score_tier2_1170.py <preds.json> [--refs <grail_test_predictions.csv>]
  preds.json: {substrate_smiles: [ranked_pred_smiles]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_match_sensitivity import load_grail_csv, score

DEFAULT_REFS = ROOT / "artifacts" / "multiseed_full5000_seed0" / "predictions" / "test_predictions.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", help="tier-2 preds json {substrate: [preds]}")
    ap.add_argument("--refs", default=str(DEFAULT_REFS), help="GRAIL CSV giving substrate->true products")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15])
    args = ap.parse_args()

    method_preds = json.loads(Path(args.preds).read_text())
    _, reals = load_grail_csv(Path(args.refs))

    # score() iterates over the substrates in `reals`; keep only those the tool also has an entry for
    # (missing -> empty pred list -> recall 0, same as the n=150 protocol's handling of no-output).
    shared = [s for s in reals if s in method_preds]
    print(f"preds={len(method_preds)}  refs={len(reals)}  shared substrates scored={len(shared)}", flush=True)
    reals_shared = {s: reals[s] for s in reals}  # score over the FULL reference set (missing tool preds = 0)

    by_mode = score(method_preds, reals_shared, args.ks)
    taut = by_mode.get("inchikey_tautomer", {})
    print("\n=== recall under inchikey_tautomer (full-1170) ===")
    for k in args.ks:
        print(f"  recall@{k}: {taut.get(f'recall@{k}')}")
    print(f"  mean_output: {taut.get('mean_output')}")
    print("\nall modes:", json.dumps({m: v.get(f'recall@15') for m, v in by_mode.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

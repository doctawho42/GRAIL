"""Budget-matched leaderboard: recall@{5,10,15} + precision + output size per method.

Reads results/match_sensitivity_5method.json (tautomer-InChIKey protocol) and emits a view that
controls for the output budget: SyGMa's high recall is bought with ~74 predictions/substrate
(precision 0.029), whereas GRAIL reaches 0.365 recall at ~8.7 predictions (precision 0.109). At a
matched small budget (recall@5) the recall gap narrows. Writes results/budget_matched_leaderboard.json
and prints the table.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "match_sensitivity_5method.json"
OUT = ROOT / "results" / "budget_matched_leaderboard.json"
PROTO = "inchikey_tautomer"
ORDER = ["GRAIL", "SyGMa", "BioTransformer", "MetaPredictor", "MetaTrans"]


def main() -> int:
    d = json.loads(DATA.read_text())
    bm = d["by_method"]
    rows = {}
    for m in ORDER:
        t = bm[m][PROTO]
        rows[m] = {
            "recall@5": t["recall@5"],
            "recall@10": t["recall@10"],
            "recall@15": t["recall@15"],
            "precision": t["precision"],
            "mean_output": t["mean_output"],
        }
    report = {
        "protocol": PROTO,
        "n_substrates_note": "GRAIL & SyGMa on n≈1170; tier-2 (BioTransformer, MetaPredictor, "
                             "MetaTrans) on the n=150 shared subset.",
        "by_method": rows,
        "reading": "recall@15 is not budget-fair: SyGMa's 0.554 is emitted over ~74 predictions "
                   "(precision 0.029), ~7x GRAIL's ~8.7 (precision 0.109). At recall@5 (matched "
                   "small budget) the recall spread narrows. Precision (recall per prediction) "
                   "reorders the board toward the low-output methods.",
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"{'method':<15}{'r@5':>8}{'r@10':>8}{'r@15':>8}{'precision':>11}{'mean_out':>10}")
    for m in ORDER:
        r = rows[m]
        print(f"{m:<15}{r['recall@5']:>8.3f}{r['recall@10']:>8.3f}{r['recall@15']:>8.3f}"
              f"{r['precision']:>11.3f}{r['mean_output']:>10.1f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

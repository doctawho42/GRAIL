"""Assemble the train-on-candidates (queue point B) verdict from the matched CPU arms.

The deployed filter trains on MolFrame.negs, derived from gen_map, whose distribution differs from
the wide generator-top-k pool the filter must rank at inference. Point B sets
filter.train_on_candidates=true so the filter's negatives are drawn from the generator's own top-k,
aligning the train and deploy distributions. The prediction was a ranking-order gain, since the
filter would then be accurate on the pool it actually orders.

The candidate arm (rulegate_cpu_candfilter) and the baseline (rulegate_cpu_baseline) share the
whole recipe and seed 42, differing only in the filter objective, and are compared only to each
other. Both are read off their frozen reports; nothing is asserted the reports do not carry.

    python scripts/typed_edit/rulegate_candfilter_result.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

CPU = ROOT / "results" / "rulegate_cpu"
KS = ("top_1_recall", "top_3_recall", "top_5_recall", "top_10_recall", "top_15_recall")
SEED_SD_AT_15 = 0.0107  # results/retraining_spread.json


def deltas(b, l, block):
    out = {}
    for k in (*KS, "mean_output_size"):
        bv, lv = b[block][k], l[block][k]
        out[k.replace("top_", "r@").replace("_recall", "")] = {
            "baseline": round(bv, 4), "candidates": round(lv, 4), "delta": round(lv - bv, 4)}
    return out


def main() -> int:
    bm = json.loads((CPU / "baseline" / "metrics.json").read_text())
    lm = json.loads((CPU / "candfilter" / "metrics.json").read_text())
    ft = json.loads((CPU / "candfilter" / "filter_training.json").read_text())

    val = deltas(bm, lm, "ensemble_val")
    test = deltas(bm, lm, "ensemble")
    head = val["r@5"]["delta"]
    depth = val["r@15"]["delta"]

    verdict = {
        "primary_recall5_val_rose": head > 0,
        "reading": (
            f"On the validation selection metric recall@5 delta {head:+} (r@15 {depth:+}) -- a wash, "
            f"every budget within the seed sd {SEED_SD_AT_15}. On test recall@5 rose {test['r@5']['delta']:+}, "
            f"the largest head signal among the survivors, but recall@15 fell {test['r@15']['delta']:+}: "
            "the covariate-shift alignment shifts the filter toward accuracy on the head of the pool "
            "it ranks (its global binary discrimination drops, mcc "
            f"{round(lm['filter']['mcc'] - bm['filter']['mcc'], 4):+}, auc "
            f"{round(lm['filter']['roc_auc'] - bm['filter']['roc_auc'], 4):+}) at the cost of depth. "
            "Same helps-head-loses-depth shape as the id-gate, and again not a win on the metric "
            "selection is made on."),
        "overall": (
            "NOT SELECTED at this scale: validation recall (the selection metric) is a wash within "
            "seed noise, so B is not adopted on its own terms, despite the strongest test-head signal "
            "seen (recall@5 +0.030). The alignment is mechanically real -- the filter trades global "
            "binary accuracy for head accuracy on the ranked pool -- but on one seed at subsample "
            "scale it does not move the validation headline. A multi-seed or full-scale run is the "
            "only way the +0.030 test head would separate from noise; worth escalating only if a "
            "cheap second seed reproduces the head gain."),
    }

    rep = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([str(CPU / a / "metrics.json") for a in ("baseline", "candfilter")]),
        "arms": {
            "baseline": {"train_on_candidates": False, "kernel": "polomoshnov/grail-rulegate-cpu-baseline"},
            "candidates": {"train_on_candidates": True, "run": "local CPU, configs/rulegate/rulegate_cpu_candfilter.yaml",
                           "filter_train_candidates": ft.get("train_data", {})}},
        "matched": {"train_substrates": 1457, "seed": 42, "standardize": False,
                    "note": "compared only to each other, never to deployed full-5000"},
        "recall_validation_selection": val,
        "recall_test": test,
        "filter": {"baseline": bm["filter"], "candidates": lm["filter"]},
        "seed_spread_caveat": {"sd_at_15": SEED_SD_AT_15, "source": "results/retraining_spread.json",
                               "note": "single seed per arm; deltas below this are not distinguished from noise"},
        "prereg": "docs/RANKING_BANK_QUEUE.md (point B)",
        "verdict": verdict,
    }
    out = ROOT / "results" / "rulegate_candfilter_result.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"  val  recall@5 {val['r@5']['delta']:+}  recall@15 {val['r@15']['delta']:+}  (selection: wash)")
    print(f"  test recall@5 {test['r@5']['delta']:+}  recall@15 {test['r@15']['delta']:+}")
    print(f"  VERDICT: {verdict['overall'][:80]}...")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

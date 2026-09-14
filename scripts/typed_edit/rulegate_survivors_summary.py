"""The four ranking-side retraining levers, side by side, and the frontier they share.

Four configurations were trained on the same small CPU subsample (1457 substrates, seed 42,
standardize false, all early-stopped) and compared only to the matched baseline:

  baseline      id_gate_lambda 0, the deployed recipe
  id_gate       id_gate_lambda 8 -- support-gate the per-template id (representation)
  candfilter    filter.train_on_candidates -- filter negatives from the generator's top-k (objective)
  diffreadout   filter difference_readout -- append (prod-sub) to the single-mode readout (features)

Three different places in the pipeline, one repeated result: each lifts recall at the head (small k)
and loses it at depth (k=15, the reported budget). None lifts the whole curve. This tabulates the
recall@k delta of each arm against the baseline at every budget, so the head-vs-depth frontier is
visible in one place, and states the reading. Read against the seed-to-seed sd (0.0107 at k=15,
results/retraining_spread.json): most single-arm deltas are within it, so the frontier is a shape,
not a set of individually significant wins.

    python scripts/typed_edit/rulegate_survivors_summary.py
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
ARMS = {
    "id_gate": CPU / "lambda8" / "metrics.json",
    "candfilter": CPU / "candfilter" / "metrics.json",
    "diffreadout": CPU / "diffreadout" / "metrics.json",
    # Queue point C. Its matched base is the id-gate arm, not the plain baseline, because the type
    # term is added to the gated identity; the deltas below are still taken against the baseline so
    # every arm is read on one ruler, and the C-against-id_gate contrast is reported separately.
    "typeid": CPU / "typeid" / "metrics.json",
}
KS = ("top_1_recall", "top_3_recall", "top_5_recall", "top_10_recall", "top_15_recall")
SEED_SD_AT_15 = 0.0107


def main() -> int:
    base = json.loads((CPU / "baseline" / "metrics.json").read_text())
    arms = {k: json.loads(p.read_text()) for k, p in ARMS.items()}

    def klabel(k):
        return k.replace("top_", "r@").replace("_recall", "")

    # validation is the selection metric; test reported alongside
    table = {"validation_ensemble_val": {}, "test_ensemble": {}}
    for block, dst in (("ensemble_val", "validation_ensemble_val"), ("ensemble", "test_ensemble")):
        for k in KS:
            row = {"baseline": round(base[block][k], 4)}
            for name, m in arms.items():
                row[name] = round(m[block][k] - base[block][k], 4)  # delta vs baseline
            table[block and dst][klabel(k)] = row

    # head = mean delta at k in {1,3}, depth = delta at k15, per arm on validation
    head_depth = {}
    for name, m in arms.items():
        d1 = m["ensemble_val"]["top_1_recall"] - base["ensemble_val"]["top_1_recall"]
        d3 = m["ensemble_val"]["top_3_recall"] - base["ensemble_val"]["top_3_recall"]
        d15 = m["ensemble_val"]["top_15_recall"] - base["ensemble_val"]["top_15_recall"]
        head_depth[name] = {"head_delta_mean_k1_k3": round((d1 + d3) / 2, 4),
                            "depth_delta_k15": round(d15, 4),
                            "trades_head_for_depth": bool((d1 + d3) / 2 > 0 and d15 < 0)}

    # Point C refines the id-gate arm rather than the plain baseline, so its own contrast is against
    # that arm. Reported as its own field instead of by swapping the ruler under one row, which
    # would leave the table comparing four arms against two different references.
    against_matched_base = None
    if "typeid" in arms and "id_gate" in arms:
        against_matched_base = {
            "base": "id_gate",
            "why": ("the type term is added to the gated identity, so the arm it refines is the "
                    "id-gate arm and not the baseline the other rows are read against"),
            "validation_delta": {klabel(k): round(arms["typeid"]["ensemble_val"][k]
                                                  - arms["id_gate"]["ensemble_val"][k], 4)
                                 for k in KS},
        }

    rep = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([str(CPU / "baseline" / "metrics.json"), *[str(p) for p in ARMS.values()]]),
        "matched": {"train_substrates": 1457, "seed": 42, "standardize": False,
                    "compared_to": "the matched baseline only, never deployed full-5000"},
        "recall_delta_vs_baseline": table,
        "head_vs_depth": head_depth,
        "seed_spread_caveat": {"sd_at_15": SEED_SD_AT_15, "source": "results/retraining_spread.json"},
        "typeid_against_its_matched_base": against_matched_base,
        "reading": (
            "Four arms, and they do not all fail the same way. Three of them -- representation "
            "(the support-gated rule id), filter objective (train-on-candidates) and filter "
            "features (difference-readout) -- lift recall at the head (k=1,3) and give it back by "
            "k=15: none raises the reported budget, each slides along a head-versus-depth frontier "
            "rather than lifting it, and most single-arm deltas sit inside the seed spread. The "
            "fourth, the type-shared rule id, is a stronger negative than that: it loses at both "
            "ends, and it is worse than the arm it refines at every validation budget, so a pooled "
            "type identity neither recovers the depth the gate costs nor keeps the head. Filter "
            "discrimination is flat across all of them, so what moves is the generator's ranking. A "
            "headline gain would need a change to what the generator ranks, not another way to "
            "represent a rule or to read a candidate."),
        "verdict": ("no arm selected: three trade head for depth within noise, and the "
                    "type-shared id loses at both ends"),
    }
    out = ROOT / "results" / "rulegate_survivors_summary.json"
    out.write_text(json.dumps(rep, indent=2))

    print("recall@k delta vs baseline (validation, the selection metric):")
    print(f"  {'k':>5} " + " ".join(f"{n:>11}" for n in arms))
    for k in KS:
        r = table["validation_ensemble_val"][klabel(k)]
        print(f"  {klabel(k):>5} " + " ".join(f"{r[n]:>+11.4f}" for n in arms))
    print("\nhead(k1,k3) vs depth(k15):")
    for n, hd in head_depth.items():
        print(f"  {n:>11}: head {hd['head_delta_mean_k1_k3']:+} depth {hd['depth_delta_k15']:+} "
              f"{'(trades head for depth)' if hd['trades_head_for_depth'] else ''}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

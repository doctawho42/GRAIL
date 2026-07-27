#!/usr/bin/env python3
"""Was the Set-GFlowNet judged on the wrong endpoint, and would a set endpoint rescue it?

The Set-GFlowNet optimises log R(S) = beta * (TP - lam*|S|), which is set agreement with a size
penalty, and it was reported on recall@15, which rewards exactly the enumeration the penalty is
meant to suppress. That is a real mismatch and it deserves the check. Its three saved runs hold
aggregate metrics only -- no per-substrate predictions, no checkpoints -- so set F1 cannot be
recomputed directly. What they do hold is `set_size_calibration`, the mean sampled set size minus
the true annotated set size, and that turns out to settle the question anyway.

Two things are computed here.

1. What the reward actually instructs. Beta cancels in the argmax, so adding a candidate whose hit
   probability is p changes the reward by p - lam: the reward says "emit anything above lam". The
   F1-optimal rule is to emit anything above F1/2. Comparing the two says whether the objective was
   mis-specified for a set endpoint, or specified fine and not achieved.

2. What set F1 the saved runs imply. From mean emitted size, mean reference size and the reported
   recall, the pooled F1 follows. This is an estimate from aggregates, not a re-scoring: the size
   statistic is over all sampled forests while recall is over the top-15 truncation of the best one,
   so the two describe different sets and the estimate is optimistic for the method.

Also computes the ceiling that incomplete annotation puts on any set metric, which bounds this whole
line of work: a predictor that emits exactly the true metabolite set is scored against the annotated
subset, so with annotation completeness c it measures Jaccard c and F1 2c/(1+c), whatever it does.
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
OUT = ROOT / "results" / "gflownet_set_endpoint.json"


def main() -> int:
    runs = [json.loads((ROOT / f"results/gflownet_seed{s}_overnight.json").read_text()) for s in SEEDS]
    cfg = runs[0]["config"]
    beta, lam, max_size = float(cfg["beta"]), float(cfg["lam"]), int(cfg["max_size"])

    recall = [r["metrics"]["gflownet_recall@15"] for r in runs]
    beam = [r["metrics"]["beam_recall@15"] for r in runs]
    rerank = [r["metrics"]["reranker_recall@15"] for r in runs]
    calib = [r["metrics"]["set_size_calibration"] for r in runs]

    truth = json.loads((ROOT / "results/test_references.json").read_text())
    ref_sizes = [len(v) for v in truth.values() if v]
    mean_ref = st.mean(ref_sizes)

    mean_emitted = st.mean(calib) + mean_ref
    mean_recall = st.mean(recall)
    # Pooled F1 from means: TP = recall * refs, F1 = 2*TP / (emitted + refs).
    tp = mean_recall * mean_ref
    f1_est = 2 * tp / (mean_emitted + mean_ref)
    jac_est = tp / (mean_emitted + mean_ref - tp)
    # The best F1 the emitted size permits, i.e. if every reference were recovered.
    f1_ceiling_at_size = 2 * mean_ref / (mean_emitted + mean_ref)

    # What the reward instructs versus what a set endpoint wants.
    reward_threshold = lam                      # emit while P(hit) > lam
    f1_threshold = f1_est / 2                   # emit while P(hit) > F1/2
    observed_precision = tp / mean_emitted

    rep = {
        "note": "estimates from saved aggregates; the runs hold no per-substrate predictions",
        "config": {"beta": beta, "lam": lam, "max_size": max_size,
                   "n_substrates_per_seed": runs[0]["metrics"]["n_substrates"]},
        "recall_at_15": {"gflownet": round(st.mean(recall), 4), "gflownet_sd": round(st.stdev(recall), 4),
                         "beam": round(st.mean(beam), 4), "reranker": round(st.mean(rerank), 4)},
        "set_size": {"calibration_mean": round(st.mean(calib), 3),
                     "calibration_sd": round(st.stdev(calib), 3),
                     "mean_reference_size": round(mean_ref, 3),
                     "implied_mean_emitted": round(mean_emitted, 2),
                     "max_size_cap": max_size,
                     "cap_saturation": round(mean_emitted / max_size, 3)},
        "set_endpoint_estimate": {"f1": round(f1_est, 4), "jaccard": round(jac_est, 4),
                                  "f1_ceiling_at_this_size": round(f1_ceiling_at_size, 4),
                                  "implied_precision": round(observed_precision, 4)},
        "thresholds": {"reward_implies_emit_above": reward_threshold,
                       "f1_optimal_emit_above": round(f1_threshold, 4),
                       "reward_is_more_conservative_than_f1": bool(reward_threshold > f1_threshold)},
        "annotation_ceiling": {
            "explanation": "a predictor emitting exactly the true set measures Jaccard c and F1 2c/(1+c)",
            "by_completeness": {str(c): {"jaccard_max": c, "f1_max": round(2 * c / (1 + c), 4)}
                                for c in (0.9, 0.75, 0.5, 0.3, 0.1)},
        },
    }
    OUT.write_text(json.dumps(rep, indent=1))

    print(f"reward: log R(S) = {beta} * (TP - {lam}*|S|), max_size = {max_size}\n")
    print(f"recall@15 (3 seeds, n={rep['config']['n_substrates_per_seed']:.0f}): "
          f"gflownet {st.mean(recall):.4f}, beam {st.mean(beam):.4f}, reranker {st.mean(rerank):.4f}")
    print(f"\nset size: reference {mean_ref:.2f}, emitted {mean_emitted:.2f} "
          f"(calibration {st.mean(calib):+.2f}), cap {max_size} "
          f"-> saturation {mean_emitted / max_size:.1%}")
    print(f"\nimplied set endpoint: F1 ~ {f1_est:.4f}, Jaccard ~ {jac_est:.4f}, precision ~ {observed_precision:.4f}")
    print(f"  best F1 the emitted size allows even at perfect recall: {f1_ceiling_at_size:.4f}")
    print(f"\nthresholds: reward emits above P={reward_threshold:.3f}; F1 wants above P={f1_threshold:.3f}"
          f"  -> reward is {'MORE' if reward_threshold > f1_threshold else 'LESS'} conservative than F1 wants")
    print("\nannotation ceiling on any set metric:")
    for c, v in rep["annotation_ceiling"]["by_completeness"].items():
        print(f"  completeness {c}: Jaccard <= {v['jaccard_max']}, F1 <= {v['f1_max']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

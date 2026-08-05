#!/usr/bin/env python3
"""What the propensity correction actually does to the rule weights.

The paper reports the weight distribution the inverse-propensity correction produces -- its range,
its mean, and the share of rules it up-weights -- and no artifact held any of it. The numbers came
from a training run whose logs were not kept, so a reader could not check them and neither could
we. SELF_CLAIMS row 11 carries that as an open item.

It does not need the run. The weight is a closed-form function of three things: each rule's positive
count over the training substrates, the number of substrates, and the two propensity constants. The
first comes from the label cache scripts/label_density.py already gates against the published
per-rule counts; the other two are config defaults. So this recomputes the vector exactly as
grail_metabolism/model/generator.py builds it, including the normalisation over rules that carry a
positive and the [0.1, 25] clamp, and writes the distribution the appendix quotes.

Reproducing the construction rather than importing it is deliberate: the model would have to be
instantiated and its records assembled to reach the same code path, and that pulls in checkpoints
and preprocessing this artifact should not depend on. The gate below is what makes the copy
trustworthy -- the per-rule counts must reproduce the four figures results/rule_train_positives.json
already publishes, or the label matrix is not the one the generator trained against.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "artifacts" / "preprocessed" / "train" / "ea9ee257861324be" / "reaction_labels.pt"
# grail_metabolism/config.py: GeneratorOptimConfig.propensity_a / .propensity_b
A, B = 0.55, 1.5
CLAMP = (0.1, 25.0)
# results/rule_train_positives.json -- the gate that says this is the right label matrix
PUBLISHED = {"train_substrates": 4787, "rules": 7581, "never_positive": 4271,
             "pos_eq_1": 1520, "pos_ge2": 1790, "useful_ge1": 3310}


def _code_version() -> dict:
    import subprocess
    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--out", default=str(ROOT / "results" / "propensity_weights.json"))
    args = ap.parse_args()

    d = torch.load(args.labels, map_location="cpu", weights_only=False)
    subs = list(d)
    M = np.asarray([d[s] for s in subs], dtype=np.int8)
    n_sub, n_rules = M.shape
    per_rule = M.sum(axis=0)

    got = {"train_substrates": n_sub, "rules": n_rules,
           "never_positive": int((per_rule == 0).sum()), "pos_eq_1": int((per_rule == 1).sum()),
           "pos_ge2": int((per_rule >= 2).sum()), "useful_ge1": int((per_rule >= 1).sum())}
    bad = {k: (PUBLISHED[k], got[k]) for k in PUBLISHED if PUBLISHED[k] != got[k]}
    print("label matrix against the published per-rule counts:")
    for k in PUBLISHED:
        print(f"  {k:18} published {PUBLISHED[k]:>6}  recomputed {got[k]:>6}"
              f"  {'OK' if not bad.get(k) else 'MISMATCH'}")
    if bad:
        raise SystemExit(f"this is not the label matrix behind the published counts ({bad})")

    positives = torch.as_tensor(per_rule, dtype=torch.float32)
    c = (math.log(max(float(n_sub), 2.0)) - 1.0) * (B + 1.0) ** A
    propensity = 1.0 / (1.0 + c * torch.exp(-A * torch.log(positives + B)))
    inverse = 1.0 / propensity.clamp_min(1e-6)
    observed = positives > 0.0
    scale = inverse[observed].mean() if observed.any() else inverse.mean()
    w = (inverse / scale.clamp_min(1e-6)).clamp(*CLAMP).numpy()

    up = float((w > 1.0).mean())
    rep = {
        "config": {**_code_version(), "labels": str(Path(args.labels).relative_to(ROOT)),
                   "propensity_a": A, "propensity_b": B, "clamp": list(CLAMP),
                   "n_train_substrates": int(n_sub), "n_rules": int(n_rules),
                   "source": "closed form of grail_metabolism/model/generator.py, propensity block",
                   "gate": "per-rule counts reproduce results/rule_train_positives.json"},
        "weight": {"min": round(float(w.min()), 4), "max": round(float(w.max()), 4),
                   "mean": round(float(w.mean()), 4), "median": round(float(np.median(w)), 4),
                   "share_up_weighted": round(up, 4),
                   "share_at_clamp_low": round(float((w <= CLAMP[0] + 1e-9).mean()), 4),
                   "share_at_clamp_high": round(float((w >= CLAMP[1] - 1e-9).mean()), 4)},
        "weight_by_positive_count": {
            str(k): round(float(w[per_rule == k].mean()), 4)
            for k in (0, 1, 2, 5, 10) if (per_rule == k).any()},
    }
    wv = rep["weight"]
    print(f"\npropensity weights over {n_rules} rules:")
    print(f"  range {wv['min']} to {wv['max']}, mean {wv['mean']}, median {wv['median']}")
    print(f"  up-weighted (>1): {100*wv['share_up_weighted']:.1f}%")
    print(f"  at the clamps: {100*wv['share_at_clamp_low']:.2f}% low, "
          f"{100*wv['share_at_clamp_high']:.2f}% high")
    print("  mean weight by positive count:", rep["weight_by_positive_count"])
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

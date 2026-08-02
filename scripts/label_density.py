#!/usr/bin/env python3
"""How sparse is the rule-selection label matrix, on both of its margins?

The paper motivates the extreme-multi-label framing with a per-substrate figure -- how many of the
7,581 rules carry a positive label for one substrate -- and reports the per-rule margin separately
in the negative-results appendix. Neither margin had a producer. results/rule_train_positives.json
records the per-rule side and was written by no committed script, and the per-substrate side was
recorded nowhere at all, so the number the main text leans on could not be checked.

Checking it changed it. The claim was "one to three rules per substrate carry a positive label,
some 0.03% of the label space". Measured on the label cache the generator actually trains against,
the median is five and the mean eleven, and only a quarter of substrates fall in the one-to-three
band. The framing survives -- a median of five positives out of 7,581 is 0.07% of the label space,
which is still extreme multi-label with a near-empty positive set -- but the numbers were wrong by
a factor of two to four, in the direction that made the paper's own case look stronger.

Both margins come out of one pass over artifacts/preprocessed/train/.../reaction_labels.pt, a dict
of substrate SMILES to a 0/1 vector over the bank. That cache is a preprocessing artifact and is
too large to commit, so this script's output is the committed record and the gate below is what
ties it to the cache: the per-rule counts it recomputes must reproduce the four already-published
figures exactly, or the cache is not the one those numbers came from.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "artifacts" / "preprocessed" / "train" / "ea9ee257861324be" / "reaction_labels.pt"
# published in results/rule_train_positives.json; the gate that says this is the right cache
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
    ap.add_argument("--out", default=str(ROOT / "results" / "label_density.json"))
    args = ap.parse_args()

    d = torch.load(args.labels, map_location="cpu", weights_only=False)
    subs = list(d)
    M = np.asarray([d[s] for s in subs], dtype=np.int8)
    n_sub, n_rules = M.shape
    print(f"label matrix: {n_sub} substrates x {n_rules} rules", flush=True)

    per_rule = M.sum(axis=0)
    got = {"train_substrates": n_sub, "rules": n_rules,
           "never_positive": int((per_rule == 0).sum()),
           "pos_eq_1": int((per_rule == 1).sum()),
           "pos_ge2": int((per_rule >= 2).sum()),
           "useful_ge1": int((per_rule >= 1).sum())}
    bad = {k: (PUBLISHED[k], got[k]) for k in PUBLISHED if PUBLISHED[k] != got[k]}
    print("\nper-rule margin, against the published record:")
    for k in PUBLISHED:
        print(f"  {k:18} published {PUBLISHED[k]:>6}  recomputed {got[k]:>6}"
              f"  {'OK' if not bad.get(k) else 'MISMATCH'}")
    if bad:
        raise SystemExit(f"this cache does not reproduce the published per-rule counts ({bad}) -- "
                         f"it is not the one those numbers came from")

    per_sub = M.sum(axis=1)
    pct = {f"p{q}": int(np.percentile(per_sub, q)) for q in (5, 25, 50, 75, 95)}
    rep = {
        "config": {**_code_version(), "labels": str(Path(args.labels).relative_to(ROOT))},
        "n_substrates": int(n_sub), "n_rules": int(n_rules),
        "per_rule": {**got, "gate": "reproduces results/rule_train_positives.json exactly"},
        "per_substrate": {
            "mean": round(float(per_sub.mean()), 2), "median": int(np.median(per_sub)),
            "min": int(per_sub.min()), "max": int(per_sub.max()), "percentiles": pct,
            "share_zero": round(float((per_sub == 0).mean()), 4),
            "share_1_to_3": round(float(((per_sub >= 1) & (per_sub <= 3)).mean()), 4),
            "label_space_share_mean": round(float(per_sub.mean() / n_rules), 6),
            "label_space_share_median": round(float(np.median(per_sub) / n_rules), 6),
        },
    }
    ps = rep["per_substrate"]
    print(f"\nper-substrate margin (the one the main text quotes):")
    print(f"  mean {ps['mean']}, median {ps['median']}, range {ps['min']}--{ps['max']}")
    print(f"  percentiles {pct}")
    print(f"  share with 1--3 positives: {100*ps['share_1_to_3']:.1f}%")
    print(f"  label-space share: mean {100*ps['label_space_share_mean']:.3f}%, "
          f"median {100*ps['label_space_share_median']:.3f}%")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

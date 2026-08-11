#!/usr/bin/env python3
r"""Is the emission rule's threshold a finding or an argmax over the split it is reported on?

The pool-relative rule emits while a candidate scores within $\alpha$ of the leader, and $\alpha=0.5$
is where its advantage over the best global constant is largest. That the advantage sits on a
plateau six values wide is an argument and not a defence: the value reported is still the one that
maximised the quantity being reported, on the split it is reported on, which is the shape of every
threshold that later fails to reproduce.

So the choice is made where it can be checked. On each of many random splits, $\alpha$ and the
competing constant $k$ are both chosen on the training part and the two are compared only on the
held-out part, and the paired difference is accumulated over splits. If the plateau is real the
held-out gain matches the reported one; if $0.5$ was a fit to this split, it does not.

Nothing is retrained: the candidate pools and their scores are the frozen artifact the rest of the
emission appendix uses, and the only thing being selected is a threshold.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

ALPHAS = (0.95, 0.9, 0.8, 0.75, 0.6, 0.5, 0.4, 0.3, 0.25, 0.1)
KS = (1, 2, 3, 5, 8, 15)
SEED = 0


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


def _host():
    spec = importlib.util.spec_from_file_location(
        "setsize_headroom", ROOT / "scripts" / "setsize_headroom.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["setsize_headroom"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--train-share", type=float, default=0.8)
    ap.add_argument("--out", default=str(ROOT / "results" / "emission_crossfit.json"))
    args = ap.parse_args()

    host = _host()
    from grail_metabolism.metrics import _tautomer_inchikey

    cache: dict = {}

    def keyer(s):
        if s not in cache:
            try:
                cache[s] = _tautomer_inchikey(s)
            except Exception:
                cache[s] = None
        return cache[s]

    rows = json.loads((ROOT / "results/scored_predictions.json").read_text())["rows"]
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    truth_keys = {s: {k for k in (keyer(y) for y in ys) if k} for s, ys in truth.items()}

    # per-substrate F1 under each candidate policy, computed once; a split is then a mask over it.
    # gap_rule and evaluate are imported from the script that reports the headline, so the rule
    # being cross-fitted is that rule and not a re-implementation of it.
    per_alpha = {a: np.asarray(host.evaluate(rows, truth_keys, host.gap_rule(a), keyer)[2],
                               dtype=float) for a in ALPHAS}
    per_k = {k: np.asarray(host.evaluate(rows, truth_keys, (lambda s, r, k=k: k), keyer)[2],
                           dtype=float) for k in KS}
    n = len(next(iter(per_alpha.values())))
    rng = np.random.default_rng(SEED)

    gains, picked_alpha, picked_k = [], [], []
    for _ in range(args.splits):
        tr = rng.random(n) < args.train_share
        te = ~tr
        if te.sum() < 20 or tr.sum() < 20:
            continue
        a_star = max(ALPHAS, key=lambda a: per_alpha[a][tr].mean())
        k_star = max(KS, key=lambda k: per_k[k][tr].mean())
        gains.append(float(per_alpha[a_star][te].mean() - per_k[k_star][te].mean()))
        picked_alpha.append(a_star)
        picked_k.append(k_star)

    g = np.array(gains)
    lo, hi = np.quantile(g, .025), np.quantile(g, .975)
    share_half = float(np.mean([a == 0.5 for a in picked_alpha]))
    rep = {"config": {**_code_version(), "splits": len(g), "train_share": args.train_share,
                      "seed": SEED, "alphas": list(ALPHAS), "constants": list(KS),
                      "note": "both the threshold and the competing constant are chosen on the "
                              "training part of each split and compared only on the held-out part"},
           "held_out_gain_mean": round(float(g.mean()), 4),
           "held_out_gain_ci95": [round(float(lo), 4), round(float(hi), 4)],
           "share_of_splits_choosing_alpha_0.5": round(share_half, 3),
           "alpha_chosen": {str(a): picked_alpha.count(a) for a in ALPHAS if a in picked_alpha},
           "constant_chosen": {str(k): picked_k.count(k) for k in KS if k in picked_k},
           "separated_from_zero": bool(lo > 0)}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"  {len(g)} splits, threshold and constant both chosen on the training part")
    print(f"  held-out gain {rep['held_out_gain_mean']:+.4f} {rep['held_out_gain_ci95']}"
          f"  {'separated' if rep['separated_from_zero'] else 'NOT separated'}")
    print(f"  alpha=0.5 was the training-part argmax in {share_half:.0%} of splits")
    print(f"  alphas chosen: {rep['alpha_chosen']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

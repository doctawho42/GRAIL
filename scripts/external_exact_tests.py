#!/usr/bin/env python3
"""Do the external rejections survive a test that does not assume normality?

The p-values feeding the external multiplicity correction are derived from stored bootstrap
intervals under a normal approximation, se = (hi - lo) / 2z. That is a reasonable shortcut when a
decision sits far from its threshold, and a poor one when it does not: the smallest of them is a
tail probability near four standard deviations, read off an interval computed from thirty-seven
paired differences that are discrete, bounded and lattice-valued. A normal tail at that distance is
exactly where the approximation is least trustworthy.

Nothing about the design requires it. The interaction is a paired quantity over substrates, so
under the null that a criterion step moves two methods equally, the sign of each substrate's
difference-of-differences is exchangeable. Flipping those signs gives an exact test in the sense
that matters here, with no distributional assumption at all, and the same per-substrate vectors
also give a bias-corrected and accelerated bootstrap interval.

Reported per interaction: the observed effect, the sign-flip p-value, the BCa interval, and whether
the decision the paper records changes. The point of the run is the last column.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import RDLogger

from grail_metabolism.metrics import _tautomer_inchikey
from scripts.gloryx_rank_flip_ci import (DATA, PRED_FILES, load_gloryx, per_substrate_recall,
                                         sygma_predict)

RDLogger.DisableLog("rdApp.*")

K = 15
N_PERM, N_BOOT, SEED = 200000, 20000, 0
STEPS = [("inchikey", "inchi_no_stereo"), ("inchi_no_stereo", "inchikey_tautomer")]
CRITERIA = sorted({c for s in STEPS for c in s})
Z = 1.959964


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


def sign_flip_p(d: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    """Two-sided p under sign exchangeability of the paired difference-of-differences.

    Substrates contributing exactly zero carry no information about the sign and are dropped, which
    is what an exact paired test does; the effective n is reported so a reader can see it.
    """
    nz = d[d != 0.0]
    if nz.size == 0:
        return 1.0
    obs = abs(nz.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, nz.size))
    null = np.abs((signs * nz).mean(axis=1))
    # add one to numerator and denominator, so a p-value is never reported as exactly zero
    return float((1 + (null >= obs - 1e-15).sum()) / (1 + n_perm))


def bca_interval(d: np.ndarray, rng: np.random.Generator, n_boot: int, alpha=0.05):
    """Bias-corrected and accelerated percentile interval for the mean of a paired vector."""
    n = d.size
    theta = float(d.mean())
    idx = rng.integers(0, n, (n_boot, n))
    boots = d[idx].mean(axis=1)
    prop = float((boots < theta).mean())
    if prop <= 0.0 or prop >= 1.0:
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi), None, None
    from math import erf, sqrt

    def ppf(p):
        # inverse standard normal by bisection; scipy is not a dependency of this file
        lo_, hi_ = -8.0, 8.0
        for _ in range(200):
            mid = (lo_ + hi_) / 2
            if 0.5 * (1 + erf(mid / sqrt(2))) < p:
                lo_ = mid
            else:
                hi_ = mid
        return (lo_ + hi_) / 2

    z0 = ppf(prop)
    jack = np.array([np.delete(d, i).mean() for i in range(n)])
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = float(num / den) if den > 0 else 0.0
    zl, zu = ppf(alpha / 2), ppf(1 - alpha / 2)
    def adj(zq):
        v = z0 + (z0 + zq) / max(1e-12, 1 - a * (z0 + zq))
        return min(max(0.5 * (1 + erf(v / math.sqrt(2))), 1e-6), 1 - 1e-6)
    lo, hi = np.quantile(boots, [adj(zl), adj(zu)])
    return float(lo), float(hi), z0, a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "external_exact_tests.json"))
    args = ap.parse_args()

    reals = load_gloryx(DATA / "gloryx_test.json")
    allsubs = sorted(s for s in reals if reals[s])
    methods = {n: {s: json.loads(p.read_text()).get(s, []) for s in allsubs}
               for n, p in PRED_FILES.items()}
    methods["SyGMa"] = sygma_predict(allsubs, K)
    names = sorted(methods)
    print(f"external set: {len(allsubs)} substrates, {len(names)} methods", flush=True)

    vec = {(n, c): per_substrate_recall(pr, reals, allsubs, c, K)
           for n, pr in methods.items() for c in CRITERIA}

    rng = np.random.default_rng(SEED)
    rows = []
    for c1, c2 in STEPS:
        for a, b in combinations(names, 2):
            d = (vec[(a, c2)] - vec[(a, c1)]) - (vec[(b, c2)] - vec[(b, c1)])
            p_perm = sign_flip_p(d, rng, N_PERM)
            lo, hi, z0, acc = bca_interval(d, rng, N_BOOT)
            # the approximation the paper currently uses, for comparison on the same vector
            bidx = rng.integers(0, d.size, (N_BOOT, d.size))
            bt = d[bidx].mean(axis=1)
            plo, phi = np.quantile(bt, [0.025, 0.975])
            se = (phi - plo) / (2 * Z)
            z = abs(d.mean()) / se if se > 0 else float("inf")
            p_norm = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
            rows.append({"step": f"{c1} -> {c2}", "pair": f"{a} vs {b}",
                         "delta": round(float(d.mean()), 4),
                         "n_informative": int((d != 0).sum()),
                         "p_signflip": round(p_perm, 6),
                         "p_normal_approx": round(float(p_norm), 6),
                         "bca_ci95": [round(lo, 4), round(hi, 4)],
                         "percentile_ci95": [round(float(plo), 4), round(float(phi), 4)],
                         "bca_excludes_zero": bool(lo * hi > 0),
                         "percentile_excludes_zero": bool(plo * phi > 0)})

    # Holm over the family the paper declares externally, under each p-value in turn.
    def holm(key):
        order = sorted(rows, key=lambda r: r[key])
        m, out, alive = len(order), [], True
        for i, r in enumerate(order):
            thr = 0.05 / (m - i)
            ok = alive and r[key] <= thr
            alive = ok
            out.append((r["step"], r["pair"], ok))
        return [f"{s} | {p}" for s, p, ok in out if ok]

    surv_norm, surv_perm = holm("p_normal_approx"), holm("p_signflip")
    agree = set(surv_norm) == set(surv_perm)
    print(f"\n  {len(rows)} interactions in the external family")
    print(f"  surviving Holm under the normal approximation: {len(surv_norm)}")
    print(f"  surviving Holm under the sign-flip test:       {len(surv_perm)}")
    print(f"  the two agree on which survive: {agree}")
    for r in sorted(rows, key=lambda r: r["p_signflip"])[:6]:
        flag = "" if r["bca_excludes_zero"] == r["percentile_excludes_zero"] else "   <- interval verdict differs"
        print(f"    {r['step']:34} {r['pair']:34} {r['delta']:+.4f} "
              f"p_perm {r['p_signflip']:.5f}  p_norm {r['p_normal_approx']:.5f}{flag}")

    rep = {"config": {**_code_version(), "n_substrates": len(allsubs), "k": K,
                      "n_perm": N_PERM, "n_boot": N_BOOT, "seed": SEED,
                      "steps": [f"{a} -> {b}" for a, b in STEPS],
                      "note": "sign-flip on the paired difference-of-differences; no normality "
                              "assumption, and BCa beside the percentile interval"},
           "rows": rows, "holm_normal_approx": surv_norm, "holm_signflip": surv_perm,
           "decisions_agree": agree,
           "interval_verdicts_agree": all(r["bca_excludes_zero"] == r["percentile_excludes_zero"]
                                          for r in rows)}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

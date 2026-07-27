#!/usr/bin/env python3
"""Holm-Bonferroni over the tested protocol x method interactions.

The paper previously declined to correct for multiplicity. It does not need to decline: this
applies the correction and reports what survives. p-values are derived from the stored
paired-bootstrap intervals under a normal approximation (se = (hi-lo)/2z), which is adequate here
because every decision is far from its threshold; the intervals themselves remain the primary
evidence.

The internal family is reported twice: over the two pairs actually computed, and conservatively
over all ten pairs the five-method table admits, since only two were computed and a reviewer is
entitled to treat the wider family as the reference.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "multiplicity_holm.json"
Z = 1.959964


def p_from_ci(mean: float, lo: float, hi: float) -> float:
    se = (hi - lo) / (2 * Z)
    if se <= 0:
        return 0.0
    z = abs(mean) / se
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def holm(tests, m=None, alpha=0.05):
    """tests: list of (name, p). m defaults to len(tests); a larger m is the conservative family."""
    m = m or len(tests)
    out, rejected = [], True
    for i, (name, p) in enumerate(sorted(tests, key=lambda t: t[1])):
        thr = alpha / (m - i)
        if rejected and p < thr:
            out.append({"pair": name, "p": p, "threshold": thr, "rejected": True})
        else:
            rejected = False
            out.append({"pair": name, "p": p, "threshold": thr, "rejected": False})
    return out


def main() -> int:
    rep = {"alpha": 0.05, "p_value_note": "normal approximation from the paired-bootstrap CI"}

    ext = json.loads((ROOT / "results" / "gloryx_rank_flip_ci.json").read_text())["pairwise"]
    tests = []
    for pair, v in ext.items():
        it = v["interaction_b_extra_gain"]
        tests.append((pair, p_from_ci(it["mean"], *it["ci95"])))
    rep["external"] = {"n_pairs_tested": len(tests), "family_size": len(tests),
                       "result": holm(tests)}

    internal = []
    for f, pair in [("rank_flip_ci.json", "GRAIL_vs_BioTransformer"),
                    ("rank_flip_ci_metatrans_sygma.json", "MetaTrans_vs_SyGMa")]:
        d = json.loads((ROOT / "results" / f).read_text())
        it = d["interaction_B_extra_gain_from_normalization"]
        internal.append((pair, p_from_ci(it["mean"], *it["ci95"])))
    n_possible = len(list(combinations(range(5), 2)))
    rep["internal_as_tested"] = {"n_pairs_tested": 2, "family_size": 2,
                                 "result": holm(internal)}
    rep["internal_conservative"] = {"n_pairs_tested": 2, "family_size": n_possible,
                                    "note": f"corrected as if all {n_possible} pairs of the five-method table were the family",
                                    "result": holm(internal, m=n_possible)}

    for k in ("external", "internal_as_tested", "internal_conservative"):
        r = rep[k]
        surv = sum(1 for x in r["result"] if x["rejected"])
        r["n_surviving"] = surv
        print(f"{k:24} family m={r['family_size']:2}  survive {surv}/{len(r['result'])}")
        for x in r["result"]:
            print(f"   {x['pair']:34} p={x['p']:.2e}  thr={x['threshold']:.4f}  "
                  f"{'rejected' if x['rejected'] else 'not rejected'}")
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

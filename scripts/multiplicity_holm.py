#!/usr/bin/env python3
"""Holm-Bonferroni over the tested protocol x method interactions.

The paper previously declined to correct for multiplicity. It does not need to decline: this
applies the correction and reports what survives. p-values are derived from the stored
paired-bootstrap intervals under a normal approximation (se = (hi-lo)/2z), which is adequate here
because every decision is far from its threshold; the intervals themselves remain the primary
evidence.

The internal family is reported three times: over the two pairs actually computed, over all ten
pairs the five-method table admits, and over the full grid that was actually searched. The search
was two-dimensional -- method pairs crossed with criterion steps, and two steps are reported
(canonical->tautomer and canonical->InChIKey) -- so a family fixed by the method-pair axis alone
undercounts it by a factor of two. The same applies externally, where six pairs were tested across
two steps (InChIKey->no-stereo and no-stereo->tautomer).

Holm-adjusted p-values are reported alongside the raw ones, so a reader can check a decision at any
alpha without redoing the step-down.
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
    """tests: list of (name, p). m defaults to len(tests); a larger m is the conservative family.

    Also reports the Holm-adjusted p-value, the running maximum of (m-i)*p capped at 1, which is
    the quantity a reader can compare against any alpha directly.
    """
    m = m or len(tests)
    out, rejected, running = [], True, 0.0
    for i, (name, p) in enumerate(sorted(tests, key=lambda t: t[1])):
        thr = alpha / (m - i)
        running = max(running, min(1.0, (m - i) * p))
        row = {"pair": name, "p": p, "p_holm": running, "threshold": thr}
        if rejected and p < thr:
            row["rejected"] = True
        else:
            rejected = False
            row["rejected"] = False
        out.append(row)
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
    rep["external_grid"] = {
        "n_pairs_tested": len(tests), "family_size": 2 * len(tests),
        "note": "the grid actually searched: six method pairs crossed with the two criterion steps "
                "the external table reports (InChIKey->no-stereo and no-stereo->tautomer)",
        "result": holm(tests, m=2 * len(tests))}

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
    rep["internal_grid"] = {
        "n_pairs_tested": 2, "family_size": 2 * n_possible,
        "note": f"the grid actually searched: all {n_possible} pairs of the five-method table "
                "crossed with the two criterion steps reported (canonical->tautomer and "
                "canonical->InChIKey)",
        "result": holm(internal, m=2 * n_possible)}

    # The second comparison declared confirmatory in the paper is the learned-versus-prior rule
    # selection gap. It is not a protocol x method interaction and so belongs to neither family
    # above; it was declared on its own and searched over nothing, so its family is of size one and
    # the Holm-adjusted p equals the raw p. Reporting it that way is the honest answer -- a
    # comparison declared corrected has to appear in the multiplicity table with its threshold.
    sel = json.loads((ROOT / "results" / "prior_vs_learned.json").read_text())
    g = sel["bootstrap_ci"]["gaps"]["learned_only__minus__prior_only__gen"]
    rep["selection_confirmatory"] = {
        "n_pairs_tested": 1, "family_size": 1,
        "note": "the learned-versus-prior rule-selection gap declared confirmatory alongside the "
                "differential sensitivity; a family of one, so Holm p = raw p",
        "estimand": "recall@15(learned selector) - recall@15(frequency prior), paired, n=245",
        "delta": g["delta"], "ci95": g["ci95"],
        "result": holm([("learned_selector_vs_frequency_prior",
                         p_from_ci(g["delta"], *g["ci95"]))])}

    for k in ("external", "external_grid", "internal_as_tested", "internal_conservative",
              "internal_grid", "selection_confirmatory"):
        r = rep[k]
        surv = sum(1 for x in r["result"] if x["rejected"])
        r["n_surviving"] = surv
        print(f"{k:24} family m={r['family_size']:2}  survive {surv}/{len(r['result'])}")
        for x in r["result"]:
            print(f"   {x['pair']:34} p={x['p']:.2e}  p_holm={x['p_holm']:.2e}  "
                  f"thr={x['threshold']:.5f}  "
                  f"{'rejected' if x['rejected'] else 'not rejected'}")
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

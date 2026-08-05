#!/usr/bin/env python3
"""How certain is the gap between the two rule banks' coverage ceilings?

The paper's central comparison is that GRAIL's bank reaches 0.735 of the annotated chemistry and
SyGMa's 0.542, and that the ordering of realised recall is the opposite. The two ceilings have each
been reported with their own interval, and the gap between them with none -- so by this manuscript's
own standard the comparison it is built on is the one comparative claim it never certified. The
pre-submission audit (paper/SELF_CLAIMS.md, row 4) flags it; this closes it.

Nothing here re-derives a ceiling. Both arms already exist per substrate: GRAIL's are the Cfull/U
records persisted by factorize_recall.py, SyGMa's are set arithmetic over its frozen predictions and
the precomputed key table, exactly as decompose_sygma.py computes them. The two runs cover the same
1170 substrates and align on exact SMILES, so the difference can be resampled paired -- which is the
whole point, since the arms share references and substrate difficulty and an unpaired interval would
inherit variance the comparison does not have.

Micro (ratio of sums) throughout, matching both sources. The estimand is coverage_GRAIL minus
coverage_SyGMa recomputed inside each bootstrap replicate, not a difference of two independent
resamplings.

Gates, fixed before running: each arm must reproduce its published marginal to within 0.001, and the
two arms must agree on U -- the reference count per substrate -- for every substrate, since they
score the same references under the same matcher. Either failing means the alignment is wrong and
the number would be meaningless.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MATCH, N_BOOT, SEED = "inchikey_tautomer", 10000, 0
# SyGMa's marginal moved from 0.5422 to 0.5391 when the parent compound was dropped from its
# frozen predictions (scripts/sygma_fulltest_predictions.py): a handful of substrates annotate a
# reference that is tautomer-identical to the substrate itself, and re-emitting the parent was
# scoring as a hit on those.
PUBLISHED = {"GRAIL": 0.7355, "SyGMa": 0.5391}
TOL = 0.001
OUT = ROOT / "results" / "ceiling_gap_ci.json"


def keyed(items, table):
    """Per-item keys in rank order, deduplicated -- the paper's matcher convention."""
    out, seen = [], set()
    for it in items:
        k = table.get(it)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def main() -> int:
    grail = json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    preds = json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text())
    table = json.loads((ROOT / "results" / "key_tables" / f"{MATCH}.json").read_text())

    g_by_sub = {r["sub"]: r for r in grail}
    subs = sorted(s for s in truth if truth[s] and s in preds and s in g_by_sub)
    print(f"substrates scored by both banks: {len(subs)}", flush=True)
    if len(subs) != len(g_by_sub) or len(subs) != len([s for s in truth if truth[s] and s in preds]):
        raise SystemExit("substrate sets differ between the two arms -- alignment is not exact")

    U_g = np.array([g_by_sub[s]["U"] for s in subs])
    C_g = np.array([g_by_sub[s]["Cfull"] for s in subs])
    U_s, C_s = [], []
    for s in subs:
        refs = {table[r] for r in truth[s] if table.get(r)}
        pool = set(keyed(preds[s], table))
        U_s.append(len(refs))
        C_s.append(len(refs & pool))
    U_s, C_s = np.array(U_s), np.array(C_s)

    # Gate 1: the two arms must be scoring the same references.
    bad = int((U_g != U_s).sum())
    print(f"substrates where the two arms disagree on the reference count: {bad}")
    if bad:
        raise SystemExit(f"U differs on {bad} substrates -- the arms are not scoring the same references")

    U = U_g
    cov = lambda C, idx: C[idx].sum() / U[idx].sum()
    full = np.arange(len(subs))
    point = {"GRAIL": float(cov(C_g, full)), "SyGMa": float(cov(C_s, full))}

    # Gate 2: each arm must reproduce the number the paper prints.
    print(f"{'arm':10}{'recomputed':>12}{'published':>12}")
    for arm, v in point.items():
        print(f"{arm:10}{v:12.4f}{PUBLISHED[arm]:12.4f}")
        if abs(v - PUBLISHED[arm]) > TOL:
            raise SystemExit(f"{arm} recomputes to {v:.4f} against a published {PUBLISHED[arm]} "
                             f"-- refusing to report a gap between numbers that are not the paper's")

    gap = point["GRAIL"] - point["SyGMa"]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    boot = np.array([cov(C_g, i) - cov(C_s, i) for i in idx])
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    rep = {"match": MATCH, "n_substrates": len(subs), "n_boot": N_BOOT, "seed": SEED,
           "estimand": "micro coverage_bank(GRAIL) - micro coverage_bank(SyGMa), paired on substrate",
           "coverage": {k: round(v, 4) for k, v in point.items()},
           "gap": round(float(gap), 4), "ci95": [round(lo, 4), round(hi, 4)],
           "excludes_zero": bool(lo > 0 or hi < 0)}
    print(f"\ngap = {gap:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"{'SIG' if rep['excludes_zero'] else 'n.s.'}")
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""What the GRAIL x MetaTox ensemble is worth once both arms are given the same output budget.

Two things make the existing complementarity measurement unsafe to build on, and both were declared
in advance by the script that produced it.

The first is a budget. It scores each method at k=15 and their union at whatever that union
contains, so the union is allowed twice the output. This paper's own finding is that recall rewards
a method for emitting more, so an ensemble compared that way is credited for its budget as much as
for its complementarity, and the gain cannot be read as either. Every arm here gets the same total
budget, and the ensemble's gain is what survives that.

The second is the falsifier the earlier run named: it used MetaTox layer 1 WITHOUT its SMIRKS-rule
variant, and recorded that if the SMIRKS version overlapped GRAIL's rule bank more, the
complementarity should shrink. The SMIRKS predictions have since arrived, so that is re-run here
rather than left as a caveat.

Arms, all on the substrates where both methods predict and references exist:

  GRAIL, MetaTox            each alone, truncated to the budget
  union, unmatched          both at the full budget each, which is what the earlier figure reported
  union, matched            interleaved by rank and truncated to ONE budget, so the ensemble is
                            charged for its output exactly as a single method is

Interleaving by rank is the policy a deployment would use without a joint scorer, and it is fixed
here rather than chosen from the outcome: rank 1 of each, then rank 2 of each, deduplicated by the
paper's matching key, first occurrence winning.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _tautomer_inchikey as _tk

KEYS = ROOT / "results" / "key_tables" / "inchikey_tautomer.json"
BUDGETS = (5, 10, 15, 30)
N_BOOT, SEED = 10000, 0


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


def interleave(a: list, b: list) -> list:
    """Rank 1 of each, then rank 2 of each, and so on. Fixed before the measurement, not after."""
    out = []
    for x, y in itertools.zip_longest(a, b):
        if x is not None:
            out.append(x)
        if y is not None:
            out.append(y)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grail", default="results/scored_predictions.json")
    ap.add_argument("--metatox", default="results/metatox_smirks_preds.json")
    ap.add_argument("--out", default=str(ROOT / "results" / "ensemble_budget_matched.json"))
    args = ap.parse_args()

    cache = json.loads(KEYS.read_text()) if KEYS.exists() else {}

    def key(s):
        k = cache.get(s)
        if k is None:
            try:
                k = _tk(s)
            except Exception:
                k = None
            cache[s] = k
        return k

    grail = {r["sub"]: [c["smiles"] for c in r["candidates"]]
             for r in json.loads((ROOT / args.grail).read_text())["rows"]}
    mt = json.loads((ROOT / args.metatox).read_text())
    mt = mt["predictions"] if "predictions" in mt else mt
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    subs = sorted(set(truth) & set(grail) & set(mt))
    print(f"population: {len(subs)} substrates carrying references and both methods' predictions",
          flush=True)

    def ranked(pool, s):
        """The method's own order, parent and duplicates dropped before any budget is applied."""
        out, seen, pk = [], set(), key(s)
        for p in pool.get(s, []):
            k = key(p)
            if k is None or k in seen or k == pk:
                continue
            seen.add(k)
            out.append(k)
        return out

    rows = []
    for s in subs:
        refs = {key(y) for y in truth[s]} - {None}
        if not refs:
            continue
        g, m = ranked(grail, s), ranked(mt, s)
        row = {"sub": s, "u": len(refs)}
        for b in BUDGETS:
            gb, mb = g[:b], m[:b]
            pool = list(dict.fromkeys(g + m))
            # what a perfect ranker over the joint pool would reach at this budget: the headroom a
            # joint scorer competes for, and the number that says whether one is worth building
            hit = [c for c in pool if c in refs]
            arms = {"grail": gb, "metatox": mb,
                    "union_unmatched": list(dict.fromkeys(gb + mb)),
                    "union_matched": list(dict.fromkeys(interleave(g, m)))[:b],
                    "union_oracle": hit[:b],
                    "union_pool": pool}
            for name, cand in arms.items():
                row[f"{name}@{b}"] = len(refs & set(cand))
                row[f"n_{name}@{b}"] = len(cand)
        rows.append(row)

    U = np.array([r["u"] for r in rows], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))

    def micro(name, b):
        H = np.array([r[f"{name}@{b}"] for r in rows], dtype=float)
        return float(H.sum() / U.sum()), H

    def paired(a, b_, budget):
        _, Ha = micro(a, budget)
        _, Hb = micro(b_, budget)
        d = Ha - Hb
        bt = np.array([d[j].sum() / max(U[j].sum(), 1) for j in idx])
        return {"delta": round(float(d.sum() / U.sum()), 4),
                "ci95": [round(float(np.quantile(bt, .025)), 4),
                         round(float(np.quantile(bt, .975)), 4)]}

    rep = {"config": {**_code_version(), "n_substrates": len(rows),
                      "references": int(U.sum()), "match": "inchikey_tautomer",
                      "aggregation": "micro, ratio of sums", "n_boot": N_BOOT, "seed": SEED,
                      "metatox_variant": "SMIRKS, the falsifier the earlier run named",
                      "interleave": "rank 1 of each, then rank 2, deduplicated; fixed in advance"},
           "by_budget": {}}
    print(f"\n  {'budget':>6}  {'GRAIL':>7} {'MetaTox':>8} {'union@2b':>9} {'union@b':>8} "
          f"{'oracle@b':>8}   {'pool':>9}")
    for b in BUDGETS:
        vals = {n: round(micro(n, b)[0], 4)
                for n in ("grail", "metatox", "union_unmatched", "union_matched", "union_oracle",
                          "union_pool")}
        emitted = {n: round(float(np.mean([r[f"n_{n}@{b}"] for r in rows])), 1)
                   for n in ("grail", "metatox", "union_unmatched", "union_matched", "union_oracle",
                          "union_pool")}
        best = "grail" if vals["grail"] >= vals["metatox"] else "metatox"
        rep["by_budget"][b] = {
            "recall": vals, "mean_emitted": emitted, "better_single": best,
            "union_matched_minus_better_single": paired("union_matched", best, b),
            "union_unmatched_minus_better_single": paired("union_unmatched", best, b)}
        print(f"  {b:>6}  {vals['grail']:>7} {vals['metatox']:>8} {vals['union_unmatched']:>9} "
              f"{vals['union_matched']:>8} {vals['union_oracle']:>8}   {emitted['union_pool']:>9}")

    print(f"\n  the ensemble's gain over the better single method, at the SAME budget:")
    for b in BUDGETS:
        v = rep["by_budget"][b]
        u = v["union_matched_minus_better_single"]
        w = v["union_unmatched_minus_better_single"]
        print(f"    k={b:>2}  matched {u['delta']:+.4f} {u['ci95']}      "
              f"unmatched (twice the output) {w['delta']:+.4f} {w['ci95']}")

    rep["per_substrate"] = rows
    Path(args.out).write_text(json.dumps(rep, indent=1))
    KEYS.parent.mkdir(parents=True, exist_ok=True)
    KEYS.write_text(json.dumps(cache))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

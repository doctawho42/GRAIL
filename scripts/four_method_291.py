#!/usr/bin/env python3
"""Four methods on one population, spanning an order of magnitude in output size.

The paper's central empirical claim is that a leaderboard in this field is driven by how many
candidates a method emits, and until now the evidence rested on one outlier: SyGMa at 81 against
everyone else's 8 to 14. One point is a coincidence waiting to be named. The SMIRKS-variant MetaTox
predictions add a second high-output method at 36, on 291 substrates where all four methods and the
references are present, which turns the claim into something with a slope rather than a gap.

Everything is scored through the paper's own matcher and its five criteria, on frozen predictions,
so only the match rule and the budget vary and never the chemistry. Two hazards are checked rather
than assumed: a method that returns its own substrate among its predictions inflates nothing here
but has silently inflated pools in this project before, and a method whose predictions are not
deduplicated gets a larger emitted count for free.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _tautomer_inchikey as _tk

KEYS = ROOT / "results" / "key_tables" / "inchikey_tautomer.json"
N_BOOT, SEED = 10000, 0
KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)


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


def load_pools() -> dict:
    grail = {r["sub"]: [c["smiles"] for c in r["candidates"]]
             for r in json.loads((ROOT / "results/scored_predictions.json").read_text())["rows"]}
    return {
        "GRAIL": grail,
        "MetaPredictor": json.loads((ROOT / "artifacts/tier2_1170/metapredictor_preds.json").read_text()),
        "MetaTox": json.loads((ROOT / "results/metatox_smirks_preds.json").read_text())["predictions"],
        "SyGMa": json.loads((ROOT / "results/sygma_fulltest_predictions.json").read_text()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "four_method_291.json"))
    args = ap.parse_args()

    cache = json.loads(KEYS.read_text())
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    pools = load_pools()
    subs = sorted(set(truth) & set.intersection(*(set(p) for p in pools.values())))
    print(f"population: {len(subs)} substrates carrying references and all four prediction sets")

    def key(s: str):
        k = cache.get(s)
        if k is None:
            try:
                k = _tk(s)
            except Exception:
                k = None
            cache[s] = k
        return k

    # hazard 1: a method returning the substrate itself. It cannot score, but it consumes a slot.
    parent_in_own = {}
    for name, pool in pools.items():
        n = sum(1 for s in subs if key(s) in {key(p) for p in pool.get(s, [])[:60]})
        parent_in_own[name] = n
    print("  substrates where a method returns its own parent:",
          {k: v for k, v in parent_in_own.items()})

    # Per-substrate rows, kept so a margin can carry an interval. This paper's own standard is
    # that a reordering is a description until a paired interval makes it a claim; the sweep below
    # is a reordering claim and was reported without one.
    rows = {name: {"U": [], **{k: [] for k in KS}} for name in pools}
    per = {}
    for name, pool in pools.items():
        U = 0
        hits = {k: 0 for k in KS}
        emitted = {k: 0 for k in KS}
        raw, dedup = 0, 0
        for s in subs:
            refs = {key(y) for y in truth.get(s, [])} - {None}
            if not refs:
                continue
            U += len(refs)
            seq, seen = [], set()
            for p in pool.get(s, []):
                k = key(p)
                if k is None or k in seen or k == key(s):
                    continue           # drop the parent and any duplicate before the budget bites
                seen.add(k)
                seq.append(k)
            raw += len(pool.get(s, []))
            dedup += len(seq)
            rows[name]["U"].append(len(refs))
            for k in KS:
                h = len(refs & set(seq[:k]))
                hits[k] += h
                emitted[k] += min(k, len(seq))
                rows[name][k].append(h)
        per[name] = {"references": U,
                     "raw_predictions": raw, "after_dedup_and_parent_drop": dedup,
                     "mean_emitted_uncapped": round(dedup / len(subs), 2),
                     "recall": {k: round(hits[k] / max(U, 1), 4) for k in KS},
                     "mean_emitted": {k: round(emitted[k] / len(subs), 2) for k in KS}}

    order = sorted(per, key=lambda m: -per[m]["mean_emitted_uncapped"])
    print(f"\n  {'method':15} {'emitted':>8}  " + "  ".join(f"r@{k:<3}" for k in KS))
    for name in order:
        v = per[name]
        print(f"  {name:15} {v['mean_emitted_uncapped']:>8}  "
              + "  ".join(f"{v['recall'][k]:<5}" for k in KS))

    seen_orders = {}
    for k in KS:
        o = tuple(sorted(per, key=lambda m: -per[m]["recall"][k]))
        seen_orders.setdefault(o, []).append(k)
    print(f"\n  distinct orderings across the budget sweep: {len(seen_orders)}")
    for o, ks in seen_orders.items():
        print(f"    {' > '.join(o)}   at k in {ks}")

    # Paired bootstrap on every pairwise margin at every budget. The substrates are shared by all
    # four methods by construction of the population, so one index draw resamples every method
    # together and the margin is a paired quantity rather than a difference of two marginals.
    import itertools
    import numpy as np
    U = np.array(rows[next(iter(rows))]["U"], dtype=float)
    assert all(np.array_equal(U, np.array(r["U"], dtype=float)) for r in rows.values()), \
        "the four methods are not scored against the same per-substrate reference counts"
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(U), (N_BOOT, len(U)))
    denom = U[idx].sum(axis=1)
    margins = {}
    for a, b in itertools.combinations(sorted(rows), 2):
        for k in KS:
            d = np.array(rows[a][k], dtype=float) - np.array(rows[b][k], dtype=float)
            bt = d[idx].sum(axis=1) / np.maximum(denom, 1)
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            margins[f"{a} vs {b} @ {k}"] = {
                "margin": round(float(d.sum() / max(U.sum(), 1)), 4),
                "ci95": [round(lo, 4), round(hi, 4)], "separable": bool(lo * hi > 0)}
    # A sign change is a description; a sign change between two margins that each separate from
    # zero is a reversal of a result. The distinction is the one this paper insists on elsewhere and
    # it has to be applied to our own claim, so it is computed rather than asserted.
    reversals = {}
    for a, b in itertools.combinations(sorted(rows), 2):
        vals = [(k, margins[f"{a} vs {b} @ {k}"]) for k in KS]
        pos = [(k, v) for k, v in vals if v["margin"] > 0 and v["separable"]]
        neg = [(k, v) for k, v in vals if v["margin"] < 0 and v["separable"]]
        if len({v["margin"] > 0 for _, v in vals}) > 1:
            reversals[f"{a} vs {b}"] = {
                "certified_both_ends": bool(pos and neg),
                "ahead_at": (pos[0][0] if pos else None), "behind_at": (neg[-1][0] if neg else None),
                "ahead_margin": (pos[0][1] if pos else None),
                "behind_margin": (neg[-1][1] if neg else None)}
    n_strong = sum(v["certified_both_ends"] for v in reversals.values())
    print(f"\n  {len(reversals)} pairs change sign across the sweep, "
          f"{n_strong} across margins certified at both ends")
    for k_, v in reversals.items():
        if v["certified_both_ends"]:
            print(f"    {k_:32} ahead at k={v['ahead_at']} by {v['ahead_margin']['margin']:+.4f}, "
                  f"behind at k={v['behind_at']} by {v['behind_margin']['margin']:+.4f}")

    # Section 3 calls a difference certified only when its interval excludes zero AND it survives
    # Holm in a declared family. This sweep had intervals and no family, so nothing in it could be
    # certified under the paper's own rule while the abstract leaned on it. The family is the grid
    # the axis admits: every method pair at every budget. The p-values are sign-flip rather than a
    # normal tail on a stored interval, since the per-substrate differences are here.
    rng2 = np.random.default_rng(SEED)
    pvals = {}
    for a, b in itertools.combinations(sorted(rows), 2):
        for k in KS:
            d = np.array(rows[a][k], dtype=float) - np.array(rows[b][k], dtype=float)
            nz = d[d != 0.0]
            if nz.size == 0:
                pvals[f"{a} vs {b} @ {k}"] = 1.0
                continue
            obs = abs(nz.mean())
            signs = rng2.choice([-1.0, 1.0], size=(20000, nz.size))
            null = np.abs((signs * nz).mean(axis=1))
            pvals[f"{a} vs {b} @ {k}"] = float((1 + (null >= obs - 1e-15).sum()) / 20001)
    order = sorted(pvals, key=lambda k_: pvals[k_])
    m_fam, alive, holm = len(order), True, {}
    for i, k_ in enumerate(order):
        thr = 0.05 / (m_fam - i)
        ok = alive and pvals[k_] <= thr
        alive = ok
        holm[k_] = {"p": round(pvals[k_], 6), "threshold": round(thr, 6), "survives": bool(ok)}
    n_holm = sum(v["survives"] for v in holm.values())
    # a sign change is certified only if BOTH ends survive the correction, not merely the interval
    for k_, v in reversed(list(reversals.items())):
        a_, b_ = v["ahead_at"], v["behind_at"]
        both = (a_ is not None and b_ is not None
                and holm[f"{k_} @ {a_}"]["survives"] and holm[f"{k_} @ {b_}"]["survives"])
        v["certified_both_ends_after_holm"] = bool(both)
    n_holm_rev = sum(v.get("certified_both_ends_after_holm", False) for v in reversals.values())
    print(f"  Holm over the {m_fam}-cell grid: {n_holm} margins survive")
    print(f"  sign changes with BOTH ends surviving Holm: {n_holm_rev}")

    # The same instrument as scripts/robust_order.py, on the budget axis alone: a pair survives
    # when the method the field's budget ranks higher is ahead at every budget in the sweep.
    # Macro F1 at each budget, which recall alone cannot show: the method that leads at the field's
    # budget emits three times what the next one does, and F1 is where that is paid for.
    f1 = {}
    for name in rows:
        f1[name] = {}
        for k in KS:
            vals = []
            for i in range(len(rows[name]["U"])):
                u = rows[name]["U"][i]
                if not u:
                    continue
                h = rows[name][k][i]
                e = min(k, per[name]["after_dedup_and_parent_drop"] / max(len(rows[name]["U"]), 1))
                e = max(e, 1e-9)
                pr, rc = h / e, h / u
                vals.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
            f1[name][k] = round(float(np.mean(vals)) if vals else 0.0, 4)
    print("\n  macro F1 by budget:")
    for name in sorted(f1):
        print(f"    {name:15} " + "  ".join(f"k{k}:{f1[name][k]:.3f}" for k in (1, 5, 15, 50)))

    published = sorted(rows, key=lambda m: -per[m]["recall"][15])
    prank = {m: i for i, m in enumerate(published)}
    robust = {}
    for a, b in itertools.combinations(sorted(rows), 2):
        hi_, lo_ = (a, b) if prank[a] < prank[b] else (b, a)
        cells = {}
        for k in KS:
            v = margins[f"{min(hi_, lo_)} vs {max(hi_, lo_)} @ {k}"]
            sign = 1 if hi_ < lo_ else -1
            cells[k] = {"margin": round(sign * v["margin"], 4),
                        "positive": (sign * v["margin"]) > 0,
                        "certified": v["separable"] and (sign * v["margin"]) > 0}
        robust[f"{hi_} over {lo_}"] = {
            "dominates": all(c["positive"] for c in cells.values()),
            "certified": all(c["certified"] for c in cells.values()),
            "budgets_that_reverse_it": [k for k, c in cells.items() if not c["positive"]]}
    n_p = len(robust)
    n_d = sum(v["dominates"] for v in robust.values())
    n_c = sum(v["certified"] for v in robust.values())
    print(f"\n  robust order over the budget grid: {n_d}/{n_p} pairs survive every budget, "
          f"{n_c} certified in every budget")

    n_sep = sum(v["separable"] for v in margins.values())
    print(f"\n  {n_sep} of {len(margins)} pairwise margins separate from zero at 95%")
    # the specific claim the sweep is quoted for: is the mover's position certified at either end?
    mover_rows = {kk: v for kk, v in margins.items() if "MetaTox" in kk}
    print(f"  of those involving the mover, {sum(v['separable'] for v in mover_rows.values())} "
          f"of {len(mover_rows)} are separable")

    rep = {"config": {**_code_version(), "n_substrates": len(subs), "match": "inchikey_tautomer",
                      "n_boot": N_BOOT, "seed": SEED,
                      "aggregation": "micro, ratio of sums", "k_sweep": list(KS),
                      "population": "the 291 MetaTox submission substrates, all of which carry "
                                    "references and predictions from every method here",
                      "parent_and_duplicates": "dropped before the budget is applied"},
           "parent_returned_by_method": parent_in_own,
           "per_method": per,
           "orderings": {" > ".join(o): ks for o, ks in seen_orders.items()},
           "pairwise_margins": margins,
           "n_margins": len(margins), "n_separable": n_sep,
           "macro_f1_by_budget": f1,
           "robust_order": {"published_order": published, "pairs": robust,
                            "n_pairs": n_p, "n_dominating": n_d, "n_certified": n_c,
                            "robustness": round(n_d / max(n_p, 1), 4),
                            "certified_robustness": round(n_c / max(n_p, 1), 4)},
           "sign_changes": reversals, "n_certified_both_ends": n_strong,
           "holm": {"family_size": m_fam, "n_surviving": n_holm,
                    "n_sign_changes_certified_both_ends": n_holm_rev, "by_cell": holm},
           "n_pairs_changing_sign": len(reversals)}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

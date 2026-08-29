#!/usr/bin/env python3
"""What the parent-drop convention costs and buys, per arm.

The comparison table declares that a prediction equal to the substrate is dropped before the
budget, for every method alike. Every other declared choice in this paper is swept -- the matching
criterion, the output budget, the grouping -- and this one was declared and never measured. It is
not a neutral bookkeeping rule: returning the parent consumes a slot, so dropping it promotes
whatever sat behind it, and an arm that returns the parent often gains more than one that never
does.

It can also cost. Two of these substrates carry their own tautomer key among their references, so
for those the convention discards a hit that the annotation counts.

Both arms of the difference are reported per method, with the paired interval, and with the count
of substrates whose returned list contains the parent at all.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
CAP = 100
N_BOOT, SEED = 10000, 0
COMPARATORS = {
    "MetaTox": ("results/metatox_smirks_preds.json", "predictions"),
    "SyGMa": ("results/sygma_fulltest_predictions.json", None),
    "MetaPredictor": ("artifacts/tier2_1170/metapredictor_preds.json", None),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/parent_drop_effect.json"))
    args = ap.parse_args()

    from bank_without_selection import _dedup, _key as tautkey

    pools, refs = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    small = json.loads((ROOT / "results/widepools_k30/all.json").read_text())["pools"]
    subs = sorted(s for s in pools if refs.get(s) and s in small)
    parent = {s: tautkey(s) for s in subs}

    def ranked(pool, s):
        keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
        return [c["key"] for c in rrf_order(keep) if c.get("key")]

    arms = {}
    for s in subs:
        arms.setdefault("GRAIL exhaustive", {})[s] = ranked(pools[s], s)
        arms.setdefault("GRAIL interactive", {})[s] = ranked(small[s], s)
    for name, (rel, key) in COMPARATORS.items():
        blob = json.loads((ROOT / rel).read_text())
        preds = blob[key] if key else blob
        for s in subs:
            arms.setdefault(name, {})[s] = [k for k in _dedup(preds.get(s, []), max(KS) + 20)
                                            if k][:max(KS) + 20]

    U = np.array([len(refs[s]) for s in subs], dtype=float)
    N = float(U.sum())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    den = np.maximum(U[idx].sum(axis=1), 1)

    # how often the parent is returned at all, and where
    presence = {}
    for a, per in arms.items():
        ranks = [per[s].index(parent[s]) + 1 for s in subs if parent[s] in per[s]]
        presence[a] = {"substrates_returning_the_parent": len(ranks),
                       "of": len(subs),
                       "share": round(len(ranks) / len(subs), 4),
                       "median_rank_when_returned": int(np.median(ranks)) if ranks else None,
                       "within_top_15": int(sum(1 for r in ranks if r <= 15))}

    # references that are the substrate's own key, which the convention discards
    self_ref = [s for s in subs if parent[s] in set(refs[s])]

    out = {}
    for a, per in arms.items():
        row = {}
        for k in KS:
            def hits(drop):
                v = []
                for s in subs:
                    seq = per[s]
                    if drop:
                        seq = [x for x in seq if x != parent[s]]
                    v.append(len(set(seq[:k]) & set(refs[s])))
                return np.array(v, dtype=float)
            with_, without = hits(True), hits(False)
            d = with_ - without
            bt = d[idx].sum(axis=1) / den
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            row[str(k)] = {
                "with_convention": round(float(with_.sum() / N), 4),
                "without": round(float(without.sum() / N), 4),
                "effect": round(float(d.sum() / N), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "separates": bool(lo > 0 or hi < 0),
            }
        out[a] = row

    rep = {"provenance": stamp(__file__),
           "population": {"n": len(subs), "n_references": N, "source": "the comparison set"},
           "aggregation": "micro, ratio of sums", "cap": CAP, "n_boot": N_BOOT, "seed": SEED,
           "convention": ("a prediction whose tautomer key equals the substrate's is removed "
                          "before the budget is applied"),
           "parent_returned": presence,
           "substrates_whose_own_key_is_a_reference": len(self_ref),
           "effect": out,
           "reading": ("the effect is what the convention gives an arm: positive where dropping "
                       "the parent promotes a hit into the window, negative where the parent was "
                       "itself an annotated reference")}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"substrates whose own key is among their references: {len(self_ref)} of {len(subs)}\n")
    print(f"{'arm':<20}{'returns parent':>16}{'median rank':>13}{'in top 15':>11}")
    for a, p in presence.items():
        print(f"{a:<20}{p['substrates_returning_the_parent']:>10} /{p['of']:<4}"
              f"{str(p['median_rank_when_returned']):>13}{p['within_top_15']:>11}")
    print(f"\neffect of the convention, micro recall\n{'arm':<20}" + "".join(f"{k:>9}" for k in KS))
    for a in out:
        cells = "".join(f"{out[a][str(k)]['effect']:>+9.4f}" for k in KS)
        print(f"{a:<20}{cells}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

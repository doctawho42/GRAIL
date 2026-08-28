#!/usr/bin/env python3
"""Does the comparison's verdict survive the matching criterion, or is it an artefact of one?

The paper's claim is that a recall figure depends on evaluation choices its literature leaves
unstated, and that reporting one cell hides that dependence. The claim is made about other
people's numbers, and the honest test is our own: the comparison table is computed under a
tautomer-aware key, four other criteria are declared alongside it, and nothing so far has checked
whether the three-way division -- SyGMa ahead at the head, neither side separating in the middle,
GRAIL ahead at depth -- is a property of the systems or of that one key.

The arms, the population, the parent-drop convention and the pool cap are exactly the deployment
table's. Only the criterion moves.

A verdict per budget is one of three, read from the paired interval and never from the point
estimate: `trails` where the strongest comparator leads with the interval excluding zero, `leads`
where GRAIL's better arm does, `neither` where the interval covers zero.

    python scripts/typed_edit/criterion_sweep.py
    python scripts/typed_edit/criterion_sweep.py --limit 10     # timing probe
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

CRITERIA = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
CAP = 100
N_BOOT, SEED = 10000, 0
COMPARATORS = {
    "MetaTox": ("results/metatox_smirks_preds.json", "predictions"),
    "SyGMa": ("results/sygma_fulltest_predictions.json", None),
    "MetaPredictor": ("artifacts/tier2_1170/metapredictor_preds.json", None),
}


def load(spec):
    pools, refs = {}, {}
    for f in sorted(glob.glob(spec)) or [spec]:
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    return pools, refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exhaustive", default="results/widepools_implicit/w*.json")
    ap.add_argument("--interactive", default=str(ROOT / "results/widepools_k30/all.json"))
    ap.add_argument("--out", default=str(ROOT / "results/criterion_sweep.json"))
    ap.add_argument("--limit", type=int, default=0, help="substrates, for a timing probe")
    args = ap.parse_args()

    from bank_without_selection import _key as _cached_tautomer_key
    from grail_metabolism.metrics import _match_keys
    from run_match_sensitivity import _dedup_canon
    from vs_metatox import population

    def key_of(smiles, crit):
        """One structure's key under one criterion.

        The tautomer key is the project's default and by far the most expensive: canonicalisation
        runs at roughly seven structures a second, which put this criterion alone at about two
        hours. `bank_without_selection._key` reads the 108,967-entry table the rest of the project
        shares and computes only on a miss, so the same key comes back in seconds. The other four
        criteria are cheap and go through the metrics helper unchanged.
        """
        if crit == "inchikey_tautomer":
            return _cached_tautomer_key(smiles)
        return next(iter(_match_keys([smiles], crit)))

    big, refs_b = load(args.exhaustive)
    small, refs_s = load(args.interactive)
    refs = {**refs_b, **refs_s}
    _, truth, _ = population()
    subs = sorted(s for s in set(big) & set(small) if refs.get(s) and truth.get(s))
    if args.limit:
        subs = subs[:args.limit]
    print(f"{len(subs)} substrates", file=sys.stderr, flush=True)

    # The structures each arm returns, in its own order, before any key is taken. Ranking is
    # settled here so the criterion cannot change what is ranked, only what counts as a hit.
    def ordered(pool):
        return [c["smiles"] for c in rrf_order(sorted(pool, key=lambda c: -c["generator"])[:CAP])]

    arms = {"GRAIL exhaustive": {s: ordered(big[s]) for s in subs},
            "GRAIL interactive": {s: ordered(small[s]) for s in subs}}
    for name, (rel, key) in COMPARATORS.items():
        blob = json.loads((ROOT / rel).read_text())
        preds = blob[key] if key else blob
        arms[name] = {s: list(preds.get(s, []))[:max(KS) + 20] for s in subs}

    U_all = {}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    ours = ("GRAIL exhaustive", "GRAIL interactive")

    out, t0 = {}, time.perf_counter()
    for crit in CRITERIA:
        print(f"  {crit} ...", file=sys.stderr, flush=True)
        real = {s: {k for k in (key_of(t, crit) for t in truth[s]) if k} for s in subs}
        # the parent-drop convention, applied under this criterion so it means the same thing
        parent = {s: key_of(s, crit) for s in subs}

        def keyed(arm):
            out_ = {}
            for s in subs:
                ks, seen = [], set()
                for sm in _dedup_canon(arms[arm][s]):
                    k = key_of(sm, crit)
                    if k and k != parent[s] and k not in seen:
                        seen.add(k); ks.append(k)
                out_[s] = ks
            return out_

        K = {a: keyed(a) for a in arms}
        U = np.array([len(real[s]) for s in subs], dtype=float)
        U_all[crit] = float(U.sum())
        denom = np.maximum(U[idx].sum(axis=1), 1)

        def hits(a, k):
            return np.array([len(set(K[a][s][:k]) & real[s]) for s in subs], dtype=float)

        rec = {a: {k: round(float(hits(a, k).sum() / max(U.sum(), 1)), 4) for k in KS}
               for a in arms}
        verdicts, margins = {}, {}
        for k in KS:
            best_ours = max(ours, key=lambda a: rec[a][k])
            best_other = max((a for a in arms if a not in ours), key=lambda a: rec[a][k])
            d = hits(best_ours, k) - hits(best_other, k)
            bt = d[idx].sum(axis=1) / denom
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            verdicts[k] = "leads" if lo > 0 else ("trails" if hi < 0 else "neither")
            margins[k] = {"ours": best_ours, "theirs": best_other,
                          "gap": round(float(d.sum() / max(U.sum(), 1)), 4),
                          "ci95": [round(lo, 4), round(hi, 4)]}
        out[crit] = {"n_references": U_all[crit],
                     "recall_micro": {a: {str(k): rec[a][k] for k in KS} for a in arms},
                     "verdict_by_budget": {str(k): verdicts[k] for k in KS},
                     "margin_by_budget": {str(k): margins[k] for k in KS}}
        print(f"    {'  '.join(f'{k}:{verdicts[k]}' for k in KS)}", file=sys.stderr, flush=True)

    ref = out["inchikey_tautomer"]["verdict_by_budget"]
    disagree = {c: [k for k in ref if out[c]["verdict_by_budget"][k] != ref[k]]
                for c in CRITERIA if c != "inchikey_tautomer"}
    rep = {
        "provenance": stamp(__file__),
        "population": {"n": len(subs), "source": "the comparison set of results/four_method_291.json"},
        "aggregation": "micro, ratio of sums",
        "cap": CAP, "n_boot": N_BOOT, "seed": SEED,
        "note": ("ranking is fixed before the key is taken, so the criterion changes what counts "
                 "as a hit and never what is ranked; the parent-drop convention is re-derived "
                 "under each criterion so it means the same thing in every column"),
        "criteria": CRITERIA,
        "reference_criterion": "inchikey_tautomer",
        "by_criterion": out,
        "budgets_whose_verdict_moves": disagree,
        "n_budgets_moving": {c: len(v) for c, v in disagree.items()},
        "reading": ("a verdict that holds across every criterion is a property of the systems; a "
                    "verdict that moves is a property of the criterion, and reporting one cell "
                    "would have hidden which"),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out} in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Is the between-group headroom about formula, or about the transformation?

The oracle that motivated H8, H12 and H14 was measured over groups defined by the candidate's
molecular formula, and all three ways of spending that signal are now closed: H8's blocked design
spends more than it draws, H12's third reciprocal-rank term is bounded to the same +0.0286 a
perfect ranking reaches, and H14's gate is worse than the cap it must beat even with a perfect
ranking. Before declaring the +0.1729 unreachable, the grouping itself is worth checking, because
formula may be a proxy.

This recomputes the same oracle under three partitions of one pool:

  formula     the candidate's molecular formula, which is what was measured
  type        the reaction type of substrate -> candidate, the bond-delta multiset the bank's
              coverage is defined on and the vocabulary H1 registers its label space in
  both        their conjunction, an upper bound on either

If the headroom under type is comparable or larger, it is reachable along a line already
registered and no fourth mechanism is needed. If there is headroom under formula and none under
type, the signal is about elemental composition and belongs as a feature of the filter rather than
as a stage of its own.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

KS = (1, 5, 10, 15, 20, 30, 50)
N_BOOT, SEED = 10000, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="results/widepools_implicit/w*.json")
    ap.add_argument("--out", default=str(ROOT / "results/oracle_by_grouping.json"))
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import rdMolDescriptors

    from coverage_gap_types import pair_to_type

    pools, refs = {}, {}
    for f in sorted(glob.glob(args.pools)):
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    subs = sorted(s for s in pools if refs.get(s))
    print(f"{len(subs)} substrates", file=sys.stderr, flush=True)

    def formula(smiles, cache={}):
        if smiles not in cache:
            m = Chem.MolFromSmiles(smiles)
            cache[smiles] = rdMolDescriptors.CalcMolFormula(m) if m else smiles
        return cache[smiles]

    PART = ("formula", "type", "both")
    hits = {p: {k: [] for k in KS} for p in ("fusion",) + PART}
    n_ref, sizes = [], {p: [] for p in PART}
    untyped = 0
    t0 = time.perf_counter()

    for n, s in enumerate(subs, 1):
        if n % 25 == 0:
            print(f"  {n}/{len(subs)} ({time.perf_counter() - t0:.0f}s)",
                  file=sys.stderr, flush=True)
        real = set(refs[s])
        n_ref.append(len(real))
        fused = rrf_order(pools[s])
        sub_mol = Chem.MolFromSmiles(s)

        labels = {}
        for c in fused:
            f = formula(c["smiles"])
            t = None
            if sub_mol is not None:
                m = Chem.MolFromSmiles(c["smiles"])
                if m is not None:
                    try:
                        t = pair_to_type(sub_mol, m)
                    except Exception:
                        t = None
            if t is None:
                untyped += 1
            tk = json.dumps(t, sort_keys=True) if t is not None else f"untyped:{f}"
            labels[id(c)] = {"formula": f, "type": tk, "both": f + "|" + tk}

        for k in KS:
            hits["fusion"][k].append(len(set(c["key"] for c in fused[:k]) & real))

        for p in PART:
            g = defaultdict(list)
            for c in fused:
                g[labels[id(c)][p]].append(c)
            sizes[p].append(len(g))
            order = sorted(g, key=lambda x: not any(c["key"] in real for c in g[x]))
            flat = [c["key"] for x in order for c in g[x]]
            for k in KS:
                hits[p][k].append(len(set(flat[:k]) & real))

    U = np.array(n_ref, dtype=float)
    N = float(U.sum())
    H = {a: {k: np.array(v[k], dtype=float) for k in KS} for a, v in hits.items()}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    den = np.maximum(U[idx].sum(axis=1), 1)

    def contrast(a, b):
        d = a - b
        bt = d[idx].sum(axis=1) / den
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(d.sum() / N), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    kk = args.k
    head = {p: contrast(H[p][kk], H["fusion"][kk]) for p in PART}
    rep = {"provenance": stamp(__file__),
           "population": {"n": len(subs), "n_references": N,
                          "source": "the 291 of results/four_method_291.json"},
           "aggregation": "micro, ratio of sums", "k": kk,
           "note": "every arm blocks its groups, which is what the oracle over a partition means; "
                   "the fusion arm interleaves and is the baseline all three are read against",
           "untyped_candidates": untyped,
           "groups_per_substrate": {p: round(st.mean(sizes[p]), 1) for p in PART},
           "recall_micro": {str(k): {a: round(float(H[a][k].sum() / N), 4)
                                     for a in ("fusion",) + PART} for k in KS},
           "headroom_over_fusion": head,
           "reading": ("headroom under type comparable to or larger than under formula means the "
                       "signal is about the transformation and is reachable through the typed "
                       "label space H1 registers; headroom under formula alone means it is about "
                       "elemental composition and belongs in the filter's features")}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n{'k':>4}{'fusion':>10}{'formula':>10}{'type':>10}{'both':>10}")
    for k in KS:
        r = rep["recall_micro"][str(k)]
        print(f"{k:>4}{r['fusion']:>10.4f}{r['formula']:>10.4f}{r['type']:>10.4f}{r['both']:>10.4f}")
    print(f"\nheadroom over fusion at k={kk}:")
    for p in PART:
        c = head[p]
        print(f"  {p:<10}{c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"{'*' if c['excludes_zero'] else ' '}   "
              f"{rep['groups_per_substrate'][p]} groups/substrate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from collections import Counter, defaultdict
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
    ap.add_argument("--cap", type=int, default=100,
                    help="the H9 cap. The oracle asks how well the emitted set could be ordered, "
                         "and the deployed configuration orders the hundred this keeps; typing "
                         "the whole 794-candidate pool measures a set nothing ranks and costs "
                         "65 hours, the tail being one substrate whose 4,614 candidates take "
                         "869 ms each to type.")
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

    # A finer partition wins at an oracle for free: in the limit of one candidate per group the
    # oracle is a perfect ranker. Type has 51.4 groups per substrate against formula's 41.5, so
    # its advantage has to be shown to be more than granularity. The control is a RANDOM partition
    # matched to type's group-size distribution on each substrate: if type beats it, the advantage
    # is the chemistry and not the count.
    PART = ("formula", "type", "both", "random_matched")
    import random as _random
    rng_local = _random.Random(SEED)
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
        keep = sorted(pools[s], key=lambda c: -c["generator"])[:args.cap]
        fused = rrf_order(keep)
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

        # the control: shuffle the candidates and cut them into blocks whose sizes are exactly
        # the sizes of the type groups, so the partition has the same granularity and no meaning
        block_sizes = sorted(Counter(labels[id(c)]["type"] for c in fused).values(),
                             reverse=True)
        order = list(range(len(fused)))
        rng_local.shuffle(order)
        pos, gi = 0, 0
        for sz in block_sizes:
            for j in order[pos:pos + sz]:
                labels[id(fused[j])]["random_matched"] = f"r{gi}"
            pos += sz
            gi += 1
        for c in fused:
            labels[id(c)].setdefault("random_matched", f"r{gi}")

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
    # Each arm's interval against fusion is a margin, and two margins that overlap do not
    # separate. The claim "type beats the granularity-matched control" is a paired contrast
    # between the arms themselves, on the same substrates, and it is the only one that decides.
    between = {f"{a}-{b}": contrast(H[a][kk], H[b][kk]) for a, b in
               (("type", "random_matched"), ("formula", "random_matched"),
                ("type", "formula"), ("both", "type"))}
    rep = {"provenance": stamp(__file__),
           "population": {"n": len(subs), "n_references": N,
                          "source": "the 291 of results/four_method_291.json"},
           "aggregation": "micro, ratio of sums", "k": kk, "cap": args.cap,
           "cap_note": "every arm including the fusion baseline is computed on the capped pool, "
                       "so the three partitions are compared on the set the deployed "
                       "configuration actually ranks",
           "note": "every arm blocks its groups, which is what the oracle over a partition means; "
                   "the fusion arm interleaves and is the baseline all three are read against",
           "untyped_candidates": untyped,
           "groups_per_substrate": {p: round(st.mean(sizes[p]), 1) for p in PART},
           "recall_micro": {str(k): {a: round(float(H[a][k].sum() / N), 4)
                                     for a in ("fusion",) + PART} for k in KS},
           "headroom_over_fusion": head,
           "contrasts_between_arms": between,
           "reproducibility": ("pair_to_type runs an MCS under a wall-clock timeout and returns "
                              "None when it is cancelled, so the typing is load-dependent and "
                              "this artifact is not byte-reproducible. Two runs on the same "
                              "machine differed in 2 of ~29,100 typings; every headroom and "
                              "every k=15 figure was identical to four decimals, and the only "
                              "cell that moved was random_matched at k=1, by 0.0015"),
           "control_note": ("random_matched is a random partition whose group-size multiset is "
                            "exactly type's on each substrate, so it holds granularity fixed and "
                            "carries no chemistry; type-random_matched is the test of whether the "
                            "advantage is the transformation or the count of groups"),
           "reading": ("headroom under type comparable to or larger than under formula means the "
                       "signal is about the transformation and is reachable through the typed "
                       "label space H1 registers; headroom under formula alone means it is about "
                       "elemental composition and belongs in the filter's features")}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n{'k':>4}{'fusion':>10}{'formula':>10}{'type':>10}{'both':>10}")
    for k in KS:
        r = rep["recall_micro"][str(k)]
        print(f"{k:>4}{r['fusion']:>10.4f}{r['formula']:>10.4f}{r['type']:>10.4f}{r['both']:>10.4f}")
    print(f"\nbetween arms at k={kk}:")
    for name, c in between.items():
        print(f"  {name:<26}{c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"{'*' if c['excludes_zero'] else ' '}")
    print(f"\nheadroom over fusion at k={kk}:")
    for p in PART:
        c = head[p]
        print(f"  {p:<10}{c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"{'*' if c['excludes_zero'] else ' '}   "
              f"{rep['groups_per_substrate'][p]} groups/substrate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

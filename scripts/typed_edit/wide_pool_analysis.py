#!/usr/bin/env python3
"""Three questions the wide pool answers and the deployed pool cannot.

The pool the pipeline ships is 10.8 candidates deep and reaches 0.3495 of the references. The
bank without a selector is 588.7 deep. Every question about ranking is a different question at
that depth, and two of them could not be asked at all before: the selector removes the isomers,
so within-type ordering had 122 testable cases in 1,170 substrates.

  the ceiling      what a perfect ranker over this pool would reach, uncapped
  the split        retention divided into ordering BETWEEN transformation types and WITHIN one,
                   by two oracles: each group lifted, or the groups reordered
  the combination  whether any recombination of the two frozen scores beats their product

Candidates that share a molecular formula on one substrate are isomers of each other: they
differ in where the edit landed and in nothing else, which is regioselectivity. The grouping is
a proxy and lumps distinct transformations that share a formula delta; that lumping moves work
out of the between-group problem and into the within-group one, so a poor within-group result is
the conservative reading.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

RDLogger.DisableLog("rdApp.*")
POOLS = ROOT / "results" / "wide_pools.json"
BUDGETS = [1, 3, 5, 10, 15, 30, 50]
K = 15


def formula(smiles, cache={}):
    if smiles not in cache:
        m = Chem.MolFromSmiles(smiles)
        cache[smiles] = CalcMolFormula(m) if m is not None else "?"
    return cache[smiles]


def hits(keys, real, k=None):
    """Number of references recovered in the first k, which micro and macro both need."""
    return len(set(keys[:k] if k else keys) & real)


def rec(keys, real, k=None):
    return hits(keys, real, k) / len(real) if real else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "wide_pool_analysis.json"))
    ap.add_argument("--pools", default="",
                    help="glob of pool shards; empty reads the single merged POOLS file")
    args = ap.parse_args()

    if args.pools:
        import glob as _glob
        pools, refs, src = {}, {}, args.pools
        for f in sorted(_glob.glob(args.pools)):
            d = json.loads(Path(f).read_text())
            pools.update(d["pools"]); refs.update(d["references"])
    else:
        blob = json.loads(POOLS.read_text())
        pools, refs, src = blob["pools"], blob["references"], str(POOLS)
    subs = sorted(s for s in pools if refs.get(s))
    print(f"{len(subs)} substrates, pool mean "
          f"{st.mean([len(pools[s]) for s in subs]):.1f}", file=sys.stderr, flush=True)

    ceiling, per, ranks, sizes, combo = [], [], [], [], defaultdict(list)
    n_refs, h_ceiling, h_per, h_combo = [], [], [], defaultdict(list)
    for n, s in enumerate(subs, 1):
        if n % 25 == 0:
            print(f"  {n}/{len(subs)}", file=sys.stderr, flush=True)
        cands = sorted(pools[s], key=lambda c: -c["combined"])
        real = set(refs[s])
        keys = [c["key"] for c in cands]
        ceiling.append(rec(keys, real))
        n_refs.append(len(real))
        h_ceiling.append(hits(keys, real))

        groups = defaultdict(list)
        for i, c in enumerate(cands):
            groups[formula(c["smiles"])].append(i)
        for g, idxs in groups.items():
            positions = [j for j, i in enumerate(idxs) if keys[i] in real]
            if positions and len(idxs) > 1:
                ranks.append(positions[0] + 1)
                sizes.append(len(idxs))

        best = {g: min(i) for g, i in groups.items()}
        by_score = sorted(groups, key=lambda g: best[g])
        hit_first = sorted(groups, key=lambda g: (not any(keys[i] in real for i in groups[g]),
                                                  best[g]))

        def flat(order, within):
            out = []
            for g in order:
                out.extend(within(groups[g]))
            return [keys[i] for i in out]

        plain = lambda idxs: idxs
        lift = lambda idxs: sorted(idxs, key=lambda i: keys[i] not in real)
        orders = {"as_ranked": flat(by_score, plain),
                  "oracle_within": flat(by_score, lift),
                  "oracle_between": flat(hit_first, plain),
                  "oracle_both": flat(hit_first, lift)}
        row = {a: rec(o, real, K) for a, o in orders.items()}
        row.update({"pool": ceiling[-1], "n_groups": len(groups)})
        per.append(row)
        h_per.append({a: hits(o, real, K) for a, o in orders.items()})

        # the frozen-score recombinations, at the same budget
        for arm, key in (("product", lambda c: -c["combined"]),
                         ("filter", lambda c: -c["filter"]),
                         ("generator", lambda c: -c["generator"])):
            ordered = [c["key"] for c in sorted(cands, key=key)]
            combo[arm].append(rec(ordered, real, K))
            h_combo[arm].append(hits(ordered, real, K))
        rrf = rrf_order(cands)
        combo["rrf"].append(rec([c["key"] for c in rrf], real, K))
        h_combo["rrf"].append(hits([c["key"] for c in rrf], real, K))

    pool = round(st.mean(ceiling), 4)
    N = sum(n_refs)
    pool_micro = round(sum(h_ceiling) / N, 4)
    ARMS = ("as_ranked", "oracle_within", "oracle_between", "oracle_both")
    arms = {a: {"recall@15": round(st.mean([r[a] for r in per]), 4),
                "retention": round(st.mean([r[a] for r in per]) / pool, 4),
                "recall@15_micro": round(sum(h[a] for h in h_per) / N, 4),
                "retention_micro": round(sum(h[a] for h in h_per) / N / pool_micro, 4)}
            for a in ARMS}
    rep = {
        "provenance": stamp(__file__),
        "population": {"n": len(subs), "pool": src},
        "aggregation": "macro, the mean of per-substrate recall; the _micro fields are the "
                       "ratio of sums, which is what the MetaTox comparison reports",
        "pool_ceiling_uncapped": pool,
        "pool_ceiling_uncapped_micro": pool_micro,
        "n_references": N,
        "mean_pool": round(st.mean([len(pools[s]) for s in subs]), 1),
        "mean_groups_per_substrate": round(st.mean([r["n_groups"] for r in per]), 1),
        "arms": arms,
        "closes": {a: round(arms[a]["recall@15"] - arms["as_ranked"]["recall@15"], 4)
                   for a in arms if a != "as_ranked"},
        "within_group_rank": {
            "n_cases_with_siblings": len(ranks),
            "mean_rank": round(st.mean(ranks), 2) if ranks else None,
            "median_rank": st.median(ranks) if ranks else None,
            "top1_share": round(sum(1 for r in ranks if r == 1) / len(ranks), 4) if ranks else None,
            "mean_group_size": round(st.mean(sizes), 2) if sizes else None,
        },
        "score_combinations_at_15": {a: round(st.mean(v), 4) for a, v in combo.items()},
        "score_combinations_at_15_micro": {a: round(sum(h_combo[a]) / N, 4) for a in combo},
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\npool ceiling (uncapped) {pool}   mean pool {rep['mean_pool']}   "
          f"groups/substrate {rep['mean_groups_per_substrate']}")
    for a, v in arms.items():
        print(f"  {a:<16} recall@15 {v['recall@15']:.4f}  retention {v['retention']:.4f}")
    print(f"\nwhat each oracle closes: {rep['closes']}")
    w = rep["within_group_rank"]
    print(f"reference inside its formula group: top-1 in {w['top1_share']}, mean rank "
          f"{w['mean_rank']}, {w['n_cases_with_siblings']} cases, mean group {w['mean_group_size']}")
    print(f"score combinations at k=15, macro: {rep['score_combinations_at_15']}")
    print(f"score combinations at k=15, micro: {rep['score_combinations_at_15_micro']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def rec(keys, real, k=None):
    return len(set(keys[:k] if k else keys) & real) / len(real) if real else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "wide_pool_analysis.json"))
    args = ap.parse_args()

    blob = json.loads(POOLS.read_text())
    pools, refs = blob["pools"], blob["references"]
    subs = sorted(s for s in pools if refs.get(s))
    print(f"{len(subs)} substrates, pool mean "
          f"{st.mean([len(pools[s]) for s in subs]):.1f}", file=sys.stderr, flush=True)

    ceiling, per, ranks, sizes, combo = [], [], [], [], defaultdict(list)
    for n, s in enumerate(subs, 1):
        if n % 25 == 0:
            print(f"  {n}/{len(subs)}", file=sys.stderr, flush=True)
        cands = sorted(pools[s], key=lambda c: -c["combined"])
        real = set(refs[s])
        keys = [c["key"] for c in cands]
        ceiling.append(rec(keys, real))

        groups = defaultdict(list)
        for i, c in enumerate(cands):
            groups[formula(c["smiles"])].append(i)
        for g, idxs in groups.items():
            hits = [j for j, i in enumerate(idxs) if keys[i] in real]
            if hits and len(idxs) > 1:
                ranks.append(hits[0] + 1)
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
        row = {"as_ranked": rec(flat(by_score, plain), real, K),
               "oracle_within": rec(flat(by_score, lift), real, K),
               "oracle_between": rec(flat(hit_first, plain), real, K),
               "oracle_both": rec(flat(hit_first, lift), real, K),
               "pool": ceiling[-1], "n_groups": len(groups)}
        per.append(row)

        # the frozen-score recombinations, at the same budget
        for arm, key in (("product", lambda c: -c["combined"]),
                         ("filter", lambda c: -c["filter"]),
                         ("generator", lambda c: -c["generator"])):
            combo[arm].append(rec([c["key"] for c in sorted(cands, key=key)], real, K))
        f = {id(c): i for i, c in enumerate(sorted(cands, key=lambda x: -x["filter"]))}
        g = {id(c): i for i, c in enumerate(sorted(cands, key=lambda x: -x["generator"]))}
        rrf = sorted(cands, key=lambda c: -(1 / (60 + f[id(c)]) + 1 / (60 + g[id(c)])))
        combo["rrf"].append(rec([c["key"] for c in rrf], real, K))

    pool = round(st.mean(ceiling), 4)
    arms = {a: {"recall@15": round(st.mean([r[a] for r in per]), 4),
                "retention": round(st.mean([r[a] for r in per]) / pool, 4)}
            for a in ("as_ranked", "oracle_within", "oracle_between", "oracle_both")}
    rep = {
        "provenance": stamp(__file__),
        "population": {"n": len(subs), "pool": "results/wide_pools.json"},
        "pool_ceiling_uncapped": pool,
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
    print(f"score combinations at k=15: {rep['score_combinations_at_15']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

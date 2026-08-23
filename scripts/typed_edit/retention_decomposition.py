#!/usr/bin/env python3
"""Retention is one number over two different problems. This separates them.

Ordering a pool of candidates for one substrate asks two questions that are not the same task.
Telling a hydroxylation from an N-dealkylation is a choice between transformation types, and the
signal for it is strong and explicit. Telling a hydroxylation at C3 from one at C5 is
regioselectivity, and appendix E.1 records that it is not solved.

Candidates that are isomers of one another -- the same molecular formula on the same substrate
-- are exactly the second problem: they differ in where the edit landed and in nothing else.
Grouping the pool by formula therefore splits the ordering into a between-group part and a
within-group part, and two oracles say what each is worth:

  as ranked        the deployed order
  oracle within    each group reordered so its annotated member leads, group order untouched
  oracle between   groups reordered so the annotated one leads, order inside each untouched
  oracle both      the ceiling of the pool

The gap each oracle closes is what a perfect ranker of that kind would buy, in recall@15 on the
pool the pipeline already produces. Nothing is trained and no candidate is added.

The grouping is a proxy and is stated as one: identical formula catches every regioisomer, and
it also lumps together transformations that happen to share a formula delta, an N-oxidation with
a C-hydroxylation. That lumping moves work from the between-group problem into the within-group
one, so a poor within-group result is the conservative reading and a good one is not.
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

from grail_metabolism.metrics import _match_keys  # noqa: E402
from scripts.run_benchmark import load_test_map  # noqa: E402

RDLogger.DisableLog("rdApp.*")
DUMP = ROOT / "results" / "scored_predictions.json"
MATCH = "inchikey_tautomer"
K = 15


def formula(smiles, cache={}):
    if smiles not in cache:
        m = Chem.MolFromSmiles(smiles)
        cache[smiles] = CalcMolFormula(m) if m is not None else "?"
    return cache[smiles]


def recall(ranked, real, k):
    return len(set(ranked[:k]) & real) / len(real) if real else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "retention_decomposition.json"))
    args = ap.parse_args()

    tm = load_test_map(None, 42)
    rows = {r["sub"]: r["candidates"] for r in json.loads(DUMP.read_text())["rows"]}
    subs = sorted(s for s in rows if s in tm and tm[s])

    per, ranks, sizes = [], [], []
    for n, s in enumerate(subs, 1):
        if n % 200 == 0:
            print(f"  {n}/{len(subs)}", file=sys.stderr, flush=True)
        cands = sorted(rows[s], key=lambda c: -c["combined"])
        if not cands:
            continue
        keys = [next(iter(_match_keys([c["smiles"]], MATCH))) for c in cands]
        real = _match_keys(tm[s], MATCH)
        groups = defaultdict(list)
        for i, c in enumerate(cands):
            groups[formula(c["smiles"])].append(i)

        # where each annotated reference sits inside its own formula group
        for g, idxs in groups.items():
            hits = [j for j, i in enumerate(idxs) if keys[i] in real]
            if hits and len(idxs) > 1:
                ranks.append(hits[0] + 1)
                sizes.append(len(idxs))

        def flatten(order_groups, order_within):
            out = []
            for g in order_groups:
                idxs = groups[g]
                out.extend(order_within(g, idxs))
            return [keys[i] for i in out]

        best = {g: min(idxs) for g, idxs in groups.items()}
        by_score = sorted(groups, key=lambda g: best[g])
        hit_first = sorted(groups, key=lambda g: (not any(keys[i] in real for i in groups[g]),
                                                  best[g]))
        plain = lambda g, idxs: idxs
        lift = lambda g, idxs: (sorted(idxs, key=lambda i: keys[i] not in real))

        arms = {
            "as_ranked": flatten(by_score, plain),
            "oracle_within": flatten(by_score, lift),
            "oracle_between": flatten(hit_first, plain),
            "oracle_both": flatten(hit_first, lift),
        }
        row = {a: recall(v, real, K) for a, v in arms.items()}
        row["pool"] = recall(arms["as_ranked"], real, None)
        row["n_groups"] = len(groups)
        row["n_candidates"] = len(cands)
        per.append(row)

    def mean(k):
        return round(st.mean([r[k] for r in per]), 4)

    pool = mean("pool")
    out = {a: {"recall@15": mean(a), "retention": round(mean(a) / pool, 4) if pool else None}
           for a in ("as_ranked", "oracle_within", "oracle_between", "oracle_both")}
    rep = {
        "provenance": stamp(__file__),
        "population": {"n": len(per), "pool": "results/scored_predictions.json",
                       "note": "the deployed pool; retention is within it"},
        "match": MATCH, "k": K,
        "grouping": "identical molecular formula on the same substrate, a proxy for one "
                    "transformation type that catches every regioisomer and lumps distinct "
                    "types sharing a formula delta",
        "arms": out,
        "closes": {a: round(out[a]["recall@15"] - out["as_ranked"]["recall@15"], 4)
                   for a in out if a != "as_ranked"},
        "within_group_rank_of_the_reference": {
            "n_cases_with_siblings": len(ranks),
            "mean_rank": round(st.mean(ranks), 2) if ranks else None,
            "median_rank": st.median(ranks) if ranks else None,
            "top1_share": round(sum(1 for r in ranks if r == 1) / len(ranks), 4) if ranks else None,
            "mean_group_size": round(st.mean(sizes), 2) if sizes else None,
        },
        "mean_groups_per_substrate": round(st.mean([r["n_groups"] for r in per]), 2),
        "mean_candidates_per_substrate": round(st.mean([r["n_candidates"] for r in per]), 2),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\npool recall {pool}")
    for a, v in out.items():
        print(f"  {a:<16} recall@15 {v['recall@15']:.4f}  retention {v['retention']:.4f}")
    print(f"\nwhat each oracle closes, in recall@15: {rep['closes']}")
    w = rep["within_group_rank_of_the_reference"]
    print(f"reference's rank inside its own formula group: mean {w['mean_rank']}, "
          f"top-1 in {w['top1_share']} of {w['n_cases_with_siblings']} cases "
          f"(mean group {w['mean_group_size']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

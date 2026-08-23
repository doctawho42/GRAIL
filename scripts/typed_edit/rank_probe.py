#!/usr/bin/env python3
"""Is there signal in the scores the pipeline already computes, or does it need new features?

The project now rests on one unbuilt component: a ranker that must close about thirteen points
of top-15 retention. Everything measured so far has trimmed the design space around it. This
probes the assumption without training anything.

The obvious probe -- softmax-normalise the scores within a substrate and re-rank -- cannot
work, and it is worth saying why rather than running it. Softmax within a substrate divides
every candidate by the same denominator and applies a monotone exponential, so it preserves the
order of the candidates it normalises. Top-k by probability is top-k by raw score, and recall@k
is unchanged by construction. A listwise LOSS changes what a model learns; a listwise
normalisation of frozen scores changes nothing that a top-k metric can see.

What can move the ranking is a recombination that treats candidates differently:

  product        filter x generator, the deployed ranking
  filter         the filter alone
  generator      the generator alone
  rrf            reciprocal-rank fusion of the two, which is scale-free and rank-based
  gen_calibrated the generator z-scored WITHIN ITS RULE across substrates before the product.
                 A rule whose score runs high everywhere is not evidence about this substrate;
                 this removes that offset, and unlike a per-substrate rescaling it reorders,
                 because different candidates belong to different rules.

Retention here is recall@15 over recall at the pool's own size, both computed on the same
released per-candidate dump, so the arms differ only in how the same numbers are ordered.
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

from grail_metabolism.metrics import _match_keys  # noqa: E402
from scripts.run_benchmark import load_test_map  # noqa: E402

DUMP = ROOT / "results" / "scored_predictions.json"
MATCH = "inchikey_tautomer"
K = 15


def recall_at(ranked_keys, real, k=None):
    got = set(ranked_keys[:k] if k else ranked_keys)
    return len(got & real) / len(real) if real else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "rank_probe.json"))
    args = ap.parse_args()

    tm = load_test_map(None, 42)
    rows = {r["sub"]: r["candidates"] for r in json.loads(DUMP.read_text())["rows"]}
    subs = sorted(s for s in rows if s in tm and tm[s])
    print(f"{len(subs)} substrates, pool from the released per-candidate dump",
          file=sys.stderr, flush=True)

    # the generator's offset per rule, over every substrate it fires on
    by_rule = defaultdict(list)
    has_rule = all("rule" in c for s in subs[:50] for c in rows[s][:5])
    for s in subs:
        for c in rows[s]:
            by_rule[c.get("rule", c["smiles"])].append(c["generator"])
    stats = {r: (st.mean(v), st.pstdev(v) or 1.0) for r, v in by_rule.items()}

    def order(cands, arm):
        if arm == "product":
            key = lambda c: c["combined"]
        elif arm == "filter":
            key = lambda c: c["filter"]
        elif arm == "generator":
            key = lambda c: c["generator"]
        elif arm == "gen_calibrated":
            def key(c):
                m, sd = stats[c.get("rule", c["smiles"])]
                return c["filter"] * ((c["generator"] - m) / sd)
        else:  # reciprocal-rank fusion
            f = {id(c): i for i, c in enumerate(sorted(cands, key=lambda x: -x["filter"]))}
            g = {id(c): i for i, c in enumerate(sorted(cands, key=lambda x: -x["generator"]))}
            key = lambda c: -(1 / (60 + f[id(c)]) + 1 / (60 + g[id(c)]))
            return sorted(cands, key=key)
        return sorted(cands, key=lambda c: -key(c))

    arms = ["product", "filter", "generator", "rrf", "gen_calibrated"]
    keys_cache = {s: {c["smiles"]: next(iter(_match_keys([c["smiles"]], MATCH)))
                      for c in rows[s]} for s in subs}
    real = {s: _match_keys(tm[s], MATCH) for s in subs}

    out = {}
    for arm in arms:
        r15, rinf = [], []
        for s in subs:
            ranked = [keys_cache[s][c["smiles"]] for c in order(rows[s], arm)]
            r15.append(recall_at(ranked, real[s], K))
            rinf.append(recall_at(ranked, real[s]))
        m15, minf = st.mean(r15), st.mean(rinf)
        out[arm] = {"recall@15": round(m15, 4), "pool_recall": round(minf, 4),
                    "retention": round(m15 / minf, 4) if minf else None}
        print(f"  {arm:<16} recall@15 {m15:.4f}  pool {minf:.4f}  "
              f"retention {m15 / minf if minf else 0:.4f}", file=sys.stderr, flush=True)

    base = out["product"]["retention"]
    rep = {
        "provenance": stamp(__file__),
        "why_not_softmax": "softmax within a substrate is a monotone rescaling by a constant "
                           "denominator, so it preserves the order of the candidates it "
                           "normalises and cannot change recall@k. A listwise loss changes what "
                           "a model learns; a listwise normalisation of frozen scores does not.",
        "population": {"n": len(subs), "pool": "results/scored_predictions.json",
                       "note": "the deployed pool, not the selector-free pool of "
                               "bank_without_selection; retention here is within this pool"},
        "match": MATCH, "k": K, "rule_field_present": has_rule,
        "arms": out,
        "best_arm": max(out, key=lambda a: out[a]["retention"] or 0),
        "gain_over_product": {a: round((out[a]["retention"] or 0) - base, 4) for a in arms},
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("provenance",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Recall and wall-clock against the generator's rule budget.

The envelope sweep shows the generator is what grows with substrate size, and the generator's
cost is set by how many of the 7,581 templates it applies. That number is already a parameter --
`top_k` -- and the shipped checkpoint records 30 for it while every pool this project has built
passed 7,581. This measures what lies between.

It cannot be simulated from pools already built. The candidate aggregation is noisy-or, so a
candidate's score is one minus the product of one minus the score of every rule that produces
it; cutting the rule budget removes producers and therefore changes the surviving candidates'
scores rather than merely filtering them. Each budget has to be run.

This is exploratory and is recorded as such. A curve is not a hypothesis, and the operating
point read off it will be fixed by a stated rule and checked where it was not chosen, on the
same terms as H9.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

BUDGETS = (30, 100, 300, 1000, 3000, 7581)
KS = (1, 5, 10, 15, 20, 30, 50)
POOL_CAP = 100          # the cap H9 registers, applied to every arm alike
GEN = ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"
FILT = ROOT / "artifacts/full5000_implicit/checkpoints/filter.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default=str(GEN))
    ap.add_argument("--filter-ckpt", default=str(FILT))
    ap.add_argument("--pools", default=str(ROOT / "results/wide_pools.json"))
    ap.add_argument("--n", type=int, default=50, help="substrates drawn from the 291")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/rule_budget_curve.json"))
    args = ap.parse_args()

    import random

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from bank_without_selection import _key, _load
    from grail_metabolism.config import FilterConfig, GeneratorConfig
    from grail_metabolism.workflows.factory import build_filter, build_generator

    blob = json.loads(Path(args.pools).read_text())
    refs = blob["references"]
    pool_subs = sorted(s for s in blob["pools"] if refs.get(s))
    # a declared draw: sampling without replacement, so a different n is a different set
    subs = sorted(random.Random(args.seed).sample(pool_subs, min(args.n, len(pool_subs))))
    real = {s: set(refs[s]) for s in subs}
    N = sum(len(real[s]) for s in subs)
    print(f"{len(subs)} substrates, {N} references", file=sys.stderr, flush=True)

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    filt = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))
    generator.generate_scored_with_details("CCO", top_k=7581, threshold=None,
                                           compute_sites=False)   # warm the rule-graph cache

    from grail_metabolism.utils import preparation as prep

    rows = {}
    for b in BUDGETS:
        # Standardisation is 94 to 99 per cent of the generator's cost and it is memoised, so an
        # arm that runs after another finds the shared products already paid for. Whichever
        # order the arms run in, the later ones would be measured warm. Clearing the caches
        # between them makes every arm cold, which is the state a service meets a molecule in.
        prep._standardize_smiles_cached.cache_clear()
        prep._canonicalize_smiles_cached.cache_clear()
        hits = {k: 0 for k in KS}
        t_gen = t_filt = 0.0
        n_raw = n_capped = 0
        for i, s in enumerate(subs, 1):
            t0 = time.perf_counter()
            det = generator.generate_scored_with_details(s, top_k=b, threshold=None,
                                                         compute_sites=False)
            t_gen += time.perf_counter() - t0
            det.sort(key=lambda d: (-d[1], d[0]))
            n_raw += len(det)
            keep = det[:POOL_CAP]
            n_capped += len(keep)
            cands = [d[0] for d in keep]
            t1 = time.perf_counter()
            fs = filt.score_batch(s, cands) if cands else []
            t_filt += time.perf_counter() - t1
            pool, seen = [], set()
            for (sm, g, *_), f in zip(keep, fs):
                k = _key(sm)
                if k and k not in seen:
                    seen.add(k)
                    pool.append({"key": k, "generator": float(g), "filter": float(f)})
            ordered = [c["key"] for c in rrf_order(pool)]
            for k in KS:
                hits[k] += len(set(ordered[:k]) & real[s])
            if i % 10 == 0:
                print(f"  top_k={b}: {i}/{len(subs)}  gen {t_gen:.0f}s  filt {t_filt:.0f}s",
                      file=sys.stderr, flush=True)
        rows[str(b)] = {"recall_micro": {str(k): round(hits[k] / N, 4) for k in KS},
                        "t_generator_s": round(t_gen, 1), "t_filter_s": round(t_filt, 1),
                        "s_per_substrate": round((t_gen + t_filt) / len(subs), 2),
                        "mean_pool_raw": round(n_raw / len(subs), 1),
                        "mean_pool_after_cap": round(n_capped / len(subs), 1)}
        r = rows[str(b)]
        print(f"top_k={b:>5}  r@15={r['recall_micro']['15']:.4f}  "
              f"{r['s_per_substrate']:>7.2f}s/substrate  pool {r['mean_pool_raw']:.0f}",
              file=sys.stderr, flush=True)
        Path(args.out).write_text(json.dumps(
            {"provenance": stamp(__file__),
             "status": "EXPLORATORY. A curve, not a hypothesis. The operating point read off it "
                       "must be fixed by a stated rule and checked where it was not chosen.",
             "population": {"n": len(subs), "n_references": N, "seed": args.seed,
                            "drawn_from": "the 291 of results/four_method_291.json"},
             "aggregation": "micro, ratio of sums",
             "pool_cap": POOL_CAP, "note": "the H9 cap is applied to every arm alike, so the "
                                           "arms differ only in the rule budget",
             "by_rule_budget": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

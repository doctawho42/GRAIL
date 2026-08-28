#!/usr/bin/env python3
"""Does knowing WHICH substrate you have help choose rules, or would any thirty do?

The generator's job in stage one is selection: it scores all \\numBankRules{} templates against
the substrate and the deployed interactive mode applies the best thirty. What that selection buys
has never been measured against an alternative of the same size. The rule budget prediction (P3)
compared thirty against the whole bank, which is a question about size, not about choice.

Three ways to choose thirty rules, with everything downstream held fixed -- the same application,
the same standardisation, the same filter, the same reciprocal rank fusion:

    learned    the generator's top thirty for THIS substrate, which is what ships
    prior      the thirty with the highest training-frequency log-odds, the same thirty for every
               substrate; this is the non-learned baseline the generator has to beat, and the
               model already carries it as the `rule_prior_logits` buffer it adds at weight 0.4
    random     thirty drawn without replacement per substrate, seeded

If the learned arm does not beat the prior arm, the generator is reproducing a frequency table
and the substrate-specific part of stage one is not earning its cost.

    python scripts/typed_edit/selection_ablation_deployed.py --arms prior random
"""
from __future__ import annotations

import argparse
import glob
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

BUDGET = 30
SEED = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--filter-ckpt", default=str(ROOT / "artifacts/full5000_priors/checkpoints/filter.pt"))
    ap.add_argument("--arms", nargs="+",
                    default=["prior_applicable", "random_applicable", "random"])
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/selection_pools_deployed.json"))
    args = ap.parse_args()

    import numpy as np
    import torch
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    from bank_without_selection import _key, _load
    from grail_metabolism.config import FilterConfig, GeneratorConfig
    from grail_metabolism.workflows.factory import build_filter, build_generator

    # the population and references of the comparison set, read from the pools the paper reports
    pools_ref, refs = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        d = json.loads(Path(f).read_text())
        pools_ref.update(d["pools"]); refs.update(d["references"])
    subs = sorted(s for s in pools_ref if refs.get(s))
    if args.limit:
        subs = subs[:args.limit]
    print(f"{len(subs)} substrates", file=sys.stderr, flush=True)

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    filt = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))
    n_rules = len(generator.rule_names)

    prior = generator.rule_prior_logits.detach().cpu().numpy()
    rng = np.random.default_rng(SEED)
    real_prepare = generator._prepare_generation

    def prepare_with(indices):
        """Return a _prepare_generation that yields OUR rule choice and nothing else.

        Wrapping the selection rather than the enumeration keeps everything downstream
        identical: the same application, normalisation and noisy-or aggregation run over a
        different set of rules. There is no hook for this on the model, and inventing one that
        the model ignores would have left every arm running the learned selection.
        """
        def wrapped(sub, top_k, threshold):
            mol, scores, ranked = real_prepare(sub, None, None)
            if mol is None:
                return None, None, None
            chosen = [int(i) for i in indices]
            return mol, scores, sorted(chosen, key=lambda i: float(scores[i]), reverse=True)
        return wrapped

    out = {a: {} for a in args.arms}
    applicable_n = []
    t0 = time.perf_counter()
    for n, s in enumerate(subs, 1):
        if n % 20 == 0:
            print(f"  {n}/{len(subs)} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr, flush=True)
        # the rules whose SMARTS matches this substrate, which is the pool the learned arm
        # selects from; drawing at random from the whole bank instead measures applicability
        # filtering rather than choice
        _, mask = generator.score_rules(s, return_mask=True)
        applicable = np.where(np.asarray(mask) > 0.0)[0]
        applicable_n.append(int(applicable.size))

        for arm in args.arms:
            if arm == "prior_applicable":
                # among the rules that CAN fire here, the thirty with the highest training
                # frequency. The global top thirty by prior is not a baseline but a strawman:
                # none of them is applicable to the first substrate of this set, so that arm
                # fails on applicability rather than on choice and returns nothing.
                pool = applicable if applicable.size else np.arange(n_rules)
                idx = pool[np.argsort(-prior[pool])[:args.budget]]
            elif arm == "random_applicable":
                pool = applicable if applicable.size else np.arange(n_rules)
                idx = rng.choice(pool, size=min(args.budget, pool.size), replace=False)
            else:
                idx = rng.choice(n_rules, size=args.budget, replace=False)
            generator._prepare_generation = prepare_with(idx)
            try:
                det = generator.generate_scored_with_details(
                    s, top_k=args.budget, threshold=None, compute_sites=False)
            finally:
                generator._prepare_generation = real_prepare
            cands = [d[0] for d in det]
            fs = filt.score_batch(s, cands) if cands else []
            rows, seen = [], set()
            for (sm, g, *_), f in zip(det, fs):
                k = _key(sm)
                if k and k not in seen:
                    seen.add(k)
                    rows.append({"smiles": sm, "generator": float(g), "filter": float(f),
                                 "combined": float(g) * float(f), "key": k})
            out[arm][s] = rows
        Path(args.out).write_text(json.dumps(
            {"provenance": stamp(__file__), "budget": args.budget, "seed": SEED,
             "arms": args.arms, "n_substrates": len(out[args.arms[0]]),
             "mean_applicable_rules": round(float(np.mean(applicable_n)), 1),
             "note": "pools for the non-learned selection arms; the learned arm is "
                     "results/widepools_k30/all.json",
             "pools": out, "references": {s: refs[s] for s in out[args.arms[0]]}}, indent=1))
    print(f"wrote {args.out} in {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

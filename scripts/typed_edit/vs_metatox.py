#!/usr/bin/env python3
"""The bank without a selector against MetaTox, on the substrates MetaTox was run on.

The system this work replaces is MetaTox, and MetaTox is not the field. On the 291 substrates
where all four methods and the references exist, MetaTox leads only from k=15 upward; at k=1, 3
and 5 it is last of the four, because PASS assigns a metabolite-likeness score to 5,177 of its
10,601 predictions and the rest keep file order. More than half of its output is not ranked at
all, and a toxicologist reads the first five rows.

So the question that decides the release is not whether the bank beats SyGMa at the budget the
literature reports. It is whether the bank, with its selector removed and its own scorer
ranking, beats MetaTox at the budgets a user actually looks at.

Everything here is frozen except the pool, which is built the way `bank_without_selection.py`
builds it: the whole bank applied without a selector and without the calibrated threshold, which
is itself a selector, ranked by filter times generator.

The gate is the population. The MetaTox recall this reproduces has to match
`results/four_method_291.json` at every budget, or this is a different set of substrates or a
different matcher and the comparison means nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from bank_without_selection import (  # noqa: E402
    _dedup, _load, recall_at,
)
from grail_metabolism.config import FilterConfig, GeneratorConfig  # noqa: E402
from grail_metabolism.workflows.factory import build_filter, build_generator  # noqa: E402

FOUR = ROOT / "results" / "four_method_291.json"
METATOX = ROOT / "results" / "metatox_smirks_preds.json"
TRUTH = ROOT / "results" / "test_references.json"
POOLS = {"MetaPredictor": ROOT / "artifacts/tier2_1170/metapredictor_preds.json",
         "SyGMa": ROOT / "results/sygma_fulltest_predictions.json"}
GRAIL_DUMP = ROOT / "results" / "scored_predictions.json"
BUDGETS = [1, 3, 5, 8, 10, 15, 20, 30, 50]
N_BOOT, SEED = 10000, 0


def population():
    """The same 291 four_method_291.py uses: references and all four prediction sets."""
    truth = json.loads(TRUTH.read_text())
    pools = {"MetaTox": json.loads(METATOX.read_text())["predictions"],
             "GRAIL": {r["sub"]: 1 for r in json.loads(GRAIL_DUMP.read_text())["rows"]}}
    for name, path in POOLS.items():
        pools[name] = json.loads(path.read_text())
    subs = sorted(set(truth) & set.intersection(*(set(p) for p in pools.values())))
    return subs, truth, pools["MetaTox"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts/full5000_priors/checkpoints/generator.pt"))
    ap.add_argument("--filter-ckpt", default=str(ROOT / "artifacts/full5000_single/checkpoints/filter.pt"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "vs_metatox.json"))
    args = ap.parse_args()

    subs, truth, mtx = population()
    if args.limit:
        subs = subs[: args.limit]
    print(f"population: {len(subs)} substrates", file=sys.stderr, flush=True)

    pool = Pool(args.threads)
    cap = max(BUDGETS)
    refs = {s: set(_dedup(truth[s], None, pool)) for s in subs}
    mt = {s: _dedup(mtx.get(s, []), cap, pool) for s in subs}

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    filt = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))

    ranked, sizes, t = {}, [], time.perf_counter()
    for i, s in enumerate(subs, 1):
        if i == 1 or i % 25 == 0 or i == len(subs):
            print(f"  pool {i}/{len(subs)} ({time.perf_counter()-t:.0f}s)",
                  file=sys.stderr, flush=True)
        det = generator.generate_scored_with_details(s, top_k=7581, threshold=None,
                                                     compute_sites=False)
        det.sort(key=lambda d: (-d[1], d[0]))
        sizes.append(len(det))
        cands = [d[0] for d in det]
        fs = filt.score_batch(s, cands) if cands else []
        order = sorted(zip(cands, [float(a) * float(d[1]) for a, d in zip(fs, det)]),
                       key=lambda x: -x[1])
        ranked[s] = _dedup([c for c, _ in order], cap, pool)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    by_budget = {}
    for b in BUDGETS:
        g = np.array([recall_at(ranked[s], refs[s], b) for s in subs])
        m = np.array([recall_at(mt[s], refs[s], b) for s in subs])
        d = g - m
        bt = d[idx].mean(axis=1)
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        by_budget[str(b)] = {"bank": round(float(g.mean()), 4),
                             "metatox": round(float(m.mean()), 4),
                             "gap": round(float(d.mean()), 4),
                             "ci95": [round(lo, 4), round(hi, 4)],
                             "excludes_zero": bool(lo > 0 or hi < 0)}

    # The gate: MetaTox here must be MetaTox there. It compares an average over 291 substrates,
    # so it is meaningless on a truncated run and is not run there -- announced rather than
    # silently skipped, because a gate that quietly stops applying is the failure this
    # repository keeps finding.
    four = json.loads(FOUR.read_text())["per_method"]["MetaTox"]["recall"]
    if args.limit:
        mism = []
        gate_note = (f"not applied: --limit {args.limit} scores a subset of the 291 the "
                     f"committed averages are taken over")
    else:
        mism = [f"k={b}: here {by_budget[str(b)]['metatox']} vs committed {four[str(b)]}"
                for b in BUDGETS if str(b) in four
                and abs(by_budget[str(b)]["metatox"] - four[str(b)]) > 1e-9]
        gate_note = "applied over the full population"

    rep = {"provenance": stamp(__file__),
           "population": {"n": len(subs), "source": "the 291 of results/four_method_291.json"},
           "pool": {"mean_raw": round(float(np.mean(sizes)), 1),
                    "mean_unique": round(float(np.mean([len(v) for v in ranked.values()])), 1),
                    "coverage": round(float(np.mean(
                        [len(set(ranked[s]) & refs[s]) / len(refs[s]) for s in subs
                         if refs[s]])), 4)},
           "gate": {"reproduces_four_method_291_metatox": not mism and not args.limit,
                    "note": gate_note, "mismatches": mism},
           "by_budget": by_budget, "n_boot": N_BOOT, "seed": SEED}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    pool.close()

    print(f"\npool: raw {rep['pool']['mean_raw']}, unique {rep['pool']['mean_unique']}, "
          f"coverage {rep['pool']['coverage']}\n")
    print(f"{'k':>4}{'bank':>9}{'MetaTox':>10}{'gap':>9}   interval")
    for b in BUDGETS:
        r = by_budget[str(b)]
        print(f"{b:>4}{r['bank']:>9.4f}{r['metatox']:>10.4f}{r['gap']:>+9.4f}   "
              f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
              f"{'separated' if r['excludes_zero'] else ''}")
    if mism:
        print(f"\nGATE FAILED: {mism}")
        return 1
    print(f"\ngate: {gate_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

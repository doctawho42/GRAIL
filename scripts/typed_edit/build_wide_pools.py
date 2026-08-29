#!/usr/bin/env python3
"""The wide pool, dumped once with everything three analyses need.

Building the selector-free pool costs about eight hours single-threaded, and it has now been
built three times because each run saved only what its own metric wanted. This saves the pool
itself: every candidate with its SMILES, its two component scores and its match key, uncapped,
so a metric that changes afterwards costs seconds rather than a night.

  the ceiling on this population   needs the pool UNCAPPED; the last run capped at 50
  the within/between decomposition needs SMILES, to group isomers by formula; the last run
                                   saved keys, from which no formula can be recovered
  the score-combination probe      needs the two components apart, not their product

Sharded by substrate, because `--threads` in the earlier scripts parallelises only the key
canonicalisation and the bank application is what takes the time.

    python scripts/typed_edit/build_wide_pools.py --start 0 --end 49 --out .../w0.json
    python scripts/typed_edit/build_wide_pools.py --merge 'results/widepools/w*.json'
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

from bank_without_selection import _key, _load  # noqa: E402
from grail_metabolism.config import FilterConfig, GeneratorConfig  # noqa: E402
from grail_metabolism.workflows.factory import build_filter, build_generator  # noqa: E402
from vs_metatox import population  # noqa: E402


def merge(pattern, out):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("no shard matched", file=sys.stderr)
        return 1
    pools, refs, slices = {}, {}, []
    for p in paths:
        d = json.loads(Path(p).read_text())
        pools.update(d["pools"])
        refs.update(d["references"])
        slices.append(tuple(d["slice"]))
        print(f"  + {Path(p).name}: {d['slice']} {len(d['pools'])} substrates", file=sys.stderr)
    subs, _, _ = population()
    covered = set()
    for a, b in slices:
        covered |= set(range(a, b))
    missing = sorted(set(range(len(subs))) - covered)
    if missing:
        print(f"FAIL: the shards do not tile the population; {len(missing)} substrates are in "
              f"none of them", file=sys.stderr)
        return 1
    if set(pools) != set(subs):
        print(f"FAIL: merged {len(pools)} substrates, the population is {len(subs)}",
              file=sys.stderr)
        return 1
    Path(out).write_text(json.dumps(
        {"provenance": stamp(__file__), "match": "inchikey_tautomer",
         "note": "whole bank, no selector, no calibrated threshold, uncapped; candidates in "
                 "rank order by filter x generator, deduplicated by match key, first SMILES kept",
         "slices": [list(s) for s in sorted(slices)],
         "n_substrates": len(pools), "pools": pools, "references": refs}, indent=1))
    import statistics as st
    sizes = [len(v) for v in pools.values()]
    print(f"\n{len(pools)} substrates, pool mean {st.mean(sizes):.1f} median "
          f"{st.median(sizes)} max {max(sizes)}")
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--merge", default="")
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts/full5000_priors/checkpoints/generator.pt"))
    ap.add_argument("--filter-ckpt", default=str(ROOT / "artifacts/full5000_single/checkpoints/filter.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "wide_pools.json"))
    ap.add_argument("--top-k", type=int, default=7581,
                    help="rule budget; 7581 is the whole bank, 30 is what the checkpoint records")
    ap.add_argument("--present", choices=("stored", "standardised"), default="stored",
                    help="how the substrate is handed to the matcher: as the corpus stores it, "
                         "or as the declared standardiser draws it. The pool and the references "
                         "stay keyed by the corpus string either way, so the two runs score "
                         "against one annotation and are paired substrate by substrate.")
    args = ap.parse_args()

    if args.merge:
        return merge(args.merge, args.out)

    subs, truth, _ = population()
    sl = subs[args.start:(args.end or None)]
    print(f"substrates [{args.start}:{args.end or len(subs)}] of {len(subs)}",
          file=sys.stderr, flush=True)

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    filt = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))

    presented = {}
    if args.present == "standardised":
        from rdkit import Chem
        from grail_metabolism.utils.preparation import standardize_mol
        for key in sl:
            try:
                presented[key] = Chem.MolToSmiles(standardize_mol(Chem.MolFromSmiles(key)))
            except Exception:
                presented[key] = key

    pools, refs, t = {}, {}, time.perf_counter()
    for i, s in enumerate(sl, 1):
        if i == 1 or i % 5 == 0 or i == len(sl):
            print(f"  {i}/{len(sl)} ({time.perf_counter() - t:.0f}s)", file=sys.stderr, flush=True)
        shown = presented.get(s, s)
        det = generator.generate_scored_with_details(shown, top_k=args.top_k, threshold=None,
                                                     compute_sites=False)
        det.sort(key=lambda d: (-d[1], d[0]))
        cands = [d[0] for d in det]
        fs = filt.score_batch(shown, cands) if cands else []
        scored = sorted(({"smiles": c, "generator": float(g[1]), "filter": float(f),
                          "combined": float(f) * float(g[1])}
                         for c, g, f in zip(cands, det, fs)),
                        key=lambda x: -x["combined"])
        seen, out = set(), []
        for c in scored:
            k = _key(c["smiles"])
            if not k or k in seen:
                continue
            seen.add(k)
            out.append({**c, "key": k})
        pools[s] = out
        refs[s] = sorted({k for k in (_key(p) for p in truth[s]) if k})

    Path(args.out).write_text(json.dumps(
        {"slice": [args.start, args.end or len(subs)], "top_k": args.top_k,
         "present": args.present,
         "pools": pools, "references": refs},
        indent=1))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

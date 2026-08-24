#!/usr/bin/env python3
"""The selector-free pool on the validation split, for the check H7 registers.

H7 fixes reciprocal rank fusion as the way the two component scores are combined and predicts it
beats their product by at least +0.05 of micro recall@15. The 291 MetaTox substrates cannot
settle that, because the rule was chosen on them. This builds the same pool on validation
substrates, which nothing in this project has read for this purpose.

The population is a declared draw: `--cap` substrates from the clean validation split with
`--seed` recorded, because sampling runs without replacement and a different cap yields a
different set of the same size rather than a subset.
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
from grail_metabolism.config import DatasetConfig, FilterConfig, GeneratorConfig  # noqa: E402
from grail_metabolism.workflows.data import load_dataset_bundle  # noqa: E402
from grail_metabolism.workflows.factory import build_filter, build_generator  # noqa: E402

CAP, SEED = 300, 0


def population(cap=CAP, seed=SEED):
    ds = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf",
        train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf",
        val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf",
        test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=8, max_val_substrates=cap, max_test_substrates=8,
        sampling_seed=seed)
    vmap = load_dataset_bundle(ds).val.map
    return sorted(s for s in vmap if vmap[s]), vmap


def merge(pattern, out):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("no shard matched", file=sys.stderr)
        return 1
    pools, refs, slices = {}, {}, []
    for p in paths:
        d = json.loads(Path(p).read_text())
        pools.update(d["pools"]); refs.update(d["references"]); slices.append(tuple(d["slice"]))
        print(f"  + {Path(p).name}: {d['slice']} {len(d['pools'])}", file=sys.stderr)
    subs, _ = population()
    covered = set()
    for a, b in slices:
        covered |= set(range(a, b))
    if sorted(set(range(len(subs))) - covered):
        print("FAIL: the shards do not tile the population", file=sys.stderr)
        return 1
    Path(out).write_text(json.dumps(
        {"provenance": stamp(__file__), "match": "inchikey_tautomer", "split": "validation",
         "population": {"cap": CAP, "seed": SEED, "n": len(pools)},
         "slices": [list(s) for s in sorted(slices)],
         "pools": pools, "references": refs}, indent=1))
    print(f"wrote {out} with {len(pools)} substrates")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--merge", default="")
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--filter-ckpt", default=str(ROOT / "artifacts/full5000_priors/checkpoints/filter.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "val_pools.json"))
    args = ap.parse_args()

    if args.merge:
        return merge(args.merge, args.out)

    subs, vmap = population()
    sl = subs[args.start:(args.end or None)]
    print(f"validation substrates [{args.start}:{args.end or len(subs)}] of {len(subs)}",
          file=sys.stderr, flush=True)

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    filt = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))

    pools, refs, t = {}, {}, time.perf_counter()
    for i, s in enumerate(sl, 1):
        if i == 1 or i % 5 == 0 or i == len(sl):
            print(f"  {i}/{len(sl)} ({time.perf_counter() - t:.0f}s)", file=sys.stderr, flush=True)
        det = generator.generate_scored_with_details(s, top_k=7581, threshold=None,
                                                     compute_sites=False)
        det.sort(key=lambda d: (-d[1], d[0]))
        cands = [d[0] for d in det]
        fs = filt.score_batch(s, cands) if cands else []
        scored = sorted(({"smiles": c, "generator": float(g[1]), "filter": float(f),
                          "combined": float(f) * float(g[1])}
                         for c, g, f in zip(cands, det, fs)), key=lambda x: -x["combined"])
        seen, out = set(), []
        for c in scored:
            k = _key(c["smiles"])
            if k and k not in seen:
                seen.add(k); out.append({**c, "key": k})
        pools[s] = out
        refs[s] = sorted({k for k in (_key(p) for p in vmap[s]) if k})

    Path(args.out).write_text(json.dumps(
        {"slice": [args.start, args.end or len(subs)], "pools": pools, "references": refs},
        indent=1))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

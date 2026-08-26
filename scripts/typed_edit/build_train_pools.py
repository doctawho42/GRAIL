"""Pools on the training split, for the group scorer H8 registers.

H8 is trained on the training split and selected on validation, so it needs pools there. The rule
budget is 1000 rather than the whole bank: the curve in results/rule_budget_curve.json reports
the same mean pool of 636.0 and the same micro recall@15 of 0.5761 at 1000, 3000 and 7581, so
above a thousand templates the candidate set does not change and the extra passes are spent
producing duplicates.

The population is a declared draw -- `--cap` substrates with `--seed` recorded -- because
sampling runs without replacement and a different cap yields a different set of the same size
rather than a subset. Shards write their own file and the artifact is written after every
substrate, so an interrupted shard keeps what it has done.
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

CAP, SEED, TOP_K = 400, 0, 1000

# The pipeline does not finish on every substrate. results/cost_envelope.json times 106 of them
# and the smallest that did not complete in 600 seconds has 42 heavy atoms, while everything
# below 40 completed; four of this draw's substrates, at 45, 51, 58 and 60, blocked their shards
# indefinitely. The training population is therefore restricted to substrates of at most 40
# heavy atoms -- the largest round threshold strictly below any observed failure, chosen from
# the envelope measurement rather than from anything about the model being trained. It removes
# 37 of 386, and the restriction is a property of population() so that every caller, every
# shard and the merge's tiling check see one population.
MAX_HEAVY = 40


def population(cap=CAP, seed=SEED, max_heavy=MAX_HEAVY):
    ds = DatasetConfig(
        train_sdf="grail_metabolism/data/train.sdf",
        train_triples="grail_metabolism/data/train_triples.txt",
        val_sdf="grail_metabolism/data/val.sdf",
        val_triples="grail_metabolism/data/val_triples.txt",
        test_sdf="grail_metabolism/data/test.sdf",
        test_triples="grail_metabolism/data/test_triples.txt",
        rules_path="grail_metabolism/resources/extended_smirks.txt",
        use_clean_splits=True, standardize=False,
        max_train_substrates=cap, max_val_substrates=8, max_test_substrates=8,
        sampling_seed=seed)
    tmap = load_dataset_bundle(ds).train.map
    keep = sorted(s for s in tmap if tmap[s])
    if max_heavy:
        from rdkit import Chem
        sized = []
        for s in keep:
            m = Chem.MolFromSmiles(s)
            if m is not None and m.GetNumHeavyAtoms() <= max_heavy:
                sized.append(s)
        keep = sized
    return keep, tmap


def merge(pattern, out, cap, seed, top_k):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("no shard matched", file=sys.stderr)
        return 1
    pools, refs, slices = {}, {}, []
    for p in paths:
        d = json.loads(Path(p).read_text())
        pools.update(d["pools"]); refs.update(d["references"]); slices.append(tuple(d["slice"]))
        print(f"  + {Path(p).name}: {d['slice']} {len(d['pools'])}", file=sys.stderr)
    subs, _ = population(cap, seed)
    covered = set()
    for a, b in slices:
        covered |= set(range(a, b))
    absent = sorted(set(range(len(subs))) - covered)
    if absent:
        print(f"FAIL: the shards do not tile the population; absent: {absent[:20]}"
              f"{' ...' if len(absent) > 20 else ''}", file=sys.stderr)
        return 1
    Path(out).write_text(json.dumps(
        {"provenance": stamp(__file__), "split": "train", "top_k": top_k,
         "population": {"cap": cap, "seed": seed, "n": len(pools),
                        "max_heavy_atoms": MAX_HEAVY,
                        "restriction": "substrates above the cap are excluded from the training "
                                       "population because the pipeline does not terminate on "
                                       "them; see results/cost_envelope.json"},
         "slices": [list(s) for s in sorted(slices)],
         "pools": pools, "references": refs}, indent=1))
    print(f"wrote {out} with {len(pools)} substrates")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--merge", default="")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--filter-ckpt", default=str(ROOT / "artifacts/full5000_implicit/checkpoints/filter.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/train_pools.json"))
    args = ap.parse_args()

    if args.merge:
        return merge(args.merge, args.out, args.cap, args.seed, args.top_k)

    subs, tmap = population(args.cap, args.seed)
    sl = subs[args.start:(args.end or None)]
    print(f"train substrates [{args.start}:{args.end or len(subs)}] of {len(subs)}",
          file=sys.stderr, flush=True)

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    filt = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))

    pools, refs, t = {}, {}, time.perf_counter()

    def dump():
        Path(args.out).write_text(json.dumps(
            {"slice": [args.start, args.end or len(subs)], "top_k": args.top_k,
             "pools": pools, "references": refs}, indent=1))

    for i, s in enumerate(sl, 1):
        if i == 1 or i % 5 == 0 or i == len(sl):
            print(f"  {i}/{len(sl)} ({time.perf_counter() - t:.0f}s)", file=sys.stderr, flush=True)
        det = generator.generate_scored_with_details(s, top_k=args.top_k, threshold=None,
                                                     compute_sites=False)
        det.sort(key=lambda d: (-d[1], d[0]))
        cands = [d[0] for d in det]
        fs = filt.score_batch(s, cands) if cands else []
        seen, out = set(), []
        for c, g, f in zip(cands, det, fs):
            k = _key(c)
            if k and k not in seen:
                seen.add(k)
                out.append({"smiles": c, "generator": float(g[1]), "filter": float(f),
                            "combined": float(f) * float(g[1]), "key": k})
        pools[s] = out
        refs[s] = sorted({k for k in (_key(p) for p in tmap[s]) if k})
        dump()          # after every substrate: an interrupted shard keeps what it has
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

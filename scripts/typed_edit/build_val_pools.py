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

from _provenance import record_inputs, stamp  # noqa: E402

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


def merge(pattern, out, allow_absent=()):
    """Merge shards into one artifact.

    The shards must tile the declared population. An index may be absent only if it is named
    on the command line, and the artifact then records which indices are absent and why the
    caller said so — an absence that is not declared is a failure, not a footnote.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("no shard matched", file=sys.stderr)
        return 1
    pools, refs, slices, ckpts, budgets = {}, {}, [], [], set()
    for p in paths:
        d = json.loads(Path(p).read_text())
        pools.update(d["pools"]); refs.update(d["references"]); slices.append(tuple(d["slice"]))
        if d.get("checkpoints"):
            ckpts.append(d["checkpoints"])
        budgets.add(d.get("top_k"))
    # The rule budget is the merged pool's defining parameter and it was recorded nowhere: a
    # reader of the merged file, and the budget curve that reads it, took the budget from the
    # directory name. Shards built at different budgets are not one pool and are refused.
    if len(budgets) > 1:
        print(f"FAIL: the shards were built at different rule budgets {sorted(budgets)}; the "
              f"merged pool would not have one", file=sys.stderr)
        return 1
        print(f"  + {Path(p).name}: {d['slice']} {len(d['pools'])}", file=sys.stderr)
    subs, vmap = population()
    # Absence is read off the pools, not off the slices. A shard may pass over a substrate inside
    # its own range -- the whole bank does not finish the peptide at index 83 in any time worth
    # spending -- and slice arithmetic calls that index covered, so the artifact would have
    # recorded no absence while holding one substrate fewer than it declares.
    held = set(pools)
    absent = sorted(i for i, s in enumerate(subs) if s not in held)
    undeclared = [i for i in absent if i not in set(allow_absent)]
    if undeclared:
        print(f"FAIL: the shards do not tile the population; absent and undeclared: "
              f"{undeclared}", file=sys.stderr)
        return 1
    if absent:
        print(f"absent by declaration: {absent}", file=sys.stderr)
    # Which models scored the merged pool, and a refusal if the shards do not agree: a pool
    # assembled from two runs is not one experiment and nothing downstream could tell.
    distinct = {json.dumps(c, sort_keys=True) for c in ckpts}
    if len(distinct) > 1:
        print("FAIL: the shards were scored by different checkpoints; the merged pool would be "
              "two experiments in one file", file=sys.stderr)
        return 1
    Path(out).write_text(json.dumps(
        {"provenance": stamp(__file__), "match": "inchikey_tautomer", "split": "validation",
         "checkpoints": json.loads(distinct.pop()) if distinct else None,
         "shards_recording_no_checkpoint": len(paths) - len(ckpts),
         "top_k": budgets.pop() if len(budgets) == 1 else None,
         "inputs": record_inputs(paths),
         "population": {"cap": CAP, "seed": SEED, "declared_n": len(subs),
                        "n": len(pools),
                        "absent_indices": absent,
                        "absent_substrates": [subs[i] for i in absent],
                        "absent_references": {subs[i]: sorted(
                            {k for k in (_key(pr) for pr in vmap[subs[i]]) if k})
                            for i in absent}},
         "slices": [list(s) for s in sorted(slices)],
         "pools": pools, "references": refs}, indent=1))
    print(f"wrote {out} with {len(pools)} substrates")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--merge", default="")
    # One substrate of this draw, a 515-character peptide at index 83, does not finish the whole
    # bank in any time worth spending; the committed artifact records it as absent for the same
    # reason. Naming it here rather than waiting on it keeps a shard from blocking the rest.
    ap.add_argument("--skip", default="",
                    help="comma-separated population indices this shard will not attempt; they "
                         "must then be declared to --absent at merge time or the merge refuses")
    ap.add_argument("--absent", default="",
                    help="comma-separated population indices the merge may lack")
    ap.add_argument("--gen-ckpt", default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--filter-ckpt", default=str(ROOT / "artifacts/full5000_implicit/checkpoints/filter.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "val_pools.json"))
    ap.add_argument("--standardise", choices=("every-product", "survivors"),
                    default="every-product",
                    help="every-product is what ships: the enumeration standardises each product "
                         "of each rule. survivors is the H13 arm: deduplicate on the cheap "
                         "canonical form during enumeration and standardise only the candidates "
                         "that survive the H9 cap, which is where the filter sees them.")
    ap.add_argument("--tautomer-budget", type=int, default=0,
                    help="H15: standardise the survivors with a private enumerator at this "
                         "budget. 0 keeps the shipped 1000. The matching key never uses it: "
                         "_tautomer_inchikey and this share preparation's cache and its global "
                         "enumerator, so lowering that would move the keys, which H15 forbids.")
    ap.add_argument("--cap", type=int, default=100,
                    help="the H9 cap, applied before the filter because the generator score is "
                         "known before any pair graph is built")
    ap.add_argument("--top-k", type=int, default=7581,
                    help="rule budget; 7581 is the whole bank, 30 is what the checkpoint records")
    args = ap.parse_args()

    if args.merge:
        allow = tuple(int(x) for x in args.absent.split(",") if x.strip())
        return merge(args.merge, args.out, allow)

    subs, vmap = population()
    sl = subs[args.start:(args.end or None)]
    print(f"validation substrates [{args.start}:{args.end or len(subs)}] of {len(subs)}",
          file=sys.stderr, flush=True)

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    filt = _load(Path(args.filter_ckpt), lambda a, r: build_filter(FilterConfig(**a)))

    if args.standardise == "survivors":
        # the enumeration's deduplication key becomes the cheap canonical form; the candidate
        # score is a noisy-or over the rules sharing a key, so this changes the scores as well as
        # the cost, which is why H13 is a hypothesis and not a refactor
        generator.gen_normalization = "canonical"

    from grail_metabolism.utils.preparation import _standardize_smiles_cached

    def make_bounded_standardiser(budget):
        """standardize_mol's pipeline with an enumerator of its own.

        preparation's `_TAUTOMER_ENUMERATOR` is a module singleton and `_standardize_smiles_cached`
        memoises on the SMILES alone, so lowering the budget there would silently re-key every
        match: `_tautomer_inchikey` calls the same cached function. This keeps a private
        enumerator and a private cache, and the global pair is left exactly as it ships.
        """
        from functools import lru_cache

        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        enum = rdMolStandardize.TautomerEnumerator()
        enum.SetMaxTautomers(budget)
        enum.SetMaxTransforms(budget)
        uncharger = rdMolStandardize.Uncharger()

        @lru_cache(maxsize=262144)
        def standardise(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(smiles)
            out = enum.Canonicalize(uncharger.uncharge(
                rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(mol))))
            if out is None:
                raise ValueError(smiles)
            return Chem.MolToSmiles(out, isomericSmiles=False)

        return standardise

    standardise_survivor = (make_bounded_standardiser(args.tautomer_budget)
                            if args.tautomer_budget else _standardize_smiles_cached)

    pools, refs, t = {}, {}, time.perf_counter()
    timing = []

    # Persisting after every substrate is only half of surviving a kill; the other half is not
    # starting the slice again. A shard that already holds pools for part of its slice keeps them
    # and skips those substrates, so a run interrupted at hour two resumes at hour two.
    resume = Path(args.out)
    if resume.exists():
        try:
            prior = json.loads(resume.read_text())
            if prior.get("slice") == [args.start, args.end or len(subs)] \
                    and prior.get("top_k") == args.top_k \
                    and prior.get("standardise") == args.standardise:
                pools.update(prior.get("pools") or {})
                refs.update(prior.get("references") or {})
                timing.extend(prior.get("generator_seconds") or [])
                print(f"  resuming with {len(pools)} substrates already built",
                      file=sys.stderr, flush=True)
        except Exception:
            pass

    import hashlib

    def _digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()[:16]

    def dump():
        """Persist after every substrate. A shard killed on the peptide at index 83 used to
        take its other 48 with it, twice."""
        Path(args.out).write_text(json.dumps(
            {"provenance": stamp(__file__),
             "slice": [args.start, args.end or len(subs)], "top_k": args.top_k,
             "standardise": args.standardise, "tautomer_budget": args.tautomer_budget or 1000,
             "cap": args.cap if args.standardise == "survivors" else None,
             "checkpoints": {"generator": {"path": str(Path(args.gen_ckpt).relative_to(ROOT)),
                                           "sha256_16": _digest(args.gen_ckpt)},
                             "filter": {"path": str(Path(args.filter_ckpt).relative_to(ROOT)),
                                        "sha256_16": _digest(args.filter_ckpt)}},
             "generator_seconds": timing, "pools": pools, "references": refs}, indent=1))
    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    for i, s in enumerate(sl, 1):
        if i == 1 or i % 5 == 0 or i == len(sl):
            print(f"  {i}/{len(sl)} ({time.perf_counter() - t:.0f}s)", file=sys.stderr, flush=True)
        if args.start + i - 1 in skip:
            print(f"  skipping index {args.start + i - 1} by request", file=sys.stderr, flush=True)
            continue
        if s in pools:
            continue
        t_gen = time.perf_counter()
        det = generator.generate_scored_with_details(s, top_k=args.top_k, threshold=None,
                                                     compute_sites=False)
        det.sort(key=lambda d: (-d[1], d[0]))
        enum_s = time.perf_counter() - t_gen
        std_s = 0.0
        if args.standardise == "survivors":
            det = det[:args.cap]        # H9's cap, before the expensive part rather than after
            t_std = time.perf_counter()
            fixed = []
            for d in det:
                try:
                    fixed.append((standardise_survivor(d[0]),) + tuple(d[1:]))
                except Exception:
                    fixed.append(d)
            det = fixed
            std_s = time.perf_counter() - t_std
        cands = [d[0] for d in det]
        fs = filt.score_batch(s, cands) if cands else []
        timing.append({"substrate": s, "seconds": round(enum_s + std_s, 3),
                       "enumerate": round(enum_s, 3), "standardise_survivors": round(std_s, 3),
                       "candidates": len(cands)})
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
        dump()

    # The same writer the loop uses. A second literal here once wrote the shard again without
    # the checkpoint block, so the record of which models scored a pool survived only when the
    # run was killed and was lost by every run that finished.
    dump()
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

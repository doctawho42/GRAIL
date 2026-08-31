#!/usr/bin/env python3
"""The chemistry an unrecoverable filter removed, typed against the chemistry it kept.

The corpus was assembled by a selection no script in this repository performs and no record
identifies, and the Supporting Information measures what that selection did to one source it can
still see: of MetXBioDB's annotated pairs, well under half are inside the corpus. That measurement
is reported and then goes nowhere, and it bears on the conclusion this work reaches hardest. If
the missing selection removed a *class* of chemistry, it would manufacture exactly the finding the
Conclusions rest on, a shortfall concentrated in transformation types the corpus does not hold.

MetXBioDB ships with BioTransformer and is on disk in full, so both halves can be typed: the pairs
the corpus kept and the pairs it dropped, under the same radius-0 reaction type the census uses.
If the two type distributions agree, the filter was not selecting on transformation type and the
alternative explanation is closed by measurement rather than by assertion. If they disagree, the
reader needs to know before reading the containment claim.

The comparison is a permutation test on the total variation distance between the two type
distributions: the labels kept and dropped are shuffled and the distance recomputed, which asks
whether a split of this size would separate the types this far by chance.

    python scripts/typed_edit/metxbiodb_drop_set.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

N_PERM, SEED = 10000, 0
METX = ROOT / "artifacts/tier2/biotransformer/database/MetXBioDB-1-0.json"


def _worker(job):
    """(inside, type-key) for one MetXBioDB pair, or None where it cannot be typed."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from coverage_gap_types import pair_to_type

    inside, sub_inchi, prod_inchi = job
    sub = Chem.MolFromInchi(sub_inchi)
    prod = Chem.MolFromInchi(prod_inchi)
    if sub is None or prod is None:
        return None
    try:
        t = pair_to_type(sub, prod)
    except Exception:
        return None
    if t is None:
        return None
    return inside, json.dumps(t, sort_keys=True)


def corpus_pairs():
    """Skeleton-keyed positive pairs of the whole corpus, the population the filter selected into."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    def skeleton(smiles):
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return Chem.MolToInchiKey(mol)[:14]
        except Exception:
            return None

    data = ROOT / "grail_metabolism" / "data"
    pairs = set()
    for split in ("train", "val", "test"):
        sdf, triples = data / f"{split}.sdf", data / f"{split}_triples_clean.txt"
        if not sdf.exists() or not triples.exists():
            continue
        index_to_smiles, cur, key = {}, {}, None
        with open(sdf, errors="replace") as handle:
            for line in handle:
                if line.startswith(">"):
                    match = re.match(r">\s*<([^>]+)>", line)
                    key = match.group(1) if match else None
                    continue
                if key is not None:
                    value = line.strip()
                    if value == "":
                        key = None
                    else:
                        cur.setdefault(key, value)
                    continue
                if line.startswith("$$$$"):
                    try:
                        index_to_smiles[int(cur["Index"])] = cur.get("SMILES")
                    except Exception:
                        pass
                    cur, key = {}, None
        with open(triples) as handle:
            for line in handle:
                parts = line.split()
                if len(parts) != 3 or parts[2] != "1":
                    continue
                a = skeleton(index_to_smiles.get(int(parts[0])))
                b = skeleton(index_to_smiles.get(int(parts[1])))
                if a and b:
                    pairs.add((a, b))
    return pairs


def skeleton_from_inchi(value):
    from rdkit import Chem

    if not value or not isinstance(value, str) or not value.startswith("InChI="):
        return None
    mol = Chem.MolFromInchi(value)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)[:14]
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "metxbiodb_drop_set.json"))
    args = ap.parse_args()

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    inside_corpus = corpus_pairs()
    blob = json.loads(METX.read_text())
    rows = list((blob.get("biotransformations") or {}).values())

    jobs, seen = [], set()
    for row in rows:
        substrate = row.get("Substrate") or {}
        sub_inchi = substrate.get("InChI") if isinstance(substrate, dict) else None
        a = skeleton_from_inchi(sub_inchi)
        if not a:
            continue
        for product in (row.get("Products") or []):
            prod_inchi = product.get("InChI") if isinstance(product, dict) else None
            b = skeleton_from_inchi(prod_inchi)
            if not b or (a, b) in seen:
                continue
            seen.add((a, b))
            jobs.append(((a, b) in inside_corpus, sub_inchi, prod_inchi))

    kept_n = sum(1 for j in jobs if j[0])
    print(f"{len(jobs)} distinct MetXBioDB pairs; {kept_n} inside the corpus, "
          f"{len(jobs) - kept_n} dropped", flush=True)

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    ctx = multiprocessing.get_context("spawn")
    kept, dropped = Counter(), Counter()
    typed, t0 = 0, time.perf_counter()
    with ctx.Pool(workers) as pool:
        for n, res in enumerate(pool.imap_unordered(_worker, jobs, 16), 1):
            if res is not None:
                typed += 1
                (kept if res[0] else dropped)[res[1]] += 1
            if n % 250 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} ({time.perf_counter() - t0:.0f}s) typed {typed}",
                      flush=True)

    universe = sorted(set(kept) | set(dropped))
    nk, nd = sum(kept.values()), sum(dropped.values())
    if not nk or not nd:
        print("FAIL: one side is empty; nothing to compare", file=sys.stderr)
        return 1

    def tv(a_counts, b_counts, a_n, b_n):
        return 0.5 * sum(abs(a_counts.get(t, 0) / a_n - b_counts.get(t, 0) / b_n)
                         for t in universe)

    observed = tv(kept, dropped, nk, nd)

    # The permutation null: the same pairs, the same split size, the kept/dropped label shuffled.
    labels = np.array([1] * nk + [0] * nd)
    codes = np.array([universe.index(t) for t in universe for _ in range(kept[t])]
                     + [universe.index(t) for t in universe for _ in range(dropped[t])])
    rng = np.random.default_rng(SEED)
    ge = 0
    for _ in range(N_PERM):
        perm = rng.permutation(labels)
        a = np.bincount(codes[perm == 1], minlength=len(universe)) / nk
        b = np.bincount(codes[perm == 0], minlength=len(universe)) / nd
        if 0.5 * np.abs(a - b).sum() >= observed:
            ge += 1
    p = (ge + 1) / (N_PERM + 1)

    dropped_types_absent_from_kept = sum(c for t, c in dropped.items() if t not in kept)

    # The question the containment claim actually raises. Of the test references whose type
    # neither the bank nor the training annotation holds -- the cell the Conclusions rest on --
    # how many have a type that IS present among the pairs this source held and the corpus
    # dropped? That mass is not chemistry the literature lacks. It is chemistry a file in this
    # repository holds and the assembly step did not carry across, so it is recoverable by
    # re-deriving the corpus rather than by finding a new one.
    recoverable = None
    absent_path = ROOT / "results/missing_types_in_train.json"
    if absent_path.exists():
        blob = json.loads(absent_path.read_text())
        per_type = blob.get("references_per_absent_type") or {}
        in_dropped = {ty: n for ty, n in per_type.items() if ty in dropped}
        in_kept_only = {ty: n for ty, n in per_type.items()
                        if ty not in dropped and ty in kept}
        # The same question one granularity coarser. A type key this strict can miss an overlap
        # that a chemist would call the same transformation, and the containment claim it feeds
        # is varied over four definitions, so this is varied too rather than reported at one.
        def deep(x):
            return tuple(deep(i) for i in x) if isinstance(x, list) else x

        def coarse(key):
            return str(sorted(frozenset(deep(e[0]) for e in json.loads(key))))

        dropped_coarse = {coarse(ty) for ty in dropped}
        in_dropped_coarse = {ty: n for ty, n in per_type.items()
                             if coarse(ty) in dropped_coarse}

        recoverable = {
            "source_of_the_absent_cell": str(absent_path.relative_to(ROOT)),
            "references_whose_type_neither_the_bank_nor_training_holds":
                int(sum(per_type.values())),
            "of_those_whose_type_this_source_dropped": int(sum(in_dropped.values())),
            "share": round(sum(in_dropped.values()) / max(sum(per_type.values()), 1), 4),
            "of_those_whose_type_this_source_kept_but_the_splits_lack":
                int(sum(in_kept_only.values())),
            "distinct_types_recoverable": len(in_dropped),
            "of_those_whose_type_this_source_dropped_counts_ignored":
                int(sum(in_dropped_coarse.values())),
            "share_counts_ignored": round(
                sum(in_dropped_coarse.values()) / max(sum(per_type.values()), 1), 4),
            "reading": ("a test reference in this count needs a transformation type that one of "
                        "the corpus's own four sources records and the assembly step did not "
                        "carry across; it bounds from below how much of the shortfall is the "
                        "assembly's rather than the literature's, since only one source of four "
                        "can be measured this way"),
        }

    report = {
        "provenance": stamp(__file__),
        "question": ("whether the unrecoverable selection that assembled the corpus removed a "
                     "class of chemistry, which would manufacture the novel-type shortfall this "
                     "work reports rather than measure it"),
        "source": {"file": str(METX.relative_to(ROOT)),
                   "distinct_pairs": len(jobs),
                   "inside_the_corpus": kept_n,
                   "dropped": len(jobs) - kept_n},
        "typing": ("radius-0 reaction type by the mining route, the same instrument the census "
                   "and the containment claim use"),
        "typed": {"kept": nk, "dropped": nd, "distinct_types": len(universe)},
        "distinct_types_kept": len(kept),
        "distinct_types_dropped": len(dropped),
        "total_variation_distance": round(observed, 4),
        "permutation": {"n": N_PERM, "seed": SEED, "p_value": round(p, 4),
                        "null": "the kept/dropped label shuffled over the same typed pairs"},
        "dropped_references_whose_type_is_absent_from_the_kept_half":
            dropped_types_absent_from_kept,
        "share_of_dropped_whose_type_the_kept_half_lacks": round(
            dropped_types_absent_from_kept / max(nd, 1), 4),
        "top_types_kept": [[t, c] for t, c in kept.most_common(8)],
        "top_types_dropped": [[t, c] for t, c in dropped.most_common(8)],
        "recoverable_from_the_dropped_half": recoverable,
        "reading": (
            "A filter selecting on transformation type would leave the two halves with different "
            "type distributions. The distance between them and the chance of seeing a distance "
            "that large under a shuffled label are reported, so the alternative explanation for "
            "the novel-type shortfall is decided by measurement. A large p-value does not prove "
            "the filter was blind to chemistry; it bounds how strongly this source can show that "
            "it was not."),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\ntyped {typed} of {len(jobs)} pairs: {nk} kept, {nd} dropped")
    print(f"  distinct types      : {len(kept)} kept, {len(dropped)} dropped, "
          f"{len(universe)} together")
    print(f"  total variation     : {observed:.4f}")
    print(f"  permutation p       : {p:.4f} over {N_PERM} shuffles")
    print(f"  dropped pairs whose type the kept half never shows: "
          f"{dropped_types_absent_from_kept} ({dropped_types_absent_from_kept / max(nd,1):.1%})")
    if recoverable:
        print(f"\nof the "
              f"{recoverable['references_whose_type_neither_the_bank_nor_training_holds']} test "
              f"references whose type neither the bank nor training holds, "
              f"{recoverable['of_those_whose_type_this_source_dropped']} "
              f"({recoverable['share']:.1%}) have a type this one source held and the corpus "
              f"dropped; with bond counts ignored, "
              f"{recoverable['of_those_whose_type_this_source_dropped_counts_ignored']} "
              f"({recoverable['share_counts_ignored']:.1%})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

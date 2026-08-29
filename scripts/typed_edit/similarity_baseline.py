#!/usr/bin/env python3
"""Does the deployed ranking beat similarity to the substrate, which needs no training at all?

Metabolism preserves the scaffold. A metabolite differs from its parent by an added oxygen, a
cleaved bond, a conjugate, so it stays close to what it came from, and closeness is a fingerprint
away. Any learned ranker over (substrate, candidate) pairs is competing against that baseline
whether or not it is reported, and nobody in this comparison reports it.

An earlier artifact, results/scaffold_baseline.json, asked this of the product ordering and found
the trained filter did not separate from similarity. That ordering is no longer deployed: the
paper's first registered prediction replaced the product with a rank fusion, worth +0.0733. So the
question has to be asked again of the ranking that ships.

Every arm re-ranks the SAME pool, so the pool, the matching rule and the budget are fixed and only
the order varies:

    fusion          the deployed reciprocal rank fusion of filter and generator orderings
    similarity      Tanimoto to the substrate, Morgan radius 2, no training and no labels
    dissimilarity   the same reversed, a control that the direction is not arbitrary
    random          a seeded permutation, to show what the pool alone is worth

    python scripts/typed_edit/similarity_baseline.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

from bank_without_selection import _key as _tautkey  # noqa: E402

KS = (1, 5, 10, 15, 30, 50)
CAP = 100
N_BOOT, SEED = 10000, 0


def load(spec):
    pools, refs = {}, {}
    for f in sorted(glob.glob(spec)) or [spec]:
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    return pools, refs


def run(pools, refs, subs, label, k_report):
    from rdkit import Chem, DataStructs, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp: dict = {}

    def fingerprint(s):
        if s not in fp:
            m = Chem.MolFromSmiles(s)
            fp[s] = gen.GetFingerprint(m) if m is not None else None
        return fp[s]

    rng = np.random.default_rng(SEED)
    arms = ("fusion", "similarity", "dissimilarity", "random")
    hits = {a: {k: [] for k in KS} for a in arms}
    U = []
    for s in subs:
        real = set(refs[s])
        U.append(len(real))
        keep = sorted(pools[s], key=lambda c: -c["generator"])[:CAP]
        fused = rrf_order(keep)
        fs = fingerprint(s)
        sims = [DataStructs.TanimotoSimilarity(fs, fc) if (fs and fc) else 0.0
                for fc in (fingerprint(c["smiles"]) for c in fused)]
        order = {
            "fusion": list(range(len(fused))),
            "similarity": sorted(range(len(fused)), key=lambda i: -sims[i]),
            "dissimilarity": sorted(range(len(fused)), key=lambda i: sims[i]),
            "random": list(rng.permutation(len(fused))),
        }
        # the parent-drop convention of the comparison table, so every arm here sits on the
        # same axis as the tables it is read beside
        pk = _tautkey(s)
        for a, idx in order.items():
            keys = [fused[i]["key"] for i in idx
                    if fused[i].get("key") and fused[i]["key"] != pk]
            for k in KS:
                hits[a][k].append(len(real & set(keys[:k])))

    U = np.array(U, dtype=float)
    N = float(U.sum())
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    den = np.maximum(U[idx].sum(axis=1), 1)

    def contrast(a, b, k):
        d = np.array(hits[a][k], dtype=float) - np.array(hits[b][k], dtype=float)
        bt = d[idx].sum(axis=1) / den
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(d.sum() / N), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "separates": bool(lo > 0 or hi < 0)}

    rec = {a: {str(k): round(float(np.array(hits[a][k]).sum() / N), 4) for k in KS} for a in arms}
    vs = {a: {str(k): contrast("fusion", a, k) for k in KS}
          for a in arms if a != "fusion"}
    return {"population": {"n": len(subs), "n_references": N, "source": label},
            "recall_micro": rec, "fusion_minus": vs,
            "verdict_at_k": {
                str(k): ("the deployed ranking separates from an untrained similarity ordering"
                         if vs["similarity"][str(k)]["separates"] and
                         vs["similarity"][str(k)]["gap"] > 0
                         else "it does not") for k in KS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/similarity_baseline.json"))
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    out = {}
    for label, spec in (("comparison set", "results/widepools_implicit/w*.json"),
                        ("validation draw", "results/val_pools.json")):
        path = spec if "*" in spec else str(ROOT / spec)
        if not (glob.glob(path) or Path(path).exists()):
            print(f"  {label}: no pool at {spec}, skipped", file=sys.stderr)
            continue
        pools, refs = load(path)
        subs = sorted(s for s in pools if refs.get(s))
        print(f"  {label}: {len(subs)} substrates", file=sys.stderr, flush=True)
        out[label] = run(pools, refs, subs, label, args.k)

    rep = {"provenance": stamp(__file__), "aggregation": "micro, ratio of sums",
           "cap": CAP, "n_boot": N_BOOT, "seed": SEED,
           "note": ("every arm re-ranks one pool, so pool, matching and budget are fixed and only "
                    "the order varies; the fusion arm is the deployed ranking"),
           "by_population": out}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    for label, r in out.items():
        print(f"\n## {label}  (n={r['population']['n']})")
        print(f"{'k':>4}" + "".join(f"{a:>15}" for a in r["recall_micro"]))
        for k in KS:
            print(f"{k:>4}" + "".join(f"{r['recall_micro'][a][str(k)]:>15.4f}"
                                     for a in r["recall_micro"]))
        print("  fusion minus similarity:")
        for k in KS:
            c = r["fusion_minus"]["similarity"][str(k)]
            print(f"    k={k:<3} {c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
                  f"{'  *' if c['separates'] else ''}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

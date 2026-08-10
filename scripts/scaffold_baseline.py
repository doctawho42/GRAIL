#!/usr/bin/env python3
"""Does the learned filter rank better than similarity to the substrate?

Metabolism preserves the scaffold. A metabolite differs from its parent by an added oxygen, a
cleaved bond, a conjugate: it stays close to what it came from, and closeness is a fingerprint
distance away. A learned filter that scores (substrate, product) pairs is therefore competing
against a baseline that needs no training and no labels at all, and nobody in this comparison has
reported that baseline.

Every arm here re-ranks the SAME candidate pool that the deployed system produced, so the pool, the
matching rule and the budget are held fixed and the only thing varying is the order:

    deployed        the trained filter's combined score
    similarity      Tanimoto to the substrate, Morgan radius 2, no training
    dissimilarity   the same, reversed, as a control that the direction is not arbitrary
    random          a fixed permutation, to show what the pool alone is worth

If similarity ranks as well as the filter, the filter has learned the scaffold and little else, and
that is worth knowing before anyone builds a larger one.
"""
from __future__ import annotations

import argparse, json, pathlib, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0


def _code_version() -> dict:
    import subprocess
    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "results" / "scaffold_baseline.json"))
    args = ap.parse_args()

    from grail_metabolism.metrics import _tautomer_inchikey
    cache: dict = {}

    def key(s):
        if s not in cache:
            try:
                cache[s] = _tautomer_inchikey(s)
            except Exception:
                cache[s] = None
        return cache[s]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp: dict = {}

    def fingerprint(s):
        if s not in fp:
            m = Chem.MolFromSmiles(s)
            fp[s] = gen.GetFingerprint(m) if m is not None else None
        return fp[s]

    rows = json.loads((ROOT / "results/scored_predictions.json").read_text())["rows"]
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    rng = np.random.default_rng(SEED)

    arms = {"deployed": [], "similarity": [], "dissimilarity": [], "random": []}
    for row in rows:
        sub = row["sub"]
        refs = {k for k in (key(y) for y in truth.get(sub, [])) if k}
        if not refs:
            continue
        cands = row["candidates"]
        fs = fingerprint(sub)
        sims = []
        for c in cands:
            fc = fingerprint(c["smiles"])
            sims.append(DataStructs.TanimotoSimilarity(fs, fc) if (fs and fc) else 0.0)
        order = {
            "deployed": list(range(len(cands))),
            "similarity": sorted(range(len(cands)), key=lambda i: -sims[i]),
            "dissimilarity": sorted(range(len(cands)), key=lambda i: sims[i]),
            "random": list(rng.permutation(len(cands))),
        }
        for name, idx in order.items():
            got = {k for k in (key(cands[i]["smiles"]) for i in idx[: args.k]) if k}
            arms[name].append(len(refs & got) / max(len(refs), 1))

    n = len(arms["deployed"])
    bi = rng.integers(0, n, (N_BOOT, n))
    base = np.array(arms["deployed"])
    rep = {"config": {**_code_version(), "k": args.k, "n_substrates": n, "n_boot": N_BOOT,
                      "seed": SEED, "fingerprint": "Morgan radius 2, 2048 bits",
                      "note": "same pool, same matching rule, same budget; only the order varies"},
           "arms": {}}
    for name, v in arms.items():
        v = np.array(v)
        d = v - base
        bt = d[bi].mean(axis=1)
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        rep["arms"][name] = {"recall": round(float(v.mean()), 4),
                             "vs_deployed": round(float(d.mean()), 4),
                             "ci95": [round(lo, 4), round(hi, 4)],
                             "certified": bool(lo * hi > 0)}
        print(f"  {name:14} recall@{args.k} {v.mean():.4f}   against the filter "
              f"{d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""How much is thrown away by scoring candidates independently of their siblings?

Every metabolite predictor in this comparison scores a candidate as a (substrate, product) pair and
emits whatever clears a threshold or a fixed budget. The label it is trained against is not a
per-pair event. Enzymes compete for one substrate, so the products observed for it are alternatives
drawn from a shared pool, and how many are observed is itself a quantity: on the clean test split
the median substrate has one annotated metabolite and the mean has 2.22, while the deployed system
emits 8.4 candidates for every substrate alike.

A per-pair sigmoid has no mechanism to express that. A model normalised over siblings does, and it
would have to be worth something. This measures the something, on frozen scores, with nothing
retrained:

    fixed budget       the deployed policy, every substrate truncated at the same k
    oracle count       every substrate truncated at its own number of annotated metabolites
    predicted count    the same, with the count predicted from the substrate rather than looked up

The first two bracket what any set-size-calibrated model can win at fixed ranking quality. The
third asks whether the count is predictable at all, which decides whether that headroom is
reachable: it is estimated on the training split and applied to test, so it is a forecast and not a
fit. Nothing here changes the ranking, so any gain is attributable to the size of the emitted set
and to nothing else.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

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


def _prf(hits: int, emitted: int, refs: int):
    p = hits / emitted if emitted else 0.0
    r = hits / refs if refs else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate(rows, truth_keys, counts, keyer):
    """counts[sub] -> how many candidates to emit. Macro precision, recall and F1."""
    P, R, F = [], [], []
    for row in rows:
        sub = row["sub"]
        refs = truth_keys.get(sub)
        if not refs:
            continue
        k = max(1, int(counts(sub, row)))
        emitted = [c["smiles"] for c in row["candidates"][:k]]
        keys = {keyer(s) for s in emitted} - {None}
        hits = len(keys & refs)
        p, r, f = _prf(hits, len(keys), len(refs))
        P.append(p); R.append(r); F.append(f)
    return np.array(P), np.array(R), np.array(F)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "setsize_headroom.json"))
    ap.add_argument("--match", default="tautomer", choices=("tautomer", "inchikey"),
                    help="a result that only holds under the most permissive rule is not a result")
    ap.add_argument("--population", default="all", choices=("all", "parents", "metabolites"),
                    help="the split silently mixes parent drugs with their own metabolites")
    args = ap.parse_args()

    from grail_metabolism.metrics import _tautomer_inchikey
    from rdkit.Chem import inchi

    cache: dict = {}

    def keyer(s):
        if s not in cache:
            try:
                if args.match == "tautomer":
                    cache[s] = _tautomer_inchikey(s)
                else:
                    m = Chem.MolFromSmiles(s)
                    cache[s] = inchi.MolToInchiKey(m) if m is not None else None
            except Exception:
                cache[s] = None
        return cache[s]

    scored = json.loads((ROOT / "results/scored_predictions.json").read_text())
    rows = scored["rows"]
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    if args.population != "all":
        smi = [Chem.MolToSmiles(m) if m is not None else None
               for m in Chem.SDMolSupplier(str(ROOT / "grail_metabolism/data/test.sdf"))]
        prods = set()
        for line in (ROOT / "grail_metabolism/data/test_triples_clean.txt").read_text().splitlines():
            a_, b_, r_ = line.split()
            a_, b_ = int(a_), int(b_)
            if r_ == "1" and a_ < len(smi) and b_ < len(smi) and smi[a_] and smi[b_]:
                prods.add(smi[b_])
        keep = (lambda s: s in prods) if args.population == "metabolites" else (lambda s: s not in prods)
        truth = {s: v for s, v in truth.items() if keep(s)}
        rows = [r for r in rows if r["sub"] in truth]
    truth_keys = {s: {k for k in (keyer(y) for y in ys) if k} for s, ys in truth.items()}

    # the count predictor: trained on the training split, applied to test. Deliberately the
    # simplest thing that could work, so that a gain is attributable to calibrating the size and
    # not to a model of the chemistry.
    tri = ROOT / "grail_metabolism/data/train_triples_clean.txt"
    sdf = ROOT / "grail_metabolism/data/train.sdf"
    feats, targets = [], []
    if tri.exists() and sdf.exists():
        smi = [Chem.MolToSmiles(m) if m is not None else None
               for m in Chem.SDMolSupplier(str(sdf))]
        per: dict = {}
        for line in tri.read_text().splitlines():
            a, b, r = line.split()
            if r != "1":
                continue
            a, b = int(a), int(b)
            if a < len(smi) and b < len(smi) and smi[a] and smi[b]:
                per.setdefault(smi[a], set()).add(smi[b])
        for s, ys in per.items():
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            feats.append([m.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumRings(m),
                          rdMolDescriptors.CalcNumRotatableBonds(m), 1.0])
            targets.append(len(ys))
    beta = None
    if len(feats) > 50:
        X, y = np.array(feats, dtype=float), np.array(targets, dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    def predicted(sub, row):
        if beta is None:
            return 2
        m = Chem.MolFromSmiles(sub)
        if m is None:
            return 2
        x = np.array([m.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumRings(m),
                      rdMolDescriptors.CalcNumRotatableBonds(m), 1.0], dtype=float)
        return int(round(min(max(float(x @ beta), 1.0), 15.0)))

    # The architectural question is not whether fewer is better but whether HOW MANY should depend
    # on the substrate. A global constant answers no; a rule relative to the pool answers yes only
    # if it beats the best constant. The gap rule emits while a candidate scores within a factor of
    # the leader, which is what normalising over siblings does at inference.
    def gap_rule(alpha):
        def f(sub, row):
            cs = row["candidates"]
            if not cs:
                return 1
            top = cs[0]["combined"]
            n_ = 1
            for c in cs[1:]:
                if c["combined"] >= alpha * top:
                    n_ += 1
                else:
                    break
            return n_
        return f

    arms = {}
    for k in (1, 2, 3, 5, 8, 15):
        arms[f"fixed k={k}"] = evaluate(rows, truth_keys, lambda s, r, k=k: k, keyer)
    arms["oracle count"] = evaluate(rows, truth_keys,
                                    lambda s, r: len(truth_keys.get(s, ())), keyer)
    arms["predicted count"] = evaluate(rows, truth_keys, predicted, keyer)
    for alpha in (0.9, 0.75, 0.5, 0.25):
        arms[f"gap rule a={alpha}"] = evaluate(rows, truth_keys, gap_rule(alpha), keyer)

    rng = np.random.default_rng(SEED)
    base = arms["fixed k=15"]
    n = len(base[0])
    idx = rng.integers(0, n, (N_BOOT, n))

    rep = {"config": {**_code_version(), "n_substrates": int(n), "n_boot": N_BOOT, "seed": SEED,
                      "match": args.match, "population": args.population,
                      "note": "ranking untouched in every arm; only the number emitted changes",
                      "count_model": "least squares on heavy atoms, rings, rotatable bonds, "
                                     "fitted on the training split"},
           "arms": {}}
    for name, (P, R, F) in arms.items():
        d = F - base[2]
        bt = d[idx].mean(axis=1)
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        rep["arms"][name] = {
            "precision": round(float(P.mean()), 4), "recall": round(float(R.mean()), 4),
            "f1": round(float(F.mean()), 4),
            "f1_gain_over_k15": round(float(d.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)], "certified": bool(lo * hi > 0)}
        print(f"  {name:18} P {P.mean():.4f}  R {R.mean():.4f}  F1 {F.mean():.4f}  "
              f"dF1 {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")

    if beta is not None:
        rep["count_model_coefficients"] = [round(float(b), 5) for b in beta]
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

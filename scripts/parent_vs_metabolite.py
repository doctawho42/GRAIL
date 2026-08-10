#!/usr/bin/env python3
"""The test split is two populations, and nobody says which one a number is about.

Every substrate in this benchmark is scored the same way, and they are not the same kind of thing.
Some are parent drugs, the compound a patient swallows. Others are metabolites of a parent that is
also in the split, so the model is being asked to predict the second step of a chain whose first
step it was separately asked to predict. The two are chemically different populations: a metabolite
has already been oxidised or conjugated once, it is smaller here by four heavy atoms at the median,
and the transformations still available to it are not the ones available to its parent.

A benchmark that mixes them reports one number for two questions. This measures whether that
matters:

    per method, recall on the parents and recall on the metabolites
    the paired difference between the two populations, per method
    whether the ORDERING of methods differs between them, which is what a leaderboard asserts

The populations are disjoint by construction, so the between-population contrast is unpaired and
gets an unpaired interval; the ordering question is answered inside each population, where it is
paired. A method that is better on parents and worse on metabolites is not better; it is better at
one of the two questions the benchmark is silently averaging.
"""
from __future__ import annotations

import argparse
import itertools
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

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED, K = 10000, 0, 15


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


def load_predictions() -> dict:
    out = {}
    sp = ROOT / "results/scored_predictions.json"
    if sp.exists():
        out["GRAIL"] = {r["sub"]: [c["smiles"] for c in r["candidates"][:K]]
                        for r in json.loads(sp.read_text())["rows"]}
    for name, path in (("SyGMa", "results/sygma_fulltest_predictions.json"),
                       ("MetaPredictor", "artifacts/tier2_1170/metapredictor_preds.json")):
        q = ROOT / path
        if not q.exists():
            continue
        d = json.loads(q.read_text())
        d = d.get("predictions", d)
        # a prediction file is substrate -> ranked list; anything else is a summary and is skipped
        d = {s: v for s, v in d.items() if isinstance(v, (list, tuple))}
        if d:
            out[name] = {s: list(v)[:K] for s, v in d.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "parent_vs_metabolite.json"))
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

    # the split: a substrate that is itself an annotated product of another substrate here
    smi = [Chem.MolToSmiles(m) if m is not None else None
           for m in Chem.SDMolSupplier(str(ROOT / "grail_metabolism/data/test.sdf"))]
    products = set()
    for line in (ROOT / "grail_metabolism/data/test_triples_clean.txt").read_text().splitlines():
        a, b, r = line.split()
        a, b = int(a), int(b)
        if r == "1" and a < len(smi) and b < len(smi) and smi[a] and smi[b]:
            products.add(smi[b])

    truth = json.loads((ROOT / "results/test_references.json").read_text())
    preds = load_predictions()
    methods = sorted(preds)
    groups = {"parents": [s for s in truth if s not in products],
              "metabolites": [s for s in truth if s in products]}
    print(f"{len(methods)} methods; parents {len(groups['parents'])}, "
          f"metabolites {len(groups['metabolites'])}")

    per: dict = {}
    for g, subs in groups.items():
        subs = [s for s in subs if all(s in preds[m] for m in methods)]
        per[g] = {"n": len(subs), "subs": subs}
        for m in methods:
            v = []
            for s in subs:
                refs = {k for k in (key(y) for y in truth[s]) if k}
                got = {k for k in (key(y) for y in preds[m][s]) if k}
                v.append(len(refs & got) / max(len(refs), 1))
            per[g][m] = np.array(v)

    rng = np.random.default_rng(SEED)
    rep = {"config": {**_code_version(), "k": K, "n_boot": N_BOOT, "seed": SEED,
                      "match": "inchikey_tautomer", "methods": methods,
                      "split": "a substrate counts as a metabolite when it is an annotated product "
                               "of another substrate in the same split"},
           "by_population": {}, "orderings": {}, "pairs": {}}

    for g in groups:
        subs = per[g]["subs"]
        idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
        rep["by_population"][g] = {"n": len(subs)}
        for m in methods:
            bt = per[g][m][idx].mean(axis=1)
            rep["by_population"][g][m] = {
                "recall": round(float(per[g][m].mean()), 4),
                "ci95": [round(float(np.quantile(bt, .025)), 4),
                         round(float(np.quantile(bt, .975)), 4)]}
        rep["orderings"][g] = sorted(methods, key=lambda m: -float(per[g][m].mean()))
        for a, b in itertools.combinations(methods, 2):
            d = per[g][a] - per[g][b]
            bt = d[idx].mean(axis=1)
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            rep["pairs"].setdefault(f"{a} vs {b}", {})[g] = {
                "margin": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "certified": bool(lo * hi > 0)}

    # The sharper question is not whether the ordering changes but whether the two populations are
    # the same task. A method that loses on metabolites what it keeps on parents is being averaged
    # over two questions, and the size of that gap differs by method: an interaction, unpaired
    # because the populations are disjoint, so the two bootstraps are drawn independently.
    ip = rng.integers(0, len(per["parents"]["subs"]), (N_BOOT, len(per["parents"]["subs"])))
    im = rng.integers(0, len(per["metabolites"]["subs"]), (N_BOOT, len(per["metabolites"]["subs"])))
    drop = {}
    for m in methods:
        bt = per["parents"][m][ip].mean(axis=1) - per["metabolites"][m][im].mean(axis=1)
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        drop[m] = {"parents_minus_metabolites": round(float(per["parents"][m].mean()
                                                            - per["metabolites"][m].mean()), 4),
                   "ci95": [round(lo, 4), round(hi, 4)], "certified": bool(lo * hi > 0)}
    inter = {}
    for a, b in itertools.combinations(methods, 2):
        bt = ((per["parents"][a][ip].mean(axis=1) - per["metabolites"][a][im].mean(axis=1))
              - (per["parents"][b][ip].mean(axis=1) - per["metabolites"][b][im].mean(axis=1)))
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        inter[f"{a} vs {b}"] = {"delta": round(float(bt.mean()), 4),
                                "ci95": [round(lo, 4), round(hi, 4)],
                                "certified": bool(lo * hi > 0)}
    rep["population_drop"] = drop
    rep["drop_differs_by_method"] = inter
    print("\n  recall lost from parents to metabolites, by method:")
    for m, v in drop.items():
        print(f"    {m:16} {v['parents_minus_metabolites']:+.4f} {v['ci95']}  "
              f"certified {v['certified']}")
    print("  the loss differs by method:")
    for k, v in inter.items():
        print(f"    {k:34} {v['delta']:+.4f} {v['ci95']}  certified {v['certified']}")

    same = rep["orderings"]["parents"] == rep["orderings"]["metabolites"]
    flips = [p for p, v in rep["pairs"].items()
             if v["parents"]["margin"] * v["metabolites"]["margin"] < 0]
    both = [p for p in flips
            if rep["pairs"][p]["parents"]["certified"] and rep["pairs"][p]["metabolites"]["certified"]]
    rep["ordering_is_the_same"] = same
    rep["pairs_that_change_sign"] = flips
    rep["pairs_certified_on_both_sides"] = both

    print(f"\n  parents      : {' > '.join(rep['orderings']['parents'])}")
    print(f"  metabolites  : {' > '.join(rep['orderings']['metabolites'])}")
    print(f"  same ordering: {same}")
    print(f"  pairs changing sign between the two populations: {len(flips)} {flips}")
    print(f"  and certified on both sides: {len(both)} {both}")
    for g in groups:
        print(f"  {g:12} " + "  ".join(f"{m} {rep['by_population'][g][m]['recall']:.4f}"
                                       for m in methods))
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

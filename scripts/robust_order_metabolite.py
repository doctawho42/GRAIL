#!/usr/bin/env python3
"""The robust order on the metabolite leaderboard, over criteria as well as budgets.

scripts/robust_order.py measures the share of a leaderboard's pairwise claims that survive every
cell of a declared grid, on two retrosynthesis leaderboards where the grid is four matching criteria
by four budgets. The metabolite comparison in this paper has so far been measured over budgets
alone, which is half a grid and makes the two domains not comparable on the very quantity being
proposed. This closes that: the same instrument, the same definition, five criteria by four budgets
on the clean test split, from the frozen predictions of three methods.

Domination is the intersection of the per-cell orders and is therefore transitive; certified
domination additionally requires each per-cell paired interval to exclude zero and is reported
separately, because an interval condition is not.
"""
from __future__ import annotations

import argparse, itertools, json, pathlib, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import inchi, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
KS = (1, 5, 10, 15)
MODES = ("canonical", "inchikey", "nostereo", "tanimoto1", "tautomer")


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
    ap.add_argument("--out", default=str(ROOT / "results" / "robust_order_metabolite.json"))
    args = ap.parse_args()

    from grail_metabolism.metrics import _tautomer_inchikey
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cache: dict = {}

    def keys(s):
        """All five equivalence relations for one structure, computed once."""
        if s in cache:
            return cache[s]
        m = Chem.MolFromSmiles(s)
        if m is None:
            cache[s] = None
            return None
        try:
            ik = inchi.MolToInchiKey(m)
        except Exception:
            ik = None
        out = {"canonical": Chem.MolToSmiles(m),
               "inchikey": ik,
               "nostereo": ik.split("-")[0] if ik else None,
               "tautomer": None, "fp": gen.GetFingerprint(m)}
        try:
            out["tautomer"] = _tautomer_inchikey(s)
        except Exception:
            pass
        cache[s] = out
        return out

    preds = {}
    sp = ROOT / "results/scored_predictions.json"
    preds["GRAIL"] = {r["sub"]: [c["smiles"] for c in r["candidates"]]
                      for r in json.loads(sp.read_text())["rows"]}
    for name, path in (("SyGMa", "results/sygma_fulltest_predictions.json"),
                       ("MetaPredictor", "artifacts/tier2_1170/metapredictor_preds.json")):
        d = json.loads((ROOT / path).read_text())
        d = d.get("predictions", d)
        preds[name] = {s: list(v) for s, v in d.items() if isinstance(v, (list, tuple))}

    truth = json.loads((ROOT / "results/test_references.json").read_text())
    methods = sorted(preds)
    subs = [s for s in truth if truth[s] and all(s in preds[m] for m in methods)]
    print(f"{len(methods)} methods, {len(subs)} substrates, "
          f"{len(MODES)} criteria x {len(KS)} budgets", flush=True)

    cells = [(mo, k) for mo in MODES for k in KS]
    hits = {(m, c): np.zeros(len(subs)) for m in methods for c in cells}
    for j, s in enumerate(subs):
        refs = [keys(y) for y in truth[s]]
        refs = [r for r in refs if r]
        for m in methods:
            got = [keys(y) for y in preds[m][s][: max(KS)]]
            got = [(r, i + 1) for i, r in enumerate(got) if r]
            for mo in MODES:
                first = None
                for r, rank in got:
                    hit = False
                    if mo == "tanimoto1":
                        hit = any(DataStructs.TanimotoSimilarity(r["fp"], t["fp"]) >= 1.0
                                  for t in refs)
                    else:
                        hit = r.get(mo) is not None and any(r[mo] == t.get(mo) for t in refs)
                    if hit:
                        first = rank
                        break
                if first is None:
                    continue
                for k in KS:
                    if first <= k:
                        hits[(m, (mo, k))][j] = 1.0
        if (j + 1) % 200 == 0:
            print(f"  {j + 1}/{len(subs)}", flush=True)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    published_cell = ("tautomer", 15)
    published = sorted(methods, key=lambda m: -float(hits[(m, published_cell)].mean()))
    rank = {m: i for i, m in enumerate(published)}

    pairs, n_dom, n_cert = {}, 0, 0
    for a, b in itertools.combinations(methods, 2):
        hi_, lo_ = (a, b) if rank[a] < rank[b] else (b, a)
        per = {}
        for c in cells:
            d = hits[(hi_, c)] - hits[(lo_, c)]
            bt = d[idx].mean(axis=1)
            l, h = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            per[str(c)] = {"margin": round(float(d.mean()), 4), "positive": bool(d.mean() > 0),
                           "certified": bool(l > 0)}
        dom = all(v["positive"] for v in per.values())
        cert = all(v["certified"] for v in per.values())
        n_dom += dom
        n_cert += cert
        pairs[f"{hi_} over {lo_}"] = {
            "dominates": dom, "certified": cert,
            "cells_that_reverse_it": [k for k, v in per.items() if not v["positive"]],
            "per_cell": per}

    rep = {"config": {**_code_version(), "n_boot": N_BOOT, "seed": SEED, "n_substrates": len(subs),
                      "grid": {"criteria": list(MODES), "budgets": list(KS)},
                      "published_cell": str(published_cell)},
           "published_order": published, "n_pairs": len(pairs),
           "n_dominating": n_dom, "n_certified": n_cert,
           "robustness": round(n_dom / max(len(pairs), 1), 4),
           "certified_robustness": round(n_cert / max(len(pairs), 1), 4), "pairs": pairs}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\n  published order at {published_cell}: {' > '.join(published)}")
    print(f"  dominate every cell: {n_dom}/{len(pairs)}   certified: {n_cert}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

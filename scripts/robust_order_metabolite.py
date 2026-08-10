#!/usr/bin/env python3
"""The robust order on the metabolite leaderboard, over criteria as well as budgets.

scripts/robust_order.py measures the share of a leaderboard's pairwise claims that survive every
cell of a declared grid, on two retrosynthesis leaderboards where the grid is four matching criteria
by four budgets. The metabolite comparison in this paper has so far been measured over budgets
alone, which is half a grid and makes the two domains not comparable on the very quantity being
proposed. This closes that: the same instrument, the same definition, the same four criteria by
four budgets, on the clean test split, from the frozen predictions of three methods.

The classification, the intervals and the sub-grid curve come from `robust_order.analyse`, imported
rather than restated, so that the rows of the paper's table are not two implementations of one
definition. Only the construction of the hit matrix differs, because the domains store predictions
differently.
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
from rdkit.Chem import inchi, rdFingerprintGenerator

from robust_order import analyse

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
KS = (1, 5, 10, 15)
# The grid runs over the four criteria that are identity relations induced by a canonical form.
# TANIMOTO1 is reported in the criteria table, because published work compares Morgan fingerprints
# and the number is informative, but it is excluded here: at radius 2 over 2048 bits it returns 1.0
# for decane against undecane and for D- against L-alanine, so it is a similarity threshold and not
# a candidate answer to "are these the same molecule". A cell that identifies two different
# compounds cannot be a convention a published ordering might have been computed under.
MODES = ("canonical", "inchikey", "nostereo", "tautomer")


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
        """Every criterion for one structure, computed once."""
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

    published_cell = ("tautomer", 15)
    sub_grids = {"criteria only, at the published budget":
                     [(m, published_cell[1]) for m in MODES],
                 "budgets only, at the published criterion":
                     [(published_cell[0], k) for k in KS],
                 "the product": cells}
    r = analyse(hits, methods, cells, published_cell, sub_grids)

    rep = {"config": {**_code_version(), "n_boot": N_BOOT, "seed": SEED, "n_substrates": len(subs),
                      "grid": {"criteria": list(MODES), "budgets": list(KS)},
                      "instrument": "robust_order.analyse"}, **r}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\n  published order at {published_cell}: {' > '.join(r['published_order'])}")
    print(f"  dominate every cell: {r['n_dominating']}/{r['n_pairs']} = {r['robustness']} "
          f"{r['robustness_ci95']}")
    print(f"  separated in every cell: {r['n_separated_in_every_cell']}   "
          f"reversed with an interval: {r['reversed_with_an_interval']}")
    print(f"  tiers {r['tiers_distinguished']} of {r['n_systems']};  "
          f"distinct orderings {r['distinct_orderings_across_the_grid']} of {r['n_cells']} cells")
    print(f"  among pairs its own cell resolves: {r['robustness_among_resolved']} "
          f"({r['n_resolved_in_the_published_cell']} pairs)")
    for lab, v in r["sub_grids"].items():
        print(f"    {lab:48} {v['n_cells']:2} cells -> {v['n_dominating']}/{r['n_pairs']} "
              f"= {v['share']}, {v['tiers']} tiers, {v['distinct_orderings']} orderings")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

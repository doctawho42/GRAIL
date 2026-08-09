#!/usr/bin/env python3
"""A ranker trained on the joint pool itself, rather than borrowed from one arm of it.

Everything tried so far orders the union with what the two methods report about their own
candidates, or with a model trained on one of them. Both plateau: the pool holds 0.674 of the
references and no such policy passes 0.54 at k=15, though agreement followed by the deployed filter
reaches 0.299 at k=5 against MetaTox's 0.218.

This trains on the union. The label is defined on the joint pool, so a candidate is positive if it
is a reference and negative otherwise regardless of which bank proposed it -- which is the thing no
borrowed model has, since each was trained with its own bank's negatives.

Three feature groups, kept separable so the ablation says what pays:

  borrowed   what the two methods report: agreement, each reciprocal rank, MetaTox's likeness
             score, GRAIL's own filter score where it has one
  transfer   the deployed pair filter applied to every candidate including those from the other
             bank, which separates references at +0.124 there against +0.267 on its own
  chemistry  the transformation itself: changes in mass, lipophilicity, polar surface, rings,
             heavy atoms and hydrogen-bond counts, the Morgan similarity of product to substrate,
             and the share of the product covered by its maximum common substructure with it

Honesty constraint, unchanged and stated in the artifact: MetaTox's predictions exist for one
population, so there is no disjoint split. Every number here is grouped five-fold cross-validation
over substrates -- each candidate scored by a model that never saw its substrate -- and is reported
as cross-validated, never as held out. A single population also means the comparison against MetaTox
is on the same substrates the folds are drawn from, which bounds what this can be claimed to show:
it is evidence that the union carries orderable signal, not a deployment result.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFMCS, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

from grail_metabolism.metrics import _tautomer_inchikey as _tk

RDLogger.DisableLog("rdApp.*")
KEYS = ROOT / "results" / "key_tables" / "inchikey_tautomer.json"
BUDGETS = (1, 3, 5, 10, 15, 30)
N_BOOT, SEED, FOLDS = 10000, 0, 5
MFP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

BORROWED = ["both", "g_recip", "m_recip", "m_score", "g_filter"]
TRANSFER = ["pair_score"]
CHEMISTRY = ["d_mw", "d_logp", "d_tpsa", "d_rings", "d_heavy", "d_hbd", "d_hba",
             "tanimoto", "mcs_share"]


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


def chemistry(sub_mol, prod_smiles: str) -> dict:
    """What the transformation does, rather than who proposed it."""
    p = Chem.MolFromSmiles(prod_smiles)
    if p is None or sub_mol is None:
        return {k: float("nan") for k in CHEMISTRY}
    try:
        fp_s, fp_p = MFP.GetFingerprint(sub_mol), MFP.GetFingerprint(p)
        tani = DataStructs.TanimotoSimilarity(fp_s, fp_p)
        mcs = rdFMCS.FindMCS([sub_mol, p], timeout=2, ringMatchesRingOnly=True)
        share = (mcs.numAtoms / p.GetNumHeavyAtoms()) if p.GetNumHeavyAtoms() else 0.0
        return {"d_mw": Descriptors.MolWt(p) - Descriptors.MolWt(sub_mol),
                "d_logp": Crippen.MolLogP(p) - Crippen.MolLogP(sub_mol),
                "d_tpsa": rdMolDescriptors.CalcTPSA(p) - rdMolDescriptors.CalcTPSA(sub_mol),
                "d_rings": rdMolDescriptors.CalcNumRings(p) - rdMolDescriptors.CalcNumRings(sub_mol),
                "d_heavy": p.GetNumHeavyAtoms() - sub_mol.GetNumHeavyAtoms(),
                "d_hbd": rdMolDescriptors.CalcNumHBD(p) - rdMolDescriptors.CalcNumHBD(sub_mol),
                "d_hba": rdMolDescriptors.CalcNumHBA(p) - rdMolDescriptors.CalcNumHBA(sub_mol),
                "tanimoto": tani, "mcs_share": share}
    except Exception:
        return {k: float("nan") for k in CHEMISTRY}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/pair_chemistry_ranker.json",
                    help="only for its config; the pair scores are recomputed from the artifact")
    ap.add_argument("--out", default=str(ROOT / "results" / "union_ranker.json"))
    args = ap.parse_args()

    cache = json.loads(KEYS.read_text()) if KEYS.exists() else {}

    def key(s):
        k = cache.get(s)
        if k is None:
            try:
                k = _tk(s)
            except Exception:
                k = None
            cache[s] = k
        return k

    graw = {r["sub"]: r["candidates"]
            for r in json.loads((ROOT / "results/scored_predictions.json").read_text())["rows"]}
    mt = json.loads((ROOT / "results/metatox_smirks_preds.json").read_text())
    mscored = mt["predictions_with_scores"]
    truth = json.loads((ROOT / "results/test_references.json").read_text())

    # the deployed pair filter's scores, already computed over the whole joint pool
    import torch
    from grail_metabolism.config import FilterConfig
    from grail_metabolism.workflows.factory import build_filter
    state = torch.load(ROOT / "artifacts/full5000_single/checkpoints/filter.pt",
                       map_location="cpu", weights_only=False)
    filt = build_filter(FilterConfig(**state["arch"]))
    filt.load_state_dict(state["state_dict"], strict=False)
    filt.eval()

    subs = sorted(set(truth) & set(graw) & set(mscored))
    print(f"population: {len(subs)} substrates", flush=True)

    per, t0 = [], time.time()
    for n, s in enumerate(subs, 1):
        refs = {key(y) for y in truth[s]} - {None}
        if not refs:
            continue
        pk, cand = key(s), {}
        for i, c in enumerate(graw[s]):
            k = key(c["smiles"])
            if k is None or k == pk or k in cand:
                continue
            cand[k] = {"smiles": c["smiles"], "g_rank": i, "m_rank": None,
                       "m_score": np.nan, "g_filter": float(c.get("filter") or 0.0)}
        for i, row in enumerate(mscored[s]):
            k = key(row[0])
            if k is None or k == pk:
                continue
            if k in cand:
                if cand[k]["m_rank"] is None:
                    cand[k].update(m_rank=i, m_score=float(row[1]))
            else:
                cand[k] = {"smiles": row[0], "g_rank": None, "m_rank": i,
                           "m_score": float(row[1]), "g_filter": np.nan}
        if not cand:
            continue
        keys = list(cand)
        with torch.no_grad():
            ps = filt.score_batch(s, [cand[k]["smiles"] for k in keys])
        sm = Chem.MolFromSmiles(s)
        for k, sc in zip(keys, list(ps)):
            c = cand[k]
            c["pair_score"] = float(sc)
            c["both"] = 1.0 if (c["g_rank"] is not None and c["m_rank"] is not None) else 0.0
            c["g_recip"] = 1.0 / (1.0 + c["g_rank"]) if c["g_rank"] is not None else 0.0
            c["m_recip"] = 1.0 / (1.0 + c["m_rank"]) if c["m_rank"] is not None else 0.0
            c.update(chemistry(sm, c["smiles"]))
            c["y"] = 1 if k in refs else 0
        per.append({"sub": s, "refs": refs, "cand": cand})
        if n % 50 == 0:
            print(f"  featurised {n}/{len(subs)} ({time.time() - t0:.0f}s)", flush=True)

    total = sum(len(r["cand"]) for r in per)
    pos = sum(c["y"] for r in per for c in r["cand"].values())
    print(f"\n  {total} candidates, {pos} of them references ({pos / total:.3%})", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier

    def run(feats: list[str], label: str):
        """Grouped five-fold: every candidate scored by a model that never saw its substrate."""
        scores = [None] * len(per)
        for fold in range(FOLDS):
            test = {i for i in range(len(per)) if i % FOLDS == fold}
            X, y = [], []
            for i, r in enumerate(per):
                if i in test:
                    continue
                for c in r["cand"].values():
                    X.append([c[f] for f in feats])
                    y.append(c["y"])
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                               random_state=SEED, class_weight="balanced")
            m.fit(np.array(X, dtype=float), np.array(y))
            for i in sorted(test):
                ks = list(per[i]["cand"])
                Xi = np.array([[per[i]["cand"][k][f] for f in feats] for k in ks], dtype=float)
                scores[i] = dict(zip(ks, m.predict_proba(Xi)[:, 1]))
        return label, scores

    ARMS = [(BORROWED, "borrowed only"),
            (BORROWED + TRANSFER, "borrowed + transferred filter"),
            (CHEMISTRY, "chemistry only"),
            (BORROWED + TRANSFER + CHEMISTRY, "all three groups")]

    U = np.array([len(r["refs"]) for r in per], dtype=float)
    hits = {}
    for feats, label in ARMS:
        _, sc = run(feats, label)
        H = {b: np.zeros(len(per)) for b in BUDGETS}
        for j, r in enumerate(per):
            ranked = sorted(r["cand"], key=lambda k: -(sc[j] or {}).get(k, 0.0))
            for b in BUDGETS:
                H[b][j] = len(r["refs"] & set(ranked[:b]))
        hits[label] = H
        print(f"  fitted: {label}", flush=True)

    # the two references this has to beat, recomputed here on the same rows
    for label, keyfn in (("MetaTox alone",
                          lambda c: (0, c["m_rank"]) if c["m_rank"] is not None else (1, 0)),
                         ("agreement then pair chemistry",
                          lambda c: (0 if c["both"] else 1, -c["pair_score"]))):
        H = {b: np.zeros(len(per)) for b in BUDGETS}
        for j, r in enumerate(per):
            ranked = sorted(r["cand"], key=lambda k: keyfn(r["cand"][k]))
            for b in BUDGETS:
                H[b][j] = len(r["refs"] & set(ranked[:b]))
        hits[label] = H
    O = {b: np.zeros(len(per)) for b in BUDGETS}
    for j, r in enumerate(per):
        present = sum(1 for k in r["cand"] if k in r["refs"])
        for b in BUDGETS:
            O[b][j] = min(present, b)
    hits["oracle"] = O

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per), (N_BOOT, len(per)))

    def rate(H):
        return round(float(H.sum() / U.sum()), 4)

    def paired(A, B):
        d = A - B
        bt = np.array([d[j].sum() / max(U[j].sum(), 1) for j in idx])
        return {"delta": round(float(d.sum() / U.sum()), 4),
                "ci95": [round(float(np.quantile(bt, .025)), 4),
                         round(float(np.quantile(bt, .975)), 4)]}

    ORDER = ["MetaTox alone", "agreement then pair chemistry", "borrowed only",
             "borrowed + transferred filter", "chemistry only", "all three groups", "oracle"]
    table = {k: {b: rate(hits[k][b]) for b in BUDGETS} for k in ORDER}
    print(f"\n  {'ranker':32}" + "".join(f"{('k=' + str(b)):>9}" for b in BUDGETS))
    for k in ORDER:
        print(f"  {k:32}" + "".join(f"{table[k][b]:>9}" for b in BUDGETS))

    gains = {k: {b: paired(hits[k][b], hits["MetaTox alone"][b]) for b in BUDGETS}
             for k in ORDER if k not in ("MetaTox alone", "oracle")}
    print(f"\n  against MetaTox alone, paired over substrates")
    for k, g in gains.items():
        print(f"  {k:32}" + "".join(f"{g[b]['delta']:>+9.4f}" for b in BUDGETS))
    best_prior = "agreement then pair chemistry"
    vs_prior = {b: paired(hits["all three groups"][b], hits[best_prior][b]) for b in BUDGETS}
    print(f"\n  all three groups against the best policy that needed no fitting")
    print(f"  {'':32}" + "".join(f"{vs_prior[b]['delta']:>+9.4f}" for b in BUDGETS))

    rep = {"config": {**_code_version(), "n_substrates": len(per), "candidates": total,
                      "references": int(U.sum()), "positive_rate": round(pos / total, 5),
                      "folds": FOLDS, "n_boot": N_BOOT, "seed": SEED,
                      "validation": "grouped cross-validation over substrates; MetaTox predictions "
                                    "exist for one population, so this is cross-validated and not "
                                    "held out, and it evidences orderable signal rather than a "
                                    "deployment result",
                      "feature_groups": {"borrowed": BORROWED, "transfer": TRANSFER,
                                         "chemistry": CHEMISTRY}},
           "recall": table, "gain_over_metatox": gains,
           "all_groups_over_best_unfitted": vs_prior}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    KEYS.write_text(json.dumps(cache))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

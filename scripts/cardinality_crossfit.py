#!/usr/bin/env python3
"""Can a per-substrate cut be predicted out of sample, or is the oracle unreachable?

`cardinality_oracle.py` measures the ceiling: knowing the F1-optimal cut per substrate is worth
48-76% relative on macro F1 for every method here. A ceiling chosen with the answer in hand is not
a target, so this asks the question that decides whether a cardinality head is worth training: how
much of that ceiling survives when the cut must be predicted from the substrate alone, out of
sample.

Three arms, all evaluated on held-out folds so none of them sees its own answer:

  constant   the best single k, chosen on the training folds -- the honest baseline a fixed
             output budget represents
  predicted  k-hat regressed on cheap substrate descriptors from the training folds
  oracle     k* per substrate, reported for reference as the unreachable bound

The descriptors are deliberately cheap: heavy-atom count, ring and aromatic-ring counts, rotatable
bonds, TPSA, logP, and counts of the functional groups metabolism most often acts on. A graph
encoder should beat them, so a failure here is not proof that the direction is dead, but a success
here is proof that it is alive, and the gap between constant and predicted is the quantity a head
would have to beat to justify itself.

Reports the same split of the headroom the oracle script found: how much comes from knowing the
true set size (a property of the substrate, learnable in principle) versus knowing where the hits
landed in the ranking (a property of the ranker, not visible to a substrate-only head).
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
MODE = "inchikey_tautomer"
KMAX, FOLDS, SEED = 30, 5, 0
OUT = ROOT / "results" / "cardinality_crossfit.json"


def descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        float(mol.GetNumHeavyAtoms()),
        float(rdMolDescriptors.CalcNumRings(mol)),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Crippen.MolLogP(mol)),
        float(Descriptors.NumHDonors(mol)),
        float(Descriptors.NumHAcceptors(mol)),
        float(sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ("C", "H"))),
        # sites the common phase-I/II transformations act on
        float(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H]")))),
        float(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")))),
        float(len(mol.GetSubstructMatches(Chem.MolFromSmarts("c1ccccc1")))),
    ]


def keyed(items, table):
    seen, out = set(), []
    for s in items:
        k = table.get(s)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def f1_at(pred_keys, real, k):
    tp = len(set(pred_keys[:k]) & real)
    n = min(k, len(pred_keys))
    return 2 * tp / (n + len(real)) if (n + len(real)) else 0.0


def main() -> int:
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    table = json.loads((ROOT / "results" / "key_tables" / f"{MODE}.json").read_text())
    grail = {r["sub"]: r["deployed_top15"]
             for r in json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]}
    methods = {
        "GRAIL": grail,
        "SyGMa": json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text()),
        "MetaPredictor": json.loads((ROOT / "artifacts" / "tier2_1170" / "metapredictor_preds.json").read_text()),
    }

    rep = {"mode": MODE, "k_max": KMAX, "folds": FOLDS, "seed": SEED, "methods": {}}
    for name, preds in methods.items():
        rows, feats = [], []
        for s in sorted(set(preds) & set(truth)):
            if not truth[s]:
                continue
            d = descriptors(s)
            if d is None:
                continue
            pk = keyed(preds.get(s, []), table)
            real = {k for k in (table.get(x) for x in truth[s]) if k}
            if not real:
                continue
            curve = [f1_at(pk, real, k) for k in range(1, KMAX + 1)]
            rows.append({"curve": curve, "kstar": int(np.argmax(curve)) + 1,
                         "n_ref": len(real), "full": f1_at(pk, real, len(pk))})
            feats.append(d)
        X = np.array(feats)
        curves = np.array([r["curve"] for r in rows])
        kstar = np.array([r["kstar"] for r in rows])
        nref = np.array([r["n_ref"] for r in rows])

        f_const, f_pred, f_card = [], [], []
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        for tr, te in kf.split(X):
            k_const = int(np.argmax(curves[tr].mean(axis=0))) + 1     # best constant, fit on train
            f_const += [curves[i, k_const - 1] for i in te]

            model = GradientBoostingRegressor(random_state=SEED).fit(X[tr], kstar[tr])
            khat = np.clip(np.rint(model.predict(X[te])).astype(int), 1, KMAX)
            f_pred += [curves[i, k - 1] for i, k in zip(te, khat)]

            card = GradientBoostingRegressor(random_state=SEED).fit(X[tr], nref[tr])
            chat = np.clip(np.rint(card.predict(X[te])).astype(int), 1, KMAX)
            f_card += [curves[i, k - 1] for i, k in zip(te, chat)]

        oracle = float(curves.max(axis=1).mean())
        deployed = st.mean([r["full"] for r in rows])
        entry = {
            "n": len(rows),
            "macro_f1": {
                "as_emitted": round(deployed, 4),
                "best_constant_oos": round(float(np.mean(f_const)), 4),
                "predicted_kstar_oos": round(float(np.mean(f_pred)), 4),
                "predicted_cardinality_oos": round(float(np.mean(f_card)), 4),
                "oracle_kstar": round(oracle, 4),
            },
        }
        c = entry["macro_f1"]["best_constant_oos"]
        entry["gain_over_constant"] = {
            "predicted_kstar": round(entry["macro_f1"]["predicted_kstar_oos"] - c, 4),
            "predicted_cardinality": round(entry["macro_f1"]["predicted_cardinality_oos"] - c, 4),
            "oracle": round(oracle - c, 4),
        }
        entry["oracle_share_reached"] = (
            round(entry["gain_over_constant"]["predicted_kstar"] / entry["gain_over_constant"]["oracle"], 3)
            if entry["gain_over_constant"]["oracle"] > 0 else None
        )
        rep["methods"][name] = entry

    OUT.write_text(json.dumps(rep, indent=1))
    print(f"criterion {MODE}, {FOLDS}-fold out of sample, k in 1..{KMAX}\n")
    print(f"{'method':15}{'emitted':>9}{'const':>9}{'pred k*':>9}{'pred |S|':>10}{'oracle':>9}{'reached':>9}")
    for name, m in rep["methods"].items():
        f = m["macro_f1"]
        share = m["oracle_share_reached"]
        print(f"{name:15}{f['as_emitted']:9.4f}{f['best_constant_oos']:9.4f}"
              f"{f['predicted_kstar_oos']:9.4f}{f['predicted_cardinality_oos']:10.4f}"
              f"{f['oracle_kstar']:9.4f}{(f'{share:.1%}' if share is not None else 'n/a'):>9}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

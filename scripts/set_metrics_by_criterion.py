#!/usr/bin/env python3
"""Precision, recall, F1 and Jaccard for every method under every matching criterion.

The paper has led with recall. The field also reports F1 and Jaccard, and a method that emits
eighty candidates to catch a few more references is rewarded by recall and punished by the others.
This computes all four on each method's deployed output set -- no top-k truncation, since
precision and Jaccard are set quantities -- macro-averaged over substrates, under all five
criteria, with a paired bootstrap over substrates.

Two populations: the 150-substrate shared subset where all five methods have predictions, and the
full clean test split where three do.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
N_BOOT, SEED = 10000, 0
OUT = ROOT / "results" / "set_metrics_by_criterion.json"


def keyset(items, table):
    return {k for k in (table.get(s) for s in items) if k}


def per_substrate(preds, truth, subs, table):
    P, R, F, J = (np.zeros(len(subs)) for _ in range(4))
    for i, s in enumerate(subs):
        pred, real = keyset(preds.get(s, []), table), keyset(truth[s], table)
        if not real:
            continue
        tp = len(pred & real)
        P[i] = tp / len(pred) if pred else 0.0
        R[i] = tp / len(real)
        F[i] = 2 * tp / (len(pred) + len(real)) if (pred or real) else 0.0
        J[i] = tp / len(pred | real) if (pred or real) else 0.0
    return {"precision": P, "recall": R, "f1": F, "jaccard": J}


def main() -> int:
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    grail_full = {r["sub"]: r["deployed_top15"]
                  for r in json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]}
    t2 = ROOT / "artifacts" / "tier2"
    pops = {
        "shared150": {
            "GRAIL": grail_full,
            "SyGMa": json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text()),
            "BioTransformer": json.loads((t2 / "biotransformer_preds.json").read_text()),
            "MetaTrans": json.loads((t2 / "metatrans_preds.json").read_text()),
            "MetaPredictor": json.loads((t2 / "metapredictor_preds.json").read_text()),
        },
        "full1170": {
            "GRAIL": grail_full,
            "SyGMa": json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text()),
            "MetaPredictor": json.loads((ROOT / "artifacts" / "tier2_1170" / "metapredictor_preds.json").read_text()),
        },
    }
    rep = {"n_boot": N_BOOT, "seed": SEED, "modes": MODES, "populations": {}}
    rng = np.random.default_rng(SEED)

    for pop, methods in pops.items():
        subs = sorted(set.intersection(*(set(m) for m in methods.values())) & set(truth))
        subs = [s for s in subs if truth[s]]
        print(f"\n=== {pop}: {len(subs)} substrates, {len(methods)} methods ===", flush=True)
        idx = [rng.integers(0, len(subs), len(subs)) for _ in range(N_BOOT)]
        IDX = np.array(idx)
        rep["populations"][pop] = {"n_substrates": len(subs), "methods": sorted(methods), "by_mode": {}}
        for mode in MODES:
            table = json.loads((ROOT / "results" / "key_tables" / f"{mode}.json").read_text())
            entry, vecs = {}, {}
            for name, preds in methods.items():
                v = vecs[name] = per_substrate(preds, truth, subs, table)
                entry[name] = {}
                for k, arr in v.items():
                    b = np.array([arr[i].mean() for i in idx])
                    entry[name][k] = {"point": round(float(arr.mean()), 4),
                                      "ci95": [round(float(np.quantile(b, .025)), 4),
                                               round(float(np.quantile(b, .975)), 4)]}
                entry[name]["mean_output"] = round(float(np.mean([len(keyset(preds.get(s, []), table)) for s in subs])), 2)
            rep["populations"][pop]["by_mode"][mode] = entry
            order = {k: sorted(methods, key=lambda m: -entry[m][k]["point"]) for k in ("recall", "f1", "jaccard", "precision")}
            rep["populations"][pop]["by_mode"][mode]["_ranking"] = order
            # Paired differences on the same resample indices: a marginal-CI overlap is not a test,
            # and adjacent ranks are exactly where the ordering claim needs one.
            names, diffs = sorted(methods), {}
            for a_i, a in enumerate(names):
                for b in names[a_i + 1:]:
                    for k in ("recall", "f1", "jaccard", "precision"):
                        da, db = vecs[a][k], vecs[b][k]
                        bt = (da - db)[IDX].mean(axis=1)
                        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
                        diffs[f"{a}-{b}|{k}"] = {"point": round(float(da.mean() - db.mean()), 4),
                                                 "ci95": [round(lo, 4), round(hi, 4)],
                                                 "excludes_zero": bool(lo > 0 or hi < 0)}
            rep["populations"][pop]["by_mode"][mode]["_paired_diffs"] = diffs
            if mode == "inchikey_tautomer":
                print(f"  {mode}:", flush=True)
                for m in order["f1"]:
                    e = entry[m]
                    print(f"    {m:16} P={e['precision']['point']:.3f} R={e['recall']['point']:.3f} "
                          f"F1={e['f1']['point']:.3f} J={e['jaccard']['point']:.3f} out={e['mean_output']:.1f}", flush=True)
                print(f"    order by recall: {order['recall']}", flush=True)
                print(f"    order by F1    : {order['f1']}", flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

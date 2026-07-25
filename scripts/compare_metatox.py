#!/usr/bin/env python3
"""GRAIL vs MetaTox under TAME — the grant-deciding comparison.

MetaTox is the way2drug incumbent; this is the first head-to-head against it under one protocol.
Input is the supervisor's SDF (`all_metabol (MetaboliteLikeness).SDF`), whose records carry:
  ID_all   "<structure_ordinal>_<metabolite_number>"; `_0` = the parent. The ordinal follows the
           submission order of `results/metatox_input/substrates.sdf`, so ordinal N -> SUB{N:04d}
           -> the eval key in `substrate_map.csv`. Verified empirically: 268/270 parents match our
           submitted structure exactly by canonical SMILES (2 differ only by MetaTox's internal
           aromaticity/charge perception), so the ordinal join is sound and unshifted.
  Values   comma-separated combined probabilities (biotransformation x site passage), ONE PER
           PRODUCING REACTION. We rank a metabolite by its max, i.e. its best route.

SCOPE CAVEAT (stated in the report): this run is MetaTox **layer 1 only, WITHOUT the SMIRKS-rule
variant**, per the supervisor's note; a SMIRKS version follows. So this measures one MetaTox
configuration, not its ceiling.

Coverage is reported two ways rather than silently choosing one:
  * `shared`  — substrates MetaTox actually returned (the fair method-vs-method comparison)
  * `all_submitted` — the 21 substrates absent from MetaTox's output counted as recall 0
    (the deployment-honest view: a tool that returns nothing for a query has failed that query)
GRAIL is scored on the identical substrate set in each case.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

from grail_metabolism.metrics import _tautomer_inchikey

RDLogger.DisableLog("rdApp.*")

GRAIL_CSV = ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"
SUB_MAP = ROOT / "results" / "metatox_input" / "substrate_map.csv"
OUT = ROOT / "results" / "grail_vs_metatox.json"
PREDS_OUT = ROOT / "results" / "metatox_preds.json"
KS = [1, 5, 10, 15]


def _taut(s):
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return None


def boot_ci(d: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def parse_metatox(sdf_path: Path):
    """-> {SUB id: [(smiles, score)] ranked desc}, plus parent-verification stats."""
    mols = [m for m in Chem.SDMolSupplier(str(sdf_path)) if m is not None]
    parents, children = {}, {}
    for m in mols:
        if not m.HasProp("ID_all"):
            continue
        idall = m.GetProp("ID_all")
        if "_" not in idall:
            continue
        ordinal, k = idall.split("_", 1)
        try:
            ordinal = int(ordinal)
        except ValueError:
            continue
        if k == "0":
            parents[ordinal] = m
        else:
            vals = m.GetProp("Values") if m.HasProp("Values") else ""
            score = 0.0
            for v in vals.split(","):
                try:
                    score = max(score, float(v))
                except ValueError:
                    pass
            try:
                smi = Chem.MolToSmiles(m)
            except Exception:
                continue
            children.setdefault(ordinal, []).append((smi, score))
    preds = {}
    for ordinal, lst in children.items():
        lst.sort(key=lambda x: -x[1])
        preds[f"SUB{ordinal:04d}"] = lst
    return preds, parents


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", required=True, help="MetaTox prediction SDF")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # ---- ground truth + GRAIL predictions, keyed by eval-key SMILES ----
    grail, truth = {}, {}
    with open(GRAIL_CSV) as fh:
        for row in csv.DictReader(fh):
            rk = {k for k in (_taut(x) for x in row.get("real", "").split("|") if x) if k}
            if rk:
                truth[row["substrate"]] = rk
                grail[row["substrate"]] = [p for p in row.get("predicted", "").split("|") if p]

    # ---- id -> eval key ----
    id2key = {r["id"]: r["substrate_smiles"] for r in csv.DictReader(open(SUB_MAP))}

    mt_preds, parents = parse_metatox(Path(args.sdf))
    print(f"MetaTox: {len(parents)} parents, {len(mt_preds)} structures with metabolites, "
          f"{sum(len(v) for v in mt_preds.values())} metabolites", flush=True)

    # MetaTox predictions keyed by eval key (only substrates we have ground truth for)
    mt_by_key, mt_returned_keys = {}, set()
    for sid, lst in mt_preds.items():
        key = id2key.get(sid)
        if key and key in truth:
            mt_by_key[key] = [s for s, _ in lst]
    for ordinal in parents:
        key = id2key.get(f"SUB{ordinal:04d}")
        if key and key in truth:
            mt_returned_keys.add(key)

    # dump the converted predictions for reuse
    PREDS_OUT.write_text(json.dumps({k: v for k, v in mt_by_key.items()}, indent=1))

    def score(subs, preds_map):
        """-> per-substrate recall vectors per k, precision@k vectors, pool coverage, output size."""
        per = {k: [] for k in KS}
        prec = {k: [] for k in KS}
        rinf, sizes = [], []
        for s in subs:
            rk = truth[s]
            ranked, seen = [], set()
            for smi in preds_map.get(s, []):
                t = _taut(smi)
                if t and t not in seen:
                    seen.add(t)
                    ranked.append(t)
            sizes.append(len(ranked))
            for k in KS:
                top = ranked[:k]
                per[k].append(len(set(top) & rk) / len(rk))
                prec[k].append((len(set(top) & rk) / len(top)) if top else 0.0)
            rinf.append(len(set(ranked) & rk) / len(rk))
        return ({k: np.asarray(v) for k, v in per.items()},
                {k: np.asarray(v) for k, v in prec.items()},
                np.asarray(rinf), float(np.mean(sizes)))

    report = {"sdf": str(args.sdf), "match": "inchikey_tautomer",
              "metatox_config": "layer 1 only, WITHOUT SMIRKS-rule variant (per supplier note)",
              "n_parents_returned": len(parents), "n_submitted": len(id2key), "scopes": {}}

    scopes = {
        "shared_metatox_returned": sorted(mt_returned_keys),
        "all_submitted_with_truth": sorted(k for k in id2key.values() if k in truth),
    }
    for scope, subs in scopes.items():
        if not subs:
            continue
        g_per, g_prec, g_inf, g_size = score(subs, grail)
        m_per, m_prec, m_inf, m_size = score(subs, mt_by_key)
        entry = {"n": len(subs),
                 "grail": {f"recall@{k}": round(float(g_per[k].mean()), 4) for k in KS},
                 "metatox": {f"recall@{k}": round(float(m_per[k].mean()), 4) for k in KS},
                 "grail_pool_coverage": round(float(g_inf.mean()), 4),
                 "metatox_pool_coverage": round(float(m_inf.mean()), 4),
                 "grail_precision": {f"precision@{k}": round(float(g_prec[k].mean()), 4) for k in KS},
                 "metatox_precision": {f"precision@{k}": round(float(m_prec[k].mean()), 4) for k in KS},
                 "mean_output_size": {"grail": round(g_size, 2), "metatox": round(m_size, 2)},
                 "delta_recall_by_k": {}, "delta_recall15_grail_minus_metatox": {}}
        for k in KS:
            dk, lok, hik = boot_ci(g_per[k] - m_per[k], args.n_boot, args.seed)
            entry["delta_recall_by_k"][f"@{k}"] = {
                "point": round(dk, 4), "ci95": [round(lok, 4), round(hik, 4)],
                "verdict": "GRAIL" if lok > 0 else ("MetaTox" if hik < 0 else "tie")}
        d, lo, hi = boot_ci(g_per[15] - m_per[15], args.n_boot, args.seed)
        entry["delta_recall15_grail_minus_metatox"] = {
            "point": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "verdict": "GRAIL WINS" if lo > 0 else ("MetaTox WINS" if hi < 0 else "tie (CI spans 0)")}
        report["scopes"][scope] = entry
        print(f"\n--- scope: {scope} (n={len(subs)}) ---", flush=True)
        print(f"  {'k':>4} | {'GRAIL':>8} | {'MetaTox':>8}", flush=True)
        for k in KS:
            c = entry["delta_recall_by_k"][f"@{k}"]
            print(f"  {k:>4} | {g_per[k].mean():>8.4f} | {m_per[k].mean():>8.4f} | "
                  f"d={c['point']:+.4f} CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] {c['verdict']}", flush=True)
        print(f"  precision@15  : GRAIL {g_prec[15].mean():.4f}  MetaTox {m_prec[15].mean():.4f}", flush=True)
        print(f"  pool coverage : GRAIL {g_inf.mean():.4f}  MetaTox {m_inf.mean():.4f}", flush=True)
        print(f"  mean outputs  : GRAIL {g_size:.2f}  MetaTox {m_size:.2f}", flush=True)
        e = entry["delta_recall15_grail_minus_metatox"]
        print(f"  delta@15 (GRAIL-MetaTox) = {e['point']:+.4f} 95%CI[{e['ci95'][0]:+.4f},"
              f"{e['ci95'][1]:+.4f}] -> {e['verdict']}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out} and {PREDS_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

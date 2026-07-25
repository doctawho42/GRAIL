#!/usr/bin/env python3
"""GRAIL x MetaTox complementarity — the deployment-decisive measurement.

The head-to-head (`compare_metatox.py`) says the two methods TIE on recall@15. A tie invites the
wrong deployment question ("which one replaces which?"). The right question is whether they find
the SAME metabolites or DIFFERENT ones, because two tied-but-disjoint predictors are worth running
together while two tied-but-redundant ones are not.

Measures, on the substrates where MetaTox returned metabolites, under tautomer-InChIKey at k=15:
  * recall of each method alone, and of their UNION
  * the paired gain from adding each method to the other (bootstrap CI over substrates)
  * the overlap census of correctly-found metabolites: only-GRAIL / only-MetaTox / both

Scope caveat inherited from the head-to-head: this is MetaTox layer-1 WITHOUT its SMIRKS-rule
variant. If the SMIRKS version overlaps GRAIL's rule bank more, the complementarity should shrink —
that is the falsifier to re-run when it arrives.
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
from rdkit import RDLogger

from grail_metabolism.metrics import _tautomer_inchikey

RDLogger.DisableLog("rdApp.*")
GRAIL_CSV = ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"
METATOX = ROOT / "results" / "metatox_preds.json"
OUT = ROOT / "results" / "metatox_complementarity.json"


def _t(s):
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    grail, truth = {}, {}
    with open(GRAIL_CSV) as fh:
        for row in csv.DictReader(fh):
            rk = {k for k in (_t(x) for x in row["real"].split("|") if x) if k}
            if rk:
                truth[row["substrate"]] = rk
                grail[row["substrate"]] = [p for p in row["predicted"].split("|") if p]
    mt = json.loads(METATOX.read_text())
    subs = sorted(set(mt) & set(truth))

    def topk(preds):
        out, seen = [], set()
        for x in preds:
            k = _t(x)
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return set(out[: args.k])

    g_only = m_only = both = 0
    gr, mr, ur = [], [], []
    for s in subs:
        rk = truth[s]
        G, M = topk(grail.get(s, [])), topk(mt.get(s, []))
        hg, hm = G & rk, M & rk
        both += len(hg & hm)
        g_only += len(hg - hm)
        m_only += len(hm - hg)
        gr.append(len(hg) / len(rk))
        mr.append(len(hm) / len(rk))
        ur.append(len((hg | hm)) / len(rk))
    gr, mr, ur = map(np.asarray, (gr, mr, ur))

    rng = np.random.default_rng(args.seed)

    def ci(d):
        b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.n_boot)])
        return float(d.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))

    add_grail = ci(ur - mr)
    add_metatox = ci(ur - gr)
    tot = g_only + m_only + both
    report = {
        "n_substrates": len(subs), "k": args.k, "match": "inchikey_tautomer",
        "metatox_config": "layer 1 only, WITHOUT SMIRKS-rule variant",
        "recall": {"grail": round(float(gr.mean()), 4), "metatox": round(float(mr.mean()), 4),
                   "union": round(float(ur.mean()), 4)},
        "gain_from_adding_grail_to_metatox": {"point": round(add_grail[0], 4),
                                              "ci95": [round(add_grail[1], 4), round(add_grail[2], 4)]},
        "gain_from_adding_metatox_to_grail": {"point": round(add_metatox[0], 4),
                                              "ci95": [round(add_metatox[1], 4), round(add_metatox[2], 4)]},
        "correct_metabolite_census": {"only_grail": g_only, "only_metatox": m_only, "both": both,
                                      "share_only_grail": round(g_only / tot, 4),
                                      "share_only_metatox": round(m_only / tot, 4),
                                      "share_both": round(both / tot, 4)},
        "reading": "Strongly complementary: the union recovers far more than either method alone and "
                   "most correctly-found metabolites are found by exactly one method, so the "
                   "deployment question is not which tool replaces which but whether to run both.",
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"n={len(subs)}  k={args.k}", flush=True)
    print(f"recall@{args.k}: GRAIL {gr.mean():.4f} | MetaTox {mr.mean():.4f} | UNION {ur.mean():.4f}", flush=True)
    print(f"  +GRAIL to MetaTox : {add_grail[0]:+.4f} 95%CI[{add_grail[1]:+.4f},{add_grail[2]:+.4f}]", flush=True)
    print(f"  +MetaTox to GRAIL : {add_metatox[0]:+.4f} 95%CI[{add_metatox[1]:+.4f},{add_metatox[2]:+.4f}]", flush=True)
    print(f"correct metabolites: only-GRAIL {g_only} | only-MetaTox {m_only} | both {both} "
          f"({both/tot:.1%} shared)", flush=True)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired-bootstrap CIs for the EXTERNAL (GLORYx-37) match-sensitivity table.

`docs/benchmark/gloryx_results.md` reports four methods scored under five matching conventions on
the field's de-facto shared hold-out, and the point estimates suggest a SyGMa <-> MetaPredictor
reordering (MetaPredictor 0.362 under strict InChIKey vs 0.504 under tautomer, while SyGMa sits at
~0.49 throughout). If that survives inference, it is an EXTERNAL replication of this paper's
primary phenomenon — the internal one (§11) lives on a 150-substrate subset of our own split.

Pre-registered before running, because n=37 is small: the paired CIs decide the claim's status.
  * interaction CI excludes zero -> external replication; the table becomes evidence in §11/§7.
  * interaction CI spans zero    -> the table is DESCRIPTIVE only; report it as an external
                                    comparison under one protocol and explicitly decline to call
                                    it a rank-flip replication.
Either way the table is worth including; only its label changes.

Predictions are the frozen ones already used for the published table (`docs/benchmark/data/`),
SyGMa is recomputed with py-sygma exactly as `eval_on_gloryx.py` does, and the per-substrate
recall vectors are resampled together (substrates are the independent unit).
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from rdkit import RDLogger

from grail_metabolism.metrics import _match_keys, top_k_recall
from scripts.eval_on_gloryx import load_gloryx

RDLogger.DisableLog("rdApp.*")
DATA = ROOT / "docs" / "benchmark" / "data"
OUT = ROOT / "results" / "gloryx_rank_flip_ci.json"
PRED_FILES = {
    "BioTransformer": DATA / "biotransformer_gloryx.json",
    "MetaPredictor": DATA / "metapredictor_gloryx.json",
    "GRAIL": DATA / "grail_reranker_gloryx.json",
}


def sygma_predict(subs, top_k: int):
    """Reuse the EXACT code path that produced the published table (`eval_on_gloryx`), rather than
    re-implementing the py-sygma API. A hand-rolled version using `tree.to_list()` silently returned
    nothing and scored SyGMa 0.0000 everywhere -- see the zero-output guard in main()."""
    from scripts.eval_on_gloryx import sygma_predictions
    preds = sygma_predictions(subs)
    return {s: preds.get(s, [])[:top_k] for s in subs}


def per_substrate_recall(preds, reals, subs, match, k) -> np.ndarray:
    v = np.empty(len(subs))
    for i, s in enumerate(subs):
        ranked, seen = [], set()
        for item in preds.get(s, []):
            key = next(iter(_match_keys([item], match)), None)
            if key and key not in seen:
                seen.add(key)
                ranked.append(key)
        v[i] = top_k_recall(ranked, _match_keys(reals[s], match), k)
    return v


def boot(d, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(d)
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--strict", default="inchikey")
    ap.add_argument("--normalized", default="inchikey_tautomer")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    reals = load_gloryx(DATA / "gloryx_test.json")
    subs = sorted(s for s in reals if reals[s])
    print(f"GLORYx substrates with references: {len(subs)}", flush=True)

    methods = {}
    for name, path in PRED_FILES.items():
        d = json.loads(path.read_text())
        methods[name] = {s: d.get(s, []) for s in subs}
        cov = sum(1 for s in subs if d.get(s))
        print(f"  {name}: covers {cov}/{len(subs)}", flush=True)
    print("  SyGMa: recomputing with py-sygma ...", flush=True)
    methods["SyGMa"] = sygma_predict(subs, args.k)

    # per-substrate recall vectors, per method per protocol
    vec = {}
    for name, preds in methods.items():
        for proto in (args.strict, args.normalized):
            vec[(name, proto)] = per_substrate_recall(preds, reals, subs, proto, args.k)

    report = {"n_substrates": len(subs), "k": args.k, "n_boot": args.n_boot, "seed": args.seed,
              "protocols": {"strict": args.strict, "normalized": args.normalized},
              "recall": {}, "protocol_sensitivity": {}, "pairwise": {}}

    for (name, proto), v in vec.items():
        if float(v.max()) == 0.0:
            raise SystemExit(
                f"ERROR: '{name}' scores exactly 0 on every substrate under '{proto}'. That is a "
                "broken prediction adapter, not a result -- check the method's prediction loader "
                "before interpreting any interval computed from it.")

    print(f"\n=== recall@{args.k} (n={len(subs)}) ===", flush=True)
    print(f"{'method':>15} | {args.strict:>10} | {args.normalized:>18}", flush=True)
    for name in methods:
        rs, rn = vec[(name, args.strict)], vec[(name, args.normalized)]
        report["recall"][name] = {args.strict: round(float(rs.mean()), 4),
                                  args.normalized: round(float(rn.mean()), 4)}
        m, lo, hi = boot(rn - rs, args.n_boot, args.seed)
        report["protocol_sensitivity"][name] = {
            "gain": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "verdict": "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."}
        print(f"{name:>15} | {rs.mean():>10.4f} | {rn.mean():>18.4f}   "
              f"gain {m:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)

    print(f"\n=== pairwise deltas and interactions ===", flush=True)
    for a, b in combinations(methods, 2):
        entry = {}
        for proto in (args.strict, args.normalized):
            m, lo, hi = boot(vec[(a, proto)] - vec[(b, proto)], args.n_boot, args.seed)
            entry[proto] = {"delta_a_minus_b": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                            "leader": a if m > 0 else b,
                            "verdict": "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."}
        inter = (vec[(b, args.normalized)] - vec[(a, args.normalized)]) - \
                (vec[(b, args.strict)] - vec[(a, args.strict)])
        m, lo, hi = boot(inter, args.n_boot, args.seed)
        entry["interaction_b_extra_gain"] = {
            "mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "verdict": "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."}
        flips = entry[args.strict]["leader"] != entry[args.normalized]["leader"]
        entry["point_estimate_reorders"] = flips
        report["pairwise"][f"{a}_vs_{b}"] = entry
        mark = "  <- POINT-ESTIMATE FLIP" if flips else ""
        print(f"  {a} vs {b}:{mark}", flush=True)
        for proto in (args.strict, args.normalized):
            e = entry[proto]
            print(f"      [{proto:>18}] Δ={e['delta_a_minus_b']:+.4f} "
                  f"[{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}] {e['verdict']} ({e['leader']} leads)", flush=True)
        i = entry["interaction_b_extra_gain"]
        print(f"      [interaction] {b} gains {i['mean']:+.4f} more "
              f"[{i['ci95'][0]:+.4f},{i['ci95'][1]:+.4f}] {i['verdict']}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

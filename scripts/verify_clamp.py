#!/usr/bin/env python3
"""Does the deployed pipeline reproduce the committed hit count, and what does the clamp hide?

results/recall_factorization.json cannot regenerate its own H from its own stored deployed output:
the paper's matcher run over the artifact's `deployed_top15` returns 689 hits where the artifact
stores 677, and in 11 of the 12 disagreeing substrates a predicted SMILES is byte-identical to a
reference, so those are missed under every one of the five matching criteria.

The cause is not the matcher. scripts/factorize_recall.py:241 clamps

    h_i = min(h_i, cbud_i)

to defend the nesting the decomposition needs, H <= Cbud <= Cfull. So a substrate whose deployed
output hits a reference that the generator POOL was not credited with has its H pushed to zero. The
deployed output is a subset of that pool by construction, so a pool credited with less than its own
output is the thing to explain, and the clamp hides it rather than fixing it.

This re-runs the deployed pipeline -- the same checkpoints, the same operating point, through
factorize_recall's own builders -- and records, per substrate, the unclamped H, the pool count, and
whether the nesting holds on its own. For the substrates where it does not, it asks the question the
clamp forecloses: is the reference actually absent from the pool, or is it present under a SMILES
whose tautomer key the pool pass wrote differently?
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from factorize_recall import (CANDIDATE_TOP_K, FILTER_CANDIDATE_CAP, K, build_dataset_config,
                              build_deployed_model, tautomer_hits)
from grail_metabolism.workflows.data import load_dataset_bundle
from grail_metabolism.metrics import _tautomer_inchikey as _tk


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


def _keys(smiles_iter):
    out = set()
    for s in smiles_iter:
        try:
            k = _tk(s)
        except Exception:
            k = None
        if k:
            out.add(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-test", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "clamp_verification.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    committed = json.loads((ROOT / "results/recall_factorization.json").read_text())
    stored = {r["sub"]: r for r in committed["per_substrate"]}

    model = build_deployed_model(copy_prior=True)
    bundle = load_dataset_bundle(build_dataset_config(args.max_test))
    items = list(bundle.test.map.items())
    print(f"test substrates: {len(items)}", flush=True)

    gen = model.generator
    threshold = getattr(gen, "calibrated_threshold", None)
    rows, t0 = [], time.perf_counter()
    for i, (sub, prods) in enumerate(items, 1):
        if i == 1 or i % 50 == 0 or i == len(items):
            print(f"  [torch] {i}/{len(items)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        ref = list(prods)
        if not ref:
            continue
        scored = gen.generate_scored(sub, top_k=CANDIDATE_TOP_K, threshold=threshold)
        pool = [s for s, _ in scored]
        top = model.generate(sub, top_k=CANDIDATE_TOP_K, threshold=threshold, max_output=K,
                             gate_by_filter=False, filter_candidate_cap=FILTER_CANDIDATE_CAP)
        h = tautomer_hits(top, ref)
        cbud = tautomer_hits(pool, ref)
        row = {"sub": sub, "H_unclamped": h, "Cbud": cbud, "pool": len(pool), "emitted": len(top)}
        if h > cbud:
            # the clamp would fire here; ask whether the output is genuinely outside the pool
            row["output_subset_of_pool"] = set(top).issubset(set(pool))
            row["hit_keys_in_pool_keys"] = bool(
                (_keys(top) & _keys(ref)) <= _keys(pool))
        rows.append(row)

    U = {s: r["U"] for s, r in stored.items()}
    tot_u = sum(U.get(r["sub"], 0) for r in rows)
    h_unclamped = sum(r["H_unclamped"] for r in rows)
    h_clamped = sum(min(r["H_unclamped"], r["Cbud"]) for r in rows)
    violations = [r for r in rows if r["H_unclamped"] > r["Cbud"]]

    rep = {"config": {**_code_version(), "n_substrates": len(rows), "threads": args.threads,
                      "operating_point": {"top_k": CANDIDATE_TOP_K, "max_output": K,
                                          "filter_candidate_cap": FILTER_CANDIDATE_CAP,
                                          "gate_by_filter": False},
                      "denominator": "U from results/recall_factorization.json per_substrate"},
           "sum_U": tot_u,
           "H_unclamped": h_unclamped, "H_clamped": h_clamped,
           "committed_H": sum(r["H"] for r in stored.values()),
           "micro_recall_unclamped": round(h_unclamped / max(tot_u, 1), 4),
           "micro_recall_clamped": round(h_clamped / max(tot_u, 1), 4),
           "committed_micro_recall": round(sum(r["H"] for r in stored.values())
                                           / max(sum(r["U"] for r in stored.values()), 1), 4),
           "nesting_violations": len(violations),
           "violations_where_output_is_a_subset_of_the_pool":
               sum(1 for r in violations if r.get("output_subset_of_pool")),
           "violation_detail": violations[:20]}
    print(f"\nsum U {tot_u}")
    print(f"  H unclamped {h_unclamped} -> micro recall {rep['micro_recall_unclamped']}")
    print(f"  H clamped   {h_clamped} -> micro recall {rep['micro_recall_clamped']}")
    print(f"  committed   {rep['committed_H']} -> micro recall {rep['committed_micro_recall']}")
    print(f"  substrates where H would exceed Cbud: {len(violations)}, "
          f"of which the emitted list is a subset of the pool in "
          f"{rep['violations_where_output_is_a_subset_of_the_pool']}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

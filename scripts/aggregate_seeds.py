#!/usr/bin/env python3
"""Aggregate Stage-2a reranker gate JSONs across seeds into mean±std recall@k.

Reads one or more ``reranker_gate_bi*.json`` files (each the output of
``run_reranker_gate.py``) and prints per-seed numbers plus the mean±std across
seeds for reranker / generator-alone / oracle recall@{5,10,12,15}. This is the
headline format the methodology asks for (mean±std over >=3 seeds).

Usage:
  # default: every test-split gate json under results/
  python scripts/aggregate_seeds.py
  # explicit files (any split):
  python scripts/aggregate_seeds.py results/reranker_gate_bi_test*.json
  # the val gate across seeds:
  python scripts/aggregate_seeds.py --glob 'results/reranker_gate_bi_seed*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KS = (5, 10, 12, 15)
SERIES = ("reranker", "generator", "oracle")


def _load(paths: list[str]) -> list[dict]:
    runs = []
    for p in sorted(paths):
        with open(p) as handle:
            data = json.load(handle)
        data["_path"] = p
        runs.append(data)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate reranker gate JSONs across seeds.")
    ap.add_argument("paths", nargs="*", help="Explicit gate JSON paths (overrides --glob).")
    ap.add_argument(
        "--glob", default="results/reranker_gate_bi_test*.json",
        help="Glob for gate JSONs when no explicit paths are given.",
    )
    args = ap.parse_args()

    paths = args.paths or glob.glob(str(ROOT / args.glob)) or glob.glob(args.glob)
    if not paths:
        sys.exit(f"no gate JSONs matched (paths={args.paths!r} glob={args.glob!r})")

    runs = _load(paths)
    # Guard: mixing eval splits in one aggregate is almost always a mistake.
    splits = {r.get("config", {}).get("eval_split", "val") for r in runs}
    if len(splits) > 1:
        print(f"WARNING: mixing eval splits {sorted(splits)} in one aggregate", flush=True)

    seeds = [r.get("seed") for r in runs]
    split = sorted(splits)[0] if len(splits) == 1 else "mixed"
    n_eval = {int(r["metrics"]["n_substrates"]) for r in runs}
    print(f"\n==== Stage-2a reranker: {len(runs)} run(s), split={split}, "
          f"seeds={seeds}, n_substrates={sorted(n_eval)} ====\n", flush=True)

    # Per-seed table.
    header = "seed | " + " | ".join(f"rr@{k}" for k in KS)
    print(header, flush=True)
    for r in runs:
        cells = " | ".join(f"{r['metrics'][f'reranker_recall@{k}']:.4f}" for k in KS)
        print(f"{str(r.get('seed')):>4} | {cells}", flush=True)

    # Mean±std across seeds, per series and k.
    print("\nmean±std across seeds (recall@k):", flush=True)
    print("series     | " + " | ".join(f"{'@'+str(k):>15}" for k in KS), flush=True)
    for series in SERIES:
        cells = []
        for k in KS:
            vals = [r["metrics"][f"{series}_recall@{k}"] for r in runs]
            mean = statistics.fmean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{mean:.4f}±{std:.4f}")
        print(f"{series:<10} | " + " | ".join(f"{c:>15}" for c in cells), flush=True)

    # Headline line.
    rr = [r["metrics"]["reranker_recall@15"] for r in runs]
    gg = [r["metrics"]["generator_recall@15"] for r in runs]
    rr_m, gg_m = statistics.fmean(rr), statistics.fmean(gg)
    rr_s = statistics.pstdev(rr) if len(rr) > 1 else 0.0
    lift = (rr_m - gg_m) / gg_m * 100 if gg_m else float("nan")
    print(f"\nHEADLINE recall@15 ({split}): reranker {rr_m:.4f}±{rr_s:.4f} vs "
          f"generator-alone {gg_m:.4f}  (+{lift:.1f}%)", flush=True)


if __name__ == "__main__":
    main()

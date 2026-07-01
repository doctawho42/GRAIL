#!/usr/bin/env python3
"""Aggregate Stage-2 result JSONs across seeds into mean±std recall@k.

Reads one or more results JSONs -- either ``reranker_gate_bi*.json`` (Stage-2a, from
``run_reranker_gate.py``: keys ``{reranker,generator,oracle}_recall@{5,10,12,15}``) or
``gflownet*.json`` (Stage-2b, from ``run_gflownet.py``: keys
``{gflownet,reranker,beam}_recall@{max_size}`` plus diversity scalars
``modes_discovered``, ``mean_pairwise_tanimoto``, ``n_unique_scaffolds``,
``set_size_calibration``) -- and prints per-seed numbers plus the mean±std across seeds.
This is the headline format the methodology asks for (mean±std over >=3 seeds).

The two JSON shapes differ (different series names, and the gflownet shape uses a single
k tied to ``--max-size`` rather than the fixed {5,10,12,15} of the gate). Rather than
hardcode either shape, this script auto-detects every ``{series}_recall@{k}`` key present
across the loaded runs (regex ``^(?P<series>.+)_recall@(?P<k>\\d+)$``) and aggregates
whatever it finds, so both shapes -- and any future one following the same naming
convention -- work unchanged.

Usage:
  # default: every test-split gate json under results/
  python scripts/aggregate_seeds.py
  # explicit files (any split, any shape):
  python scripts/aggregate_seeds.py results/reranker_gate_bi_test*.json
  python scripts/aggregate_seeds.py results/gflownet_test*.json
  # the val gate across seeds:
  python scripts/aggregate_seeds.py --glob 'results/reranker_gate_bi_seed*.json'
"""
from __future__ import annotations

import argparse
import glob
import re
import statistics
import sys
from pathlib import Path

try:
    import json
except ImportError:  # pragma: no cover
    raise

ROOT = Path(__file__).resolve().parents[1]

# Matches any "<series>_recall@<k>" metric key, e.g. "reranker_recall@15",
# "gflownet_recall@15", "generator_recall@5". Series names are free-form (may contain
# underscores themselves, e.g. a future "beam_search_recall@15" would parse as
# series="beam_search", k=15).
RECALL_KEY_RE = re.compile(r"^(?P<series>.+)_recall@(?P<k>\d+)$")

# Non-recall scalar metrics worth aggregating mean±std when present (currently only
# emitted by run_gflownet.py's diversity block, but any run's ``metrics`` dict may add
# more scalars later -- anything here that's missing from a run is simply skipped).
DIVERSITY_KEYS = (
    "modes_discovered",
    "mean_pairwise_tanimoto",
    "n_unique_scaffolds",
    "set_size_calibration",
)


def _load(paths: list[str]) -> list[dict]:
    runs = []
    for p in sorted(paths):
        with open(p) as handle:
            data = json.load(handle)
        data["_path"] = p
        runs.append(data)
    return runs


def _detect_series_k(runs: list[dict]) -> tuple[list[str], dict[str, list[int]], dict[str, set]]:
    """Scan every run's ``metrics`` dict for ``{series}_recall@{k}`` keys.

    Returns (series_in_first_seen_order, {series: sorted k's}, {series: set of k's per-run
    to detect disagreement}). Warns (via stdout) if runs disagree on which keys they carry.
    """
    series_ks: dict[str, set] = {}
    per_run_keysets: list[set] = []
    series_order: list[str] = []
    for r in runs:
        metrics = r.get("metrics", {})
        keyset = set()
        for key in metrics:
            m = RECALL_KEY_RE.match(key)
            if not m:
                continue
            series = m.group("series")
            k = int(m.group("k"))
            keyset.add((series, k))
            if series not in series_ks:
                series_ks[series] = set()
                series_order.append(series)
            series_ks[series].add(k)
        per_run_keysets.append(keyset)

    # Warn if runs disagree on which (series, k) keys are present.
    all_keysets = set()
    for ks in per_run_keysets:
        all_keysets |= ks
    for r, keyset in zip(runs, per_run_keysets):
        missing = all_keysets - keyset
        if missing:
            missing_str = ", ".join(f"{s}_recall@{k}" for s, k in sorted(missing))
            print(
                f"WARNING: {r.get('_path')} (seed={r.get('seed')}) is missing keys present "
                f"in other runs: {missing_str}",
                flush=True,
            )

    series_k_sorted = {s: sorted(ks) for s, ks in series_ks.items()}
    return series_order, series_k_sorted, series_ks


def _detect_diversity_keys(runs: list[dict]) -> list[str]:
    present = []
    for key in DIVERSITY_KEYS:
        if any(key in r.get("metrics", {}) for r in runs):
            present.append(key)
    return present


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Stage-2 result JSONs across seeds.")
    ap.add_argument("paths", nargs="*", help="Explicit result JSON paths (overrides --glob).")
    ap.add_argument(
        "--glob", default="results/reranker_gate_bi_test*.json",
        help="Glob for result JSONs when no explicit paths are given.",
    )
    args = ap.parse_args()

    paths = args.paths or glob.glob(str(ROOT / args.glob)) or glob.glob(args.glob)
    if not paths:
        sys.exit(f"no result JSONs matched (paths={args.paths!r} glob={args.glob!r})")

    runs = _load(paths)
    # Guard: mixing eval splits in one aggregate is almost always a mistake.
    splits = {r.get("config", {}).get("eval_split", "val") for r in runs}
    if len(splits) > 1:
        print(f"WARNING: mixing eval splits {sorted(splits)} in one aggregate", flush=True)

    seeds = [r.get("seed") for r in runs]
    split = sorted(splits)[0] if len(splits) == 1 else "mixed"
    n_eval = {int(r["metrics"]["n_substrates"]) for r in runs if "n_substrates" in r.get("metrics", {})}
    print(f"\n==== Stage-2 aggregate: {len(runs)} run(s), split={split}, "
          f"seeds={seeds}, n_substrates={sorted(n_eval)} ====\n", flush=True)

    series_order, series_ks, _ = _detect_series_k(runs)
    if not series_order:
        sys.exit("no '<series>_recall@<k>' keys found in any run's metrics -- nothing to aggregate")

    # All (series, k) pairs actually observed anywhere, in a stable order: series in
    # first-seen order, k ascending within each series.
    all_pairs: list[tuple[str, int]] = [(s, k) for s in series_order for k in series_ks[s]]

    # Per-seed table: one column per (series, k) pair actually present.
    header = "seed | " + " | ".join(f"{s}@{k}" for s, k in all_pairs)
    print(header, flush=True)
    for r in runs:
        metrics = r.get("metrics", {})
        cells = []
        for s, k in all_pairs:
            key = f"{s}_recall@{k}"
            val = metrics.get(key)
            cells.append(f"{val:.4f}" if val is not None else "   -  ")
        print(f"{str(r.get('seed')):>4} | " + " | ".join(cells), flush=True)

    # Mean±std across seeds, per series and k (skip a run for a given (series,k) if it
    # doesn't have that key rather than hard-erroring).
    print("\nmean±std across seeds (recall@k):", flush=True)
    # One row per series, one column per k that ANY series has (union), matching the
    # legacy layout when all series share the same k's (e.g. the gate JSON's {5,10,12,15}).
    all_ks = sorted({k for _, k in all_pairs})
    print("series     | " + " | ".join(f"{'@'+str(k):>15}" for k in all_ks), flush=True)
    for series in series_order:
        cells = []
        for k in all_ks:
            key = f"{series}_recall@{k}"
            vals = [r["metrics"][key] for r in runs if key in r.get("metrics", {})]
            if not vals:
                cells.append("-")
                continue
            mean = statistics.fmean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{mean:.4f}±{std:.4f}")
        print(f"{series:<10} | " + " | ".join(f"{c:>15}" for c in cells), flush=True)

    # Diversity scalars (gflownet shape only; skipped entirely if none present).
    diversity_keys = _detect_diversity_keys(runs)
    if diversity_keys:
        print("\nmean±std across seeds (diversity):", flush=True)
        for key in diversity_keys:
            vals = [r["metrics"][key] for r in runs if key in r.get("metrics", {})]
            mean = statistics.fmean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"{key:<25} | {mean:.4f}±{std:.4f}", flush=True)

    # Headline line: largest k present across all series, mean±std for every series at
    # that k. If both "reranker" and "generator" series exist there, keep the legacy
    # "vs generator-alone (+X%)" lift line; otherwise print each series' mean±std.
    headline_k = max(all_ks)
    headline_series = [s for s in series_order if headline_k in series_ks[s]]
    headline_vals = {}
    for s in headline_series:
        key = f"{s}_recall@{headline_k}"
        vals = [r["metrics"][key] for r in runs if key in r.get("metrics", {})]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        headline_vals[s] = (mean, std)

    if "reranker" in headline_vals and "generator" in headline_vals:
        rr_m, rr_s = headline_vals["reranker"]
        gg_m, _gg_s = headline_vals["generator"]
        lift = (rr_m - gg_m) / gg_m * 100 if gg_m else float("nan")
        print(f"\nHEADLINE recall@{headline_k} ({split}): reranker {rr_m:.4f}±{rr_s:.4f} vs "
              f"generator-alone {gg_m:.4f}  (+{lift:.1f}%)", flush=True)
        # Print any other series (e.g. oracle) at the headline k too.
        others = [s for s in headline_series if s not in ("reranker", "generator")]
        if others:
            extra = "  ".join(f"{s} {headline_vals[s][0]:.4f}±{headline_vals[s][1]:.4f}" for s in others)
            print(f"  (also: {extra})", flush=True)
    else:
        print(f"\nHEADLINE recall@{headline_k} ({split}):", flush=True)
        for s in headline_series:
            m, s_std = headline_vals[s]
            print(f"  {s:<10} {m:.4f}±{s_std:.4f}", flush=True)


if __name__ == "__main__":
    main()

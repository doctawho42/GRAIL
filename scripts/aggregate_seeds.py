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
hardcode either shape, this script auto-detects every ``{series}_recall@{k}``/``{series}_
union@{k}`` key present across the loaded runs (regex ``^(?P<series>.+)_(?:recall|union)@
(?P<k>\\d+)$``) and aggregates whatever it finds, so both shapes -- and any future one
following the same naming convention -- work unchanged.

Additionally (Phase 1 Plan 02, additive-only per D-EVAL05-JSONKEY): ``gflownet_union@{k}``/
``reranker_union@{k}`` co-primary keys (budget-matched union@K curve rows) are aggregated
through the SAME series/k path as ``_recall@{k}``, and ``circles@t0.4``/``circles@t0.7``/
``union_at_k_auc`` are added to ``DIVERSITY_KEYS``. The existing ``modes_discovered`` key
and ``_recall@{k}`` handling are left byte-stable -- this is purely additive so no
historical M2 aggregate output changes.

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

# Matches any "<series>_recall@<k>" OR "<series>_union@<k>" metric key, e.g.
# "reranker_recall@15", "gflownet_recall@15", "generator_recall@5",
# "gflownet_union@30", "reranker_union@50" (Phase 1 Plan 02's budget-matched union@K curve
# rows -- additive, aggregated through the SAME series/k path as the pre-existing
# "_recall@{k}" keys so the recall behavior stays byte-identical). Series names are
# free-form (may contain underscores themselves, e.g. a future "beam_search_recall@15"
# would parse as series="beam_search", k=15).
RECALL_KEY_RE = re.compile(r"^(?P<series>.+)_(?:recall|union)@(?P<k>\d+)$")

# Non-recall scalar metrics worth aggregating mean±std when present (currently only
# emitted by run_gflownet.py's diversity block, but any run's ``metrics`` dict may add
# more scalars later -- anything here that's missing from a run is simply skipped).
# circles@t0.4 / circles@t0.7 / union_at_k_auc are Phase 1 Plan 02's additive co-primary
# keys (D-EVAL03-CIRCLESKEYS / D-EVAL02-AUCNORM); modes_discovered and the original three
# stay untouched.
DIVERSITY_KEYS = (
    "modes_discovered",
    "mean_pairwise_tanimoto",
    "n_unique_scaffolds",
    "set_size_calibration",
    "circles@t0.4",
    "circles@t0.7",
    "union_at_k_auc",
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
    """Scan every run's ``metrics`` dict for ``{series}_recall@{k}`` (and, additively,
    ``{series}_union@{k}``) keys.

    A series carrying ``_union@`` keys is tracked under a distinct series label
    (``"{series}(union)"``) rather than folded into the plain ``_recall@`` series bucket
    of the same base name -- e.g. ``gflownet_recall@15`` and ``gflownet_union@30`` are two
    DIFFERENT series here, since they are different metric keys with different k-grids.
    This keeps every downstream key reconstruction (``f"{series}_recall@{k}"``) correct: a
    plain ``_recall@`` series is only ever reconstructed with ``_recall@``, and a ``_union@``
    series only ever with ``_union@``. Byte-stable for the pre-existing recall-only runs
    (which never produce a ``_union@`` key, so ``series_order``/``series_ks`` for them is
    identical to before this change).

    Returns (series_in_first_seen_order, {series: sorted k's}, {series: set of k's per-run
    to detect disagreement}). Warns (via stdout) if runs disagree on which keys they carry.
    """
    series_ks: dict[str, set] = {}
    series_suffix: dict[str, str] = {}
    per_run_keysets: list[set] = []
    series_order: list[str] = []
    for r in runs:
        metrics = r.get("metrics", {})
        keyset = set()
        for key in metrics:
            m = RECALL_KEY_RE.match(key)
            if not m:
                continue
            base_series = m.group("series")
            k = int(m.group("k"))
            suffix = "union" if key.endswith(f"_union@{k}") else "recall"
            series = base_series if suffix == "recall" else f"{base_series}(union)"
            keyset.add((series, k))
            if series not in series_ks:
                series_ks[series] = set()
                series_order.append(series)
                series_suffix[series] = suffix
            series_ks[series].add(k)
        per_run_keysets.append(keyset)

    # Warn if runs disagree on which (series, k) keys are present.
    all_keysets = set()
    for ks in per_run_keysets:
        all_keysets |= ks
    for r, keyset in zip(runs, per_run_keysets):
        missing = all_keysets - keyset
        if missing:
            missing_str = ", ".join(
                f"{s}_{series_suffix.get(s, 'recall')}@{k}" for s, k in sorted(missing)
            )
            print(
                f"WARNING: {r.get('_path')} (seed={r.get('seed')}) is missing keys present "
                f"in other runs: {missing_str}",
                flush=True,
            )

    series_k_sorted = {s: sorted(ks) for s, ks in series_ks.items()}
    return series_order, series_k_sorted, series_ks


def _metric_key(series: str, k: int) -> str:
    """Reconstruct the actual metrics-dict key for a (possibly ``(union)``-suffixed)
    series label produced by ``_detect_series_k``. ``"gflownet"`` -> ``gflownet_recall@{k}``
    (byte-stable, pre-existing behavior); ``"gflownet(union)"`` -> ``gflownet_union@{k}``."""
    if series.endswith("(union)"):
        return f"{series[: -len('(union)')]}_union@{k}"
    return f"{series}_recall@{k}"


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
            key = _metric_key(s, k)
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
            key = _metric_key(series, k)
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
    #
    # FIX A: headline_k must be selected over the PLAIN (non-"(union)") recall series
    # only. all_ks pools BOTH the plain "_recall@{k}" family (k tied to --max-size, e.g.
    # 10-15) and the "_union@{k}" family (k up to the K-grid max, e.g. 50) -- taking
    # max(all_ks) over that mixed pool resolves to the union grid's max and silently
    # drops the primary matched-budget "gflownet_recall@{max_size}"/"reranker_recall@
    # {max_size}"/"beam_recall@{max_size}" rows from the printed HEADLINE. Restricting
    # headline_k to the plain-recall series' own k's keeps that primary row visible, and
    # a SEPARATE, clearly-labeled union headline is printed alongside it so neither
    # family is silently dropped or conflated with the other.
    recall_series_ks = {s: ks for s, ks in series_ks.items() if not s.endswith("(union)")}
    union_series_ks = {s: ks for s, ks in series_ks.items() if s.endswith("(union)")}

    def _headline_block(label: str, series_ks_map: dict, k: int) -> None:
        headline_series = [s for s in series_order if s in series_ks_map and k in series_ks_map[s]]
        headline_vals = {}
        for s in headline_series:
            key = _metric_key(s, k)
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
            print(f"\n{label}{k} ({split}): reranker {rr_m:.4f}±{rr_s:.4f} vs "
                  f"generator-alone {gg_m:.4f}  (+{lift:.1f}%)", flush=True)
            # Print any other series (e.g. oracle) at the headline k too.
            others = [s for s in headline_series if s not in ("reranker", "generator")]
            if others:
                extra = "  ".join(f"{s} {headline_vals[s][0]:.4f}±{headline_vals[s][1]:.4f}" for s in others)
                print(f"  (also: {extra})", flush=True)
        else:
            print(f"\n{label}{k} ({split}):", flush=True)
            for s in headline_series:
                m, s_std = headline_vals[s]
                print(f"  {s:<10} {m:.4f}±{s_std:.4f}", flush=True)

    if recall_series_ks:
        headline_k = max(k for ks in recall_series_ks.values() for k in ks)
        _headline_block("HEADLINE recall@", recall_series_ks, headline_k)
    else:
        print("\n(no plain recall@k series found -- skipping HEADLINE recall@k block)", flush=True)

    if union_series_ks:
        union_k_max = max(k for ks in union_series_ks.values() for k in ks)
        _headline_block("HEADLINE union@", union_series_ks, union_k_max)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""What the merged-graph filter costs, measured on the featurisation rather than remembered.

The architecture appendix said the merged-graph variant makes full-data training take about
fourteen hours against forty minutes, roughly twentyfold. Those wall-clock figures came from a run
whose outputs were purged, so nothing committed carries them, and a wall clock is the wrong unit
anyway: it mixes the cost being claimed with the machine it was measured on and with everything
else that ran that day.

What is claimed is a property of the representation. The merged variant encodes a substrate and a
product as one graph with cross-edges laid down by a maximum common substructure, so it pays for an
\textsc{mcs} on every pair; the independent variant encodes the two molecules separately and pays
for neither. That difference is measurable in a minute, on the molecules the paper actually scores,
and it recurs at every candidate at inference --- which is where it decides deployment.

So this times both featurisations on the same pairs, from the released test predictions, and
reports the ratio with its spread. Nothing is trained and no checkpoint is loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


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


def pairs(limit: int) -> list:
    """Substrate--product pairs from the released test predictions, in file order."""
    src = ROOT / "artifacts/full5000_single/predictions/test_predictions.csv"
    out = []
    with open(src) as fh:
        for row in csv.DictReader(fh):
            sub = row.get("substrate")
            # the prediction column is a pipe-separated candidate list, which is exactly the
            # population the filter is asked about at inference
            for prod in (row.get("predicted") or "").split("|"):
                if sub and prod:
                    out.append((sub, prod))
                if len(out) >= limit:
                    return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "pair_cost_benchmark.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from grail_metabolism.utils.transform import from_pair, from_rdmol

    todo = []
    for s, p in pairs(args.n):
        a, b = Chem.MolFromSmiles(s), Chem.MolFromSmiles(p)
        if a is not None and b is not None:
            todo.append((a, b))
    if not todo:
        print("no pairs could be parsed", file=sys.stderr)
        return 1

    merged, independent = [], []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for a, b in todo:
            from_pair(a, b)
        merged.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for a, b in todo:
            from_rdmol(a)
            from_rdmol(b)
        independent.append(time.perf_counter() - t0)

    m, i = min(merged), min(independent)
    rep = {"config": {**_code_version(), "n_pairs": len(todo), "repeats": args.repeats,
                      "source": "artifacts/full5000_single/predictions/test_predictions.csv",
                      "note": "the fastest of the repeats is reported for each arm, which is the "
                              "usual reading of a microbenchmark; the ratio is what the appendix "
                              "quotes and the per-pair costs are given so it can be recomputed"},
           "merged_graph_seconds_per_pair": round(m / len(todo), 6),
           "independent_seconds_per_pair": round(i / len(todo), 6),
           "ratio": round(m / i, 2),
           "ratio_across_repeats": [round(x / y, 2) for x, y in zip(merged, independent)],
           "merged_seconds": [round(x, 3) for x in merged],
           "independent_seconds": [round(x, 3) for x in independent],
           "spread_of_the_ratio": round(
               statistics.pstdev([x / y for x, y in zip(merged, independent)]), 3)}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"  {len(todo)} pairs, {args.repeats} repeats")
    print(f"  merged graph      {rep['merged_graph_seconds_per_pair'] * 1000:.3f} ms per pair")
    print(f"  two independent   {rep['independent_seconds_per_pair'] * 1000:.3f} ms per pair")
    print(f"  ratio             {rep['ratio']}x  (across repeats {rep['ratio_across_repeats']})")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

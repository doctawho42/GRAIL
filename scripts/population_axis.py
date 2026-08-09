#!/usr/bin/env python3
"""Does the population reorder a leaderboard, with the method, criterion and budget held fixed?

Three of the four undeclared choices this paper names are varied with everything else fixed. The
fourth -- which population a familiar dataset name refers to -- is demonstrated rather than varied:
eleven published retrosynthesis files turn out to sit on three test sets, which shows the hazard
exists but is a fact about someone else's release rather than a measurement of ours.

It is measurable here, because the same three methods are scored under the same five criteria, by
the same estimator, on two populations of this paper's own split: the 150 substrates all five
methods share and the full 1,170. Nothing varies between them except which substrates are in.

Reported per method pair and criterion:

    gap on each population, the change between them, and whether the ordering survives

The comparison is deliberately narrow. Two populations is a weak design for a general claim, and it
is not offered as one: it says whether the axis moves the answer at all on a case where everything
else is pinned, which is more than the three-splits finding can say and less than the criterion and
budget results say.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "results" / "set_metrics_by_criterion.json"
METRICS = ("recall", "f1", "precision", "jaccard")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "population_axis.json"))
    args = ap.parse_args()

    d = json.loads(SRC.read_text())["populations"]
    pops = sorted(d, key=lambda p: d[p]["n_substrates"])
    if len(pops) < 2:
        raise SystemExit("two populations are needed and the artifact holds fewer")
    small, big = pops[0], pops[-1]
    methods = sorted(set(d[small]["methods"]) & set(d[big]["methods"]))
    modes = sorted(set(d[small]["by_mode"]) & set(d[big]["by_mode"]))
    print(f"{small} (n={d[small]['n_substrates']}) against {big} (n={d[big]['n_substrates']})")
    print(f"  methods on both: {methods}\n  criteria on both: {modes}", flush=True)

    rows, flips, ties = [], 0, 0
    for metric in METRICS:
        for mode in modes:
            for a, b in itertools.combinations(methods, 2):
                def g(pop):
                    A = d[pop]["by_mode"][mode][a][metric]["point"]
                    B = d[pop]["by_mode"][mode][b][metric]["point"]
                    return A - B
                gs, gb = g(small), g(big)
                if gs == 0 or gb == 0:
                    ties += 1
                    continue
                flip = (gs > 0) != (gb > 0)
                flips += flip
                rows.append({"metric": metric, "criterion": mode, "pair": f"{a} vs {b}",
                             f"gap_{small}": round(gs, 4), f"gap_{big}": round(gb, 4),
                             "change": round(abs(gb - gs), 4), "reordered": bool(flip)})

    print(f"\n  {len(rows)} comparisons, method and criterion and budget all held fixed")
    print(f"  the ordering changes with the population in {flips} of them")
    for r in rows:
        if r["reordered"]:
            print(f"    {r['metric']:9} {r['criterion']:18} {r['pair']:28} "
                  f"{r[f'gap_{small}']:+.4f} -> {r[f'gap_{big}']:+.4f}")

    med = sorted(r["change"] for r in rows)
    print(f"\n  median change in a pair's gap between the two populations: "
          f"{med[len(med) // 2]:.4f}")
    by_metric = {m: {"comparisons": sum(1 for r in rows if r["metric"] == m),
                     "reordered": sum(1 for r in rows if r["metric"] == m and r["reordered"])}
                 for m in METRICS}
    for m, v in by_metric.items():
        print(f"    {m:10} {v['reordered']} of {v['comparisons']} reorder")

    rep = {"config": {**_code_version(), "source": str(SRC.relative_to(ROOT)),
                      "populations": {p: d[p]["n_substrates"] for p in (small, big)},
                      "methods": methods, "criteria": modes, "metrics": list(METRICS),
                      "note": "the same methods, criteria and estimator on two populations of one "
                              "split; two populations is a weak design and is not offered as a "
                              "general claim"},
           "comparisons": len(rows), "reordered": flips, "ties_skipped": ties,
           "median_gap_change": round(med[len(med) // 2], 4),
           "by_metric": by_metric, "rows": rows}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

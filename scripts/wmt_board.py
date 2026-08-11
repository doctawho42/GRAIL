#!/usr/bin/env python3
r"""A leaderboard from outside chemistry, whose scoring convention ships as a function.

Machine translation is the domain that most resembles the ones audited here and shares none of their
data: the ranking is human, the items are sentences, and nothing about a molecule is involved. WMT's
general translation task publishes, for each system, an expert annotation of every error in every
segment --- its category and its severity --- and then turns those annotations into a number. The
turning is the convention. It is a Python function in the shared task's own repository
(``humeval/tools.py:mqm_weights``), and it makes four decisions that a paper quoting "MQM score" does
not repeat: an error in the source text is charged nothing, a creative reinterpretation is charged
nothing, a minor punctuation error is charged a tenth of an ordinary minor, and a non-translation is
charged five times a major. Each is defensible; none is the only defensible choice; and the sum over
a segment is what the ranking is built from.

So the criterion axis here is which annotations are charged, and it is a choice about membership
rather than about the metric, exactly as a structure-matching rule is: the metric stays the mean of
the per-segment sums in every cell. The second axis is whether that sum is floored. An MQM segment
score is unbounded below, so one catastrophic output can move a system's mean by more than a hundred
ordinary errors; implementations differ on whether to clip at the non-translation weight, and the
choice is never in a results table.

The published cell is the shared task's own function with no floor. It is checked against the
official ranking rather than assumed to reproduce it: if the order in that cell is not the order the
task published, the board is measuring something else and the run says so.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from robust_order import analyse  # noqa: E402

CANARY = "canary"
FLOOR = -25.0


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


def w_wmt(category: str, severity: str) -> float:
    """The shared task's own function, transcribed from humeval/tools.py:mqm_weights."""
    if severity in ("No-error", "neutral") or "Reinterpretation" in category:
        return 0.0
    if category == "Non-translation!":
        return -25.0
    if category == "Source issue":
        return 0.0
    if severity == "minor":
        return -0.1 if "Fluency/Punctuation" in category else -1.0
    if severity in ("major", "critical"):
        return -5.0
    raise ValueError(f"unweighted annotation: {category!r} {severity!r}")


def w_plain(category: str, severity: str) -> float:
    """Canonical MQM with no category exceptions: every error is charged by its severity alone."""
    if severity in ("No-error", "neutral"):
        return 0.0
    if severity == "minor":
        return -1.0
    if severity in ("major", "critical"):
        return -5.0
    raise ValueError(f"unweighted annotation: {category!r} {severity!r}")


def w_major(category: str, severity: str) -> float:
    """Only errors a reader would notice: the serious-error rate some comparisons report."""
    return -1.0 if severity in ("major", "critical") else 0.0


def w_count(category: str, severity: str) -> float:
    """Every annotated error charged alike, which is what an error count is."""
    if severity in ("No-error", "neutral"):
        return 0.0
    return -1.0


CRITERIA = {
    "wmt": ("the shared task's own weighting", w_wmt),
    "plain": ("severity alone, no category exceptions", w_plain),
    "major-only": ("major and critical errors only", w_major),
    "count": ("every annotated error charged alike", w_count),
}
FLOORS = ("none", "clipped")
PUBLISHED_CELL = ("wmt", "none")


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype=str, sep="\t", quoting=csv.QUOTE_NONE, quotechar="")
    d = d[d["doc"] != CANARY].copy()
    d["severity"] = d["severity"].fillna("No-error")
    d["category"] = d["category"].fillna("No-error")
    return d


def build_hits(d: pd.DataFrame) -> tuple[dict, list[str], list[str], list]:
    systems = sorted(d["system"].unique())
    items = sorted(d["globalSegId"].unique(), key=int)
    cells = [(c, f) for c in CRITERIA for f in FLOORS]

    hits = {}
    for criterion, (_, weight) in CRITERIA.items():
        charged = d.assign(w=[weight(c, s) for c, s in zip(d["category"], d["severity"])])
        per_seg = charged.groupby(["system", "globalSegId"])["w"].sum().unstack(fill_value=0.0)
        per_seg = per_seg.reindex(index=systems, columns=items, fill_value=0.0)
        for floor in FLOORS:
            v = per_seg.to_numpy(dtype=float)
            if floor == "clipped":
                v = np.maximum(v, FLOOR)
            for i, name in enumerate(systems):
                hits[(name, (criterion, floor))] = v[i]
    return hits, systems, items, cells


def official_order(d: pd.DataFrame) -> list[str]:
    """The task's published ranking, recomputed from its own function, to check the board against."""
    charged = d.assign(w=[w_wmt(c, s) for c, s in zip(d["category"], d["severity"])])
    per_seg = charged.groupby(["system", "globalSegId"])["w"].sum()
    return list(per_seg.groupby("system").mean().sort_values(ascending=False).index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqm", default=str(ROOT / "data/external/wmt24/humeval/"
                                                "mqm_generalMT2024_ende.tsv"))
    ap.add_argument("--lp", default="en-de")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ROOT / "results" / f"robust_order_wmt24_{args.lp}.json")

    raw = Path(args.mqm).read_bytes()
    d = load(Path(args.mqm))
    hits, systems, items, cells = build_hits(d)

    sub_grids = {"criteria only, unfloored": [(c, PUBLISHED_CELL[1]) for c in CRITERIA],
                 "the floor only, at the published criterion":
                     [(PUBLISHED_CELL[0], f) for f in FLOORS],
                 "the product": cells}
    rep = analyse(hits, systems, cells, PUBLISHED_CELL, sub_grids)

    official = official_order(d)
    agrees = official == rep["published_order"]
    rep["config"] = {
        **_code_version(), "n_items": len(items), "language_pair": args.lp,
        "source": f"WMT24 general MT, {Path(args.mqm).name}, from "
                  f"github.com/wmt-conference/wmt24-news-systems",
        "source_sha256": hashlib.sha256(raw).hexdigest(), "source_bytes": len(raw),
        "criteria": {k: v[0] for k, v in CRITERIA.items()},
        "second_axis": f"whether the per-segment sum is floored at {FLOOR}, the weight the "
                       f"published function gives a non-translation",
        "published_cell_reproduces_the_official_ranking": agrees,
        "official_ranking": official,
        "declared_conventions": [
            "quality-control rows (doc == 'canary') are dropped; they are not translations",
            "one rater scores each system-segment, so a rater subset would change the items and "
            "is not a cell",
            "a segment with no annotation for a system scores zero, which is what no error means"],
    }
    Path(out).write_text(json.dumps(rep, indent=1))

    print(f"wmt24 {args.lp}: {rep['n_systems']} systems, {rep['n_cells']} cells, "
          f"{rep['n_pairs']} pairs, {len(items)} segments")
    print(f"  published cell reproduces the official ranking: {agrees}")
    if not agrees:
        for a, b in itertools.zip_longest(official, rep["published_order"]):
            if a != b:
                print(f"     official {a} against ours {b}")
    print(f"  survive every cell:            {rep['n_dominating']}/{rep['n_pairs']} "
          f"= {rep['robustness']} {rep['robustness_ci95']}")
    print(f"  reversed with an interval:     {rep['n_contested']}, "
          f"after correction {rep['n_contested_after_correction']}")
    print(f"  unresolved, neither way:       {rep['n_unresolved']}")
    print(f"  tiers: {rep['tiers_distinguished']} {rep['tiers_ci95']} of {rep['n_systems']}")
    print(f"  distinct orderings across the grid: {rep['distinct_orderings_across_the_grid']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

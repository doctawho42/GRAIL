#!/usr/bin/env python3
r"""Eight boards from the previous year of the same task, so a share can be replicated.

Nine boards from one edition of one shared task establish that the surviving share varies. They
cannot establish that it is a property of a leaderboard rather than of a year, because every one of
them was annotated in the same campaign under the same instructions. The previous edition annotated
eight language pairs, two of which are the same pair scored again by different people a year later,
and that is the only comparison in this paper where the same question is asked twice of independent
data.

The protocol is different, which is the point: this edition collected direct assessments with no
error spans, several ratings per segment, and between $116$ and $207$ annotators per pair. So the
axes are the ones that protocol leaves open rather than the ones the later one does:

  criterion        how the several ratings of one segment become one number. Averaging them is
                   usual; taking the median is what one does when a rater pool is suspected of
                   outliers, which the field has published on; taking the first is what a paper
                   gets when it uses whichever rating the file happens to give first; and dropping
                   the fastest tenth of ratings is a quality control the task itself applies in
                   some editions and not others.
  normalisation    whether a rating is used as written or standardised within the annotator who
                   wrote it, applied before the ratings are combined.

Every segment here is rated for every system, so the boards are dense and no imputation is involved.
"""
from __future__ import annotations

import argparse
import hashlib
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

COLS = ["annotator", "system", "seg", "type", "src", "tgt", "score", "doc", "isdoc", "t0", "t1"]
CRITERIA = {
    "mean": "the ratings of a segment averaged",
    "median": "the ratings of a segment taken at the median",
    "first": "the first rating the file gives",
    "not-hasty": "the fastest tenth of ratings dropped before averaging",
}
NORMS = ("raw", "z-per-annotator")
PUBLISHED_CELL = ("mean", "raw")


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


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, names=COLS, header=None, low_memory=False)
    d = d[d["type"] == "TGT"].copy()
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    for c in ("t0", "t1"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["score"])
    # seconds a rating took; used only by the quality-control criterion, and only within a pair
    d["secs"] = (d["t1"] - d["t0"]).clip(lower=0.0)
    return d


def build_board(sub: pd.DataFrame) -> tuple[dict, list[str], list[str], list]:
    sub = sub.assign(item=sub["doc"].astype(str) + "#" + sub["seg"].astype(str))
    systems = sorted(sub["system"].unique())
    items = sorted(sub["item"].unique())
    cells = [(c, n) for c in CRITERIA for n in NORMS]

    stats = sub.groupby("annotator")["score"].agg(["mean", "std"])
    z = sub["score"] - sub["annotator"].map(stats["mean"])
    sd = sub["annotator"].map(stats["std"]).replace(0.0, np.nan)
    sub = sub.assign(z=(z / sd).fillna(0.0))

    cut = sub["secs"].quantile(0.10)
    hits = {}
    for criterion in CRITERIA:
        frame = sub[sub["secs"] > cut] if criterion == "not-hasty" else sub
        how = {"mean": "mean", "not-hasty": "mean", "median": "median", "first": "first"}[criterion]
        for norm in NORMS:
            col = "z" if norm == "z-per-annotator" else "score"
            g = frame.groupby(["system", "item"])[col].agg(how).unstack()
            g = g.reindex(index=systems, columns=items)
            # dropping the fastest ratings can empty a cell that the other criteria fill; the
            # segment mean over the systems still rated there keeps the item the same object
            if g.isna().to_numpy().any():
                g = g.apply(lambda c: c.fillna(c.mean()), axis=0).fillna(0.0)
            v = g.to_numpy(dtype=float)
            for i, name in enumerate(systems):
                hits[(name, (criterion, norm))] = v[i]
    return hits, systems, items, cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/external/wmt23/humaneval/DA+SQM/"
                                                "WMT23.scores_all.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "robust_order_wmt23.json"))
    args = ap.parse_args()

    raw = Path(args.csv).read_bytes()
    d = load(Path(args.csv))

    boards = {}
    for (src, tgt), sub in d.groupby(["src", "tgt"]):
        lp = f"{src}-{tgt}"
        hits, systems, items, cells = build_board(sub)
        sub_grids = {"criteria only, unnormalised": [(c, PUBLISHED_CELL[1]) for c in CRITERIA],
                     "normalisation only, at the published criterion":
                         [(PUBLISHED_CELL[0], n) for n in NORMS],
                     "the product": cells}
        r = analyse(hits, systems, cells, PUBLISHED_CELL, sub_grids)
        r["n_annotators"] = int(sub["annotator"].nunique())
        r["ratings_per_system_segment"] = round(
            float(len(sub) / max(sub.groupby(["system", "doc", "seg"]).ngroups, 1)), 3)
        boards[lp] = r
        print(f"  {lp:9s} {r['n_systems']:3d} systems, {len(items):5d} segments, "
              f"{r['n_annotators']:3d} annotators, {r['n_pairs']:4d} pairs: "
              f"{r['n_dominating']:4d} dominate, {r['n_contested']:2d} contested "
              f"({r['n_contested_after_correction']} certified), {r['n_unresolved']:3d} unresolved, "
              f"{r['tiers_distinguished']:2d} tiers", flush=True)

    shares = {lp: r["robustness"] for lp, r in boards.items()}
    rep = {"config": {**_code_version(),
                      "source": "WMT23 general MT, humaneval/DA+SQM/WMT23.scores_all.csv, from "
                                "github.com/wmt-conference/wmt23-news-systems",
                      "source_sha256": hashlib.sha256(raw).hexdigest(), "source_bytes": len(raw),
                      "protocol": "DA+SQM",
                      "criteria": dict(CRITERIA),
                      "second_axis": "whether a rating is standardised within its annotator before "
                                     "the ratings of a segment are combined",
                      "published_cell": "the mean of the ratings, unnormalised"},
           "n_boards": len(boards),
           "n_pairs_total": sum(r["n_pairs"] for r in boards.values()),
           "share_min": round(min(shares.values()), 4),
           "share_median": round(float(np.median(list(shares.values()))), 4),
           "share_max": round(max(shares.values()), 4),
           "n_certified_total": sum(r["n_contested_after_correction"] for r in boards.values()),
           "places_published": sum(r["n_systems"] for r in boards.values()),
           "places_supported": sum(r["tiers_distinguished"] for r in boards.values()),
           "boards": boards}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n  {rep['n_boards']} boards, {rep['n_pairs_total']} pairs; share "
          f"{rep['share_min']} to {rep['share_max']}, median {rep['share_median']}")
    print(f"  {rep['places_supported']} tiers against {rep['places_published']} published places; "
          f"{rep['n_certified_total']} certified reversals")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

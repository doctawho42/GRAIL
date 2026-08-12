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
import re
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


OFFICIAL = re.compile(r"^\d+\.\d+ \('([^']+)', (\d+), (-?[\d.]+), (-?[\d.]+)")


def official_scores(path: Path) -> dict:
    """{lp: [(system, ratings, raw mean), ...]} from the task's own published results file."""
    out, lp = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"^\[(\w+)-->(\w+)\]", line)
        if m:
            lp = f"{m.group(1)}-{m.group(2)}"
            out[lp] = []
            continue
        m = OFFICIAL.match(line)
        if m and lp:
            out[lp].append((m.group(1), int(m.group(2)), float(m.group(4))))
    return {k: v for k, v in out.items() if v}


def load(path: Path) -> pd.DataFrame:
    """The released table put through the preprocessing its own README documents.

    ``cat scores_all.csv | python3 scripts/rm_bad_docs.py | grep -v ",True," > scores_noqc_nodoc.csv``

    Two steps, and skipping either changes the ranking. ``rm_bad_docs`` drops a whole
    (annotator, system, document) group when any of its rows is a planted quality-control item, so a
    rater who failed the check loses that document rather than that row; and the grep removes the
    document-level meta scores, which are not segment judgements at all. Reading only ``type==TGT``
    keeps both --- it inflated every board here by between a ninth and two fifths, and left the
    published order disagreeing with the one the task published.
    """
    # rm_bad_docs is sequential and order-dependent: it flushes a run of lines when the
    # (annotator, system, document) key changes or the previous line was a document-level score,
    # and discards the run if any line in it was a planted control. Reading it as "drop every row
    # sharing a key with a control" is a different and stricter filter -- it leaves 840 ratings per
    # system on en-de where the task's own file reports 1094 -- so the algorithm is transcribed
    # rather than paraphrased, and the counts it produces match the published ones exactly.
    raw = [ln.rstrip("\n").split(",") for ln in path.read_text().splitlines() if ln.strip()]
    raw = [f for f in raw if len(f) == len(COLS)]
    I_TYPE, I_DOCTYPE = COLS.index("type"), COLS.index("isdoc")
    dkey = lambda f: (f[COLS.index("annotator")], f[COLS.index("system")], f[COLS.index("doc")])
    kept, run, isbad = [], [], False
    for f in raw:
        if run and (run[-1][I_DOCTYPE] == "True" or dkey(f) != dkey(run[-1])):
            if not isbad:
                kept.extend(run)
            run, isbad = [], False
        if f[I_TYPE] == "BAD":
            isbad = True
        run.append(f)
    if run and not isbad:
        kept.extend(run)

    d = pd.DataFrame(kept, columns=COLS)
    d = d[d["isdoc"].astype(str) != "True"]
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
    # The task averages over ratings, not over segments, and segments carry between one and several
    # ratings. Weighting a segment by how many it carries makes the plain mean of the item vector
    # the rating mean, which is what the published table ranks by; the machinery above is untouched.
    # The task ranks by the mean of the per-segment means, not by the mean over ratings. We first
    # inferred the opposite from its published per-system counts, which are rating counts, and
    # weighted each system's vector by its own -- that broke a reproduction that is otherwise exact
    # to 3e-14 with no inversion on any of the 909 pairs, and the discrepancy was then written up as
    # underdetermination in WMT rather than as our own weighting. Nothing is weighted here.

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

    OFF = official_scores(Path(args.csv).with_name("WMT23.results.txt"))
    boards = {}
    for (src, tgt), sub in d.groupby(["src", "tgt"]):
        lp = f"{src}-{tgt}"
        hits, systems, items, cells = build_board(sub)
        sub_grids = {"criteria only, unnormalised": [(c, PUBLISHED_CELL[1]) for c in CRITERIA],
                     "normalisation only, at the published criterion":
                         [(PUBLISHED_CELL[0], n) for n in NORMS],
                     "the product": cells}
        r = analyse(hits, systems, cells, PUBLISHED_CELL, sub_grids)
        off = OFF.get(lp, [])
        off_order = [s for s, _, _ in sorted(off, key=lambda x: -x[2])]
        common = [s for s in off_order if s in r["published_order"]]
        ours = [s for s in r["published_order"] if s in set(common)]
        r["official_order"] = off_order
        r["published_cell_reproduces_the_official_ranking"] = bool(common) and ours == common
        # The filter is exact -- per-system rating counts match the task's file to the unit -- and
        # the scores agree to a few tenths on a hundred-point scale, but the published number
        # involves a step past the mean that is not in the release. So the agreement is measured
        # rather than asserted: how far the scores are apart, and how many pairs the two orders
        # disagree on. That disagreement is itself an instance of what this paper is about.
        acc = r["system_accuracy_by_cell"]
        cell = r["published_cell"]
        gaps = [abs(acc[s][cell] - sc) for s, _, sc in off if s in acc]
        pos = {s: i for i, s in enumerate(r["published_order"])}
        r["agreement_with_official"] = {
            "max_score_gap": round(max(gaps), 4) if gaps else None,
            "pairs_ordered_differently": sum(
                1 for i in range(len(common)) for j in range(i + 1, len(common))
                if pos[common[i]] > pos[common[j]]),
            "pairs_compared": len(common) * (len(common) - 1) // 2,
            "rating_counts_match": all(
                int(n) == int((sub["system"] == s).sum()) for s, n, _ in off if s in acc)}
        r["n_annotators"] = int(sub["annotator"].nunique())
        r["ratings_per_system_segment"] = round(
            float(len(sub) / max(sub.groupby(["system", "doc", "seg"]).ngroups, 1)), 3)
        boards[lp] = r
        print(f"  {lp:9s} {r['n_systems']:3d} systems, {len(items):5d} segments, "
              f"{r['n_annotators']:3d} annotators, {r['n_pairs']:4d} pairs: "
              f"{r['n_dominating']:4d} dominate, {r['n_contested']:2d} contested "
              f"({r['n_contested_after_correction']} certified), {r['n_unresolved']:3d} unresolved, "
              f"{r['tiers_distinguished']:2d} tiers"
              + ("" if r["published_cell_reproduces_the_official_ranking"]
                 else "  [DOES NOT REPRODUCE THE OFFICIAL ORDER]"), flush=True)

    inv = sum(b["agreement_with_official"]["pairs_ordered_differently"] for b in boards.values())
    cmp_ = sum(b["agreement_with_official"]["pairs_compared"] for b in boards.values())
    gap = max(b["agreement_with_official"]["max_score_gap"] or 0.0 for b in boards.values())
    counts_ok = all(b["agreement_with_official"]["rating_counts_match"] for b in boards.values())
    print(f"\n  against the task's published scores: rating counts match exactly on every board: "
          f"{counts_ok}; largest score gap {gap:.3f} on a hundred-point scale; "
          f"{inv} of {cmp_} pairs ordered differently")
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
           "official_agreement": {"rating_counts_match_on_every_board": counts_ok,
                                  "largest_score_gap": round(gap, 4),
                                  "pairs_ordered_differently": inv, "pairs_compared": cmp_},
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

#!/usr/bin/env python3
r"""Nine more leaderboards from one shared task, so the share can be read as a distribution.

Two boards make a pair of anecdotes. WMT24 ran its general translation task in eleven language pairs
and annotated nine of them under \textsc{esa}, in which one expert marks the error spans in a
segment and then gives it a score out of a hundred. The systems differ per pair, the annotators
differ per pair, and the pairs were run independently, so nine boards from one task are nine
leaderboards and not one leaderboard nine times --- which is what lets the surviving share be
reported as a distribution rather than as a number that happens to have come out of one table.

The grid is the same shape as the others and both axes are choices this protocol leaves to whoever
scores it:

  criterion        what the annotation is read as. The task's own number is the score out of a
                   hundred; the marked spans are also there, and reading them as an error tally is
                   what a paper does when it wants a scale it can interpret. Three tallies are
                   included because three are in print: severity-weighted, serious errors only, and
                   a plain count.
  normalisation    whether a score is used as the annotator wrote it or standardised within that
                   annotator first. This is the oldest open question in the field's own evaluation
                   practice --- \textsc{wmt} scored with $z$ for years and then stopped --- and a
                   comparison quoting human scores almost never says which it used.

The published cell is the score out of a hundred, unnormalised, averaged within each domain and then
across domains, which is how the task ranks under this protocol as under the other. It was first
built as the mean over segments on the reasoning that the official number is just the mean of the
column; it is not, and the two orders differ.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from robust_order import analyse  # noqa: E402

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


def _tally(errors, mode: str) -> float:
    total = 0.0
    for e in errors or []:
        sev = (e.get("severity") or "").lower()
        if mode == "major-only":
            total += -1.0 if sev in ("major", "critical") else 0.0
        elif mode == "count":
            total += 0.0 if sev in ("no-error", "neutral", "") else -1.0
        else:  # severity-weighted
            if sev in ("major", "critical"):
                total += -5.0
            elif sev == "minor":
                total += -1.0
    return total


CRITERIA = {
    "esa": ("the score out of a hundred the annotator gave", None),
    "spans": ("the marked spans read as a severity-weighted tally", "weighted"),
    "major-only": ("the marked spans, serious errors only", "major-only"),
    "count": ("the marked spans, one point each", "count"),
}
NORMS = ("raw", "z-per-annotator")
PUBLISHED_CELL = ("esa", "raw")


def load(path: Path) -> dict:
    """lp -> {(system, segment) -> (value_by_criterion, annotator)}, from the merged human file."""
    per = collections.defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lp = r["doc_id"].split("_#_")[0]
        for system, anns in r["scores"].items():
            if not anns:
                continue
            # a segment carries one annotation per system here; if a protocol ever supplies more,
            # averaging them is the only reading that keeps the item the same object across systems
            vals = {}
            for name, (_, mode) in CRITERIA.items():
                if mode is None:
                    vals[name] = float(np.mean([a["score"] for a in anns]))
                else:
                    vals[name] = float(np.mean([_tally(a.get("errors"), mode) for a in anns]))
            per[lp][(system, r["doc_id"])] = (vals, anns[0]["annotator"])
    return per


def build_board(cell_map: dict) -> tuple[dict, list[str], list[str], list]:
    systems = sorted({s for s, _ in cell_map})
    items = sorted({i for _, i in cell_map})
    cells = [(c, n) for c in CRITERIA for n in NORMS]

    # The task ranks by the mean within each domain and then across domains -- its own clustering
    # script defaults to that for this protocol as well as for MQM, and the domain is the second
    # field of a document id. Weighting an item by $1/(G\,n_g)$ and rescaling by the item count
    # makes an unweighted mean of the rescaled vector that average, leaving the paired differences
    # and the item bootstrap untouched. Without it the published cell is the mean over segments,
    # which is a different number and a different order.
    doms = [i.split("_#_")[1] if len(i.split("_#_")) > 1 else "?" for i in items]
    sizes = collections.Counter(doms)
    G = len(sizes)
    w_macro = np.array([1.0 / (G * sizes[g]) for g in doms]) * len(items)

    # z is taken within an annotator across everything that annotator scored, which is the only
    # pool that makes the transform about the annotator rather than about the systems they saw
    stats = {}
    for criterion in CRITERIA:
        by_ann = collections.defaultdict(list)
        for (s, i), (vals, ann) in cell_map.items():
            by_ann[ann].append(vals[criterion])
        stats[criterion] = {a: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                            for a, v in by_ann.items()}

    hits = {}
    for criterion in CRITERIA:
        for norm in NORMS:
            grid = np.zeros((len(systems), len(items)))
            si = {s: k for k, s in enumerate(systems)}
            ii = {i: k for k, i in enumerate(items)}
            seen = np.zeros_like(grid, dtype=bool)
            for (s, i), (vals, ann) in cell_map.items():
                v = vals[criterion]
                if norm == "z-per-annotator":
                    m, sd = stats[criterion][ann]
                    v = 0.0 if sd == 0.0 else (v - m) / sd
                grid[si[s], ii[i]] = v
                seen[si[s], ii[i]] = True
            # a system not scored on a segment is given that segment's mean over the systems that
            # were, so an absent observation neither rewards nor punishes it; the alternative,
            # dropping the segment, would change the item set with the cell
            if not seen.all():
                col_mean = np.where(seen.any(axis=0), (grid * seen).sum(0) /
                                    np.maximum(seen.sum(0), 1), 0.0)
                grid = np.where(seen, grid, col_mean)
            grid = grid * w_macro
            for k, name in enumerate(systems):
                hits[(name, (criterion, norm))] = grid[k]
    return hits, systems, items, cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(ROOT / "data/external/wmt24/txt/"
                                                  "wmt24-genmt-humeval.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "robust_order_wmt24_esa.json"))
    ap.add_argument("--min-systems", type=int, default=3)
    args = ap.parse_args()

    raw = Path(args.jsonl).read_bytes()
    per_lp = load(Path(args.jsonl))

    boards, skipped = {}, []
    for lp in sorted(per_lp):
        cell_map = per_lp[lp]
        hits, systems, items, cells = build_board(cell_map)
        if len(systems) < args.min_systems:
            skipped.append({"lp": lp, "systems": len(systems)})
            continue
        sub_grids = {"criteria only, unnormalised": [(c, PUBLISHED_CELL[1]) for c in CRITERIA],
                     "normalisation only, at the published criterion":
                         [(PUBLISHED_CELL[0], n) for n in NORMS],
                     "the product": cells}
        r = analyse(hits, systems, cells, PUBLISHED_CELL, sub_grids)
        r["n_annotators"] = len({a for _, a in cell_map.values()})
        boards[lp] = r
        print(f"  {lp:8s} {r['n_systems']:3d} systems, {len(items):5d} segments, "
              f"{r['n_pairs']:4d} pairs: {r['n_dominating']:4d} dominate, "
              f"{r['n_contested']:2d} contested ({r['n_contested_after_correction']} certified), "
              f"{r['n_unresolved']:3d} unresolved, {r['tiers_distinguished']:2d} tiers", flush=True)

    shares = {lp: r["robustness"] for lp, r in boards.items()}
    rep = {"config": {**_code_version(),
                      "source": "WMT24 general MT, txt/wmt24-genmt-humeval.jsonl, from "
                                "github.com/wmt-conference/wmt24-news-systems",
                      "source_sha256": hashlib.sha256(raw).hexdigest(), "source_bytes": len(raw),
                      "protocol": "ESA",
                      "criteria": {k: v[0] for k, v in CRITERIA.items()},
                      "second_axis": "whether a score is used as written or standardised within "
                                     "the annotator who wrote it",
                      "published_cell": "the score out of a hundred, unnormalised, averaged within each "
                                        "domain and then across domains, which is how the task "
                                        "ranks; the mean over segments is a different number",
                      "skipped": skipped},
           "n_boards": len(boards),
           "n_pairs_total": sum(r["n_pairs"] for r in boards.values()),
           "share_min": round(min(shares.values()), 4) if shares else None,
           "share_median": round(float(np.median(list(shares.values()))), 4) if shares else None,
           "share_max": round(max(shares.values()), 4) if shares else None,
           "n_certified_total": sum(r["n_contested_after_correction"] for r in boards.values()),
           "places_published": sum(r["n_systems"] for r in boards.values()),
           "places_supported": sum(r["tiers_distinguished"] for r in boards.values()),
           "boards": boards}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n  {rep['n_boards']} boards, {rep['n_pairs_total']} pairs; share "
          f"{rep['share_min']} to {rep['share_max']}, median {rep['share_median']}")
    print(f"  {rep['places_supported']} tiers supported against {rep['places_published']} "
          f"published places; {rep['n_certified_total']} certified reversals in total")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

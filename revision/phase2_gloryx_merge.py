#!/usr/bin/env python3
"""Phase 2: GLORYx's two runs merged into one arm covering the evaluated population.

Writes results/gloryx_service_preds_evaluated1170.json.

The published run covers the 291-substrate comparison set. The Phase 2 run asks the service only
for what that run does not already hold, so its output covers 879. Neither file covers the
evaluated population, and that is the trap this module exists to close: `arm_key_lists` reads a
list arm with `preds.get(s, [])`, so declaring either file as the arm on `evaluated1170` would hand
back an empty list for every substrate it lacks and score those as misses. GLORYx would be
understated on a quarter of the population, in the one table built to make the systems comparable.

So the arm is the merge, and the merge refuses rather than fills: an incomplete union, an overlap
between the two sources, or a missing input all raise. Nothing is re-keyed, re-ordered or
truncated here -- those are the table's own steps, and doing any of them twice is how two readings
of one column drift apart.

    python revision/phase2_gloryx_merge.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PUBLISHED = ROOT / "results" / "gloryx_service_preds.json"
WIDER = ROOT / "results" / "gloryx_service_preds_1170.json"
OUT = ROOT / "results" / "gloryx_service_preds_evaluated1170.json"
POPULATION_FILE = ROOT / "results" / "test_references.json"


def _read(path):
    """One run's predictions and rank detail, through the envelope the producer writes.

    A ValueError rather than SystemExit: SystemExit derives from BaseException and would pass
    straight through a caller's `except Exception`, which is exactly where a refusal must land.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"REFUSING: {path} does not exist; a missing run is not an empty run")
    blob = json.loads(path.read_text())
    preds = blob.get("predictions") if isinstance(blob, dict) else None
    if not isinstance(preds, dict):
        raise ValueError(f"REFUSING: {path} carries no 'predictions' map")
    detail = blob.get("with_rank_and_score")
    jobs = ((blob.get("obtained_from") or {}).get("jobs") or []) if isinstance(blob, dict) else []
    return preds, (detail if isinstance(detail, dict) else {}), jobs


def _population():
    return sorted(json.loads(POPULATION_FILE.read_text()))


def merge(published_path=PUBLISHED, wider_path=WIDER, population=None):
    """Both runs as one substrate map covering the population exactly, or a refusal.

    The checks are ordered so the most informative failure wins: a missing file first, then an
    overlap between the sources, then coverage. An overlap is checked before coverage because a
    file that claims substrates belonging to the other run is not the run it says it is, and that
    is worth hearing before a count.
    """
    population = _population() if population is None else list(population)
    pub, pub_detail, pub_jobs = _read(published_path)
    wide, wide_detail, wide_jobs = _read(wider_path)

    both = set(pub) & set(wide)
    if both:
        raise ValueError(
            f"REFUSING: {len(both)} substrate(s) appear in both runs, so one of them is not the "
            f"run it reports being; the two were obtained under different submission sets and an "
            f"overlap is a fact about the artefacts rather than a tie to break. "
            f"First: {sorted(both)[0][:60]}")

    merged = {}
    source = {}
    for s, v in pub.items():
        merged[s] = v
        source[s] = "published"
    for s, v in wide.items():
        merged[s] = v
        source[s] = "wider"

    pop = set(population)
    missing = pop - set(merged)
    extra = set(merged) - pop
    if missing:
        raise ValueError(
            f"REFUSING: the merge does not cover the population: "
            f"{len(pop) - len(missing)} of {len(pop)} substrates, {len(missing)} missing. An arm "
            f"short of its population scores every absent substrate as a miss. "
            f"First missing: {sorted(missing)[0][:60]}")

    detail = {}
    detail.update({s: v for s, v in pub_detail.items()})
    detail.update({s: v for s, v in wide_detail.items()})

    return {
        "predictions": {s: merged[s] for s in population},
        "with_rank_and_score": {s: detail[s] for s in population if s in detail},
        "source_of_each_substrate": {s: source[s] for s in population},
        "n_from_published": sum(1 for s in population if source[s] == "published"),
        "n_from_wider": sum(1 for s in population if source[s] == "wider"),
        "substrates_outside_the_population": sorted(extra),
        "jobs": {"published": pub_jobs, "wider": wide_jobs},
    }


def _digest(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main() -> int:
    try:
        got = merge()
    except ValueError as e:
        print(e)
        return 1

    report = {
        "what_this_is": ("GLORYx on the evaluated population, assembled from the published "
                         "comparison-set run and the Phase 2 run over the rest"),
        "why_a_merge": ("neither run covers the population; a list arm is read with "
                        "preds.get(s, []), so a partial file would score every substrate it lacks "
                        "as a miss"),
        "population": "evaluated1170",
        "sources": {
            "published": {"path": str(PUBLISHED.relative_to(ROOT)),
                          "sha256": _digest(PUBLISHED), "substrates": got["n_from_published"]},
            "wider": {"path": str(WIDER.relative_to(ROOT)),
                      "sha256": _digest(WIDER), "substrates": got["n_from_wider"]},
        },
        "drawing": ("the substrate as the corpus stores it, which is what both runs were handed "
                    "and what the scoring joins on"),
        "n_substrates": len(got["predictions"]),
        "n_with_at_least_one_prediction": sum(1 for v in got["predictions"].values() if v),
        "mean_returned": round(sum(len(v) for v in got["predictions"].values())
                               / max(len(got["predictions"]), 1), 2),
        **got,
    }
    try:
        from _provenance import stamp
        report = {"provenance": stamp(__file__), **report}
    except Exception as e:
        report = {"provenance": {"unavailable": f"{e.__class__.__name__}: {e}"}, **report}
    OUT.write_text(json.dumps(report, indent=1))

    print(f"  from the published run : {report['n_from_published']:>5}")
    print(f"  from the Phase 2 run   : {report['n_from_wider']:>5}")
    print(f"  population covered     : {report['n_substrates']:>5}")
    print(f"  with a prediction      : {report['n_with_at_least_one_prediction']:>5}"
          f"  (mean list {report['mean_returned']})")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

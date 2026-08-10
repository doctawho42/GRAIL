#!/usr/bin/env python3
"""How tightly are real leaderboards packed, against how far a criterion moves them?

A criterion reorders a pair of methods exactly when it moves them apart by more than the gap between
them. That much is arithmetic and needs no evidence: the difference between two scores changes sign
only if the change exceeds it. What is not arithmetic, and is the whole of the empirical question,
is whether real leaderboards are packed tightly enough for the inequality to be met -- and that is a
fact about how a field's methods converge, not about the criterion.

So this measures the two quantities whose comparison decides it, on every leaderboard this paper
touches, in three domains: molecular generation, metabolite-structure prediction and retrosynthesis.
Nothing is re-scored here. Every input is a committed artifact of scores already computed under
several criteria, so the only thing this adds is the comparison.

Per leaderboard, per pair of methods and per pair of criteria:

    gap          |score_a - score_b| under the first criterion
    differential |(score_a - score_b) under the second - the same under the first|
    exchanged    whether the sign of score_a - score_b differs between the two

`differential > gap` is necessary for an exchange and not sufficient, since the movement can widen
the gap instead of closing it. Both counts are therefore reported: how often the inequality is met,
and how often an exchange follows.
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


def _load(path: str):
    p = ROOT / "results" / path
    return json.loads(p.read_text()) if p.exists() else None


def leaderboards() -> dict:
    """{name: (domain, {method: {criterion: score}})} from committed artifacts only."""
    out = {}

    for cl, label in (("cluster0", "retrosynthesis, seven-system group"),
                      ("cluster1", "retrosynthesis, three-system group")):
        d = _load(f"retro_leaderboard_{cl}.json")
        if d:
            out[label] = ("retrosynthesis",
                          {m: {c: v["top1"] for c, v in cs.items()}
                           for m, cs in d["accuracy"].items()})

    d = _load("moses_uniqueness.json")
    if d:
        # criterion -> model -> score, transposed; the metric IS the criterion here
        per = {}
        for crit, models in d["uniqueness"].items():
            for m, v in models.items():
                per.setdefault(m, {})[crit] = v["unique@10000"]
        out["molecular generation, MOSES"] = ("generation", per)

    d = _load("gloryx_criterion_ladder.json")
    if d:
        out["metabolites, external GLORYx"] = ("metabolites", d["recall"])

    d = _load("set_metrics_by_criterion.json")
    if d and "populations" in d:
        for pop, block in d["populations"].items():
            per = {}
            for crit, methods in block.items():
                if not isinstance(methods, dict):
                    continue
                for m, v in methods.items():
                    s = v.get("recall") if isinstance(v, dict) else v
                    if isinstance(s, (int, float)):
                        per.setdefault(m, {})[crit] = s
            if len(per) >= 3 and len(next(iter(per.values()))) >= 2:
                out[f"metabolites, {pop}"] = ("metabolites", per)
    return out


def analyse(scores: dict) -> list[dict]:
    methods = sorted(scores)
    crits = sorted(set.intersection(*(set(scores[m]) for m in methods)))
    rows = []
    for a, b in itertools.combinations(methods, 2):
        for c1, c2 in itertools.combinations(crits, 2):
            d1 = scores[a][c1] - scores[b][c1]
            d2 = scores[a][c2] - scores[b][c2]
            if d1 == 0:
                continue                      # already tied, so there is no order to exchange
            # A pair that becomes exactly tied under the second criterion has lost its order
            # without reversing it. Counting that as an exchange makes the arithmetic look violated,
            # because the movement then equals the gap rather than exceeding it.
            rows.append({"pair": f"{a} vs {b}", "criteria": f"{c1} vs {c2}",
                         "gap": abs(d1), "differential": abs(d2 - d1),
                         "closer_than_the_move": abs(d2 - d1) > abs(d1),
                         "tied": d2 == 0,
                         "exchanged": d2 != 0 and (d1 > 0) != (d2 > 0)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "packing_vs_differential.json"))
    args = ap.parse_args()

    boards, report, allrows = leaderboards(), {}, []
    print(f"{len(boards)} leaderboards from committed artifacts\n")
    print(f"  {'leaderboard':44} {'n':>3} {'pairs':>6} {'med gap':>8} {'med diff':>9} "
          f"{'closer':>7} {'exch':>5}")
    for name, (domain, scores) in sorted(boards.items(), key=lambda kv: kv[1][0]):
        rows = analyse(scores)
        if not rows:
            continue
        allrows += [{**r, "leaderboard": name, "domain": domain} for r in rows]
        gaps = sorted(r["gap"] for r in rows)
        diffs = sorted(r["differential"] for r in rows)
        closer = sum(r["closer_than_the_move"] for r in rows)
        exch = sum(r["exchanged"] for r in rows)
        tied = sum(r["tied"] for r in rows)
        report[name] = {"domain": domain, "methods": len(scores), "comparisons": len(rows),
                        "median_gap": round(gaps[len(gaps) // 2], 4),
                        "median_differential": round(diffs[len(diffs) // 2], 4),
                        "closer_than_the_move": closer, "exchanged": exch, "tied": tied}
        print(f"  {name[:44]:44} {len(scores):>3} {len(rows):>6} "
              f"{report[name]['median_gap']:>8.4f} {report[name]['median_differential']:>9.4f} "
              f"{closer:>7} {exch:>5}")

    # the arithmetic must hold on every row, or one of the two quantities is computed wrongly
    violations = [r for r in allrows if r["exchanged"] and not r["closer_than_the_move"]]
    if violations:
        raise SystemExit(f"{len(violations)} comparisons exchanged without the differential "
                         f"exceeding the gap, which is arithmetically impossible; one of the two "
                         f"quantities is being computed wrongly")

    closer = sum(r["closer_than_the_move"] for r in allrows)
    exch = sum(r["exchanged"] for r in allrows)
    print(f"\n  {len(allrows)} method-pair by criterion-pair comparisons across "
          f"{len({r['domain'] for r in allrows})} domains")
    print(f"  the criterion moves a pair by more than the gap between them in {closer} "
          f"({closer / len(allrows):.1%})")
    print(f"  an exchange follows in {exch} ({exch / max(closer, 1):.1%} of those, "
          f"{exch / len(allrows):.1%} of all)")
    print("  no exchange occurs where the gap survives the movement, as arithmetic requires")

    # Read as a screening test, the inequality is what a benchmark's maintainer can actually use:
    # it is computable from a published table of scores and one measurement of how far the criterion
    # moves a system, without re-scoring anything, and it flags the pairs that could exchange. Its
    # value is that it has no false negatives -- the arithmetic forbids them -- so the flagged set is
    # guaranteed to contain every pair at risk. What it costs is the false positives: movements that
    # widen a gap rather than close it.
    tp = sum(r["exchanged"] and r["closer_than_the_move"] for r in allrows)
    fp = sum(not r["exchanged"] and r["closer_than_the_move"] for r in allrows)
    fn = sum(r["exchanged"] and not r["closer_than_the_move"] for r in allrows)
    tn = len(allrows) - tp - fp - fn
    screen = {"flagged": tp + fp, "exchanged_and_flagged": tp,
              "exchanged_and_missed": fn, "not_exchanged_and_flagged": fp,
              "not_exchanged_and_not_flagged": tn,
              "sensitivity": round(tp / max(tp + fn, 1), 4),
              "specificity": round(tn / max(tn + fp, 1), 4),
              "precision": round(tp / max(tp + fp, 1), 4),
              "share_of_comparisons_flagged": round((tp + fp) / max(len(allrows), 1), 4)}
    print(f"  as a screening test: sensitivity {screen['sensitivity']}, "
          f"specificity {screen['specificity']}, precision {screen['precision']}; "
          f"it flags {screen['share_of_comparisons_flagged']:.1%} of comparisons")

    rep = {"config": {**_code_version(),
                      "inputs": "committed leaderboards; nothing is re-scored here",
                      "note": "differential > gap is necessary for an exchange and not sufficient; "
                              "the empirical question is how often real leaderboards meet it"},
           "per_leaderboard": report,
           "screening_test": screen,
           "totals": {"comparisons": len(allrows), "closer_than_the_move": closer,
                      "exchanged": exch, "tied": sum(r["tied"] for r in allrows),
                      "domains": sorted({r["domain"] for r in allrows})},
           "comparisons": allrows}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

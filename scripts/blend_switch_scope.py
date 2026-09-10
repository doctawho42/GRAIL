#!/usr/bin/env python3
"""Which committed artifacts the budget switch invalidates, derived rather than guessed.

ModelWrapper now ranks under the blend at ten candidates and below. An artifact can only move if
its content depends on the candidate aggregation, either because its producer builds a GRAIL
ranking or because it reads an artifact that does. Both are recoverable: every artifact records
the script that wrote it, and most record the files they read.

So the scope is a graph rather than a grep. Seeds are the producers that rank -- they import the
generator or the wrapper, or they combine a generator score with a filter score themselves. From
each seed the dependency edges are followed forward through the recorded inputs. An artifact is in
scope only if it is reachable AND it carries a quantity at a budget of ten or less, because
nothing above the switch changes.

Two things this deliberately does not do. It does not assume an artifact is safe because its name
looks unrelated, and it does not assume one is affected because it happens to be keyed by a small
integer: the budget test is applied to artifacts that reach the ranking, not to all of them.

Artifacts with no recorded inputs are reported separately. They cannot be placed in the graph and
are the residue a reader has to settle by hand rather than a set this file quietly calls clean.

    python scripts/blend_switch_scope.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIGHT = ("1", "3", "5", "8", "10")

# A producer ranks if it reaches the model or builds the order itself. The second half matters:
# several analyses never import the model and rank frozen pools by the two component scores, and
# those move with the aggregation exactly as the model does.
RANKS = re.compile(
    r"candidate_aggregation|_aggregate_candidate_scores|ModelWrapper|generate_scored|"
    r"build_generator|rrf_order|competition_ranks|RRF_K|\"generator\"\]\s*\*|"
    r"c\[.generator.\]|aggregate\(")


# A key of "5" can be a budget, a rule count, a shard index or a seed. What distinguishes a budget
# is the value under it: a recall, a precision or a difference of them, so a float in [-1, 1]. The
# looser test -- any key in {1,3,5,8,10} anywhere -- put artifacts about standardisation timing in
# the same list as the sweep, which is what forced this one.
def tight_budget_recall(blob) -> bool:
    """Does this artifact hold a rate keyed by a budget of ten or less?"""
    found = [False]

    def rate(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -1.0 <= float(v) <= 1.0
        if isinstance(v, dict):
            return any(rate(x) for x in list(v.values())[:20])
        return False

    def walk(o, depth=0):
        if found[0] or depth > 8:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if k in TIGHT and rate(v):
                    found[0] = True
                    return
                walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o[:200]:
                walk(v, depth + 1)

    walk(blob)
    return found[0]


def names_a_grail_arm(text: str) -> bool:
    """An artifact that never names an arm of this system cannot be reporting one."""
    return bool(re.search(r"grail|whole bank|trained budget|exhaustive|interactive", text, re.I))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "blend_switch_scope.json"))
    args = ap.parse_args()

    meta = {}
    for f in sorted((ROOT / "results").glob("*.json")):
        if f.stat().st_size > 60_000_000:
            continue
        try:
            blob = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(blob, dict):
            continue
        # Some artifacts carry a provenance dict, some a config dict, and a few carry a bare
        # string under one of those names. Anything that is not a mapping has no producer to read.
        prov = blob.get("provenance")
        if not isinstance(prov, dict):
            prov = blob.get("config") if isinstance(blob.get("config"), dict) else {}
        script = prov.get("script_path") or prov.get("script")
        if not isinstance(script, str):
            script = None
        inputs = [i.get("path") for i in (blob.get("inputs") or []) if isinstance(i, dict)]
        meta[f"results/{f.name}"] = {
            "producer": script, "inputs": [i for i in inputs if i],
            "records_inputs": bool(blob.get("inputs")),
            "tight_budget": tight_budget_recall(blob),
            "names_an_arm": names_a_grail_arm(f.read_text()[:2_000_000])}

    src = {}
    for m in meta.values():
        s = m["producer"]
        if not s or s in src:
            continue
        p = ROOT / s if "/" in s else ROOT / "scripts" / s
        src[s] = p.read_text() if p.exists() else ""

    seeds = {name for name, m in meta.items()
             if m["producer"] and RANKS.search(src.get(m["producer"], ""))}

    # Forward closure over recorded inputs: if A reads B and B moves, A moves.
    reach, changed = set(seeds), True
    while changed:
        changed = False
        for name, m in meta.items():
            if name in reach:
                continue
            if any(i in reach for i in m["inputs"]):
                reach.add(name)
                changed = True

    # Two tiers, because the producer test is coarse: it sees that a script reaches the model,
    # not what it does with it. An artifact that ranks, reports a rate at a tight budget and names
    # an arm of this system is as close to certain as a static test gets; one that misses the last
    # condition is possible and has to be settled by opening it.
    certain = sorted(n for n in reach
                     if meta[n]["tight_budget"] and meta[n]["names_an_arm"])
    possible = sorted(n for n in reach
                      if meta[n]["tight_budget"] and not meta[n]["names_an_arm"])
    in_scope = certain
    reach_only = sorted(n for n in reach if not meta[n]["tight_budget"])
    unplaceable = sorted(n for n, m in meta.items()
                         if not m["records_inputs"] and n not in seeds and m["tight_budget"])

    rep = {"artifacts_scanned": len(meta),
           "producers_that_rank": sorted({meta[n]["producer"] for n in seeds}),
           "recompute": [{"artifact": n, "producer": meta[n]["producer"],
                          "reached_directly": n in seeds} for n in certain],
           "possible": [{"artifact": n, "producer": meta[n]["producer"]} for n in possible],
           "method": "an artifact is listed if its producer builds a GRAIL ranking or it reads "
                     "one that does, and it holds a rate keyed by a budget of ten or less. The "
                     "producer test is coarse and this is an upper bound, not a diff.",
           "reaches_the_ranking_but_reads_no_tight_budget": reach_only,
           "no_recorded_inputs_and_reads_a_tight_budget": unplaceable}

    print(f"  scanned {len(meta)} artifacts")
    print(f"  producers that build a GRAIL ranking: {len(rep['producers_that_rank'])}")
    print(f"\n  MUST RECOMPUTE ({len(in_scope)}):")
    for r in rep["recompute"]:
        mark = "direct" if r["reached_directly"] else "via inputs"
        print(f"    {r['artifact']:<46} {mark:<11} {r['producer']}")
    print(f"\n  possible, but names no arm of this system ({len(possible)}): "
          f"{', '.join(Path(n).name for n in possible) or 'none'}")
    print(f"\n  reach the ranking but read no rate at a tight budget ({len(reach_only)}): "
          f"{', '.join(Path(n).name for n in reach_only) or 'none'}")
    print(f"\n  cannot be placed (no recorded inputs) and read a tight budget "
          f"({len(unplaceable)}):")
    for n in unplaceable:
        print(f"    {n:<46} {meta[n]['producer']}")

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

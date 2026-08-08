#!/usr/bin/env python3
"""The substrate populations the convention and engine measurements run on, named once.

Eleven scripts took their substrate list from results/filter_vs_prior_ci.json, an artifact of a
model comparison that happens to carry 245 rows. Nothing about the convention work requires that
subsample -- it was inherited, and the paper's largest single finding, the engine term, rests on it
while the clean test split holds 1,170 substrates and 2,597 references.

Two populations are named here so a script can be pointed at either and so the choice is recorded in
its report rather than implied by which file it happened to open:

  subsample245  the 245 rows of results/filter_vs_prior_ci.json, kept so every committed artifact
                stays reproducible as a gate
  clean_test    all 1,170 substrates of the clean test split that carry references

The nesting check is not decoration. The paper will describe the widening as replacing a measurement
with the same measurement on a superset, not as an independent replication, and that sentence is
only true if the 245 are actually inside the 1,170. They are drawn from it, but "drawn from" is a
claim about a script that ran months ago, so it is asserted here against the files as they exist.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUBSAMPLE_SOURCE = ROOT / "results" / "filter_vs_prior_ci.json"
REFERENCES = ROOT / "results" / "test_references.json"

# results/coverage_gap_types.json, which types every reference on the clean test split
CLEAN_TEST_SIZE = (1170, 2597)
SUBSAMPLE_SIZE = 245

POPULATIONS = ("subsample245", "clean_test")
NAMES = POPULATIONS


def _references() -> dict:
    return json.loads(REFERENCES.read_text())


def load_population(name: str) -> list[str]:
    """The substrates of a named population, sorted, each carrying at least one reference."""
    if name not in NAMES:
        raise SystemExit(f"unknown population {name!r}; expected one of {NAMES}")
    refs = _references()
    if name == "clean_test":
        subs = sorted(s for s, v in refs.items() if v)
        n, u = len(subs), sum(len(refs[s]) for s in subs)
        if (n, u) != CLEAN_TEST_SIZE:
            raise SystemExit(f"clean_test is {n} substrates / {u} references against the committed "
                             f"{CLEAN_TEST_SIZE[0]} / {CLEAN_TEST_SIZE[1]} -- this is not the split "
                             f"the paper reports")
        return subs
    src = json.loads(SUBSAMPLE_SOURCE.read_text())["per_substrate"]
    subs = sorted(r["sub"] for r in src if refs.get(r["sub"]))
    if len(subs) != SUBSAMPLE_SIZE:
        raise SystemExit(f"subsample245 is {len(subs)} substrates against the committed "
                         f"{SUBSAMPLE_SIZE}")
    return subs


def population_items(name: str) -> list[tuple[str, list[str]]]:
    """(substrate, its references) for a named population, in a fixed order."""
    refs = _references()
    return [(s, refs[s]) for s in load_population(name)]


def assert_nested() -> dict:
    """The subsample is a subset of the clean split, which is what lets one replace the other."""
    small, big = set(load_population("subsample245")), set(load_population("clean_test"))
    if not small <= big:
        raise SystemExit(f"{len(small - big)} of the 245 subsample substrates are absent from the "
                         f"clean test split; the widening is not a superset of the subsample and "
                         f"cannot be described as replacing it")
    return {"subsample245": len(small), "clean_test": len(big), "nested": True}


def tagged_out(default_path, name: str) -> str:
    """Where a run writes, so widening never silently overwrites the artifact a gate cites."""
    p = Path(default_path)
    return str(p) if name == "subsample245" else str(p.with_name(f"{p.stem}__{name}{p.suffix}"))


if __name__ == "__main__":
    print(assert_nested())
    for n in NAMES:
        subs = load_population(n)
        refs = _references()
        print(f"  {n:14} {len(subs):>5} substrates  {sum(len(refs[s]) for s in subs):>5} references")

#!/usr/bin/env python3
r"""Read the extrapolation-split retrosynthesis release, and establish it is one board.

\citet{Hastedt_2024} taught this paper that one release can hide three disjoint test sets, so a
second release is not taken on trust either. Five systems ship ranked predictions here; this
establishes what they share before anything is scored, and measures the overlap with the boards
already in the survey, because the union correction treats each board as its own family.

Nothing is scored here and no criterion is applied. The output is a population and its provenance.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "grail_metabolism" / "data" / "retro_extrapolation"
EVALRETRO = ROOT / "grail_metabolism" / "data" / "evalretro"
SYSTEMS = {"CF": "Chemformer", "LR": "LocalRetro", "MG": "MEGAN",
           "GR": "GraphRetro", "MT": "Molecular Transformer"}


def _code_version() -> dict:
    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def read_system(key: str) -> dict:
    """product -> ranked list of predicted reactant strings, in file order."""
    out: dict = {}
    with open(SRC / f"{key}.txt") as fh:
        for row in csv.DictReader(fh):
            p = row.get("product")
            if not p:
                continue
            out.setdefault(p, []).append(row.get("prediction") or "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "retro_extrapolation_population.json"))
    args = ap.parse_args()

    if not SRC.exists():
        print(f"no release at {SRC}", file=sys.stderr)
        return 1

    per_system, sizes = {}, {}
    for k in SYSTEMS:
        per_system[k] = read_system(k)
        sizes[SYSTEMS[k]] = {"targets": len(per_system[k]),
                             "predictions": sum(len(v) for v in per_system[k].values())}

    shared = set.intersection(*(set(v) for v in per_system.values()))
    # the released test file, which is not identical to any system's product column
    ref = {ln.split(",")[0] for ln in (SRC / "test_50k.txt").read_text().splitlines() if ln.strip()}

    # overlap with the boards already surveyed. Holm is valid under arbitrary dependence, so a
    # shared item does not invalidate the union correction; it is measured and declared because a
    # reader is entitled to know the families are not disjoint.
    overlap = {}
    for f in sorted(EVALRETRO.glob("cluster*_test.csv")):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        col = "target" if "target" in rows[0] else list(rows[0])[1]
        old = {r[col] for r in rows if r.get(col)}
        ov = shared & old
        overlap[f.stem] = {"targets": len(old), "shared_with_this_board": len(ov),
                           "share_of_that_board": round(len(ov) / max(len(old), 1), 4),
                           "share_of_this_board": round(len(ov) / max(len(shared), 1), 4)}

    rep = {"config": {**_code_version(), "source": str(SRC.relative_to(ROOT)),
                      "systems": SYSTEMS,
                      "note": "population only; nothing is scored and no criterion is applied"},
           "per_system": sizes,
           "released_test_file_targets": len(ref),
           "shared_targets": len(shared),
           "all_systems_agree_on_the_population": len({len(v) for v in per_system.values()}) == 1,
           "systems_matching_the_released_test_file":
               [SYSTEMS[k] for k, v in per_system.items() if set(v) == ref],
           "overlap_with_surveyed_boards": overlap,
           "n_systems": len(SYSTEMS),
           "n_pairs": len(SYSTEMS) * (len(SYSTEMS) - 1) // 2}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"  {len(SYSTEMS)} systems, {len(shared)} shared targets, {rep['n_pairs']} pairs")
    for name, s in sizes.items():
        print(f"    {name:<24} {s['targets']:>5} targets, {s['predictions']:>6} predictions")
    print(f"  released test file lists {len(ref)} targets; systems matching it exactly: "
          f"{rep['systems_matching_the_released_test_file'] or 'none'}")
    print("  overlap with the boards already surveyed:")
    for k, v in overlap.items():
        print(f"    {k:<16} {v['shared_with_this_board']:>5} of {v['targets']:>5} "
              f"= {v['share_of_that_board']:.1%}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

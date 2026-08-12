#!/usr/bin/env python3
r"""How much of the gap between published places and supported ones is the grid, and how much is not?

The paper's headline is that twenty-three leaderboards publish $293$ places and support $170$. Read
quickly, and beside a title about undeclared choices, that invites the reading that the choices cost
the difference. They do not, and the paper's own Section~5 insists the two ways a pair can fail must
never be added together, so the same discipline has to apply to the count they roll up into.

Three readings of "supported", each a longest chain in a relation on the same systems:

    the leaderboard's own cell separates the pair      what the benchmark can resolve at all
    every declared cell separates it                   the same, robust to the grid
    every declared cell agrees in sign                 the paper's dominance, which asks less

The first to the second is what the grid costs. Everything above the first is lost before any choice
is varied, and is a fact about the benchmark's power rather than about anyone's conventions.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from robust_order import _tiers  # noqa: E402

SOURCES = [("robust_order.json", "cluster0"), ("robust_order.json", "cluster1"),
           ("robust_order_metabolite.json", None), ("robust_order_posebusters.json", None),
           ("robust_order_wmt24_en-de.json", None), ("robust_order_wmt24_ja-zh.json", None)]
COLLECTIONS = ("robust_order_wmt24_esa.json", "robust_order_wmt23.json")


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


def boards() -> list:
    out = []
    for name, key in SOURCES:
        p = ROOT / "results" / name
        if p.exists():
            d = json.loads(p.read_text())
            out.append(d["leaderboards"][key] if key else d)
    for name in COLLECTIONS:
        p = ROOT / "results" / name
        if p.exists():
            out += list(json.loads(p.read_text())["boards"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "places_decomposition.json"))
    args = ap.parse_args()

    B = boards()
    places = own = every = dom = 0
    per = {}
    for b in B:
        systems, cell = b["published_order"], b["published_cell"]
        e_own, e_every, e_dom = {}, {}, {}
        for name, v in b["pairs"].items():
            hi, lo = name.split(" over ")
            if v["per_cell"][cell]["separated"]:
                e_own.setdefault(hi, set()).add(lo)
            if v["separated_in_every_cell"]:
                e_every.setdefault(hi, set()).add(lo)
            if v["dominates"]:
                e_dom.setdefault(hi, set()).add(lo)
        t_own, t_every, t_dom = (_tiers(systems, e_own), _tiers(systems, e_every),
                                 _tiers(systems, e_dom))
        places += b["n_systems"]; own += t_own; every += t_every; dom += t_dom
        per[b["published_cell"] + "|" + "/".join(systems[:2])] = {
            "places": b["n_systems"], "own_cell_separates": t_own,
            "every_cell_separates": t_every, "every_cell_agrees_in_sign": t_dom}

    rep = {"config": {**_code_version(), "n_boards": len(B),
                      "note": "each figure is a longest chain in a relation on the same systems; "
                              "they differ only in what an edge requires"},
           "places_published": places,
           "supported_when_its_own_cell_separates": own,
           "supported_when_every_cell_separates": every,
           "supported_when_every_cell_agrees_in_sign": dom,
           "lost_before_any_choice_is_varied": places - own,
           "lost_to_the_grid": own - every,
           "per_board": per}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"  {places} places published across {len(B)} leaderboards")
    print(f"  {own} supported when the leaderboard's own cell must separate the pair")
    print(f"  {every} when every declared cell must separate it")
    print(f"  {dom} when every declared cell must merely agree in sign (the paper's dominance)")
    print(f"\n  lost before any choice is varied: {places - own}")
    print(f"  lost to the grid on top of that:  {own - every}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

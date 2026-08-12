#!/usr/bin/env python3
r"""Correcting each board for its own grid, when the paper quotes twenty-three of them.

Every board here runs Holm over the family its own grid creates: $n$ pairs by $c$ cells, which on
the nineteen-system translation board is $1{,}368$ tests. That controls the error rate of a claim
about \emph{that} leaderboard. It does not control the error rate of the claim this paper actually
makes, which ranges over all twenty-three, and a reader entitled to ask "how many looks did you take
in total?" gets a different number: the union of the grids.

So the same procedure is run over the union. Nothing is re-scored and no board is re-analysed --
each artifact carries the sorted $p$ of every cell-level test in its family, and Holm over the
concatenation is arithmetic on those vectors. Two intermediate readings are reported beside it,
because which one a reader wants depends on what they intend to quote:

    per board            what the artifacts already report: Holm at $\alpha$ within each grid
    alpha split          Holm within each grid at $\alpha/23$, the Bonferroni split across boards
    union                Holm over the concatenated family, which is the strictest of the three

The union is not merely the strictest: it is step-down, so it stops at its first failure, and a
reversal with a small $p$ can be lost because thousands of unrelated separations sit below it. That
is a property of the procedure and not a defect of the reversal, and it is why all three are
printed rather than only the one that flatters.
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

SOURCES = [("robust_order.json", "cluster0"), ("robust_order.json", "cluster1"),
           ("robust_order_metabolite.json", None), ("robust_order_posebusters.json", None),
           ("robust_order_wmt24_en-de.json", None), ("robust_order_wmt24_ja-zh.json", None)]
COLLECTIONS = ("robust_order_wmt24_esa.json", "robust_order_wmt23.json")
ALPHA = 0.05


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
    """(label, board) for each of the leaderboards the paper counts."""
    out = []
    for name, key in SOURCES:
        p = ROOT / "results" / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out.append((f"{name[:-5]}{':' + key if key else ''}",
                    d["leaderboards"][key] if key else d))
    for name in COLLECTIONS:
        p = ROOT / "results" / name
        if not p.exists():
            continue
        for lp, b in json.loads(p.read_text())["boards"].items():
            out.append((f"{name[:-5]}:{lp}", b))
    return out


def holm(ps: list[float], alpha: float) -> int:
    """How many of the sorted $p$ a step-down Holm rejects at level ``alpha``."""
    m = len(ps)
    for step, pv in enumerate(sorted(ps)):
        if pv > alpha / (m - step):
            return step
    return m


def _surviving_reversals(board: dict, cutoff: float) -> list:
    """The board's reversal tests whose $p$ is at or below a rejection cutoff."""
    return [r for r in board["multiplicity"]["reversal_tests"] if r["p"] <= cutoff]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "union_multiplicity.json"))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    B = boards()
    missing = [lab for lab, b in B if "p_values" not in b.get("multiplicity", {})]
    if missing:
        print("these boards predate the recorded p-vector; re-run their scripts:", file=sys.stderr)
        for lab in missing:
            print("   ", lab, file=sys.stderr)
        return 1

    # the union family, and the cutoff Holm reaches in it
    pooled = sorted(pv for _, b in B for pv in b["multiplicity"]["p_values"])
    k_union = holm(pooled, args.alpha)
    cut_union = pooled[k_union - 1] if k_union else -1.0

    per = {}
    tot = {"per_board": 0, "alpha_split": 0, "union": 0}
    pairs_tot = {"per_board": 0, "alpha_split": 0, "union": 0}
    for lab, b in B:
        mult = b["multiplicity"]
        k_own = holm(mult["p_values"], args.alpha)
        k_split = holm(mult["p_values"], args.alpha / len(B))
        cut_own = mult["p_values"][k_own - 1] if k_own else -1.0
        cut_split = mult["p_values"][k_split - 1] if k_split else -1.0
        r = {mode: _surviving_reversals(b, c)
             for mode, c in (("per_board", cut_own), ("alpha_split", cut_split),
                             ("union", cut_union))}
        per[lab] = {"family_size": mult["family_size"], "n_reversal_tests":
                    len(mult["reversal_tests"]),
                    **{f"reversals_{k}": len(v) for k, v in r.items()},
                    **{f"pairs_{k}": len({x["pair"] for x in v}) for k, v in r.items()},
                    "certified_in_the_artifact": b["n_contested_after_correction"]}
        for mode in tot:
            tot[mode] += len(r[mode])
            pairs_tot[mode] += len({x["pair"] for x in r[mode]})

    # the artifacts' own count, which the paper quotes, must be the per-board reading
    quoted = sum(b["n_contested_after_correction"] for _, b in B)

    rep = {"config": {**_code_version(), "n_boards": len(B), "alpha": args.alpha,
                      "note": "no board is re-analysed; Holm is re-applied to the p-vectors the "
                              "board artifacts already carry"},
           "union_family_size": len(pooled),
           "union_tests_rejected": k_union,
           "union_cutoff_p": cut_union,
           "certified_pairs": {**pairs_tot, "quoted_in_the_paper": quoted},
           "certified_reversal_tests": tot,
           "per_board": per}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"  {len(B)} boards, {len(pooled)} cell-level tests in the union")
    print(f"  Holm over the union rejects {k_union} of them, cutoff p = {cut_union:.3g}")
    print(f"\n  certified pairs, per board:   {pairs_tot['per_board']}  (paper quotes {quoted})")
    print(f"  certified pairs, alpha/{len(B)}:    {pairs_tot['alpha_split']}")
    print(f"  certified pairs, union Holm:  {pairs_tot['union']}")
    for lab, v in per.items():
        if v["pairs_per_board"] or v["pairs_union"]:
            print(f"     {lab:34s} {v['pairs_per_board']:2d} -> {v['pairs_alpha_split']:2d} -> "
                  f"{v['pairs_union']:2d}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

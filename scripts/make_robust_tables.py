#!/usr/bin/env python3
"""Generate the appendix tables behind the robust order, so the share is reconstructible.

The instrument in scripts/robust_order.py reduces a leaderboard to two integers, and a reader who
cannot recompute those integers has to take them on trust. Everything the instrument computes is a
function of one table -- accuracy per system per cell -- so this writes that table out for each
leaderboard, together with the per-pair classification and the sub-grid curve, straight from the
committed artifacts. Nothing is retyped: if an artifact changes, the tables change with it.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "app" / "robust_tables.tex"

SHORT = {"canonical": "canon", "inchikey": "IK", "nostereo": "no-st",
         "tautomer": "taut", "tanimoto1": "Tan"}


def cell_label(s: str) -> str:
    try:
        mode, k = ast.literal_eval(s)
        return f"{SHORT.get(mode, mode)}@{k}"
    except Exception:
        return s.replace("_", " ")


def esc(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&")


def accuracy_table(name: str, r: dict) -> str:
    acc = r["system_accuracy_by_cell"]
    systems = r["published_order"]
    cells = list(next(iter(acc.values())))
    out = []
    # split wide grids so the table fits the text block
    for start in range(0, len(cells), 8):
        block = cells[start:start + 8]
        out.append("\\begin{center}\\small")
        out.append("\\begin{tabular}{l" + "c" * len(block) + "}")
        out.append("\\toprule")
        out.append("system & " + " & ".join(cell_label(c) for c in block) + " \\\\")
        out.append("\\midrule")
        for s in systems:
            out.append(esc(s) + " & " + " & ".join(f"{acc[s][c]:.3f}" for c in block) + " \\\\")
        out.append("\\bottomrule\\end{tabular}\\end{center}")
    return "\n".join(out)


def pair_table(r: dict) -> str:
    rows = []
    for p, v in r["pairs"].items():
        verdict = ("dominates" if v["dominates"]
                   else ("contested" if v["contested"] else "unresolved"))
        rev = v["cells_that_reverse_it"]
        rows.append((esc(p), verdict,
                     "yes" if v["separated_in_every_cell"] else "no",
                     str(len(rev)),
                     str(len(v["cells_that_reverse_it_with_an_interval"])),
                     "yes" if v["resolved_in_the_published_cell"] else "no"))
    out = ["\\begin{center}\\small", "\\begin{tabular}{llccccl}", "\\toprule",
           "pair (as the published cell orders it) & verdict & sep.\\ every cell & "
           "cells reversing & with an interval & own cell resolves \\\\", "\\midrule"]
    for row in rows:
        out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule\\end{tabular}\\end{center}")
    return "\n".join(out)


def subgrid_table(boards: list[tuple[str, dict]]) -> str:
    out = ["\\begin{center}\\small", "\\begin{tabular}{llcccc}", "\\toprule",
           "leaderboard & sub-grid & cells & dominate & tiers & orderings \\\\", "\\midrule"]
    for name, r in boards:
        first = True
        for lab, v in r.get("sub_grids", {}).items():
            out.append(f"{esc(name) if first else ''} & {esc(lab)} & {v['n_cells']} & "
                       f"{v['n_dominating']} of {r['n_pairs']} & {v['tiers']} & "
                       f"{v['distinct_orderings']} \\\\")
            first = False
        out.append("\\midrule")
    out[-1] = "\\bottomrule\\end{tabular}\\end{center}"
    return "\n".join(out)


def hasse(r: dict) -> str:
    """The dominance poset drawn as its transitive reduction, ranked by longest chain.

    The published table is a line; this is what is left of that line once every pair a declared
    choice can reverse has been removed. Systems on the same row are ones the grid does not order.
    """
    systems = r["published_order"]
    edge = {s: set() for s in systems}
    for p, v in r["pairs"].items():
        if v["dominates"]:
            hi_, lo_ = p.split(" over ")
            edge[hi_].add(lo_)
    memo: dict = {}

    def chain(n):
        if n not in memo:
            memo[n] = 1 + max((chain(c) for c in edge.get(n, ())), default=0)
        return memo[n]

    depth = {s: chain(s) for s in systems}
    top = max(depth.values())
    rows: dict = {}
    for s in systems:
        rows.setdefault(top - depth[s], []).append(s)
    # transitive reduction: drop a to c when a to b to c already exists
    red = {a: {c for c in edge[a] if not any(c in edge[b] for b in edge[a] if b != c)}
           for a in systems}

    # laid out left to right: tiers along x, so the whole poset is two rows tall and fits a
    # nine-page body, which a top-to-bottom drawing of the same relation does not
    out = ["\\begin{tikzpicture}[x=2.45cm,y=0.62cm,every node/.style={font=\\scriptsize}]"]
    pos = {}
    for lvl in sorted(rows):
        row = rows[lvl]
        for i, s in enumerate(row):
            y = -(i - (len(row) - 1) / 2)
            pos[s] = (lvl, y)
            out.append(f"\\node[draw,rounded corners,inner sep=2pt] ({s.replace(' ', '')}) "
                       f"at ({lvl},{y:.2f}) {{{esc(s)}}};")
        out.append(f"\\node[font=\\tiny,gray] at ({lvl},{-(len(row) + 1) / 2:.2f}) "
                   f"{{tier {lvl + 1}}};")
    for a, cs in red.items():
        for c in cs:
            if pos[a][0] < pos[c][0]:
                out.append(f"\\draw[->,gray] ({a.replace(' ', '')}) -- ({c.replace(' ', '')});")
    out.append("\\end{tikzpicture}")
    return "\n".join(out)


def main() -> int:
    boards = []
    ro = ROOT / "results/robust_order.json"
    if ro.exists():
        d = json.loads(ro.read_text())["leaderboards"]
        for cl, r in d.items():
            boards.append((f"retrosynthesis, {r['n_systems']} systems", r))
    rm = ROOT / "results/robust_order_metabolite.json"
    if rm.exists():
        r = json.loads(rm.read_text())
        boards.append((f"metabolites, {r['n_systems']} methods", r))
    if not boards:
        raise SystemExit("no robust-order artifact to render")

    body = ["% generated by scripts/make_robust_tables.py -- do not edit by hand",
            "\\paragraph{The share as a function of the grid.} Domination is an intersection over "
            "cells, so adding a cell can only remove pairs from it and the share is a property of "
            "the pair (leaderboard, grid). Each declared axis is therefore reported alone and "
            "against the product.",
            subgrid_table(boards)]
    for name, r in boards:
        body.append(f"\\paragraph{{{esc(name)}.}} Published order at {esc(r['published_cell'])}: "
                    + " $>$ ".join(esc(s) for s in r["published_order"])
                    + f". {r['n_items']} items. Accuracy by system and cell, from which every "
                      "quantity below follows:")
        body.append(accuracy_table(name, r))
        body.append("Per-pair classification. A pair is \\emph{contested} only when some cell puts "
                    "it the other way round with an interval excluding zero; a pair that is "
                    "neither dominating nor contested is one the benchmark does not resolve, "
                    "which is a different failure and is not counted as a reversal.")
        body.append(pair_table(r))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(body) + "\n")

    # the figure: the largest leaderboard's published line beside what survives of it
    big = max(boards, key=lambda b: b[1]["n_systems"])[1]
    fig = ROOT / "paper" / "app" / "robust_hasse.tex"
    fig.write_text("% generated by scripts/make_robust_tables.py -- do not edit by hand\n"
                   + hasse(big) + "\n")
    print(f"wrote {OUT} for {len(boards)} leaderboards, and {fig} "
          f"({big['n_systems']} systems, {big['tiers_distinguished']} tiers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

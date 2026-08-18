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
        # the second axis is a budget in the prediction domains and a post-processing setting in
        # docking; "@15" reads as a cut-off and would misread as one on a board that has none
        if isinstance(k, int):
            return f"{SHORT.get(mode, mode)}@{k}"
        return f"{SHORT.get(mode, mode)} / {SHORT.get(k, k)}"
    except Exception:
        return s.replace("_", " ")


def esc(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&")


def header_cell(s: str) -> str:
    """A column head stacked over as many lines as its words need.

    A cell name is a criterion crossed with a second axis, and written on one line the widest of
    them (``rmsd2+valid / energy minimization``) is alone wider than a quarter of the text block.
    Set on one line these tables ran up to 444pt past the margin. Nothing is abbreviated here: the
    words are the same words, broken where they would otherwise run into the next column.
    """
    label = cell_label(s)
    if " / " not in label:
        return label
    lines = []
    for part in label.split(" / "):
        lines += part.split(" ")
    return "\\shortstack{" + "\\\\".join(lines) + "}"


def _block_width(cells: list) -> int:
    """How many of these columns fit side by side, which depends on how wide their names are."""
    widest = max((max(len(w) for w in cell_label(c).replace(" / ", " ").split(" "))
                  for c in cells), default=0)
    return 8 if widest <= 8 else 4


def accuracy_table(name: str, r: dict) -> str:
    acc = r["system_accuracy_by_cell"]
    systems = r["published_order"]
    cells = list(next(iter(acc.values())))
    out = []
    # split wide grids so the table fits the text block
    per = _block_width(cells)
    for start in range(0, len(cells), per):
        block = cells[start:start + per]
        out.append("\\begin{center}\\small")
        out.append("\\begin{tabular}{l" + "c" * len(block) + "}")
        out.append("\\toprule")
        out.append("system & " + " & ".join(header_cell(c) for c in block) + " \\\\")
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
    # the body rows are short; it is this header that ran past the margin on one line
    head = ("\\shortstack[l]{pair (as the published\\\\cell orders it)} & verdict & "
            "\\shortstack{sep.\\ every\\\\cell} & \\shortstack{cells\\\\reversing} & "
            "\\shortstack{with an\\\\interval} & \\shortstack[l]{own cell\\\\resolves} \\\\")
    spec = ">{\\raggedright\\arraybackslash}p{0.24\\textwidth}lccccl"
    # a tabular is one unbreakable box, and the nineteen-system board has 171 pairs: set that way
    # it stood 1{,}966pt taller than the page and simply ran off it. longtable breaks across pages
    # and repeats the header, which is what a table of this length needs.
    if len(rows) > 30:
        out = ["\\begingroup\\small\\setlength{\\tabcolsep}{4pt}",
               "\\begin{longtable}{" + spec + "}",
               "\\toprule", head, "\\midrule", "\\endfirsthead",
               "\\multicolumn{6}{l}{\\small\\itshape continued from the previous page} \\\\",
               "\\toprule", head, "\\midrule", "\\endhead",
               "\\midrule",
               "\\multicolumn{6}{r}{\\small\\itshape continued on the next page} \\\\",
               "\\endfoot",
               "\\bottomrule", "\\endlastfoot"]
        for row in rows:
            out.append(" & ".join(row) + " \\\\")
        out.append("\\end{longtable}\\endgroup")
        return "\n".join(out)

    out = ["\\begin{center}\\small\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{" + spec + "}", "\\toprule", head, "\\midrule"]
    for row in rows:
        out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule\\end{tabular}\\end{center}")
    return "\n".join(out)


def subgrid_table(boards: list[tuple[str, dict]]) -> str:
    out = ["\\begin{center}\\small\\setlength{\\tabcolsep}{4pt}",
           # the sub-grid labels are phrases, so that column wraps rather than runs on
           "\\begin{tabular}{l>{\\raggedright\\arraybackslash}p{0.26\\textwidth}cccc}",
           "\\toprule",
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
    # the published place of each system, which is what makes the drawing worth printing: the
    # table's first place is not in the top tier, and that is visible only if the places are on it
    place = {s: i + 1 for i, s in enumerate(systems)}
    caption_y = -(max(len(r) for r in rows.values()) + 1) / 2
    for lvl in sorted(rows):
        row = rows[lvl]
        col = "gPublished" if lvl == 0 else "gRule"
        for i, s in enumerate(row):
            y = -(i - (len(row) - 1) / 2)
            pos[s] = (lvl, y)
            out.append(f"\\node[draw={col}, text={col}, rounded corners, inner sep=2.5pt, "
                       f"fill=white, line width={0.7 if lvl == 0 else 0.4}pt] "
                       f"({s.replace(' ', '')}) at ({lvl},{y:.2f}) "
                       f"{{{esc(s)}\\,\\textcolor{{gRule!70}}{{\\tiny({place[s]})}}}};")
        out.append(f"\\draw[{col}!45] ({lvl - 0.30},{caption_y + 0.42:.2f}) -- "
                   f"({lvl + 0.30},{caption_y + 0.42:.2f});")
        out.append(f"\\node[font=\\tiny,{col}] at ({lvl},{caption_y:.2f}) {{tier {lvl + 1}}};")
    for a, cs in red.items():
        for c in cs:
            if pos[a][0] < pos[c][0]:
                out.append(f"\\draw[->,gRule!55,line width=0.35pt] ({a.replace(' ', '')}) -- "
                           f"({c.replace(' ', '')});")
    out.append(f"\\node[font=\\tiny,gRule,anchor=west] at ({max(rows) + 0.45},{caption_y:.2f}) "
               f"{{({{\\it n}}) is the place the published table gives}};")
    out.append("\\end{tikzpicture}")
    return "\n".join(out)


def main() -> int:
    boards = []
    ro = ROOT / "results/robust_order.json"
    if ro.exists():
        d = json.loads(ro.read_text())["leaderboards"]
        for cl, r in d.items():
            boards.append((f"retrosynthesis, {r['n_systems']} systems", r))
    rx = ROOT / "results/robust_order_retro_extrapolation.json"
    if rx.exists():
        for cl, r in json.loads(rx.read_text())["leaderboards"].items():
            boards.append((f"retrosynthesis, {r['n_systems']} systems", r))
    rm = ROOT / "results/robust_order_metabolite.json"
    if rm.exists():
        r = json.loads(rm.read_text())
        boards.append((f"metabolites, {r['n_systems']} methods", r))
    for lp in ("en-de", "ja-zh"):
        rw = ROOT / f"results/robust_order_wmt24_{lp}.json"
        if rw.exists():
            r = json.loads(rw.read_text())
            boards.append((f"translation, WMT24 {lp}, {r['n_systems']} systems", r))

    rp = ROOT / "results/robust_order_posebusters.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        boards.append((f"docking, {r['n_systems']} programs", r))
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
    esa = ROOT / "results/robust_order_wmt24_esa.json"
    esa_rows = []
    if esa.exists():
        E = json.loads(esa.read_text())
        for lp, r in sorted(E["boards"].items()):
            esa_rows.append(
                f"{lp} & {r['n_systems']} & {r['n_items']} & {r.get('n_annotators', '--')} & "
                f"{r['n_dominating']} of {r['n_pairs']} & {r['n_contested']} "
                f"({r['n_contested_after_correction']}) & {r['n_unresolved']} & "
                f"{r['tiers_distinguished']} \\\\")

    w23 = ROOT / "results/robust_order_wmt23.json"
    w23_rows = []
    if w23.exists():
        W = json.loads(w23.read_text())
        for lp, r in sorted(W["boards"].items()):
            w23_rows.append(
                f"{lp} & {r['n_systems']} & {r['n_items']} & {r.get('n_annotators', '--')} & "
                f"{r.get('ratings_per_system_segment', '--')} & "
                f"{r['n_dominating']} of {r['n_pairs']} & {r['n_contested']} "
                f"({r['n_contested_after_correction']}) & {r['n_unresolved']} & "
                f"{r['tiers_distinguished']} \\\\")

    if esa_rows:
        body += ["\\paragraph{The nine \\textsc{esa} translation boards.} Their per-cell accuracy "
                 "matrices are not printed: nine boards of eleven to sixteen systems by eight cells "
                 "is four hundred numbers that no reader checks, and they are in "
                 "\\texttt{results/robust\\_order\\_wmt24\\_esa.json} in full. What the text uses "
                 "from them is here.",
                 "\\begin{center}\\footnotesize",
                 "\\begin{tabular}{lccccccc}", "\\toprule",
                 "pair & systems & segments & annotators & dominate & contested (cert.) & "
                 "unresolved & tiers \\\\", "\\midrule",
                 *esa_rows, "\\bottomrule", "\\end{tabular}", "\\end{center}", ""]
    if w23_rows:
        body += ["\\paragraph{The eight boards of the previous edition.} Same treatment, and the "
                 "column a reader wants that the later edition cannot supply is the last but one: "
                 "how many ratings a segment carries, which is what gives this protocol its "
                 "criterion axis.",
                 "\\begin{center}\\footnotesize",
                 "\\begin{tabular}{lccccccc c}", "\\toprule",
                 "pair & systems & segments & annotators & ratings & dominate & "
                 "contested (cert.) & unresolved & tiers \\\\", "\\midrule",
                 *w23_rows, "\\bottomrule", "\\end{tabular}", "\\end{center}", ""]
    OUT.write_text("\n\n".join(body) + "\n")
    # the figure in the body is the seven-system retrosynthesis board, named rather than
    # picked by size: adding a larger board must not silently redraw the paper's figure
    big = next(r for name, r in boards if name.startswith("retrosynthesis, 7"))

    fig = ROOT / "paper" / "app" / "robust_hasse.tex"
    fig.write_text("% generated by scripts/make_robust_tables.py -- do not edit by hand\n"
                   + hasse(big) + "\n")
    print(f"wrote {OUT} for {len(boards)} leaderboards, and {fig} "
          f"({big['n_systems']} systems, {big['tiers_distinguished']} tiers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

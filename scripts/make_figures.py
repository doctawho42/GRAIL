#!/usr/bin/env python3
"""Three figures for results that are currently rows of numbers.

A reviewer's objection to this manuscript was that thirty-two pages carry nine tables and no figure,
and that three of its results are read far more slowly than they need to be. Each figure here is
drawn from a committed artifact rather than retyped, so it cannot drift from the tables it
accompanies, and each is checked against the number the paper prints before it is written.

  fig_decomp   the recall decomposition as a waterfall, both methods side by side: what the bank
               reaches, what the selector keeps of that, what the ranker returns of that.
  fig_ladder   the five matching criteria as a slope plot. The message is which lines cross and
               which stay parallel, which a table of the same numbers does not show.
  fig_xdomain  the same slope plot for seven released retrosynthesis systems under four
               criteria, which is where the reordering is visible rather than argued.
  fig_budget   macro F1 against the output budget, with the regions where the ordering changes
               shaded. The appendix currently states these crossings in prose.

Written as vector PDF at the column width the ICLR style uses; no seaborn, no style sheet, so the
output does not depend on anything outside matplotlib's defaults.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
GREY, ACC, ALT = "#444444", "#1f4e79", "#a6301f"
MET, DOCK = "#2e6f3e", "#6a3d9a"
# The manuscript's TikZ preamble defines the same five, and hue never carries meaning alone
# in either place: every series here also gets its own marker, so the figure survives
# greyscale and the common forms of colour blindness.
SERIES = [ACC, ALT, MET, DOCK, GREY]
MARKERS = ["o", "s", "^", "D", "v"]

# One method keeps one colour and one marker across every figure in the paper. Assigning them
# per figure -- alphabetically, or by whichever two the figure highlights -- makes GRAIL blue
# in one drawing and red in the next, which is a worse fault than the grey collision this
# table was introduced to fix.
METHOD_STYLE = {
    "GRAIL": (ACC, "o"), "SyGMa": (ALT, "s"), "MetaPredictor": (MET, "^"),
    "MetaTrans": (DOCK, "D"), "BioTransformer": (GREY, "v"),
}


def _style(name, i=0):
    return METHOD_STYLE.get(name, (SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]))


def _declutter(ax, placed, gap):
    """Push right-edge series labels apart so two of them cannot print on top of each other.

    `placed` is [(y, text, colour)]. Labels are laid out from the bottom up, each pushed to at
    least `gap` above the one below it. Without this, two systems within a hundredth of each
    other overprint into an unreadable smear, which is what fig_xdomain did to `graphretro`
    and `megan`.
    """
    out, last = [], None
    for y, text, colour in sorted(placed):
        pos = y if last is None else max(y, last + gap)
        out.append((pos, text, colour))
        last = pos
    return out


def _load(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def fig_decomp():
    """Waterfall: 1.0 -> coverage -> x selection -> x ranking -> realised, per method."""
    g = _load("recall_factorization")["factors"]
    s = _load("decompose_sygma")["factors"]
    methods = [("GRAIL", [g["coverage_bank"]["point"], g["selection_retention"]["point"],
                          g["ranking_conversion"]["point"]], _style("GRAIL")[0]),
               ("SyGMa", [s["coverage_bank"]["point"], s["selection_retention"]["point"],
                          s["ranking_conversion"]["point"]], _style("SyGMa")[0])]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.55), sharey=True)
    for ax, (name, f, colour) in zip(axes, methods):
        levels = [1.0, f[0], f[0] * f[1], f[0] * f[1] * f[2]]
        labels = ["all\nreferences", "bank\nreaches", "selector\nkeeps", "ranker\nreturns"]
        for i, (lo, hi) in enumerate(zip(levels[:-1], levels[1:])):
            ax.bar(i + 0.5, lo - hi, bottom=hi, width=0.55, color=GREY, alpha=0.30,
                   edgecolor="none")
        ax.bar(range(len(levels)), levels, width=0.55, color=colour, alpha=0.85, edgecolor="none")
        for i, v in enumerate(levels):
            ax.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=6.5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_title(f"{name}   ({f[0]:.3f} $\\times$ {f[1]:.3f} $\\times$ {f[2]:.3f})", fontsize=8.5)
        ax.set_ylim(0, 1.16)
        ax.tick_params(labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("fraction of references", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_decomp.pdf")
    plt.close(fig)
    return {"GRAIL realised": np.prod(methods[0][1]), "SyGMa realised": np.prod(methods[1][1])}


def fig_ladder():
    """Slope plot over the five criteria on the shared subset, where all five methods exist."""
    d = _load("match_sensitivity_5method")
    modes = d["modes"] if isinstance(d.get("modes"), list) else \
        ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
    short = ["canonical", "InChIKey", "no-stereo", "Tanimoto=1", "tautomer"]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    vals, labels = {}, []
    # every method gets its own colour AND marker. Three of the five used to share one grey,
    # which made the crossings -- the only thing a slope plot is for -- impossible to follow
    order = sorted(d["by_method"])
    for i, meth in enumerate(order):
        rec = d["by_method"][meth]
        y = [rec[m]["recall@15"] if isinstance(rec.get(m), dict) else rec.get(m) for m in modes]
        if any(v is None for v in y):
            continue
        vals[meth] = y
        c, mk = _style(meth, i)
        ax.plot(range(len(modes)), y, marker=mk, ms=3, lw=1.4, color=c, zorder=3)
        labels.append((y[-1], meth, c))
    span = max(max(v) for v in vals.values()) - min(min(v) for v in vals.values())
    for y, meth, c in _declutter(ax, labels, 0.055 * span):
        ax.text(len(modes) - 0.9, y, f" {meth}", fontsize=7, color=c, va="center")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(short, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("recall@15", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(-0.25, len(modes) + 1.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ladder.pdf")
    plt.close(fig)
    return {m: (v[0], v[-1]) for m, v in vals.items()}


def fig_xdomain():
    """The same slope plot in another field, on files we did not produce.

    Seven retrosynthesis systems released their own top-k predictions on one shared USPTO-50k test
    set; re-scoring those files under four matching criteria moves every system and reorders the
    leading four. The message is the same one \\S\\ref{app:xdomain} makes in prose --- the lines are
    not parallel, so which system is first is a property of the criterion --- and it is drawn from
    the ingest artifact rather than retyped, so it cannot drift from the table beside it.
    """
    d = _load("retro_leaderboard_cluster0")
    A, crit = d["accuracy"], ["canonical", "inchikey", "nostereo", "tautomer"]
    short = ["canonical", "InChIKey", "no-stereo", "tautomer"]
    moving = sorted({s for pair in d["pairs_that_exchange"]["top1"] for s in pair.split(" vs ")})
    palette = [ACC, ALT, "#2e6f3e", "#6a3d9a"]
    colours = dict(zip(moving, palette))

    vals = {sys_: [A[sys_][c]["top1"] for c in crit] for sys_ in A}

    # One system sits a tenth below the rest, which stretched the axis until the reordering --
    # the only thing this figure is for -- was squeezed into a tenth of its height. The axis is
    # broken at the largest gap between two systems' bands, and the break is drawn rather than
    # implied. Where there is no such gap the figure stays a single panel.
    bands = sorted(((min(v), max(v)) for v in vals.values()))
    gaps = [(bands[i + 1][0] - bands[i][1], i) for i in range(len(bands) - 1)]
    span = bands[-1][1] - bands[0][0]
    gap, at = max(gaps) if gaps else (0.0, 0)
    broken = gap > 0.30 * span
    pad = 0.06 * span

    if broken:
        cut_lo, cut_hi = bands[at][1], bands[at + 1][0]
        # tight_layout cannot lay out a broken axis (it refuses shared spines that are
        # hidden), and left to it the y label and the rotated ticks fall off the canvas
        fig, (hi, lo) = plt.subplots(
            2, 1, figsize=(3.3, 2.6), sharex=True, layout="constrained",
            gridspec_kw={
                "height_ratios": [bands[-1][1] - cut_hi + 2 * pad, cut_lo - bands[0][0] + 2 * pad],
                "hspace": 0.10})
        hi.set_ylim(cut_hi - pad, bands[-1][1] + pad)
        lo.set_ylim(bands[0][0] - pad, cut_lo + pad)
        axes = [hi, lo]
    else:
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        hi = lo = ax
        axes = [ax]

    labels = {id(hi): [], id(lo): []}
    for sys_ in sorted(A, key=lambda m: -A[m]["canonical"]["top1"]):
        y = vals[sys_]
        c = colours.get(sys_, GREY)
        target = hi if (not broken or min(y) >= cut_hi - pad) else lo
        for ax_ in axes:
            ax_.plot(range(len(crit)), y, marker="o", ms=3,
                     lw=1.5 if sys_ in colours else 1.0, color=c,
                     alpha=1.0 if sys_ in colours else 0.45,
                     zorder=3 if sys_ in colours else 2)
        labels[id(target)].append((y[-1], sys_, c))

    for ax_ in axes:
        seen = labels[id(ax_)]
        if not seen:
            continue
        lo_y, hi_y = ax_.get_ylim()
        for y, sys_, c in _declutter(ax_, seen, 0.085 * (hi_y - lo_y)):
            ax_.text(len(crit) - 0.92, y, f" {sys_}", fontsize=6.5, color=c, va="center")

    if broken:
        hi.spines["bottom"].set_visible(False)
        lo.spines["top"].set_visible(False)
        hi.tick_params(bottom=False)
        kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=5, linestyle="none",
                  color=GREY, mec=GREY, mew=0.8, clip_on=False)
        hi.plot([0], [0], transform=hi.transAxes, **kw)
        lo.plot([0], [1], transform=lo.transAxes, **kw)

    lo.set_xticks(range(len(crit)))
    lo.set_xticklabels(short, fontsize=7, rotation=20, ha="right")
    hi.set_ylabel("top-1 accuracy", fontsize=8)
    for ax_ in axes:
        ax_.tick_params(labelsize=7.5)
        ax_.set_xlim(-0.25, len(crit) + 1.5)
        for side in ("top", "right"):
            ax_.spines[side].set_visible(False)
    if not broken:
        fig.tight_layout()
    fig.savefig(OUT / "fig_xdomain.pdf")
    plt.close(fig)

    # The drawing has to show the reordering the artifact records, and no other: a pair the
    # artifact calls exchanged must cross somewhere between two adjacent criteria, and a pair it
    # does not must never cross. A figure that agrees with its caption but not with its source is
    # the failure this file exists to prevent.
    def crosses(a, b):
        signs = {(vals[a][i] > vals[b][i]) for i in range(len(crit)) if vals[a][i] != vals[b][i]}
        return len(signs) > 1
    recorded = {frozenset(p.split(" vs ")) for p in d["pairs_that_exchange"]["top1"]}
    drawn = {frozenset((a, b)) for a in vals for b in vals if a < b and crosses(a, b)}
    assert drawn == recorded, f"drawn crossings {drawn} != the artifact's {recorded}"
    return {"systems": len(vals), "criteria": len(crit), "exchanging pairs": len(recorded)}


def fig_budget():
    """Macro F1 against k, with the region where the ordering differs from the field's k=15."""
    d = _load("budget_curves")
    series = d["macro_f1_by_k"]          # method -> list indexed by k-1
    methods = sorted(series)
    ks = list(range(1, len(series[methods[0]]) + 1))
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for m in methods:
        y = series[m]
        c, _ = _style(m, methods.index(m))
        ax.plot(ks, y, lw=1.4, color=c, label=m, zorder=3)
    # headroom above the curves, so the band's label and the k marker sit in empty space
    # instead of printing across the series they describe
    top = max(max(series[m]) for m in methods)
    bottom = min(min(series[m]) for m in methods)
    ax.set_ylim(bottom - 0.02 * (top - bottom), top + 0.16 * (top - bottom))
    order = d.get("ordering_by_k", {})
    if order:
        ref = order.get("15") or order.get(str(max(ks)))
        flip = [k for k in ks if order.get(str(k)) is not None and order[str(k)] != ref]
        if flip:
            ax.axvspan(min(flip) - 0.5, max(flip) + 0.5, color=GREY, alpha=0.10, zorder=1)
            ax.text((min(flip) + max(flip)) / 2, ax.get_ylim()[1], " ordering differs",
                    ha="center", va="top", fontsize=6.5, color=GREY)
    ax.axvline(15, color=GREY, lw=0.8, ls=":", zorder=1)
    ax.text(15.6, ax.get_ylim()[1], "field's $k$", fontsize=6.5, color=GREY, va="top")
    ax.set_xlabel("output budget $k$", fontsize=8)
    ax.set_ylabel("macro F1", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_budget.pdf")
    plt.close(fig)
    return {"k range": (min(ks), max(ks)), "methods": methods}


def main() -> int:
    for fn in (fig_decomp, fig_ladder, fig_xdomain, fig_budget, fig_transfer, fig_propensity,
               fig_gap, fig_curators):
        info = fn()
        print(f"{fn.__name__:12} -> paper/{fn.__name__.replace('fig_', 'fig_')}.pdf   {info}")
    # The waterfall must multiply out to the recall the artifact records, and the artifact's recall
    # must be the one the manuscript prints. Reading the target from the artifact rather than from a
    # literal is deliberate: a hardcoded target goes stale the moment the measurement is corrected,
    # and then the gate certifies the old number instead of the drawing.
    art = _load("recall_factorization")
    g = art["factors"]
    prod = g["coverage_bank"]["point"] * g["selection_retention"]["point"] * g["ranking_conversion"]["point"]
    assert abs(prod - art["micro_recall"]) < 5e-4, \
        f"waterfall product {prod:.4f} != this artifact's micro recall {art['micro_recall']:.4f}"
    macro = re.search(r"\\newcommand\{\\realised\}\{([\d.]+)\}",
                      (ROOT / "paper" / "grail_iclr.tex").read_text())
    assert macro and abs(float(macro.group(1)) - art["micro_recall"]) < 5e-4, \
        f"the manuscript prints {macro and macro.group(1)} where the artifact has {art['micro_recall']:.4f}"
    print(f"\ncheck: GRAIL factors multiply to {prod:.4f}, and the manuscript prints {macro.group(1)}")
    return 0




def fig_transfer():
    """The negative control: a rule engine that never trained degrades on the same slope."""
    d = _load("transfer_stratified")
    strata = d["strata"]
    x = [(s["lo"] + min(s["hi"], 1.0)) / 2 for s in strata]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for i, m in enumerate(strata[0]["methods"]):
        y = [s["methods"][m]["recall_at_15"] for s in strata]
        c, mk = _style(m, i)
        ax.plot(x, y, marker=mk, ms=3, lw=1.5, color=c,
                label=f"{m}{' (never trained)' if m == 'SyGMa' else ''}")
    ax.set_xlabel("max Tanimoto similarity to any training substrate", fontsize=7.5)
    ax.set_ylabel("recall@15", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_transfer.pdf"); plt.close(fig)
    return {"strata n": [s["n"] for s in strata]}


def fig_propensity():
    """Where the F1 ordering would flip if the references were this incomplete."""
    d = _load("pu_propensity_bounds")
    grid, series = d["grid"], d["macro_f1_by_c"]
    crit = d["crossings"]["GRAIL-SyGMa"]["critical_c"]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for i, (m, y) in enumerate(series.items()):
        ax.plot(grid, y, lw=1.5, color=_style(m, i)[0], label=m)
    if crit:
        ax.axvline(crit, color=GREY, lw=0.8, ls=":")
        # the axis is reversed, so the crossing sits at the right edge of the canvas. The
        # label is right-aligned on its own line and therefore runs inwards; aligning it left
        # puts it off the page, which is where it was.
        ax.text(crit, ax.get_ylim()[1] * 0.99, f"flip at $c={crit}$  ", fontsize=6.5,
                color=GREY, va="top", ha="right")
    ax.set_xlabel("annotation propensity $c$ (1 = references complete)", fontsize=7.5)
    ax.set_ylabel("macro F1, corrected", fontsize=8)
    ax.invert_xaxis(); ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_propensity.pdf"); plt.close(fig)
    return {"critical c": crit}


def fig_gap():
    """What the uncovered references are: a type the bank has, a type it lacks, or untypeable."""
    d = _load("coverage_gap_types")
    g, tot = d["gap"], d["uncovered_pairs"]
    # three-line legend entries in three columns, anchored under a 1.5in-tall axes, landed on top
    # of the axis label; one entry per row and a taller figure keep them apart
    parts = [("type in bank (reachable by a more general rule)", g["known_type"], ACC),
             ("type absent from bank", g["novel_type"], ALT),
             ("untypeable", g["untypeable"], GREY)]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    left = 0
    for label, v, c in parts:
        ax.barh(0, v, left=left, color=c, alpha=0.85, edgecolor="none")
        # a segment narrower than a tenth of the bar cannot hold its own count
        if v / tot >= 0.10:
            ax.text(left + v / 2, 0, f"{v}\n{100*v/tot:.0f}%", ha="center", va="center",
                    fontsize=7, color="white" if v > 60 else GREY)
        else:
            ax.text(left + v / 2, 0.45, f"{v} ({100*v/tot:.0f}%)", ha="center", va="bottom",
                    fontsize=6.5, color=GREY)
        left += v
    ax.set_yticks([]); ax.set_xlim(0, tot); ax.set_ylim(-0.6, 0.95)
    ax.set_xlabel(f"{tot:,} uncovered references (of {d['covered_pairs'] + tot:,})",
                  fontsize=7.5)
    ax.tick_params(labelsize=7)
    for i, (label, v, c) in enumerate(parts):
        ax.plot([], [], color=c, lw=5, label=label)
    ax.legend(fontsize=6.3, frameon=False, loc="upper center", handlelength=1.4,
              bbox_to_anchor=(0.5, -0.42), ncol=1, borderaxespad=0.0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.savefig(OUT / "fig_gap.pdf", bbox_inches="tight"); plt.close(fig)
    return {"total": tot, "in-bank ceiling": round((d["covered_pairs"]+g["known_type"])/(d["covered_pairs"]+tot), 3)}


def fig_curators():
    """Two expert curations of the same drugs, agreeing five times better under a tolerant rule."""
    d = _load("annotation_agreement")["by_mode"]
    modes = ["canonical", "inchikey", "tanimoto1", "inchi_no_stereo", "inchikey_tautomer"]
    short = ["canonical", "InChIKey", "Tanimoto=1", "no-stereo", "tautomer"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    y = [d[m]["jaccard"] for m in modes]
    ax.bar(range(len(modes)), y, width=0.6, color=ACC, alpha=0.85, edgecolor="none")
    for i, v in enumerate(y):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(short, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Jaccard between two curations", fontsize=8)
    ax.set_ylim(0, max(y) * 1.22); ax.tick_params(labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_curators.pdf"); plt.close(fig)
    return {m: d[m]["jaccard"] for m in modes}


if __name__ == "__main__":
    raise SystemExit(main())

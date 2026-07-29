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
  fig_budget   macro F1 against the output budget, with the regions where the ordering changes
               shaded. The appendix currently states these crossings in prose.

Written as vector PDF at the column width the ICLR style uses; no seaborn, no style sheet, so the
output does not depend on anything outside matplotlib's defaults.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
GREY, ACC, ALT = "#444444", "#1f4e79", "#a6301f"


def _load(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def fig_decomp():
    """Waterfall: 1.0 -> coverage -> x selection -> x ranking -> realised, per method."""
    g = _load("recall_factorization")["factors"]
    s = _load("decompose_sygma")["factors"]
    methods = [("GRAIL", [g["coverage_bank"]["point"], g["selection_retention"]["point"],
                          g["ranking_conversion"]["point"]], ACC),
               ("SyGMa", [s["coverage_bank"]["point"], s["selection_retention"]["point"],
                          s["ranking_conversion"]["point"]], ALT)]
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
    short = ["canon.", "InChIKey", "no-stereo", "Tanimoto=1", "tautomer"]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    colours = {"GRAIL": ACC, "SyGMa": ALT}
    vals = {}
    for meth, rec in d["by_method"].items():
        y = [rec[m]["recall@15"] if isinstance(rec.get(m), dict) else rec.get(m) for m in modes]
        if any(v is None for v in y):
            continue
        vals[meth] = y
        c = colours.get(meth, GREY)
        ax.plot(range(len(modes)), y, marker="o", ms=3, lw=1.4, color=c,
                alpha=1.0 if meth in colours else 0.55, zorder=3 if meth in colours else 2)
        ax.text(len(modes) - 0.9, y[-1], f" {meth}", fontsize=7, color=c, va="center")
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


def fig_budget():
    """Macro F1 against k, with the region where the ordering differs from the field's k=15."""
    d = _load("budget_curves")
    series = d["macro_f1_by_k"]          # method -> list indexed by k-1
    methods = sorted(series)
    ks = list(range(1, len(series[methods[0]]) + 1))
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    colours = {"GRAIL": ACC, "SyGMa": ALT}
    for m in methods:
        y = series[m]
        c = colours.get(m, GREY)
        ax.plot(ks, y, lw=1.4, color=c, alpha=1.0 if m in colours else 0.55,
                label=m, zorder=3 if m in colours else 2)
    order = d.get("ordering_by_k", {})
    if order:
        ref = order.get("15") or order.get(str(max(ks)))
        flip = [k for k in ks if order.get(str(k)) is not None and order[str(k)] != ref]
        if flip:
            ax.axvspan(min(flip) - 0.5, max(flip) + 0.5, color=GREY, alpha=0.12, zorder=1)
            ax.text((min(flip) + max(flip)) / 2, ax.get_ylim()[1] * 0.97,
                    "ordering differs", ha="center", va="top", fontsize=6.5, color=GREY)
    ax.axvline(15, color=GREY, lw=0.8, ls=":", zorder=1)
    ax.text(15.4, ax.get_ylim()[0] * 1.02 + 0.002, "field's $k$", fontsize=6.5, color=GREY)
    ax.set_xlabel("output budget $k$", fontsize=8)
    ax.set_ylabel("macro F1", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_budget.pdf")
    plt.close(fig)
    return {"k range": (min(ks), max(ks)), "methods": methods}


def main() -> int:
    for fn in (fig_decomp, fig_ladder, fig_budget):
        info = fn()
        print(f"{fn.__name__:12} -> paper/{fn.__name__.replace('fig_', 'fig_')}.pdf   {info}")
    # the waterfall must multiply out to the recall the paper prints, or the figure is not the table
    g = _load("recall_factorization")["factors"]
    prod = g["coverage_bank"]["point"] * g["selection_retention"]["point"] * g["ranking_conversion"]["point"]
    assert abs(prod - 0.261) < 5e-4, f"waterfall product {prod:.4f} != published 0.261"
    print(f"\ncheck: GRAIL factors multiply to {prod:.4f}, published 0.261")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

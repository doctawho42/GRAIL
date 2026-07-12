#!/usr/bin/env python
"""Render the coverage->selection->ranking recall-decomposition waterfall -- the carrying
figure of the paper. Shows, on ONE common population (MICRO frame, tautomer-InChIKey),
how the deployed recall@15 is a product of three sequential factors:

    all true (U) -> coverage (bank) -> +selection (=oracle) -> +ranking (deployed)

Reads results/recall_factorization.json (produced by the recall-factorization analysis)
and writes docs/benchmark/factorization_waterfall.{png,svg}. No dataset or model needed;
values are read from the JSON at runtime so a re-run regenerates the figure.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

BAR_LABELS = [
    "all true (U)",
    "coverage\n(bank)",
    "+selection\n(=oracle)",
    "+ranking\n(deployed)",
]
BAR_COLORS = ["#bbbbbb", "#2c6fbb", "#f0ad4e", "#2ca02c"]


def main() -> int:
    data = json.loads((ROOT / "results" / "recall_factorization.json").read_text())

    n = data["n_substrates"]
    match = data["match"]
    k = data["k"]

    coverage = data["factors"]["coverage_bank"]["point"]
    coverage_lo = data["factors"]["coverage_bank"]["lo"]
    coverage_hi = data["factors"]["coverage_bank"]["hi"]
    oracle = data["oracle_recall"]
    deployed = data["micro_recall"]

    values = [1.0, coverage, oracle, deployed]
    xs = list(range(len(values)))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    yerr = [[coverage - coverage_lo], [coverage_hi - coverage]]
    bars = ax.bar(xs, values, color=BAR_COLORS, width=0.6, zorder=3, edgecolor="#333333", linewidth=0.8)
    ax.errorbar(xs[1], coverage, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=1.3,
                capsize=5, zorder=4)

    for x, v in zip(xs, values):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=10, fontweight="bold", zorder=5)

    # oracle rerank ceiling: dashed line at oracle_recall spanning oracle -> deployed bars,
    # so the coverage->oracle gap reads as SELECTION loss and oracle->deployed as RANKING loss.
    ax.hlines(oracle, xs[2] - 0.3, xs[3] + 0.3, colors="#a33", linestyles="--", lw=1.6, zorder=4,
               label=f"oracle rerank ceiling ({oracle:.3f})")

    ax.annotate("selection loss", xy=(1.5, (coverage + oracle) / 2), ha="center", va="center",
                fontsize=8.5, color="#555555", style="italic")
    ax.annotate("ranking loss", xy=(2.5, oracle + 0.035), ha="center", va="bottom",
                fontsize=8.5, color="#a33", style="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels(BAR_LABELS, fontsize=9.5)
    ax.set_ylabel(f"recall@{k} ({match}, micro)")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Where deployed recall is lost: coverage -> selection -> ranking\n"
                 f"(micro, {match}, n={n})", fontsize=11)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    for ext in ("png", "svg"):
        out = ROOT / "docs" / "benchmark" / f"factorization_waterfall.{ext}"
        fig.savefig(out, dpi=160)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

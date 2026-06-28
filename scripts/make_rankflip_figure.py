#!/usr/bin/env python
"""Render the GLORYx-37 rank-flip figure: recall@15 for each method across the five
matching protocols, ordered strict->lenient, to show the SyGMa<->MetaPredictor #1 flip.

Reads results/gloryx_eval.json (produced by scripts/eval_on_gloryx.py) and writes
docs/benchmark/rankflip.{png,svg}. No dataset or model needed.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# strict -> lenient (left to right): this ordering makes the crossover monotone-ish
PROToCOLS = [
    ("inchikey", "InChIKey\n(strict)"),
    ("canonical", "canonical\n(LAGOM)"),
    ("tanimoto1", "Tanimoto=1\n(MetaTrans)"),
    ("inchi_no_stereo", "no-stereo\n(GLORYx)"),
    ("inchikey_tautomer", "tautomer\n(ours)"),
]
# (method, color, linewidth, z, marker) — the flipping pair bold, the rest faint
STYLE = {
    "MetaPredictor": ("#d62728", 2.6, 5, "o"),
    "SyGMa": ("#1f77b4", 2.6, 5, "s"),
    "BioTransformer": ("#7f7f7f", 1.6, 3, "^"),
    "GRAIL": ("#2ca02c", 1.6, 3, "D"),
}


def main() -> int:
    report = json.loads((ROOT / "results" / "gloryx_eval.json").read_text())
    bm = report["by_method"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = list(range(len(PROToCOLS)))
    for method, (color, lw, z, mk) in STYLE.items():
        ys = [bm[method][mode]["recall@15"] for mode, _ in PROToCOLS]
        ax.plot(xs, ys, color=color, lw=lw, zorder=z, marker=mk, ms=7,
                label=method, alpha=1.0 if lw > 2 else 0.75)

    # annotate the #1 flip between SyGMa and MetaPredictor
    mp = [bm["MetaPredictor"][m]["recall@15"] for m, _ in PROToCOLS]
    # crossover happens between 'tanimoto1' (idx2, SyGMa leads) and 'no-stereo' (idx3, MP leads)
    ax.axvspan(2.5, 4.5, color="#ffe9a8", alpha=0.45, zorder=0)
    ax.text(3.5, 0.30, "MetaPredictor #1", ha="center", va="center",
            fontsize=9, color="#a33", style="italic")
    ax.text(1.0, 0.30, "SyGMa #1", ha="center", va="center",
            fontsize=9, color="#15527a", style="italic")
    # strict-InChIKey collapse callout for MetaPredictor
    ax.annotate("stereo-strict\ncollapse 1.4×", xy=(0, mp[0]), xytext=(0.15, 0.40),
                fontsize=8, color="#a33",
                arrowprops=dict(arrowstyle="->", color="#a33", lw=1))

    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in PROToCOLS], fontsize=8.5)
    ax.set_ylabel("recall@15  (GLORYx-37)")
    ax.set_ylim(0.05, 0.56)
    ax.set_title("How you match decides who wins:\nthe #1 method flips across matching protocols",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    for ext in ("png", "svg"):
        out = ROOT / "docs" / "benchmark" / f"rankflip.{ext}"
        fig.savefig(out, dpi=160)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

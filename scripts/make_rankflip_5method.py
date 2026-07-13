"""Figure: the 5-method leaderboard reorders under the match protocol (§11 rank-flip).

Reads results/match_sensitivity_5method.json and draws recall@15 for all five methods across the
five match protocols as a slopegraph; where two methods' lines cross, the leaderboard rank flips.
The two documented flips (GRAIL x BioTransformer; MetaTrans x SyGMa) show as crossings.

Writes docs/benchmark/rankflip_5method.{svg,png}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmark"
DATA = ROOT / "results" / "match_sensitivity_5method.json"

# short protocol labels in the paper's order
PROTO_LABELS = {
    "canonical": "canonical",
    "inchikey": "InChIKey",
    "inchi_no_stereo": "InChI\nno-stereo",
    "tanimoto1": "Tanimoto=1",
    "inchikey_tautomer": "InChIKey\ntautomer",
}
COLORS = {
    "GRAIL": "#1f77b4",
    "SyGMa": "#d62728",
    "BioTransformer": "#2ca02c",
    "MetaPredictor": "#9467bd",
    "MetaTrans": "#ff7f0e",
}


def main() -> int:
    d = json.loads(DATA.read_text())
    protocols = d["modes"]
    by_method = d["by_method"]
    xs = list(range(len(protocols)))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # order methods by their mean recall so the legend reads top-to-bottom
    methods = sorted(by_method, key=lambda m: -sum(by_method[m][p]["recall@15"] for p in protocols))
    for m in methods:
        ys = [by_method[m][p]["recall@15"] for p in protocols]
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=2, color=COLORS.get(m, "#555"), label=m, zorder=3)
        ax.annotate(m, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=COLORS.get(m, "#555"), fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([PROTO_LABELS.get(p, p) for p in protocols], fontsize=9)
    ax.set_ylabel("recall@15", fontsize=11)
    ax.set_xlim(-0.3, len(protocols) - 1 + 1.4)
    ax.set_title("Leaderboard reorders under the structure-match protocol", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    n = d.get("n_substrates")
    ax.text(0.0, -0.22,
            f"GRAIL & SyGMa on n≈1170; tier-2 (BioTransformer, MetaPredictor, MetaTrans) on the "
            f"n=150 shared subset (match_sensitivity n={n}).",
            transform=ax.transAxes, fontsize=7.5, color="#555")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"rankflip_5method.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {OUT/'rankflip_5method.svg'} (+ .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Figure 1 for the ICLR draft: the match convention reorders the leaderboard, twice.

Two panels, deliberately parallel so the eye compares them directly:
  (a) INTERNAL  -- 5 methods re-scored under 5 matching conventions on the shared subset.
  (b) EXTERNAL  -- 4 methods on the GLORYx hold-out, strict InChIKey vs tautomer-InChIKey.

Both are slope plots because the claim is about *reordering*, and a slope plot makes a crossing
visible as a crossing. Panel (b) additionally annotates the per-method gain with its paired
bootstrap CI, since that is what promotes the external panel from description to evidence.

Values are read from the committed artifacts, never hard-coded, so the figure cannot drift from
the tables in the text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INTERNAL = ROOT / "results" / "match_sensitivity_fulln_paired.json"
EXTERNAL = ROOT / "results" / "gloryx_criterion_ladder.json"
OUT = ROOT / "paper" / "fig_rankflip.pdf"

PROTOS = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
PLABEL = {"canonical": "canon.", "inchikey": "InChIKey", "inchi_no_stereo": "no-stereo",
          "tanimoto1": "Tanimoto=1", "inchikey_tautomer": "tautomer"}
# colourblind-safe (Okabe-Ito), and distinguishable in greyscale print
COLOR = {"GRAIL": "#000000", "SyGMa": "#0072B2", "BioTransformer": "#D55E00",
         "MetaTrans": "#009E73", "MetaPredictor": "#CC79A7"}


def main() -> int:
    internal = json.loads(INTERNAL.read_text())["recall_at_15"]
    ext = json.loads(EXTERNAL.read_text())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5.5, 2.7), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ---- (a) internal, 5 protocols ----
    xs = range(len(PROTOS))
    for m, d in internal.items():
        ys = [d[p] for p in PROTOS]
        ax1.plot(xs, ys, marker="o", ms=3.4, lw=1.5, color=COLOR.get(m, "#555"), zorder=3,
                 label=m)
        ax1.annotate(m, (len(PROTOS) - 1, ys[-1]), xytext=(4, -2), textcoords="offset points",
                     fontsize=6.0, color=COLOR.get(m, "#555"), va="center")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([PLABEL[p] for p in PROTOS], fontsize=6.4, rotation=20, ha="right")
    ax1.set_ylabel("recall@15", fontsize=8)
    ax1.set_xlim(-0.25, len(PROTOS) - 1 + 1.5)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.grid(axis="y", ls=":", alpha=0.45, lw=0.6)
    ax1.set_title(r"(a) full test split ($n{=}1{,}170$)", fontsize=8)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # ---- (b) external GLORYx, 2 protocols, with gain CIs ----
    rec, sens = ext["recall"], ext["steps"]
    for m, v in rec.items():
        ys = [v["inchikey"], v["inchi_no_stereo"], v["inchikey_tautomer"]]
        ax2.plot([0, 1, 2], ys, marker="o", ms=3.4, lw=1.5, color=COLOR.get(m, "#555"), zorder=3)

    # Declutter the right-hand labels: MetaPredictor and SyGMa converge to ~0.50 under tautomer
    # matching (that convergence IS the panel's point), so naive annotation overlaps them and
    # hides the crossing. Push labels apart vertically while keeping their order.
    ends = sorted(((v["inchikey_tautomer"], m) for m, v in rec.items()), reverse=True)
    lo, hi = min(y for y, _ in ends), max(y for y, _ in ends)
    gap = 0.075 * (hi - lo)
    placed = []
    for y, m in ends:
        yy = y
        for py in placed:
            if abs(yy - py) < gap:
                yy = py - gap
        placed.append(yy)
        g = sens[m]["stereo"]
        ax2.annotate(f"{m} stereo {g['gain']:+.3f}",
                     (2, y), xytext=(2.08, yy), textcoords=("data", "data"),
                     fontsize=5.2, color=COLOR.get(m, "#555"), va="center",
                     arrowprops=dict(arrowstyle="-", lw=0.4,
                                     color=COLOR.get(m, "#555"), alpha=0.55,
                                     shrinkA=0, shrinkB=2))
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["InChIKey", "no-stereo", "tautomer"], fontsize=6.4)
    ax2.set_xlim(-0.12, 5.2)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.grid(axis="y", ls=":", alpha=0.45, lw=0.6)
    ax2.set_title(r"(b) external GLORYx ($n{=}37$), by rung", fontsize=8)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}", flush=True)

    # provenance echo so the caption's numbers can be eyeballed against the artifacts
    print("  full-split tautomer-vs-canonical spread:",
          {m: round(d["inchikey_tautomer"] - d["canonical"], 3) for m, d in internal.items()}, flush=True)
    print("  external stereo steps:", {m: v["stereo"]["gain"] for m, v in sens.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

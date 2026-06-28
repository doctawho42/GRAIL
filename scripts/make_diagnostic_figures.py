#!/usr/bin/env python3
"""Generate the Stage-1 diagnostic figures from existing measurements (no model runs).

Currently: the data-scaling / saturation curve with the SyGMa baseline and rule-bank
coverage ceiling as reference lines -- shows GRAIL plateaus well below both, i.e. the gap is
ranking, not data. Writes PNG + SVG under docs/benchmark/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)

# recall@15, tautomer-canonical InChIKey, rank-only, single-mode filter (clean split).
train_sizes = [400, 2418, 4787]
grail_recall = [0.10, 0.330, 0.334]
SYGMA = 0.558       # same-split SyGMa baseline (run_benchmark)
CEILING = 0.718     # rule-bank single-step recall ceiling (measured)


def main() -> int:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(CEILING, ls="--", lw=1.3, color="#888", label=f"rule-bank ceiling ({CEILING:.3f})")
    ax.axhline(SYGMA, ls="--", lw=1.3, color="#c0392b", label=f"SyGMa baseline ({SYGMA:.3f})")
    ax.plot(train_sizes, grail_recall, "o-", lw=2.2, ms=7, color="#2c6fbb", label="GRAIL (ensemble)")
    for x, y in zip(train_sizes, grail_recall):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=9)

    ax.annotate("plateau (+0.004 for 2× data)", (4787, 0.334), xytext=(2600, 0.20),
                fontsize=9, color="#2c6fbb",
                arrowprops=dict(arrowstyle="->", color="#2c6fbb", lw=1))

    ax.set_xlabel("training substrates")
    ax.set_ylabel("recall@15 (tautomer-canonical InChIKey)")
    ax.set_title("Data scaling saturates; the gap to SyGMa is ranking, not coverage")
    ax.set_ylim(0, 0.78)
    ax.set_xlim(0, 5200)
    ax.grid(alpha=0.25)
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"scaling_curve.{ext}", dpi=160)
    print(f"wrote {OUT/'scaling_curve.png'} (+ .svg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

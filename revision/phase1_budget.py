#!/usr/bin/env python3
"""Phase 1 item 2: recall against the number of candidates a system actually emits.

Reads revision/T_main.csv and writes revision/F_budget.pdf. This is the retrieval view a reader of
this literature expects, and it replaces nothing: the decomposition stays where it is and this is
shown beside it.

One line per system, one panel per matching criterion, and one row of panels per population, so a
cell that exists on the comparison set and not on the whole evaluated set is visibly missing rather
than quietly averaged in. A system with no cells on a population is absent from that panel: drawing
a comparator that was never run at zero recall would read as a measured loss instead of as no data,
which is the error this figure is most likely to be misread as making.

The x axis is the mean number of candidates the system emits at that budget, not the budget itself.
A method that emits three structures when asked for fifteen is not competing at fifteen, and the
budget axis hides that while this one shows it.

The error bars are the per-cell intervals from T_main.csv, which are computed by
revision/phase1_tmain.py rather than taken from the manuscript: the repository stores no interval
for a single arm's recall, only for contrasts.

    python revision/phase1_budget.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "revision" / "T_main.csv"
OUT = ROOT / "revision" / "F_budget.pdf"

NUMERIC = ("k", "recall", "ci_lo", "ci_hi", "mean_emitted_at_k", "mean_emitted_untruncated",
           "n_substrates", "n_references")
# Presentation order only: strictest comparison first, the default last, as the supplement's own
# criteria table orders them. Any criterion the table carries and this list does not is appended.
CRITERION_ORDER = ("exact", "canonical", "inchikey", "inchi_no_stereo", "tanimoto1",
                   "inchikey_tautomer")
POPULATION_ORDER = ("comparison291", "evaluated1170")
# GRAIL's two arms first so the eye finds them, then the comparators alphabetically.
SYSTEM_ORDER = ("whole bank", "trained budget")


def load_rows(path=TABLE):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for field in NUMERIC:
            if field in r and r[field] != "":
                value = float(r[field])
                r[field] = int(value) if field in ("k", "n_substrates", "n_references") else value
    return rows


def panels(rows):
    """The criteria the data actually carries, in presentation order."""
    present = {r["criterion"] for r in rows}
    ordered = [c for c in CRITERION_ORDER if c in present]
    return ordered + sorted(present - set(ordered))


def populations(rows):
    present = {r["population"] for r in rows}
    ordered = [p for p in POPULATION_ORDER if p in present]
    return ordered + sorted(present - set(ordered))


def systems(rows):
    present = {r["system"] for r in rows}
    ordered = [s for s in SYSTEM_ORDER if s in present]
    return ordered + sorted(present - set(ordered))


def series(rows, criterion, population):
    """{system: [{k, x, y, lo, hi} ...]} in budget order, omitting systems with no cells here."""
    out: dict = {}
    for r in rows:
        if r["criterion"] != criterion or r["population"] != population:
            continue
        out.setdefault(r["system"], []).append(
            {"k": r["k"], "x": r["mean_emitted_at_k"], "y": r["recall"],
             "lo": r.get("ci_lo"), "hi": r.get("ci_hi")})
    for points in out.values():
        points.sort(key=lambda p: p["k"])
    return out


def main() -> int:
    if not TABLE.exists():
        print(f"REFUSING: {TABLE.relative_to(ROOT)} does not exist; run revision/phase1_tmain.py")
        return 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows()
    cols, pops = panels(rows), populations(rows)
    all_systems = systems(rows)
    colours = plt.get_cmap("tab10")
    colour_of = {s: colours(i % 10) for i, s in enumerate(all_systems)}

    fig, axes = plt.subplots(len(pops), len(cols), squeeze=False,
                             figsize=(3.1 * len(cols), 3.0 * len(pops)), sharey="row")
    drawn = []
    for i, population in enumerate(pops):
        for j, criterion in enumerate(cols):
            ax = axes[i][j]
            data = series(rows, criterion, population)
            for system in all_systems:
                points = data.get(system)
                if not points:
                    continue
                xs = [p["x"] for p in points]
                ys = [p["y"] for p in points]
                lo = [p["y"] - p["lo"] for p in points]
                hi = [p["hi"] - p["y"] for p in points]
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", markersize=2.6, linewidth=1.1,
                            elinewidth=0.6, capsize=1.5, color=colour_of[system], label=system)
                drawn.append((population, criterion, system, len(points)))
            ax.set_xscale("log")
            ax.grid(alpha=0.25, linewidth=0.5)
            if i == 0:
                ax.set_title(criterion, fontsize=9)
            if j == 0:
                n = next((r["n_substrates"] for r in rows if r["population"] == population), "?")
                ax.set_ylabel(f"{population}\n(n={n})\nrecall", fontsize=8)
            if i == len(pops) - 1:
                ax.set_xlabel("mean candidates emitted", fontsize=8)
            ax.tick_params(labelsize=7)
            absent = [s for s in all_systems if s not in data]
            if absent:
                ax.text(0.98, 0.02, "absent: " + ", ".join(absent), transform=ax.transAxes,
                        fontsize=5.5, ha="right", va="bottom", color="0.35")

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen, h, l = set(), [], []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            h.append(handle)
            l.append(label)
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 7), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  panels: {len(pops)} populations x {len(cols)} criteria")
    print(f"  populations: {pops}")
    print(f"  criteria   : {cols}")
    for population in pops:
        for criterion in cols:
            data = series(rows, criterion, population)
            missing = [s for s in all_systems if s not in data]
            print(f"  {population:14} {criterion:18} lines={len(data):>2}"
                  + (f"  absent: {', '.join(missing)}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

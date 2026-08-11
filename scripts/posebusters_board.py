#!/usr/bin/env python3
"""The same instrument on a leaderboard where the reversal is already known.

Docking is the one domain where a field has already published the finding this paper generalises:
scoring a predicted pose by distance alone and scoring it by distance *and* physical validity do not
rank methods the same way. PoseBusters made that argument and released the per-pose table behind it,
which makes it the positive control the instrument needs. If the dominance order recovers a
reversal here, on data whose own authors named it, then the reversals it reports on boards nobody
has audited are the same measurement rather than a new and unvouched-for one.

Two axes, both choices a docking paper can make without reporting it:

  criterion       what counts as recovering the pose. Distance alone at the field's default
                  tolerance is the thirty-year convention; distance and validity is the source's
                  own; a tighter tolerance and a chemistry-only validity screen are both in print.
  post-processing whether the pose is energy-minimised before it is scored.

Two conventions have to be declared because the released table forces them. The two flatness checks
are recorded only in the un-minimised arm, so a criterion that used them would not exist in half the
grid; they are dropped from every cell rather than from one, which keeps the criterion the same
object in each. A row with no pose to score is a miss, not an absent observation: dropping those
rows would score each method on the subset it managed to produce output for.

The crystal ligand ships as an eighth row and is not a method: it is the reference re-scored through
the same pipeline, has no minimised arm, and is excluded. The astex set is a second population
rather than a second cell, and the grid holds the population fixed; the benchmark set is the one the
source leads with, and the astex rows are left out and said to be left out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from robust_order import N_BOOT, SEED, analyse  # noqa: E402

# The released table's own column order: the loading precondition, then eleven intramolecular
# checks, then six protein- and cofactor-contact checks, then the distance verdict.
LOAD = "docked_ligand_successfully_loaded"
FLATNESS = ("aromatic_ring_flatness_passes", "double_bond_flatness_passes")
INTRA = ("molecule_passes_rdkit_sanity_check", "molecular_formula_preserved",
         "molecular_bonds_preserved", "sp3_stereochemistry_preserved",
         "double_bond_stereochemistry_preserved", "bond_lengths_within_bounds",
         "bond_angles_within_bounds", "no_internal_clashes", "energy_ratio_within_threshold")
INTER = ("no_clashes_with_protein", "no_clashes_with_organic_cofactors",
         "no_clashes_with_inorganic_cofactors", "no_volume_clash_with_protein",
         "no_volume_clash_with_organic_cofactors", "no_volume_clash_with_inorganic_cofactors")

CRITERIA = {
    "rmsd2": ("distance alone at the field's default tolerance", 2.0, ()),
    "rmsd2+valid": ("distance and every recorded validity check", 2.0, (LOAD,) + INTRA + INTER),
    "rmsd1": ("distance alone at half the tolerance", 1.0, ()),
    "rmsd2+chem": ("distance and the intramolecular checks only", 2.0, (LOAD,) + INTRA),
}
POSTPROC = ("none", "energy minimization")
PUBLISHED_CELL = ("rmsd2+valid", "none")
CONTROL_ROW = "crystal_structures"
POPULATION = "posebuster"


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def _truthy(frame: pd.DataFrame, columns) -> np.ndarray:
    """Every named check passed. A missing verdict is a failure, because it means no pose."""
    if not columns:
        return np.ones(len(frame), dtype=bool)
    out = np.ones(len(frame), dtype=bool)
    for c in columns:
        out &= (frame[c].astype("object") == True).to_numpy(dtype=bool)  # noqa: E712
    return out


def build_hits(table: pd.DataFrame, population: str = POPULATION) -> tuple[dict, list[str], list[str], list]:
    pop = table[table["dataset"] == population]
    items = sorted(pop["pdb_id"].unique())
    systems = sorted(m for m in pop["method"].unique() if m != CONTROL_ROW)
    cells = [(c, p) for c in CRITERIA for p in POSTPROC]

    hits = {}
    for name in systems:
        for criterion, post in cells:
            _, tolerance, checks = CRITERIA[criterion]
            arm = pop[(pop["method"] == name) & (pop["post-processing"] == post)]
            arm = arm.set_index("pdb_id").reindex(items)
            near = (arm["rmsd"] <= tolerance).fillna(False).to_numpy(dtype=bool)
            hits[(name, (criterion, post))] = (near & _truthy(arm, checks)).astype(float)
    return hits, systems, items, cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/external/posebusters_paper_results.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "robust_order_posebusters.json"))
    # The source distributes two sets. A population is not a cell -- the grid holds the items fixed
    # -- so the second is run as a separate board and reported as one, which is also the only way to
    # see how much of a verdict is the items rather than the criterion.
    ap.add_argument("--population", default=POPULATION, choices=["posebuster", "astex"])
    args = ap.parse_args()

    raw = Path(args.csv).read_bytes()
    table = pd.read_csv(args.csv)
    dropped = int((table["dataset"] != args.population).sum())
    hits, systems, items, cells = build_hits(table, args.population)

    sub_grids = {"criteria only, at the published post-processing":
                     [(c, PUBLISHED_CELL[1]) for c in CRITERIA],
                 "post-processing only, at the published criterion":
                     [(PUBLISHED_CELL[0], p) for p in POSTPROC],
                 "the product": cells}
    rep = analyse(hits, systems, cells, PUBLISHED_CELL, sub_grids)
    rep["config"] = {
        **_code_version(), "n_boot": N_BOOT, "seed": SEED, "n_items": len(items),
        "source": "PoseBusters, Zenodo 8278563, posebusters_paper_results.csv, CC-BY-4.0",
        # the file is external and not redistributed here, so the artifact carries its digest and a
        # reader can confirm they are reading the same table this was computed from
        "source_sha256": hashlib.sha256(raw).hexdigest(), "source_bytes": len(raw),
        "population": f"the {args.population} set; {dropped} rows of the other set excluded, a "
                      f"second population rather than a second cell",
        "excluded_rows": f"{CONTROL_ROW}: the re-scored reference, not a method, and it has no "
                         f"minimised arm",
        "criteria": {k: v[0] for k, v in CRITERIA.items()},
        "declared_conventions": [
            f"the flatness checks {FLATNESS} are recorded only under post-processing 'none', so "
            f"they are dropped from every cell rather than from one",
            "a row with no scorable pose counts as a miss, not as an absent observation"],
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"{args.population}: {rep['n_systems']} systems, {rep['n_cells']} cells, "
          f"{rep['n_pairs']} pairs, {len(items)} items")
    print(f"  published order at {rep['published_cell']}: {' > '.join(rep['published_order'])}")
    print(f"  survive every cell:            {rep['n_dominating']}/{rep['n_pairs']} "
          f"= {rep['robustness']} {rep['robustness_ci95']}")
    print(f"  and separated in every cell:   {rep['n_separated_in_every_cell']}/{rep['n_pairs']}")
    print(f"  reversed with an interval:     {rep['reversed_with_an_interval']}")
    print(f"  unresolved, neither way:       {rep['n_unresolved']}: "
          + ", ".join(rep["unresolved"]))
    print(f"  tiers: {rep['tiers_distinguished']} {rep['tiers_ci95']} of {rep['n_systems']}")
    print(f"  distinct orderings across the grid: {rep['distinct_orderings_across_the_grid']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

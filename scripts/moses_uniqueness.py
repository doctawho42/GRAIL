#!/usr/bin/env python3
"""Does the matching criterion move the ordering of published generative models?

MOSES reports Unique@k as distinct canonical SMILES among a model's generations. Canonical SMILES
does not normalise tautomers, so a model that emits both forms of one compound is credited with
two distinct molecules. Models differ in how much they do this, which is the same mechanism that
reorders the metabolite leaderboard.

Frozen published outputs, re-scored; nothing is retrained.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
KS = [1000, 10000]
N_BOOT, SEED = 2000, 0
OUT = ROOT / "results" / "moses_uniqueness.json"


def main() -> int:
    samples = json.loads((ROOT / "results" / "moses_samples.json").read_text())
    rep = {"models": sorted(samples), "modes": MODES, "ks": KS,
           "n_generated": {m: len(v) for m, v in samples.items()},
           "uniqueness": {}, "ranking": {}}

    for mode in MODES:
        table = json.loads((ROOT / "results" / "moses_keys" / f"{mode}.json").read_text())
        rep["uniqueness"][mode] = {}
        for m, mols in samples.items():
            keys = [table.get(s) for s in mols]
            keys = [k for k in keys if k]
            entry = {}
            for k in KS:
                sub = keys[:k]
                entry[f"unique@{k}"] = round(len(set(sub)) / len(sub), 4) if sub else 0.0
            rng = np.random.default_rng(SEED)
            arr = np.array(keys, dtype=object)
            n = min(KS[-1], len(arr))
            draws = [len(set(arr[rng.integers(0, len(arr), n)])) / n for _ in range(N_BOOT)]
            entry["ci95_at_%d" % KS[-1]] = [round(float(np.quantile(draws, .025)), 4),
                                            round(float(np.quantile(draws, .975)), 4)]
            rep["uniqueness"][mode][m] = entry
        order = sorted(samples, key=lambda m: -rep["uniqueness"][mode][m][f"unique@{KS[-1]}"])
        rep["ranking"][mode] = order
        print(f"{mode:20} " + "  ".join(f"{m}:{rep['uniqueness'][mode][m][f'unique@{KS[-1]}']:.3f}"
                                        for m in order), flush=True)

    # Compare EVERY criterion against the base, not just the two endpoints: the criteria are not
    # ordered by tolerance in a way that makes canonical-vs-tautomer an upper bound on movement,
    # and the intermediate stereo-blind rung is exactly where this set moves.
    base = rep["ranking"][MODES[0]]
    rep["models_that_moved_by_mode"] = {
        mode: [m for m in base if base.index(m) != rep["ranking"][mode].index(m)]
        for mode in MODES[1:]
    }
    moved = sorted({m for v in rep["models_that_moved_by_mode"].values() for m in v})
    rep["ordering_changed"] = bool(moved)
    rep["models_that_moved"] = moved
    for mode in MODES:
        print(f"{mode:20} {rep['ranking'][mode]}", flush=True)
    print(f"ordering changed: {rep['ordering_changed']}  (moves under: "
          f"{[k for k, v in rep['models_that_moved_by_mode'].items() if v]})", flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

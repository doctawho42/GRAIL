#!/usr/bin/env python3
"""The external GLORYx table under all five matching criteria, not the three-rung ladder.

`gloryx_criterion_ladder.py` scores the external set under inchikey -> inchi_no_stereo ->
inchikey_tautomer, which is enough to attribute the gain to the stereo step but not enough to say
how many of the five defensible criteria separate the methods at all. The internal table
(`match_sensitivity_*.json`) carries all five; the external one did not, so a claim about "one rung
of five" had no artifact behind it.

This re-scores the same frozen predictions, the same loaders and the same matcher under all five
criteria, on both populations the paper uses: the whole 37-drug set, and the 13 drugs that are not
in GRAIL's training or validation split (the contamination found by `external_overlap_audit.json`;
the seen-substrate keys are read from `results/train_val_substrate_keys.json` so this runs without
the gitignored SDFs). It reports, per criterion, each method's recall@15 and the paired
MetaPredictor - SyGMa difference with a bootstrap interval, plus the stereo step per method.

Nothing here is a new run: only the equivalence relation changes across columns.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import RDLogger

from grail_metabolism.metrics import _tautomer_inchikey
from scripts.gloryx_rank_flip_ci import (
    DATA, PRED_FILES, load_gloryx, sygma_predict, per_substrate_recall, boot,
)

RDLogger.DisableLog("rdApp.*")

CRITERIA = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
K, N_BOOT, SEED = 15, 10000, 0
SEEN_KEYS = ROOT / "results" / "train_val_substrate_keys.json"
OUT = ROOT / "results" / "gloryx_criterion_grid.json"


def tk(smiles: str):
    try:
        return _tautomer_inchikey(smiles)
    except Exception:
        return None


def main() -> int:
    reals = load_gloryx(DATA / "gloryx_test.json")
    allsubs = sorted(s for s in reals if reals[s])
    seen = set(json.loads(SEEN_KEYS.read_text()))
    clean = [s for s in allsubs if tk(s) not in seen]
    print(f"n_all={len(allsubs)}  n_clean={len(clean)}", flush=True)

    methods = {n: {s: json.loads(p.read_text()).get(s, []) for s in allsubs}
               for n, p in PRED_FILES.items()}
    methods["SyGMa"] = sygma_predict(allsubs, K)

    idx = {s: i for i, s in enumerate(allsubs)}
    keep = np.array([idx[s] for s in clean])

    vec = {(n, c): per_substrate_recall(pr, reals, allsubs, c, K)
           for n, pr in methods.items() for c in CRITERIA}
    for (n, c), v in vec.items():
        if float(v.max()) == 0.0:
            raise SystemExit(f"ERROR: '{n}' scores 0 everywhere under '{c}' -- broken adapter.")

    import rdkit
    rep = {
        "k": K, "n_boot": N_BOOT, "seed": SEED, "rdkit_version": rdkit.__version__,
        "criteria": CRITERIA,
        "populations": {
            "all37": {"n": len(allsubs),
                      "note": "the whole GLORYx set; not drawn with a cap and seed"},
            "clean13": {"n": len(clean),
                        "note": "GLORYx drugs absent from GRAIL's train and val splits, keyed by "
                                "tautomer InChIKey against results/train_val_substrate_keys.json"},
        },
        "recall": {}, "stereo_step": {}, "pairwise_delta": {}, "stereo_interaction": {},
    }

    for pop, sel in (("all37", None), ("clean13", keep)):
        take = (lambda v: v) if sel is None else (lambda v: v[sel])
        rep["recall"][pop] = {
            n: {c: round(float(take(vec[(n, c)]).mean()), 4) for c in CRITERIA} for n in methods}
        rep["stereo_step"][pop] = {}
        for n in methods:
            d = take(vec[(n, "inchi_no_stereo")]) - take(vec[(n, "inchikey")])
            m, lo, hi = boot(d, N_BOOT, SEED)
            rep["stereo_step"][pop][n] = {
                "gain": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
        rep["pairwise_delta"][pop] = {}
        for a, b in combinations(methods, 2):
            entry = {}
            for c in CRITERIA:
                m, lo, hi = boot(take(vec[(a, c)]) - take(vec[(b, c)]), N_BOOT, SEED)
                entry[c] = {"delta_a_minus_b": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                            "verdict": "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."}
            rep["pairwise_delta"][pop][f"{a}_vs_{b}"] = entry
        rep["stereo_interaction"][pop] = {}
        for a, b in combinations(methods, 2):
            d = ((take(vec[(a, "inchi_no_stereo")]) - take(vec[(a, "inchikey")]))
                 - (take(vec[(b, "inchi_no_stereo")]) - take(vec[(b, "inchikey")])))
            m, lo, hi = boot(d, N_BOOT, SEED)
            rep["stereo_interaction"][pop][f"{a}_extra_gain_over_{b}"] = {
                "mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "verdict": "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."}

    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}")
    for pop in ("all37", "clean13"):
        print(f"\n=== {pop} (n={rep['populations'][pop]['n']}) recall@{K} ===")
        print(f"{'method':>15} | " + " | ".join(f"{c:>17}" for c in CRITERIA))
        for n in methods:
            print(f"{n:>15} | " + " | ".join(
                f"{rep['recall'][pop][n][c]:>17.4f}" for c in CRITERIA))
        mp = rep["pairwise_delta"][pop]["MetaPredictor_vs_SyGMa"]
        print("  MetaPredictor - SyGMa:")
        for c in CRITERIA:
            e = mp[c]
            print(f"    {c:>18}: {e['delta_a_minus_b']:+.4f} "
                  f"[{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}] {e['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

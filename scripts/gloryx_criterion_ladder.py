#!/usr/bin/env python3
"""Decompose the external InChIKey -> tautomer gain into its stereo and tautomer components.

The recommended criterion (inchikey_tautomer) relaxes stereochemistry AND tautomerism at once,
so a two-column table cannot say which one moves the methods. This inserts the paper's own
inchi_no_stereo criterion as the intermediate rung and reports both steps with paired bootstrap
intervals, reusing the same loaders, matcher and prediction files as gloryx_rank_flip_ci.py.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gloryx_rank_flip_ci import (
    DATA, PRED_FILES, load_gloryx, sygma_predict, per_substrate_recall, boot,
)

LADDER = ["inchikey", "inchi_no_stereo", "inchikey_tautomer"]
OUT = ROOT / "results" / "gloryx_criterion_ladder.json"
K, N_BOOT, SEED = 15, 10000, 0


def main() -> int:
    reals = load_gloryx(DATA / "gloryx_test.json")
    subs = sorted(s for s in reals if reals[s])

    methods = {n: {s: json.loads(p.read_text()).get(s, []) for s in subs}
               for n, p in PRED_FILES.items()}
    methods["SyGMa"] = sygma_predict(subs, K)

    vec = {(n, c): per_substrate_recall(pr, reals, subs, c, K)
           for n, pr in methods.items() for c in LADDER}
    for (n, c), v in vec.items():
        if float(v.max()) == 0.0:
            raise SystemExit(f"ERROR: '{n}' scores 0 everywhere under '{c}' -- broken adapter.")

    import rdkit
    rep = {"n_substrates": len(subs), "k": K, "n_boot": N_BOOT, "seed": SEED,
           "rdkit_version": rdkit.__version__,
           "ladder": LADDER, "recall": {}, "steps": {}}
    print(f"n={len(subs)}, recall@{K}\n")
    print(f"{'method':>15} | " + " | ".join(f"{c:>18}" for c in LADDER)
          + " | {:>24} | {:>24}".format("stereo step", "tautomer residual"))
    for n in methods:
        v = {c: vec[(n, c)] for c in LADDER}
        rep["recall"][n] = {c: round(float(v[c].mean()), 4) for c in LADDER}
        st = v["inchi_no_stereo"] - v["inchikey"]
        ta = v["inchikey_tautomer"] - v["inchi_no_stereo"]
        _, lo_s, hi_s = boot(st, N_BOOT, SEED)
        _, lo_t, hi_t = boot(ta, N_BOOT, SEED)
        rep["steps"][n] = {
            "stereo": {"gain": round(float(st.mean()), 4), "ci95": [round(lo_s, 4), round(hi_s, 4)]},
            "tautomer": {"gain": round(float(ta.mean()), 4), "ci95": [round(lo_t, 4), round(hi_t, 4)]},
        }
        print(f"{n:>15} | " + " | ".join(f"{rep['recall'][n][c]:>18.4f}" for c in LADDER)
              + f" | {st.mean():+.4f} [{lo_s:+.3f},{hi_s:+.3f}]"
              + f" | {ta.mean():+.4f} [{lo_t:+.3f},{hi_t:+.3f}]")

    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

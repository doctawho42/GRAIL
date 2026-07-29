#!/usr/bin/env python3
"""Is the external GLORYx result still there once GRAIL's own training drugs are removed?

The paper reports the criterion ladder on 37 GLORYx drugs as an external replication. Checking the
two curations against each other turned up something that check was not looking for: 19 of those 37
parents are substrates in GRAIL's training split and 5 more are in validation. Only 13 are unseen.

The contamination is one-sided, which is the part that matters. SyGMa and BioTransformer do not
train on this corpus and MetaPredictor trained on its own data, so their GLORYx numbers are
unaffected; GRAIL's are inflated by drugs it was fitted on, in its own favour. Any interaction
involving GRAIL on this set inherits that.

This recomputes the ladder on the 13 genuinely unseen parents, reusing the same loaders, matcher and
frozen predictions as the published run so the only thing that changes is which substrates are
scored. Thirteen is small and the intervals will say so; the point is not a tighter estimate but
whether the effect survives at all once the contaminated substrates are gone.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from grail_metabolism.metrics import _tautomer_inchikey
from scripts.gloryx_rank_flip_ci import (
    DATA, PRED_FILES, load_gloryx, sygma_predict, per_substrate_recall, boot,
)

RDLogger.DisableLog("rdApp.*")
LADDER = ["inchikey", "inchi_no_stereo", "inchikey_tautomer"]
K, N_BOOT, SEED = 15, 10000, 0
OUT = ROOT / "results" / "gloryx_clean_subset.json"


def tk(s):
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return None


def trained_substrate_keys() -> set:
    """Tautomer keys of every substrate GRAIL saw in training or validation."""
    keys = set()
    for split in ("train", "val"):
        sdf = ROOT / f"grail_metabolism/data/{split}.sdf"
        tri = ROOT / f"grail_metabolism/data/{split}_triples_clean.txt"
        if not sdf.exists() or not tri.exists():
            continue
        ids = set()
        with open(tri) as fh:
            for line in fh:
                a = line.split()
                if len(a) == 3:
                    ids.add(a[0])
        for mol in Chem.SDMolSupplier(str(sdf)):
            if mol is None:
                continue
            p = mol.GetPropsAsDict()
            if str(p.get("Index", "")) in ids:
                s = p.get("SMILES") or Chem.MolToSmiles(mol)
                k = tk(s) if s else None
                if k:
                    keys.add(k)
    return keys


def main() -> int:
    reals = load_gloryx(DATA / "gloryx_test.json")
    allsubs = sorted(s for s in reals if reals[s])
    seen = trained_substrate_keys()
    clean = [s for s in allsubs if tk(s) not in seen]
    print(f"GLORYx parents with references: {len(allsubs)}; unseen by GRAIL: {len(clean)}", flush=True)

    methods = {n: {s: json.loads(p.read_text()).get(s, []) for s in clean}
               for n, p in PRED_FILES.items()}
    methods["SyGMa"] = sygma_predict(clean, K)

    vec = {(n, c): per_substrate_recall(pr, reals, clean, c, K)
           for n, pr in methods.items() for c in LADDER}

    import rdkit
    rep = {"n_all": len(allsubs), "n_clean": len(clean), "k": K, "n_boot": N_BOOT, "seed": SEED,
           "rdkit_version": rdkit.__version__, "ladder": LADDER, "recall": {}, "steps": {}}
    print(f"\n{'method':16}" + "".join(f"{c:>20}" for c in LADDER) + f"{'stereo step':>24}")
    for n in methods:
        rep["recall"][n] = {c: round(float(vec[(n, c)].mean()), 4) for c in LADDER}
        d = vec[(n, "inchi_no_stereo")] - vec[(n, "inchikey")]
        _, lo, hi = boot(d, N_BOOT, SEED)
        rep["steps"][n] = {"stereo": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                           "excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"{n:16}" + "".join(f"{vec[(n,c)].mean():20.4f}" for c in LADDER)
              + f"   {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {'SIG' if (lo>0 or hi<0) else 'n.s.'}")

    # The paired interactions the paper reports, recomputed on the clean subset.
    names = list(methods)
    rep["pairwise"] = {}
    print("\ndifferential sensitivity on the stereo step (interaction), clean subset:")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da = vec[(a, "inchi_no_stereo")] - vec[(a, "inchikey")]
            db = vec[(b, "inchi_no_stereo")] - vec[(b, "inchikey")]
            _, lo, hi = boot(da - db, N_BOOT, SEED)
            sig = lo > 0 or hi < 0
            rep["pairwise"][f"{a}_vs_{b}"] = {
                "interaction": round(float((da - db).mean()), 4),
                "ci95": [round(lo, 4), round(hi, 4)], "excludes_zero": bool(sig)}
            print(f"  {a} vs {b}: {(da-db).mean():+.4f} [{lo:+.4f},{hi:+.4f}] {'SIG' if sig else 'n.s.'}")

    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

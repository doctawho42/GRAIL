#!/usr/bin/env python3
"""Is the external coverage ceiling inflated by the drugs GRAIL trained on?

The paper reports an external ceiling of 0.633 on all 37 GLORYx parents and justifies not
restricting it to the 13 unseen ones by saying that applying the bank is model-free. That
justification is false, and an adversarial review caught it. Three quarters of the bank is mined
automatically from annotated substrate-metabolite pairs, and mining reads the training split
(Appendix A). Twenty-four of the 37 parents are in that split, so templates mined from their own
annotated pairs will recover their own metabolites. Rule application involves no trained network,
but the rule set is not independent of the training data, and that is the distinction the sentence
elides.

Whether the inflation is material is a measurement, not an argument. The per-parent coverage counts
are already stored by ceiling_external_validity.py, so splitting them by whether the parent is a
training or validation substrate costs one pass over the split SDFs and no rule application at all.

Two outcomes, both worth having. If the two groups agree, the ceiling stands on all 37 and the
justification is replaced by evidence. If they do not, the external ceiling is contaminated in the
paper's own favour and has to be restricted to the thirteen, as GRAIL's recall ladder already is.
"""
from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from grail_metabolism.metrics import _tautomer_inchikey

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
CACHE = ROOT / "results" / "train_val_substrate_keys.json"
OUT = ROOT / "results" / "ceiling_seen_vs_unseen.json"


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


def tk(s):
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return None


def train_val_keys() -> set:
    """Tautomer keys of every substrate GRAIL trained or validated on. Cached: the SDFs are large."""
    if CACHE.exists():
        return set(json.loads(CACHE.read_text()))
    keys = set()
    for split in ("train", "val"):
        ids = set()
        with open(ROOT / f"grail_metabolism/data/{split}_triples_clean.txt") as fh:
            for line in fh:
                a = line.split()
                if len(a) == 3:
                    ids.add(a[0])
        for mol in Chem.SDMolSupplier(str(ROOT / f"grail_metabolism/data/{split}.sdf")):
            if mol is None:
                continue
            p = mol.GetPropsAsDict()
            if str(p.get("Index", "")) in ids:
                s = p.get("SMILES") or Chem.MolToSmiles(mol)
                k = tk(s) if s else None
                if k:
                    keys.add(k)
        print(f"  {split}: {len(keys)} cumulative keys", flush=True)
    CACHE.write_text(json.dumps(sorted(keys)))
    return keys


def main() -> int:
    rec = json.loads((ROOT / "results" / "ceiling_external_validity.json").read_text())
    published = rec["external_ceiling_uncapped"]
    per_parent = rec["external_per_parent"]
    print(f"per-parent records: {len(per_parent)}; published ceiling {published['point']:.4f}",
          flush=True)

    seen = train_val_keys()
    print(f"train+val substrate keys: {len(seen)}", flush=True)

    groups = {"seen": [], "unseen": []}
    for r in per_parent:
        k = tk(r["parent"])
        groups["seen" if (k and k in seen) else "unseen"].append(
            (int(r["recovered"]), int(r["denom"])))

    rng = np.random.default_rng(SEED)
    rep = {"config": _code_version(), "n_total": len(per_parent),
           "published_ceiling_all_37": round(float(published["point"]), 4), "groups": {}}
    print(f"\n{'group':8}{'n':>5}{'ceiling':>10}{'95% CI':>22}")
    stats = {}
    for g, rows in groups.items():
        R = np.array([a for a, _ in rows]); U = np.array([b for _, b in rows])
        idx = rng.integers(0, len(R), (N_BOOT, len(R)))
        bt = np.array([R[i].sum() / U[i].sum() for i in idx])
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        micro = float(R.sum() / U.sum())
        stats[g] = (R, U, micro)
        rep["groups"][g] = {"n": int(len(R)), "ceiling": round(micro, 4),
                            "ci95": [round(lo, 4), round(hi, 4)]}
        print(f"{g:8}{len(R):>5}{micro:>10.4f}   [{lo:.4f}, {hi:.4f}]")

    # unpaired: the two groups are different drugs, so resample each independently
    Rs, Us, ms = stats["seen"]; Ru, Uu, mu = stats["unseen"]
    i1 = rng.integers(0, len(Rs), (N_BOOT, len(Rs)))
    i2 = rng.integers(0, len(Ru), (N_BOOT, len(Ru)))
    d = np.array([Rs[a].sum() / Us[a].sum() for a in i1]) - np.array([Ru[b].sum() / Uu[b].sum() for b in i2])
    lo, hi = float(np.quantile(d, .025)), float(np.quantile(d, .975))
    rep["seen_minus_unseen"] = {"point": round(ms - mu, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "excludes_zero": bool(lo > 0 or hi < 0),
                               "note": "unpaired: the groups are different drugs"}
    print(f"\nseen minus unseen: {ms-mu:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"{'INFLATED' if lo > 0 else 'no detectable inflation'}")
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

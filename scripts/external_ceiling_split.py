#!/usr/bin/env python3
"""The external ceiling on the drugs GRAIL saw in training and on the ones it did not.

The paper reports the external coverage ceiling split by whether a GLORYx parent also appears in
GRAIL's training or validation split, because 24 of the 37 do. That split has to be recomputed
whenever the ceiling is, and the ceiling has just moved: it is now measured in the hydrogen
convention the deployed generator fires rules in rather than in the expanding one, which is the
convention the internal ceiling it is compared against uses.

The overlap keys are built exactly as scripts/external_overlap_audit.py builds them -- from the
train and validation SDF records whose Index appears in the clean triples, keyed on the record's own
SMILES property rather than on anything the loader has standardised. Reproducing that construction
rather than importing a set is what the gate below is for: the split must come out 24 and 13, or the
keys are not the ones the audit counted and neither group's coverage means what it says.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from grail_metabolism.metrics import _tautomer_inchikey

RDLogger.DisableLog("rdApp.*")
# results/external_overlap_audit.json, "GLORYx external set"
COMMITTED_SPLIT = (24, 13)


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


def _key(smiles: str):
    try:
        return _tautomer_inchikey(smiles)
    except Exception:
        return None


def trained_keys() -> set:
    keys = set()
    for split in ("train", "val"):
        sdf = ROOT / f"grail_metabolism/data/{split}.sdf"
        triples = ROOT / f"grail_metabolism/data/{split}_triples_clean.txt"
        if not sdf.exists() or not triples.exists():
            continue
        ids = {line.split()[0] for line in triples.read_text().splitlines()
               if len(line.split()) == 3}
        for mol in Chem.SDMolSupplier(str(sdf)):
            if mol is None:
                continue
            props = mol.GetPropsAsDict()
            if str(props.get("Index", "")) not in ids:
                continue
            smiles = props.get("SMILES") or Chem.MolToSmiles(mol)
            k = _key(smiles) if smiles else None
            if k:
                keys.add(k)
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "external_ceiling_split.json"))
    args = ap.parse_args()

    ext = json.loads((ROOT / "results/ceiling_external_validity.json").read_text())
    parents = ext["external_per_parent"]
    print(f"external parents: {len(parents)}", flush=True)
    keys = trained_keys()
    print(f"substrates keyed from train and validation: {len(keys)}", flush=True)

    seen = [p for p in parents if _key(p["parent"]) in keys]
    unseen = [p for p in parents if _key(p["parent"]) not in keys]
    print(f"\ngate: {len(seen)} seen / {len(unseen)} unseen against the committed "
          f"{COMMITTED_SPLIT[0]} / {COMMITTED_SPLIT[1]}")
    if (len(seen), len(unseen)) != COMMITTED_SPLIT:
        raise SystemExit("these are not the keys the overlap audit counted; neither group's "
                         "coverage means what it says")

    import numpy as np

    N_BOOT, SEED = 10000, 0
    rng = np.random.default_rng(SEED)

    def micro(group):
        """Micro coverage with a cluster bootstrap over parents, the unit of dependence."""
        r = np.array([p["recovered"] for p in group], dtype=float)
        d = np.array([p["denom"] for p in group], dtype=float)
        idx = rng.integers(0, len(group), (N_BOOT, len(group)))
        bt = r[idx].sum(axis=1) / np.maximum(d[idx].sum(axis=1), 1)
        return {"recovered": int(r.sum()), "references": int(d.sum()),
                "coverage": round(float(r.sum() / max(d.sum(), 1)), 4),
                "ci95": [round(float(np.quantile(bt, .025)), 4),
                         round(float(np.quantile(bt, .975)), 4)]}

    def unpaired_difference(a, b):
        """Seen minus unseen. The two groups are disjoint parents, so the draws are independent."""
        ra = np.array([p["recovered"] for p in a], float); da = np.array([p["denom"] for p in a], float)
        rb = np.array([p["recovered"] for p in b], float); db = np.array([p["denom"] for p in b], float)
        ia = rng.integers(0, len(a), (N_BOOT, len(a)))
        ib = rng.integers(0, len(b), (N_BOOT, len(b)))
        bt = (ra[ia].sum(1) / np.maximum(da[ia].sum(1), 1)) - (rb[ib].sum(1) / np.maximum(db[ib].sum(1), 1))
        return {"delta": round(float(ra.sum()/da.sum() - rb.sum()/db.sum()), 4),
                "ci95": [round(float(np.quantile(bt, .025)), 4),
                         round(float(np.quantile(bt, .975)), 4)]}

    rep = {"config": {**_code_version(), "match": "inchikey_tautomer",
                      "convention": ext.get("convention", "hydrogens implicit, as deployed"),
                      "source": "results/ceiling_external_validity.json external_per_parent",
                      "gate": "the split reproduces results/external_overlap_audit.json"},
           "all": micro(parents), "seen_in_training": micro(seen), "unseen": micro(unseen),
           "seen_minus_unseen": unpaired_difference(seen, unseen)}
    for name in ("all", "seen_in_training", "unseen"):
        v = rep[name]
        print(f"  {name:18} {v['recovered']:>4}/{v['references']:<4} = {v['coverage']} {v['ci95']}")
    d = rep["seen_minus_unseen"]
    print(f"  seen minus unseen  {d['delta']:+.4f} {d['ci95']}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

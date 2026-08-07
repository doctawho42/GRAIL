#!/usr/bin/env python3
"""The SMIRKS-variant MetaTox predictions, keyed back to the substrates they were run on.

The supplier's earlier delivery was layer 1 without the SMIRKS rules and returned 270 of the 291
submitted parents (results/grail_vs_metatox.json), which is why MetaTox appears in no table in this
paper. This is the SMIRKS variant, and it covers all 291.

The file carries no substrates. Records are identified only as `<substrate index>_<metabolite
index>`, so the substrates have to be recovered from the submission order in
results/metatox_input/substrate_map.csv. A positional join is exactly the kind of assumption that
silently produces a whole table of wrong numbers, so it is gated rather than assumed: a predicted
metabolite of a substrate should look like that substrate, and under an off-by-one or a shuffled
order it would not. The gate is the median Tanimoto between each substrate and its own predictions,
against the same statistic under a deliberately rotated assignment. If the true join is not far
above the rotated one, the join is wrong and nothing downstream means anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
MAP = ROOT / "results" / "metatox_input" / "substrate_map.csv"


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


def parse_sdf(path: Path) -> dict:
    """{substrate index -> [metabolite SMILES]} from an SDF whose only key is `sub_met`."""
    out: dict[int, list[str]] = {}
    supplier = Chem.SDMolSupplier(str(path), sanitize=True)
    for mol in supplier:
        if mol is None:
            continue
        raw = mol.GetPropsAsDict().get("ID")
        if raw is None or "_" not in str(raw):
            continue
        try:
            idx = int(str(raw).split("_")[0])
        except ValueError:
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if smiles:
            out.setdefault(idx, []).append(smiles)
    return out


def median_similarity(pairs, gen) -> float:
    """Median over substrates of the median Tanimoto to that substrate's own predictions."""
    import statistics
    per = []
    for sub_smiles, preds in pairs:
        m = Chem.MolFromSmiles(sub_smiles)
        if m is None or not preds:
            continue
        fp = gen.GetFingerprint(m)
        sims = []
        for p in preds[:40]:
            q = Chem.MolFromSmiles(p)
            if q is not None:
                sims.append(DataStructs.TanimotoSimilarity(fp, gen.GetFingerprint(q)))
        if sims:
            per.append(statistics.median(sims))
    return round(statistics.median(per), 4) if per else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "metatox_smirks_preds.json"))
    args = ap.parse_args()

    by_index = parse_sdf(Path(args.sdf))
    rows = list(csv.DictReader(open(MAP)))
    print(f"predictions for {len(by_index)} substrate indices; "
          f"{sum(len(v) for v in by_index.values())} metabolites", flush=True)
    print(f"submission map: {len(rows)} substrates, {rows[0]['id']} .. {rows[-1]['id']}", flush=True)
    if sorted(by_index) != list(range(1, len(rows) + 1)):
        raise SystemExit("the substrate indices are not 1..N over the submission map; the "
                         "positional join is not available")

    ordered = [(r["substrate_smiles"], by_index[i + 1]) for i, r in enumerate(rows)]
    rotated = [(r["substrate_smiles"], by_index[((i + 137) % len(rows)) + 1])
               for i, r in enumerate(rows)]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    true_sim = median_similarity(ordered, gen)
    rot_sim = median_similarity(rotated, gen)
    print(f"\ngate: median self-similarity {true_sim} against {rot_sim} under a rotated assignment")
    if true_sim < 0.5 or true_sim < rot_sim + 0.25:
        raise SystemExit("the positional join does not separate from a rotated one; the substrate "
                         "order is not what this file assumes")

    preds = {sub: sorted(set(p)) for sub, p in ordered}
    sizes = [len(v) for v in preds.values()]
    rep = {"config": {**_code_version(), "sdf": Path(args.sdf).name,
                      "variant": "SMIRKS rules, all 291 parents returned",
                      "join": "positional against results/metatox_input/substrate_map.csv",
                      "gate": {"median_self_similarity": true_sim,
                               "median_under_rotated_assignment": rot_sim}},
           "n_substrates": len(preds),
           "n_predictions": int(sum(sizes)),
           "mean_output": round(sum(sizes) / len(sizes), 2),
           "median_output": sorted(sizes)[len(sizes) // 2],
           "predictions": preds}
    print(f"\nsubstrates {rep['n_substrates']}, predictions {rep['n_predictions']}, "
          f"mean output {rep['mean_output']}, median {rep['median_output']}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

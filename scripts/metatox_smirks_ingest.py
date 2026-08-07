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


def _spectrum(props) -> tuple[float, float] | None:
    """(Pa, Pi) for the Metabolite class, which is the score the delivery is named after.

    PASS writes the pair with a comma as the decimal separator: "0,370  0,167  Metabolite". Pa is
    the method's own confidence and therefore its ranking signal; ignoring it and taking the file's
    record order would score the method in an ordering it never produced, which is the one thing a
    frozen-prediction comparison must not do.
    """
    raw = props.get("PASS_ACTIVITY_SPECTRUM")
    if raw is None:
        return None
    for line in str(raw).splitlines():
        parts = line.replace(",", ".").split()
        if len(parts) >= 3 and parts[2].lower().startswith("metabolite"):
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return None
    return None


def parse_sdf(path: Path) -> dict:
    """{substrate index -> [(smiles, Pa, Pi)]}, ranked by the method's own confidence."""
    out: dict[int, list[tuple]] = {}
    supplier = Chem.SDMolSupplier(str(path), sanitize=True)
    n_scored = n_total = n_declared = n_disagree = 0
    for mol in supplier:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        raw = props.get("ID")
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
        if not smiles:
            continue
        n_total += 1
        pa_pi = _spectrum(props)
        if pa_pi is not None:
            n_scored += 1
        # PASS writes the spectrum only where it clears its own threshold, and says so in a second
        # tag. Cross-checking the two is the gate: if they disagree, the field is being read wrong.
        declared = "1 of 1" in str(props.get("PASS_RESULT_COUNT", ""))
        n_declared += declared
        if declared != (pa_pi is not None):
            n_disagree += 1
        pa, pi = pa_pi if pa_pi else (float("nan"), float("nan"))
        out.setdefault(idx, []).append((smiles, pa, pi))
    print(f"  metabolites carrying a Metabolite spectrum: {n_scored} of {n_total}", flush=True)
    print(f"  records PASS declares above its own threshold: {n_declared}, "
          f"disagreeing with the spectrum on {n_disagree}", flush=True)
    if n_disagree:
        raise SystemExit("the two tags disagree about which predictions PASS scored; the spectrum "
                         "field is not being read as PASS wrote it")
    # scored first, by the method's own confidence; the rest keep the order the file gives them,
    # since PASS expresses no preference among predictions it declines to score
    for idx in out:
        out[idx].sort(key=lambda t: (0, -t[1]) if t[1] == t[1] else (1, 0.0))
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
            q = Chem.MolFromSmiles(p[0] if isinstance(p, tuple) else p)
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

    # Keep the method's own order -- Pa descending -- and dedup by FIRST appearance. Sorting the
    # SMILES alphabetically, as an earlier revision did, scores a method in an ordering it never
    # produced, which makes every recall@k for it a statement about the alphabet.
    preds, scored = {}, {}
    for sub, items in ordered:
        seq, seen = [], set()
        for smiles, pa, pi in items:
            if smiles in seen:
                continue
            seen.add(smiles)
            seq.append(smiles)
        preds[sub] = seq
        scored[sub] = [[smiles, round(pa, 4), round(pi, 4)] for smiles, pa, pi in items
                       if smiles in seen and (seen.discard(smiles) or True)]
    above = {sub: [t[0] for t in v if t[1] > t[2]] for sub, v in scored.items()}
    sizes = [len(v) for v in preds.values()]
    n_above = [len(v) for v in above.values()]
    rep = {"config": {**_code_version(), "sdf": Path(args.sdf).name,
                      "variant": "SMIRKS rules, all 291 parents returned",
                      "join": "positional against results/metatox_input/substrate_map.csv",
                      "gate": {"median_self_similarity": true_sim,
                               "median_under_rotated_assignment": rot_sim}},
           "n_substrates": len(preds),
           "n_predictions": int(sum(sizes)),
           "mean_output": round(sum(sizes) / len(sizes), 2),
           "median_output": sorted(sizes)[len(sizes) // 2],
           "ranking": "the method's own Pa for the Metabolite class, descending",
           "mean_output_above_threshold": round(sum(n_above) / len(n_above), 2),
           "predictions": preds,
           "predictions_with_scores": scored,
           "predictions_above_own_threshold": above}
    print(f"\nsubstrates {rep['n_substrates']}, predictions {rep['n_predictions']}, "
          f"mean output {rep['mean_output']}, median {rep['median_output']}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

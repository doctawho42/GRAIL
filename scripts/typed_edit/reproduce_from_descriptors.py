#!/usr/bin/env python3
"""The criterion sweep, recomputed from what the repository actually releases.

The reference set is released as matching descriptors rather than as structures, and the claim
that goes with it is that all five declared criteria survive the substitution. That claim is worth
a check that a reader can run, so this is it: the whole verdict grid rebuilt from
results/test_reference_descriptors.json and the frozen per-substrate predictions, with no corpus
structure read at any point, and compared cell by cell against the artifact the manuscript prints.

The predictions are this work's own output and are released as structures, so their descriptors are
computed here. The references' are read from the file. A criterion is decided by comparing the two,
which is exactly what the substitution claims is possible.

    python scripts/typed_edit/reproduce_from_descriptors.py
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
CAP = 100
CRITERIA = ("canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer")


def token(desc: dict, criterion: str):
    """The one value a criterion compares, taken from a descriptor record."""
    if criterion == "canonical":
        return desc["canonical_sha256"]
    if criterion == "inchikey":
        return desc["inchikey"]
    if criterion == "inchi_no_stereo":
        return (desc["inchikey"] or "")[:14] or None
    if criterion == "tanimoto1":
        return tuple(desc["morgan_bits"])
    if criterion == "inchikey_tautomer":
        return desc["tautomer_key"]
    raise KeyError(criterion)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "reproduce_from_descriptors.json"))
    args = ap.parse_args()

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from _rrf import rrf_order
    from reference_descriptors import descriptors

    blob = json.loads((ROOT / "results/test_reference_descriptors.json").read_text())
    refs = blob["references"]

    pools = {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        pools.update(json.loads(Path(f).read_text())["pools"])
    subs = sorted(s for s in pools if refs.get(s))

    comparators = {
        "metatox": ("results/metatox_smirks_preds.json", "predictions"),
        "sygma": ("results/sygma_fulltest_predictions.json", None),
        "metapredictor": ("artifacts/tier2_1170/metapredictor_preds.json", None),
        "biotransformer": ("results/biotransformer_allhuman_one_step_preds.json", None),
    }

    # Every structure whose descriptors this needs, computed once. The predictions are this work's
    # own output; nothing here reads a corpus structure.
    ordered = {"whole bank": {}}
    for s in subs:
        keep = sorted(pools[s], key=lambda c: -c["generator"])[:CAP]
        ordered["whole bank"][s] = [c["smiles"] for c in rrf_order(keep)]
    for name, (rel, key) in comparators.items():
        path = ROOT / rel
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        preds = raw[key] if key else raw
        ordered[name] = {s: list(preds.get(s, []))[: max(KS) + 40] for s in subs}

    cache: dict = {}

    def desc(smiles):
        if smiles not in cache:
            cache[smiles] = descriptors(smiles)
        return cache[smiles]

    rows = {}
    for criterion in CRITERIA:
        ref_tokens = {s: {t for t in (token(d, criterion) for d in refs[s]) if t} for s in subs}
        universe = float(sum(len(v) for v in ref_tokens.values()))
        per_arm = {}
        for arm, lists in ordered.items():
            hits = {k: 0 for k in KS}
            for s in subs:
                parent = token(desc(s) or {}, criterion) if desc(s) else None
                seen, seq = set(), []
                for smiles in lists[s]:
                    d = desc(smiles)
                    if d is None:
                        continue
                    tok = token(d, criterion)
                    if not tok or tok in seen or tok == parent:
                        continue
                    seen.add(tok)
                    seq.append(tok)
                for k in KS:
                    hits[k] += len(set(seq[:k]) & ref_tokens[s])
            per_arm[arm] = {str(k): round(hits[k] / max(universe, 1), 4) for k in KS}
        rows[criterion] = per_arm
        print(f"  {criterion:18s} whole bank r@15 {per_arm['whole bank']['15']}", flush=True)

    # The check: the default criterion's whole-bank column has to be the column the paper prints.
    published = json.loads((ROOT / "results/deployment_table.json").read_text())["recall_micro"]
    mismatch = {k: (rows["inchikey_tautomer"]["whole bank"][k], v["whole bank"])
                for k, v in published.items()
                if abs(rows["inchikey_tautomer"]["whole bank"][k] - v["whole bank"]) > 1e-4}

    report = {
        "provenance": stamp(__file__),
        "what_this_shows": (
            "the comparison recomputed under all five declared matching criteria from the "
            "released reference descriptors and the released per-substrate predictions, reading "
            "no corpus structure at any point"),
        "population": {"n_substrates": len(subs),
                       "n_references": int(sum(len(refs[s]) for s in subs))},
        "recall_by_criterion": rows,
        "default_matches_the_published_column": not mismatch,
        "mismatches": mismatch,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\ndefault criterion reproduces the published whole-bank column: {not mismatch}")
    if mismatch:
        for k, (a, b) in mismatch.items():
            print(f"  k={k}: from descriptors {a}, published {b}")
    print(f"wrote {args.out}")
    return 0 if not mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())

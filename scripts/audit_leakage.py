"""Read-only leakage audit of the committed clean splits -> results/leakage_fix_report.json.

`fix_splits.py --molecule-disjoint` BUILDS the clean splits (and overwrites the *_triples_clean.txt
files, which are gitignored symlinks into the shared dataset). This script is its read-only
companion: it re-derives the canonical substrate/molecule/positive-pair sets from the ALREADY
committed clean triples (via fix_splits' own functions) and verifies zero cross-split overlap,
WITHOUT touching any data file. Emits the machine-checkable audit summary the manuscript's Data &
Code Availability section refers to.

Usage: python scripts/audit_leakage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fix_splits import (
    CLEAN_TRIPLES,
    SPLIT_SPECS,
    build_canonical_triples,
    metabolite_set,
    molecule_set,
    overlap_summary,
    substrate_set,
    summarize_clean_split,
)

RESULTS = ROOT / "results" / "leakage_fix_report.json"


def main() -> int:
    splits = {}
    clean_stats = {}
    for name in ("train", "val", "test"):
        sdf_path = SPLIT_SPECS[name][0]
        clean_path = CLEAN_TRIPLES[name]
        if not clean_path.exists():
            print(f"ERROR: clean triples missing: {clean_path}", flush=True)
            return 1
        triples, _ = build_canonical_triples(name, sdf_path, clean_path)
        splits[name] = triples
        clean_stats[name] = summarize_clean_split(triples)

    overlap = overlap_summary(splits)
    substrate_disjoint = all(v["substrate_overlap"] == 0 for v in overlap.values())
    pair_disjoint = all(v["positive_pair_overlap"] == 0 for v in overlap.values())
    molecule_disjoint = all(v["molecule_overlap"] == 0 for v in overlap.values())

    # Directional structure-leak diagnostic: does an EVAL substrate appear anywhere in the TRAIN
    # molecule set (i.e. was a val/test substrate's structure seen in training, e.g. as a
    # metabolite of a different substrate)? That is the leak a structure-scoring filter could
    # exploit -- distinct from benign metabolite-recurrence across splits.
    train_mols = molecule_set(splits["train"])
    train_metabs = metabolite_set(splits["train"])
    structure_leak = {
        "val_substrate_in_train_molecules": len(substrate_set(splits["val"]) & train_mols),
        "test_substrate_in_train_molecules": len(substrate_set(splits["test"]) & train_mols),
        "val_substrate_in_train_metabolites": len(substrate_set(splits["val"]) & train_metabs),
        "test_substrate_in_train_metabolites": len(substrate_set(splits["test"]) & train_metabs),
    }

    report = {
        "audit": "read-only verification of the committed clean splits",
        "source_clean_triples": {k: str(v) for k, v in CLEAN_TRIPLES.items()},
        "clean_split_stats": clean_stats,
        "clean_overlap": overlap,
        "substrate_disjoint": substrate_disjoint,
        "positive_pair_disjoint": pair_disjoint,
        "molecule_disjoint": molecule_disjoint,
        "structure_leak": structure_leak,
        "note": "substrate_disjoint = no substrate appears in two splits (the primary leakage guard "
                "for substrate->metabolite prediction). positive_pair_disjoint = no identical "
                "(substrate, metabolite) pair recurs. molecule_disjoint would additionally require "
                "no molecule (incl. metabolites) to recur across splits; metabolites are shared "
                "across substrates so this is False and structure_leak quantifies whether any EVAL "
                "SUBSTRATE structure was seen in training.",
    }
    leak_free = substrate_disjoint and pair_disjoint
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "substrate_disjoint": substrate_disjoint,
        "positive_pair_disjoint": pair_disjoint,
        "molecule_disjoint": molecule_disjoint,
        "structure_leak": structure_leak,
        "clean_overlap": overlap,
        "sizes": {k: v["remaining_substrates"] for k, v in clean_stats.items()},
    }, indent=2), flush=True)
    print(f"\nWrote {RESULTS}", flush=True)
    return 0 if leak_free else 2


if __name__ == "__main__":
    raise SystemExit(main())

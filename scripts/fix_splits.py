#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "grail_metabolism" / "data"
RESULTS_DIR = ROOT / "results"

SPLIT_SPECS = {
    "train": (DATA_DIR / "train.sdf", DATA_DIR / "train_triples.txt"),
    "val": (DATA_DIR / "val.sdf", DATA_DIR / "val_triples.txt"),
    "test": (DATA_DIR / "test.sdf", DATA_DIR / "test_triples.txt"),
}

CLEAN_TRIPLES = {
    "train": DATA_DIR / "train_triples_clean.txt",
    "val": DATA_DIR / "val_triples_clean.txt",
    "test": DATA_DIR / "test_triples_clean.txt",
}


@dataclass(frozen=True)
class CanonicalTriple:
    sub_id: int
    prod_id: int
    label: int
    sub_smiles: str
    prod_smiles: str


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def canonicalize_with_fallback(mol: Optional[Chem.Mol], smiles: Optional[str]) -> Optional[str]:
    canonical = canonicalize_smiles(smiles or "")
    if canonical:
        return canonical
    if mol is None:
        return None
    try:
        return canonicalize_smiles(Chem.MolToSmiles(mol))
    except Exception:
        return None


def read_triples(path: Path) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid triple at {path}:{line_number}: {stripped}")
            triples.append(tuple(int(part) for part in parts))
    return triples


def load_index_to_smiles(sdf_path: Path, required_ids: Set[int]) -> Tuple[Dict[int, str], int]:
    index_to_smiles: Dict[int, str] = {}
    invalid_count = 0
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for fallback_index, mol in enumerate(supplier, start=1):
        if mol is None:
            invalid_count += 1
            continue
        try:
            index = int(mol.GetProp("Index")) if mol.HasProp("Index") else fallback_index
        except Exception:
            index = fallback_index
        if index not in required_ids:
            continue
        raw_smiles = mol.GetProp("SMILES") if mol.HasProp("SMILES") else None
        canonical = canonicalize_with_fallback(mol, raw_smiles)
        if canonical is None:
            invalid_count += 1
            continue
        index_to_smiles[index] = canonical
        if len(index_to_smiles) == len(required_ids):
            break
    return index_to_smiles, invalid_count


def build_canonical_triples(name: str, sdf_path: Path, triples_path: Path) -> Tuple[List[CanonicalTriple], Dict[str, int]]:
    print(f"[{name}] Loading triples from {triples_path}", flush=True)
    triples = read_triples(triples_path)
    required_ids = {sub_id for sub_id, _, _ in triples} | {prod_id for _, prod_id, _ in triples}
    index_to_smiles, invalid_records = load_index_to_smiles(sdf_path, required_ids)
    canonical_triples: List[CanonicalTriple] = []
    missing_ids = 0
    for sub_id, prod_id, label in triples:
        sub_smiles = index_to_smiles.get(sub_id)
        prod_smiles = index_to_smiles.get(prod_id)
        if sub_smiles is None or prod_smiles is None:
            missing_ids += 1
            continue
        canonical_triples.append(
            CanonicalTriple(
                sub_id=sub_id,
                prod_id=prod_id,
                label=label,
                sub_smiles=sub_smiles,
                prod_smiles=prod_smiles,
            )
        )
    stats = {
        "original_triples": len(triples),
        "canonical_triples": len(canonical_triples),
        "invalid_sdf_records": invalid_records,
        "missing_id_rows": missing_ids,
        "original_positive_triples": sum(1 for _, _, label in triples if label == 1),
        "canonical_positive_triples": sum(1 for row in canonical_triples if row.label == 1),
        "substrates": len({row.sub_smiles for row in canonical_triples}),
    }
    print(
        f"[{name}] {stats['canonical_triples']} canonical triples, "
        f"{stats['substrates']} substrates, {stats['canonical_positive_triples']} positive rows "
        f"(invalid SDF records: {invalid_records}, missing ids: {missing_ids})",
        flush=True,
    )
    return canonical_triples, stats


def substrate_set(triples: Sequence[CanonicalTriple]) -> Set[str]:
    return {row.sub_smiles for row in triples}


def positive_pairs(triples: Sequence[CanonicalTriple]) -> Set[Tuple[str, str]]:
    return {(row.sub_smiles, row.prod_smiles) for row in triples if row.label == 1}


def pair_overlap(left: Sequence[CanonicalTriple], right: Sequence[CanonicalTriple]) -> int:
    return len(positive_pairs(left) & positive_pairs(right))


def filter_by_disallowed_substrates(
    triples: Sequence[CanonicalTriple],
    disallowed: Set[str],
) -> Tuple[List[CanonicalTriple], int, int]:
    kept: List[CanonicalTriple] = []
    removed_triples = 0
    removed_substrates: Set[str] = set()
    for row in triples:
        if row.sub_smiles in disallowed:
            removed_triples += 1
            removed_substrates.add(row.sub_smiles)
            continue
        kept.append(row)
    return kept, removed_triples, len(removed_substrates)


def write_triples(path: Path, triples: Iterable[CanonicalTriple]) -> None:
    with open(path, "w") as handle:
        for row in triples:
            handle.write(f"{row.sub_id}\t{row.prod_id}\t{row.label}\n")


def summarize_clean_split(triples: Sequence[CanonicalTriple]) -> Dict[str, int]:
    return {
        "remaining_triples": len(triples),
        "remaining_positive_triples": sum(1 for row in triples if row.label == 1),
        "remaining_substrates": len({row.sub_smiles for row in triples}),
        "remaining_positive_pairs": len(positive_pairs(triples)),
    }


def overlap_summary(splits: Dict[str, Sequence[CanonicalTriple]]) -> Dict[str, Dict[str, int]]:
    names = ["train", "val", "test"]
    summary: Dict[str, Dict[str, int]] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = splits[left_name]
            right = splits[right_name]
            key = f"{left_name}_{right_name}"
            summary[key] = {
                "substrate_overlap": len(substrate_set(left) & substrate_set(right)),
                "positive_pair_overlap": pair_overlap(left, right),
            }
    return summary


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    original_triples: Dict[str, List[CanonicalTriple]] = {}
    split_stats: Dict[str, Dict[str, int]] = {}
    for name, (sdf_path, triples_path) in SPLIT_SPECS.items():
        triples, stats = build_canonical_triples(name, sdf_path, triples_path)
        original_triples[name] = triples
        split_stats[name] = stats

    contamination = {
        "test_substrates_in_train": len(substrate_set(original_triples["test"]) & substrate_set(original_triples["train"])),
        "test_positive_pairs_in_train": pair_overlap(original_triples["test"], original_triples["train"]),
        "val_substrates_in_train": len(substrate_set(original_triples["val"]) & substrate_set(original_triples["train"])),
        "val_positive_pairs_in_train": pair_overlap(original_triples["val"], original_triples["train"]),
        "val_substrates_in_test": len(substrate_set(original_triples["val"]) & substrate_set(original_triples["test"])),
        "val_positive_pairs_in_test": pair_overlap(original_triples["val"], original_triples["test"]),
    }

    test_substrates = substrate_set(original_triples["test"])
    val_substrates = substrate_set(original_triples["val"])

    clean_train, train_removed_for_test, train_removed_test_substrates = filter_by_disallowed_substrates(
        original_triples["train"],
        test_substrates | val_substrates,
    )
    clean_val, val_removed_for_test, val_removed_test_substrates = filter_by_disallowed_substrates(
        original_triples["val"],
        test_substrates,
    )
    clean_test = list(original_triples["test"])

    write_triples(CLEAN_TRIPLES["train"], clean_train)
    write_triples(CLEAN_TRIPLES["val"], clean_val)
    shutil.copyfile(SPLIT_SPECS["test"][1], CLEAN_TRIPLES["test"])

    clean_splits = {
        "train": clean_train,
        "val": clean_val,
        "test": clean_test,
    }
    clean_overlap = overlap_summary(clean_splits)

    report = {
        "original": split_stats,
        "contamination_before_cleaning": contamination,
        "cleaning": {
            "train": {
                "removed_triples": train_removed_for_test,
                "removed_substrates": train_removed_test_substrates,
                **summarize_clean_split(clean_train),
            },
            "val": {
                "removed_triples": val_removed_for_test,
                "removed_substrates": val_removed_test_substrates,
                **summarize_clean_split(clean_val),
            },
            "test": {
                "removed_triples": 0,
                "removed_substrates": 0,
                **summarize_clean_split(clean_test),
            },
        },
        "clean_overlap": clean_overlap,
        "verification": {
            "zero_substrate_overlap_between_clean_splits": all(
                value["substrate_overlap"] == 0 for value in clean_overlap.values()
            ),
            "zero_positive_pair_overlap_train_test": clean_overlap["train_test"]["positive_pair_overlap"] == 0,
        },
        "outputs": {name: str(path) for name, path in CLEAN_TRIPLES.items()},
    }

    report_path = RESULTS_DIR / "leakage_fix_report.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\nLeakage cleanup summary", flush=True)
    for name in ("train", "val", "test"):
        original = split_stats[name]["original_triples"]
        cleaned = report["cleaning"][name]["remaining_triples"]
        removed = report["cleaning"][name]["removed_triples"]
        substrates = report["cleaning"][name]["remaining_substrates"]
        print(
            f"{name}: original={original}, removed={removed}, remaining={cleaned}, "
            f"remaining_substrates={substrates}",
            flush=True,
        )
    print("Clean split overlaps:", flush=True)
    for name, values in clean_overlap.items():
        print(
            f"  {name}: substrate_overlap={values['substrate_overlap']}, "
            f"positive_pair_overlap={values['positive_pair_overlap']}",
            flush=True,
        )
    print(
        f"Verification: zero_substrate_overlap={report['verification']['zero_substrate_overlap_between_clean_splits']}, "
        f"zero_train_test_pair_overlap={report['verification']['zero_positive_pair_overlap_train_test']}",
        flush=True,
    )
    print(f"Wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

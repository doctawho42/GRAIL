#!/usr/bin/env python3
from __future__ import annotations

import json
import signal
import statistics
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "grail_metabolism" / "data"
RESULTS_DIR = ROOT / "results"

RULE_BANK_SPECS: List[Tuple[str, Path]] = [
    ("smirks.txt", DATA_DIR / "smirks.txt"),
    ("merged_smirks.txt", DATA_DIR / "merged_smirks.txt"),
    ("compressed_rules.smarts", ROOT / "grail_metabolism" / "compressed_rules.smarts"),
    (
        "notebooks_rules.txt",
        (ROOT / "grail_metabolism" / "resources" / "notebooks_rules.txt")
        if (ROOT / "grail_metabolism" / "resources" / "notebooks_rules.txt").exists()
        else (ROOT / "notebooks" / "rules.txt"),
    ),
]

SPLIT_SPECS: List[Tuple[str, Path, Path]] = [
    ("test", DATA_DIR / "test.sdf", DATA_DIR / "test_triples.txt"),
    ("val", DATA_DIR / "val.sdf", DATA_DIR / "val_triples.txt"),
]

LEAKAGE_SPLITS: List[Tuple[str, Path, Path]] = [
    ("train", DATA_DIR / "train.sdf", DATA_DIR / "train_triples.txt"),
    ("val", DATA_DIR / "val.sdf", DATA_DIR / "val_triples.txt"),
    ("test", DATA_DIR / "test.sdf", DATA_DIR / "test_triples.txt"),
]

RULE_TIMEOUT_SECONDS = 30.0
PRODUCT_CAP = 1000
PROGRESS_EVERY = 100


@dataclass
class DatasetSplit:
    name: str
    sdf_path: Path
    triples_path: Path
    substrate_to_products: Dict[str, Set[str]]
    substrate_set: Set[str]
    metabolite_set: Set[str]
    positive_pairs: Set[Tuple[str, str]]
    invalid_index_count: int = 0
    missing_index_count: int = 0


@dataclass
class RuleRecord:
    rule_id: int
    text: str
    parsed: bool
    rxn: Optional[object] = None
    template: Optional[Chem.Mol] = None
    parse_error: Optional[str] = None


@dataclass
class RuleEvalResult:
    applicable: bool = False
    products: Set[str] = field(default_factory=set)
    application_failed: bool = False
    timed_out: bool = False
    candidate_capped: bool = False
    sanitize_failures: int = 0


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles or not isinstance(smiles, str):
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


def read_rule_file(path: Path) -> List[str]:
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def read_positive_triples(path: Path) -> List[Tuple[int, int]]:
    positives: List[Tuple[int, int]] = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid triple at {path}:{line_number}: {stripped}")
            sub_idx, prod_idx, label = (int(part) for part in parts)
            if label == 1:
                positives.append((sub_idx, prod_idx))
    return positives


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


def load_positive_split(name: str, sdf_path: Path, triples_path: Path) -> DatasetSplit:
    print(f"[{name}] Loading positive pairs from {triples_path}", flush=True)
    positive_triples = read_positive_triples(triples_path)
    required_ids = {sub_idx for sub_idx, _ in positive_triples} | {prod_idx for _, prod_idx in positive_triples}
    index_to_smiles, invalid_index_count = load_index_to_smiles(sdf_path, required_ids)

    substrate_to_products: Dict[str, Set[str]] = defaultdict(set)
    positive_pairs: Set[Tuple[str, str]] = set()
    missing_index_count = 0
    for sub_idx, prod_idx in positive_triples:
        substrate = index_to_smiles.get(sub_idx)
        product = index_to_smiles.get(prod_idx)
        if substrate is None or product is None:
            missing_index_count += 1
            continue
        substrate_to_products[substrate].add(product)
        positive_pairs.add((substrate, product))

    split = DatasetSplit(
        name=name,
        sdf_path=sdf_path,
        triples_path=triples_path,
        substrate_to_products=dict(substrate_to_products),
        substrate_set=set(substrate_to_products.keys()),
        metabolite_set={product for products in substrate_to_products.values() for product in products},
        positive_pairs=positive_pairs,
        invalid_index_count=invalid_index_count,
        missing_index_count=missing_index_count,
    )
    print(
        f"[{name}] Loaded {len(split.substrate_set)} substrates, "
        f"{len(split.metabolite_set)} metabolites, {len(split.positive_pairs)} positive pairs "
        f"(invalid SDF records: {invalid_index_count}, missing mapped ids: {missing_index_count})",
        flush=True,
    )
    return split


def build_rule_registry(bank_rules: Dict[str, List[str]]) -> Tuple[List[RuleRecord], Dict[str, Counter[int]], Dict[int, Dict[str, int]], Dict[str, Dict[str, object]]]:
    unique_rules: Dict[str, int] = {}
    membership: Dict[str, Counter[int]] = {}
    metadata: Dict[str, Dict[str, object]] = {}

    for bank_name, rules in bank_rules.items():
        counter: Counter[int] = Counter()
        for rule_text in rules:
            if rule_text not in unique_rules:
                unique_rules[rule_text] = len(unique_rules)
            counter[unique_rules[rule_text]] += 1
        membership[bank_name] = counter

    rule_records: List[RuleRecord] = []
    for rule_text, rule_id in sorted(unique_rules.items(), key=lambda item: item[1]):
        parsed = False
        rxn = None
        template = None
        parse_error = None
        try:
            rxn = AllChem.ReactionFromSmarts(rule_text)
            if rxn is None:
                parse_error = "ReactionFromSmarts returned None"
            elif rxn.GetNumReactantTemplates() < 1:
                parse_error = "No reactant templates"
                rxn = None
            else:
                template = rxn.GetReactantTemplate(0)
                if template is None:
                    parse_error = "Missing reactant template"
                    rxn = None
                else:
                    parsed = True
        except Exception as exc:
            parse_error = str(exc)
            rxn = None
            template = None
        rule_records.append(
            RuleRecord(
                rule_id=rule_id,
                text=rule_text,
                parsed=parsed,
                rxn=rxn,
                template=template,
                parse_error=parse_error,
            )
        )

    reverse_membership: Dict[int, Dict[str, int]] = defaultdict(dict)
    for bank_name, counter in membership.items():
        for rule_id, multiplicity in counter.items():
            reverse_membership[rule_id][bank_name] = multiplicity

    for bank_name, rules in bank_rules.items():
        counter = membership[bank_name]
        metadata[bank_name] = {
            "path": str(path_for_bank(bank_name)),
            "n_rules_total": len(rules),
            "n_rules_parsed": sum(
                multiplicity
                for rule_id, multiplicity in counter.items()
                if rule_records[rule_id].parsed
            ),
        }

    return rule_records, membership, reverse_membership, metadata


def path_for_bank(bank_name: str) -> str:
    if bank_name == "union_all":
        return "UNION(" + ";".join(str(path) for _, path in RULE_BANK_SPECS) + ")"
    for candidate_name, path in RULE_BANK_SPECS:
        if candidate_name == bank_name:
            return str(path)
    raise KeyError(bank_name)


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"Timed out after {seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def evaluate_rule_on_substrate(mol: Chem.Mol, record: RuleRecord) -> RuleEvalResult:
    result = RuleEvalResult()
    if not record.parsed or record.rxn is None or record.template is None:
        return result

    try:
        result.applicable = mol.HasSubstructMatch(record.template)
    except Exception:
        result.application_failed = True
        return result

    if not result.applicable:
        return result

    try:
        with time_limit(RULE_TIMEOUT_SECONDS):
            outcomes = record.rxn.RunReactants((mol,), maxProducts=PRODUCT_CAP)
    except TimeoutError:
        result.application_failed = True
        result.timed_out = True
        return result
    except Exception:
        result.application_failed = True
        return result

    if len(outcomes) >= PRODUCT_CAP:
        result.candidate_capped = True

    for product_tuple in outcomes:
        for product in product_tuple:
            if len(result.products) >= PRODUCT_CAP:
                result.candidate_capped = True
                return result
            try:
                Chem.SanitizeMol(product)
                canonical = Chem.MolToSmiles(product)
            except Exception:
                result.sanitize_failures += 1
                continue
            if canonical:
                result.products.add(canonical)
    return result


def initialise_bank_metrics(bank_metadata: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    metrics: Dict[str, Dict[str, object]] = {}
    for bank_name, metadata in bank_metadata.items():
        metrics[bank_name] = {
            "path": metadata["path"],
            "n_rules_total": int(metadata["n_rules_total"]),
            "n_rules_parsed": int(metadata["n_rules_parsed"]),
            "total_true_recovered": 0,
            "total_true_metabolites": 0,
            "sum_applicable_rules": 0,
            "sum_candidate_products": 0,
            "per_substrate_recall": [],
            "substrates_with_hit": 0,
            "total_rule_application_failures": 0,
            "total_rule_timeouts": 0,
            "total_sanitize_failures": 0,
            "total_capped_rule_runs": 0,
        }
    return metrics


def summarise_split_metrics(metrics: Dict[str, Dict[str, object]], n_substrates: int) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for bank_name, values in metrics.items():
        recalls: List[float] = values["per_substrate_recall"]
        total_true = int(values["total_true_metabolites"])
        total_recovered = int(values["total_true_recovered"])
        summary[bank_name] = {
            "upper_bound_recall": (total_recovered / total_true) if total_true else 0.0,
            "mean_applicable_rules": (float(values["sum_applicable_rules"]) / n_substrates) if n_substrates else 0.0,
            "mean_candidates": (float(values["sum_candidate_products"]) / n_substrates) if n_substrates else 0.0,
            "median_per_substrate_recall": statistics.median(recalls) if recalls else 0.0,
            "fraction_with_hit": (float(values["substrates_with_hit"]) / n_substrates) if n_substrates else 0.0,
            "total_rule_application_failures": int(values["total_rule_application_failures"]),
            "total_rule_timeouts": int(values["total_rule_timeouts"]),
            "total_sanitize_failures": int(values["total_sanitize_failures"]),
            "total_capped_rule_runs": int(values["total_capped_rule_runs"]),
            "total_true_recovered": total_recovered,
            "total_true_metabolites": total_true,
        }
    return summary


def classify_gap(delta_mw: Optional[float]) -> str:
    if delta_mw is None:
        return "other (invalid MW)"
    if abs(delta_mw - 16.0) < 2:
        return "hydroxylation"
    if abs(delta_mw - (-14.0)) < 2:
        return "demethylation"
    if abs(delta_mw - 176.0) < 2:
        return "glucuronidation"
    if abs(delta_mw - 80.0) < 2:
        return "sulfation"
    if abs(delta_mw - (-2.0)) < 2:
        return "desaturation"
    if abs(delta_mw - (-15.0)) < 2:
        return "deamination"
    if abs(delta_mw - 42.0) < 2:
        return "acetylation"
    if abs(delta_mw - (-30.0)) < 2:
        return "deformylation"
    if abs(delta_mw - 0.0) < 2:
        return "isomerization/rearrangement"
    return f"other (ΔMW={delta_mw:.1f})"


def compute_delta_mw(substrate_smiles: str, product_smiles: str) -> Optional[float]:
    try:
        substrate = Chem.MolFromSmiles(substrate_smiles)
        product = Chem.MolFromSmiles(product_smiles)
    except Exception:
        return None
    if substrate is None or product is None:
        return None
    try:
        return float(Descriptors.MolWt(product) - Descriptors.MolWt(substrate))
    except Exception:
        return None


def evaluate_split(
    split: DatasetSplit,
    rule_records: Sequence[RuleRecord],
    reverse_membership: Dict[int, Dict[str, int]],
    bank_metadata: Dict[str, Dict[str, object]],
    union_gap_analysis: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, object]]:
    metrics = initialise_bank_metrics(bank_metadata)
    uncovered_entries: List[Dict[str, object]] = []
    gap_counter: Counter[str] = Counter()
    start = time.perf_counter()
    substrates = sorted(split.substrate_to_products.items())

    for substrate_index, (substrate_smiles, true_products) in enumerate(substrates, start=1):
        if substrate_index == 1 or substrate_index % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[{split.name}] Coverage progress: {substrate_index}/{len(substrates)} substrates "
                f"({elapsed:.1f}s elapsed)",
                flush=True,
            )

        mol = None
        try:
            mol = Chem.MolFromSmiles(substrate_smiles)
        except Exception:
            mol = None

        applicable_counts = {bank_name: 0 for bank_name in bank_metadata}
        candidate_sets = {bank_name: set() for bank_name in bank_metadata}

        if mol is not None:
            for record in rule_records:
                membership = reverse_membership.get(record.rule_id)
                if not membership:
                    continue
                eval_result = evaluate_rule_on_substrate(mol, record)
                if eval_result.application_failed:
                    for bank_name, multiplicity in membership.items():
                        metrics[bank_name]["total_rule_application_failures"] += multiplicity
                        if eval_result.timed_out:
                            metrics[bank_name]["total_rule_timeouts"] += multiplicity
                if eval_result.sanitize_failures:
                    for bank_name, multiplicity in membership.items():
                        metrics[bank_name]["total_sanitize_failures"] += eval_result.sanitize_failures * multiplicity
                if eval_result.candidate_capped:
                    print(
                        f"[{split.name}] Warning: candidate cap hit for substrate {substrate_index} "
                        f"rule_id={record.rule_id}",
                        flush=True,
                    )
                    for bank_name, multiplicity in membership.items():
                        metrics[bank_name]["total_capped_rule_runs"] += multiplicity
                if not eval_result.applicable:
                    continue
                for bank_name, multiplicity in membership.items():
                    applicable_counts[bank_name] += multiplicity
                    if eval_result.products:
                        candidate_sets[bank_name].update(eval_result.products)

        for bank_name in bank_metadata:
            recovered = len(candidate_sets[bank_name] & true_products)
            true_total = len(true_products)
            metrics[bank_name]["total_true_recovered"] += recovered
            metrics[bank_name]["total_true_metabolites"] += true_total
            metrics[bank_name]["sum_applicable_rules"] += applicable_counts[bank_name]
            metrics[bank_name]["sum_candidate_products"] += len(candidate_sets[bank_name])
            metrics[bank_name]["per_substrate_recall"].append((recovered / true_total) if true_total else 0.0)
            if recovered > 0:
                metrics[bank_name]["substrates_with_hit"] += 1

        if union_gap_analysis:
            uncovered = sorted(true_products - candidate_sets["union_all"])
            for product_smiles in uncovered:
                delta_mw = compute_delta_mw(substrate_smiles, product_smiles)
                gap_type = classify_gap(delta_mw)
                gap_counter[gap_type] += 1
                uncovered_entries.append(
                    {
                        "substrate_smi": substrate_smiles,
                        "metabolite_smi": product_smiles,
                        "delta_mw": round(delta_mw, 6) if delta_mw is not None else None,
                        "type": gap_type,
                    }
                )

    summary = summarise_split_metrics(metrics, len(substrates))
    gap_payload = {}
    if union_gap_analysis:
        total_uncovered = len(uncovered_entries)
        top_types = [
            {
                "type": gap_type,
                "count": count,
                "fraction": (count / total_uncovered) if total_uncovered else 0.0,
            }
            for gap_type, count in gap_counter.most_common(20)
        ]
        gap_payload = {
            "n_uncovered_metabolites": total_uncovered,
            "top_uncovered_types": top_types,
            "uncovered_pairs": uncovered_entries,
        }
    return summary, gap_payload


def format_float(value: float) -> str:
    return f"{value:.4f}"


def print_summary_table(rule_banks: Dict[str, Dict[str, object]]) -> None:
    headers = [
        "Bank",
        "Rules",
        "Parsed",
        "Test Recall Ceiling",
        "Val Recall Ceiling",
        "Mean Applicable",
        "Mean Candidates",
    ]
    rows = []
    for bank_name in ["smirks.txt", "merged_smirks.txt", "compressed_rules.smarts", "notebooks_rules.txt", "union_all"]:
        payload = rule_banks[bank_name]
        rows.append(
            [
                bank_name,
                str(payload["n_rules_total"]),
                str(payload["n_rules_parsed"]),
                format_float(float(payload["test_upper_bound_recall"])),
                format_float(float(payload["val_upper_bound_recall"])),
                format_float(float(payload["test_mean_applicable_rules"])),
                format_float(float(payload["test_mean_candidates"])),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render(row: Sequence[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    print(render(headers))
    print(separator)
    for row in rows:
        print(render(row))


def leakage_overlap(a: DatasetSplit, b: DatasetSplit) -> Dict[str, int]:
    return {
        "substrate_overlap": len(a.substrate_set & b.substrate_set),
        "metabolite_overlap": len(a.metabolite_set & b.metabolite_set),
        "pair_overlap": len(a.positive_pairs & b.positive_pairs),
    }


def build_rule_banks() -> Dict[str, List[str]]:
    banks: Dict[str, List[str]] = {}
    union_rules: Dict[str, None] = {}
    for bank_name, path in RULE_BANK_SPECS:
        rules = read_rule_file(path)
        banks[bank_name] = rules
        for rule in rules:
            union_rules.setdefault(rule, None)
    banks["union_all"] = list(union_rules.keys())
    return banks


def round_floats(payload: object) -> object:
    if isinstance(payload, float):
        return round(payload, 8)
    if isinstance(payload, dict):
        return {key: round_floats(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [round_floats(item) for item in payload]
    return payload


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bank_rules = build_rule_banks()
    rule_records, _membership, reverse_membership, bank_metadata = build_rule_registry(bank_rules)
    print(
        "Loaded rule banks: "
        + ", ".join(f"{name}={len(rules)}" for name, rules in bank_rules.items()),
        flush=True,
    )
    print(
        f"Unique rules across all banks: {len(rule_records)} "
        f"({sum(1 for record in rule_records if record.parsed)} parsed)",
        flush=True,
    )

    splits = {
        name: load_positive_split(name, sdf_path, triples_path)
        for name, sdf_path, triples_path in LEAKAGE_SPLITS
    }

    print(
        f"[test] Summary: {len(splits['test'].substrate_set)} unique substrates, "
        f"{len(splits['test'].metabolite_set)} unique metabolites, "
        f"{len(splits['test'].positive_pairs)} positive pairs",
        flush=True,
    )

    test_summary, gap_analysis = evaluate_split(
        splits["test"],
        rule_records,
        reverse_membership,
        bank_metadata,
        union_gap_analysis=True,
    )
    val_summary, _ = evaluate_split(
        splits["val"],
        rule_records,
        reverse_membership,
        bank_metadata,
        union_gap_analysis=False,
    )

    rule_banks_payload: Dict[str, Dict[str, object]] = {}
    for bank_name, metadata in bank_metadata.items():
        rule_banks_payload[bank_name] = {
            "path": metadata["path"],
            "n_rules_total": metadata["n_rules_total"],
            "n_rules_parsed": metadata["n_rules_parsed"],
            "test_upper_bound_recall": test_summary[bank_name]["upper_bound_recall"],
            "test_mean_applicable_rules": test_summary[bank_name]["mean_applicable_rules"],
            "test_mean_candidates": test_summary[bank_name]["mean_candidates"],
            "test_median_per_substrate_recall": test_summary[bank_name]["median_per_substrate_recall"],
            "test_fraction_with_hit": test_summary[bank_name]["fraction_with_hit"],
            "test_total_rule_application_failures": test_summary[bank_name]["total_rule_application_failures"],
            "test_total_rule_timeouts": test_summary[bank_name]["total_rule_timeouts"],
            "test_total_sanitize_failures": test_summary[bank_name]["total_sanitize_failures"],
            "test_total_capped_rule_runs": test_summary[bank_name]["total_capped_rule_runs"],
            "val_upper_bound_recall": val_summary[bank_name]["upper_bound_recall"],
            "val_mean_applicable_rules": val_summary[bank_name]["mean_applicable_rules"],
            "val_mean_candidates": val_summary[bank_name]["mean_candidates"],
            "val_median_per_substrate_recall": val_summary[bank_name]["median_per_substrate_recall"],
            "val_fraction_with_hit": val_summary[bank_name]["fraction_with_hit"],
            "val_total_rule_application_failures": val_summary[bank_name]["total_rule_application_failures"],
            "val_total_rule_timeouts": val_summary[bank_name]["total_rule_timeouts"],
            "val_total_sanitize_failures": val_summary[bank_name]["total_sanitize_failures"],
            "val_total_capped_rule_runs": val_summary[bank_name]["total_capped_rule_runs"],
        }

    train_test_overlap = leakage_overlap(splits["train"], splits["test"])
    train_val_overlap = leakage_overlap(splits["train"], splits["val"])
    val_test_overlap = leakage_overlap(splits["val"], splits["test"])

    report = {
        "rule_banks": round_floats(rule_banks_payload),
        "dataset_summary": {
            "test_unique_substrates": len(splits["test"].substrate_set),
            "test_unique_metabolites": len(splits["test"].metabolite_set),
            "test_positive_pairs": len(splits["test"].positive_pairs),
            "val_unique_substrates": len(splits["val"].substrate_set),
            "val_unique_metabolites": len(splits["val"].metabolite_set),
            "val_positive_pairs": len(splits["val"].positive_pairs),
        },
        "gap_analysis": round_floats(gap_analysis),
        "leakage_check_canonical": {
            "train_test_substrate_overlap": train_test_overlap["substrate_overlap"],
            "train_test_metabolite_overlap": train_test_overlap["metabolite_overlap"],
            "train_test_pair_overlap": train_test_overlap["pair_overlap"],
            "train_val_substrate_overlap": train_val_overlap["substrate_overlap"],
            "train_val_metabolite_overlap": train_val_overlap["metabolite_overlap"],
            "train_val_pair_overlap": train_val_overlap["pair_overlap"],
            "val_test_substrate_overlap": val_test_overlap["substrate_overlap"],
            "val_test_metabolite_overlap": val_test_overlap["metabolite_overlap"],
            "val_test_pair_overlap": val_test_overlap["pair_overlap"],
        },
    }

    output_path = RESULTS_DIR / "coverage_report.json"
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {output_path}", flush=True)
    print_summary_table(rule_banks_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

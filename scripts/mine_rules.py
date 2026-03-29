#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fix_splits import read_triples, load_index_to_smiles  # noqa: E402
from scripts.measure_coverage import (  # noqa: E402
    DatasetSplit,
    evaluate_rule_on_substrate,
    load_positive_split,
)

DATA_DIR = ROOT / "grail_metabolism" / "data"
RESOURCES_DIR = ROOT / "grail_metabolism" / "resources"
RESULTS_DIR = ROOT / "results"

TRAIN_SDF = DATA_DIR / "train.sdf"
TRAIN_TRIPLES_CLEAN = DATA_DIR / "train_triples_clean.txt"
VAL_SDF = DATA_DIR / "val.sdf"
VAL_TRIPLES_CLEAN = DATA_DIR / "val_triples_clean.txt"
TEST_SDF = DATA_DIR / "test.sdf"
TEST_TRIPLES_CLEAN = DATA_DIR / "test_triples_clean.txt"

RULE_BANK_PATHS = {
    "smirks.txt": DATA_DIR / "smirks.txt",
    "merged_smirks.txt": DATA_DIR / "merged_smirks.txt",
    "compressed_rules.smarts": ROOT / "grail_metabolism" / "compressed_rules.smarts",
    "notebooks_rules": (
        ROOT / "grail_metabolism" / "resources" / "notebooks_rules.txt"
        if (ROOT / "grail_metabolism" / "resources" / "notebooks_rules.txt").exists()
        else ROOT / "notebooks" / "rules.txt"
    ),
}

MINED_ONLY_PATH = RESOURCES_DIR / "mined_only.txt"
EXTENDED_RULES_PATH = RESOURCES_DIR / "extended_smirks.txt"
RULE_CATALOG_PATH = RESULTS_DIR / "mined_rule_catalog.json"
RULE_MINING_REPORT = RESULTS_DIR / "rule_mining_report.json"
PREVIOUS_COVERAGE_REPORT = RESULTS_DIR / "coverage_report.json"
LEAKAGE_FIX_REPORT = RESULTS_DIR / "leakage_fix_report.json"

MCS_TIMEOUT_SECONDS = 5
MINING_TIME_LIMIT_SECONDS = 4 * 60 * 60
PAIR_PROGRESS_EVERY = 500
FILTER_PROGRESS_EVERY = 200
COVERAGE_PROGRESS_EVERY = 100
RULE_PRODUCT_CAP = 1000
FILTER_RULE_PRODUCT_CAP = 250
FILTER_SAMPLE_SIZE = 50
RANDOM_SEED = 42
FALLBACK_PAIR_LIMIT = 10000


@dataclass
class MiningOutcome:
    smirks: Optional[str]
    metadata: Optional[Dict[str, object]]
    reason: Optional[str]


@dataclass
class RuleCatalogEntry:
    count: int = 0
    source_pairs: List[Tuple[str, str]] = field(default_factory=list)
    metadata_examples: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class GenericRuleRecord:
    rule_id: int
    text: str
    parsed: bool
    rxn: Optional[object] = None
    template: Optional[Chem.Mol] = None
    parse_error: Optional[str] = None


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


def load_clean_positive_pairs() -> Tuple[List[Tuple[str, str]], Dict[str, Chem.Mol]]:
    triples = read_triples(TRAIN_TRIPLES_CLEAN)
    required_ids = {sub_id for sub_id, _, _ in triples} | {prod_id for _, prod_id, _ in triples}
    index_to_smiles, invalid_count = load_index_to_smiles(TRAIN_SDF, required_ids)
    if invalid_count:
        print(f"[mine] Warning: {invalid_count} invalid SDF records encountered while loading clean train", flush=True)

    positive_pairs: List[Tuple[str, str]] = []
    mol_cache: Dict[str, Chem.Mol] = {}
    skipped_equal = 0
    skipped_missing = 0
    for sub_id, prod_id, label in triples:
        if label != 1:
            continue
        sub_smiles = index_to_smiles.get(sub_id)
        prod_smiles = index_to_smiles.get(prod_id)
        if sub_smiles is None or prod_smiles is None:
            skipped_missing += 1
            continue
        if sub_smiles == prod_smiles:
            skipped_equal += 1
            continue
        positive_pairs.append((sub_smiles, prod_smiles))
        if sub_smiles not in mol_cache:
            mol = Chem.MolFromSmiles(sub_smiles)
            if mol is not None:
                mol_cache[sub_smiles] = mol
        if prod_smiles not in mol_cache:
            mol = Chem.MolFromSmiles(prod_smiles)
            if mol is not None:
                mol_cache[prod_smiles] = mol

    print(
        f"[mine] Loaded {len(positive_pairs)} clean positive rows "
        f"(skipped identical pairs: {skipped_equal}, missing ids: {skipped_missing})",
        flush=True,
    )
    return positive_pairs, mol_cache


def find_reaction_center(
    sub_mol: Chem.Mol,
    prod_mol: Chem.Mol,
    sub_match: Sequence[int],
    prod_match: Sequence[int],
) -> Tuple[Set[int], Set[int]]:
    center_sub: Set[int] = set()
    center_prod: Set[int] = set()
    sml = list(sub_match)
    pml = list(prod_match)

    for index in range(len(sml)):
        sa = sub_mol.GetAtomWithIdx(sml[index])
        pa = prod_mol.GetAtomWithIdx(pml[index])
        if (
            sa.GetAtomicNum() != pa.GetAtomicNum()
            or sa.GetFormalCharge() != pa.GetFormalCharge()
            or sa.GetTotalNumHs() != pa.GetTotalNumHs()
            or sa.GetIsAromatic() != pa.GetIsAromatic()
            or sa.GetDegree() != pa.GetDegree()
        ):
            center_sub.add(sml[index])
            center_prod.add(pml[index])

    for left in range(len(sml)):
        for right in range(left + 1, len(sml)):
            sb = sub_mol.GetBondBetweenAtoms(sml[left], sml[right])
            pb = prod_mol.GetBondBetweenAtoms(pml[left], pml[right])
            st = sb.GetBondTypeAsDouble() if sb else 0.0
            pt = pb.GetBondTypeAsDouble() if pb else 0.0
            if st != pt:
                center_sub.update([sml[left], sml[right]])
                center_prod.update([pml[left], pml[right]])

    sub_matched = set(sml)
    prod_matched = set(pml)
    leaving = set(range(sub_mol.GetNumAtoms())) - sub_matched
    entering = set(range(prod_mol.GetNumAtoms())) - prod_matched
    center_sub.update(leaving)
    center_prod.update(entering)

    for sub_idx in list(leaving):
        for neighbor in sub_mol.GetAtomWithIdx(sub_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in sub_matched:
                try:
                    position = sml.index(neighbor_idx)
                except ValueError:
                    continue
                center_sub.add(neighbor_idx)
                center_prod.add(pml[position])

    for prod_idx in list(entering):
        for neighbor in prod_mol.GetAtomWithIdx(prod_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in prod_matched:
                try:
                    position = pml.index(neighbor_idx)
                except ValueError:
                    continue
                center_prod.add(neighbor_idx)
                center_sub.add(sml[position])

    return center_sub, center_prod


def expand_center(mol: Chem.Mol, center: Set[int], radius: int = 1) -> Set[int]:
    expanded = set(center)
    for _ in range(radius):
        frontier: Set[int] = set()
        for atom_idx in expanded:
            for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                frontier.add(neighbor.GetIdx())
        expanded |= frontier
    return expanded


def has_atom_maps(smarts: str) -> bool:
    return bool(re.search(r":\d+\]", smarts))


def build_smirks(
    sub_mol: Chem.Mol,
    prod_mol: Chem.Mol,
    sub_match: Sequence[int],
    prod_match: Sequence[int],
    exp_sub: Set[int],
    exp_prod: Set[int],
) -> Optional[str]:
    sub_rw = Chem.RWMol(Chem.RWMol(sub_mol))
    prod_rw = Chem.RWMol(Chem.RWMol(prod_mol))

    for atom in sub_rw.GetAtoms():
        atom.SetAtomMapNum(0)
    for atom in prod_rw.GetAtoms():
        atom.SetAtomMapNum(0)

    sml = list(sub_match)
    pml = list(prod_match)
    mapnum = 1

    for index in range(len(sml)):
        sub_idx = sml[index]
        prod_idx = pml[index]
        if sub_idx in exp_sub or prod_idx in exp_prod:
            sub_rw.GetAtomWithIdx(sub_idx).SetAtomMapNum(mapnum)
            prod_rw.GetAtomWithIdx(prod_idx).SetAtomMapNum(mapnum)
            mapnum += 1

    for sub_idx in sorted(exp_sub - set(sml)):
        sub_rw.GetAtomWithIdx(sub_idx).SetAtomMapNum(mapnum)
        mapnum += 1

    for prod_idx in sorted(exp_prod - set(pml)):
        prod_rw.GetAtomWithIdx(prod_idx).SetAtomMapNum(mapnum)
        mapnum += 1

    try:
        reactant_smarts = Chem.MolFragmentToSmarts(sub_rw, atomsToUse=sorted(exp_sub))
        product_smarts = Chem.MolFragmentToSmarts(prod_rw, atomsToUse=sorted(exp_prod))
    except Exception:
        return None

    if not has_atom_maps(reactant_smarts) or not has_atom_maps(product_smarts):
        return None

    smirks = f"{reactant_smarts}>>{product_smarts}"
    try:
        rxn = AllChem.ReactionFromSmarts(smirks)
        if rxn is None:
            return None
        rxn.Initialize()
    except Exception:
        return None
    return smirks


def self_test(smirks: str, sub_smi: str, prod_smi: str) -> bool:
    try:
        rxn = AllChem.ReactionFromSmarts(smirks)
        mol = Chem.MolFromSmiles(sub_smi)
        expected = canonicalize_smiles(prod_smi)
        if rxn is None or mol is None or expected is None:
            return False
        for product_tuple in rxn.RunReactants((mol,)):
            for product in product_tuple:
                try:
                    Chem.SanitizeMol(product)
                    if canonicalize_smiles(Chem.MolToSmiles(product)) == expected:
                        return True
                except Exception:
                    continue
        return False
    except Exception:
        return False


def process_pair(sub_smi: str, prod_smi: str) -> MiningOutcome:
    try:
        sub_mol = Chem.MolFromSmiles(sub_smi)
        prod_mol = Chem.MolFromSmiles(prod_smi)
    except Exception:
        return MiningOutcome(None, None, "parse_fail")

    if sub_mol is None or prod_mol is None:
        return MiningOutcome(None, None, "parse_fail")

    try:
        mcs = rdFMCS.FindMCS(
            [sub_mol, prod_mol],
            timeout=MCS_TIMEOUT_SECONDS,
            matchValences=False,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )
    except Exception:
        return MiningOutcome(None, None, "mcs_error")

    if mcs.canceled:
        return MiningOutcome(None, None, "mcs_timeout")
    if mcs.numAtoms == 0:
        return MiningOutcome(None, None, "mcs_empty")

    min_atoms = min(sub_mol.GetNumAtoms(), prod_mol.GetNumAtoms())
    if mcs.numAtoms < 0.4 * min_atoms:
        return MiningOutcome(None, None, "mcs_too_small")

    try:
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    except Exception:
        return MiningOutcome(None, None, "mcs_parse_fail")
    if mcs_mol is None:
        return MiningOutcome(None, None, "mcs_parse_fail")

    try:
        sub_matches = sub_mol.GetSubstructMatches(mcs_mol, maxMatches=10)
        prod_matches = prod_mol.GetSubstructMatches(mcs_mol, maxMatches=10)
    except Exception:
        return MiningOutcome(None, None, "no_match")

    if not sub_matches or not prod_matches:
        return MiningOutcome(None, None, "no_match")

    best = None
    best_center_size = float("inf")

    for sub_match in sub_matches[:5]:
        for prod_match in prod_matches[:5]:
            center_sub, center_prod = find_reaction_center(sub_mol, prod_mol, sub_match, prod_match)
            center_size = len(center_sub) + len(center_prod)
            if 0 < center_size < best_center_size:
                best_center_size = center_size
                best = (sub_match, prod_match, center_sub, center_prod)

    if best is None:
        return MiningOutcome(None, None, "no_center")

    sub_match, prod_match, center_sub, center_prod = best
    exp_sub = expand_center(sub_mol, center_sub, radius=1)
    exp_prod = expand_center(prod_mol, center_prod, radius=1)
    smirks = build_smirks(sub_mol, prod_mol, sub_match, prod_match, exp_sub, exp_prod)
    if smirks is None:
        return MiningOutcome(None, None, "build_fail")

    if not self_test(smirks, sub_smi, prod_smi):
        return MiningOutcome(None, None, "selftest_fail")

    return MiningOutcome(
        smirks=smirks,
        metadata={
            "source_sub": sub_smi,
            "source_prod": prod_smi,
            "center_size": int(best_center_size),
            "expanded_sub_size": len(exp_sub),
            "expanded_prod_size": len(exp_prod),
        },
        reason=None,
    )


def read_rule_file(path: Path) -> List[str]:
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def save_rule_file(path: Path, rules: Iterable[str]) -> None:
    with open(path, "w") as handle:
        for rule in rules:
            handle.write(f"{rule}\n")


def filter_rule_candidates(
    unique_rules: Dict[str, RuleCatalogEntry],
    training_substrates: Sequence[str],
) -> Tuple[List[str], Dict[str, Dict[str, float]], Counter[str]]:
    rng = random.Random(RANDOM_SEED)
    sample_substrates = list(training_substrates)
    rng.shuffle(sample_substrates)
    sample_substrates = sample_substrates[: min(FILTER_SAMPLE_SIZE, len(sample_substrates))]
    sample_mols = []
    for smiles in sample_substrates:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            sample_mols.append((smiles, mol))

    print(f"[mine] Filtering {len(unique_rules)} unique rules on {len(sample_mols)} sampled substrates", flush=True)

    kept_rules: List[str] = []
    details: Dict[str, Dict[str, float]] = {}
    rejection_stats: Counter[str] = Counter()

    for index, rule in enumerate(unique_rules.keys(), start=1):
        if index == 1 or index % FILTER_PROGRESS_EVERY == 0:
            print(f"[mine] Filter progress: {index}/{len(unique_rules)} rules, kept={len(kept_rules)}", flush=True)

        try:
            rxn = AllChem.ReactionFromSmarts(rule)
        except Exception:
            rxn = None
        if rxn is None or rxn.GetNumReactantTemplates() < 1:
            rejection_stats["parse_fail"] += 1
            continue
        template = rxn.GetReactantTemplate(0)
        if template is None:
            rejection_stats["parse_fail"] += 1
            continue

        applicable_count = 0
        total_products = 0
        for _, mol in sample_mols:
            try:
                applicable = mol.HasSubstructMatch(template)
            except Exception:
                applicable = False
            if not applicable:
                continue
            applicable_count += 1
            try:
                outcomes = rxn.RunReactants((mol,), maxProducts=FILTER_RULE_PRODUCT_CAP)
            except Exception:
                continue
            product_set: Set[str] = set()
            for product_tuple in outcomes:
                for product in product_tuple:
                    if len(product_set) >= FILTER_RULE_PRODUCT_CAP:
                        break
                    try:
                        Chem.SanitizeMol(product)
                        canonical = canonicalize_smiles(Chem.MolToSmiles(product))
                    except Exception:
                        canonical = None
                    if canonical:
                        product_set.add(canonical)
                if len(product_set) >= FILTER_RULE_PRODUCT_CAP:
                    break
            total_products += len(product_set)

        mean_products = (total_products / applicable_count) if applicable_count else 0.0
        applicability_fraction = (applicable_count / len(sample_mols)) if sample_mols else 0.0

        if mean_products > 200:
            rejection_stats["too_unselective"] += 1
            continue
        if applicability_fraction > 0.9 and mean_products > 50:
            rejection_stats["too_general_and_prolific"] += 1
            continue

        kept_rules.append(rule)
        details[rule] = {
            "applicable_count": applicable_count,
            "sample_size": len(sample_mols),
            "applicability_fraction": applicability_fraction,
            "mean_products": mean_products,
        }

    return kept_rules, details, rejection_stats


def build_rule_registry(bank_specs: Dict[str, Tuple[Path, List[str]]]) -> Tuple[List[GenericRuleRecord], Dict[str, Counter[int]], Dict[int, Dict[str, int]], Dict[str, Dict[str, object]]]:
    unique_rules: Dict[str, int] = {}
    membership: Dict[str, Counter[int]] = {}
    metadata: Dict[str, Dict[str, object]] = {}

    for bank_name, (path, rules) in bank_specs.items():
        counter: Counter[int] = Counter()
        for rule_text in rules:
            if rule_text not in unique_rules:
                unique_rules[rule_text] = len(unique_rules)
            counter[unique_rules[rule_text]] += 1
        membership[bank_name] = counter
        metadata[bank_name] = {
            "path": str(path),
            "n_rules_total": len(rules),
            "n_rules_parsed": 0,
        }

    records: List[GenericRuleRecord] = []
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
        records.append(
            GenericRuleRecord(
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
        metadata[bank_name]["n_rules_parsed"] = sum(
            multiplicity
            for rule_id, multiplicity in counter.items()
            if records[rule_id].parsed
        )
        for rule_id, multiplicity in counter.items():
            reverse_membership[rule_id][bank_name] = multiplicity

    return records, membership, reverse_membership, metadata


def measure_rule_banks(
    split: DatasetSplit,
    bank_specs: Dict[str, Tuple[Path, List[str]]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Set[Tuple[str, str]]]]:
    records, _, reverse_membership, metadata = build_rule_registry(bank_specs)
    metrics: Dict[str, Dict[str, object]] = {}
    recovered_pairs: Dict[str, Set[Tuple[str, str]]] = {name: set() for name in bank_specs}
    for bank_name, bank_meta in metadata.items():
        metrics[bank_name] = {
            "path": bank_meta["path"],
            "n_rules_total": int(bank_meta["n_rules_total"]),
            "n_rules_parsed": int(bank_meta["n_rules_parsed"]),
            "total_true_recovered": 0,
            "total_true_metabolites": 0,
            "sum_applicable_rules": 0,
            "sum_candidates": 0,
        }

    items = sorted(split.substrate_to_products.items())
    start = time.perf_counter()
    for substrate_index, (substrate_smiles, true_products) in enumerate(items, start=1):
        if substrate_index == 1 or substrate_index % COVERAGE_PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[coverage] Progress {substrate_index}/{len(items)} substrates "
                f"({elapsed:.1f}s elapsed)",
                flush=True,
            )

        try:
            mol = Chem.MolFromSmiles(substrate_smiles)
        except Exception:
            mol = None

        applicable_counts = {name: 0 for name in bank_specs}
        candidate_sets = {name: set() for name in bank_specs}

        if mol is not None:
            for record in records:
                membership = reverse_membership.get(record.rule_id)
                if not membership:
                    continue
                result = evaluate_rule_on_substrate(mol, record)
                if not result.applicable:
                    continue
                for bank_name, multiplicity in membership.items():
                    applicable_counts[bank_name] += multiplicity
                    if result.products:
                        candidate_sets[bank_name].update(result.products)

        for bank_name in bank_specs:
            recovered = candidate_sets[bank_name] & true_products
            metrics[bank_name]["total_true_recovered"] += len(recovered)
            metrics[bank_name]["total_true_metabolites"] += len(true_products)
            metrics[bank_name]["sum_applicable_rules"] += applicable_counts[bank_name]
            metrics[bank_name]["sum_candidates"] += len(candidate_sets[bank_name])
            for product_smiles in recovered:
                recovered_pairs[bank_name].add((substrate_smiles, product_smiles))

    n_substrates = len(items)
    summary: Dict[str, Dict[str, float]] = {}
    for bank_name, values in metrics.items():
        total_true = int(values["total_true_metabolites"])
        total_recovered = int(values["total_true_recovered"])
        summary[bank_name] = {
            "path": str(values["path"]),
            "n_rules": int(values["n_rules_total"]),
            "n_rules_parsed": int(values["n_rules_parsed"]),
            "test_recall_ceiling": (total_recovered / total_true) if total_true else 0.0,
            "mean_applicable_rules": (float(values["sum_applicable_rules"]) / n_substrates) if n_substrates else 0.0,
            "mean_candidates": (float(values["sum_candidates"]) / n_substrates) if n_substrates else 0.0,
        }
    return summary, recovered_pairs


def load_previous_gap_pairs() -> List[Dict[str, object]]:
    if not PREVIOUS_COVERAGE_REPORT.exists():
        return []
    with open(PREVIOUS_COVERAGE_REPORT) as handle:
        payload = json.load(handle)
    return payload.get("gap_analysis", {}).get("uncovered_pairs", [])


def load_previous_union_baseline() -> Tuple[int, float]:
    if not PREVIOUS_COVERAGE_REPORT.exists():
        return 1715, 0.3758
    with open(PREVIOUS_COVERAGE_REPORT) as handle:
        payload = json.load(handle)
    bank = payload.get("rule_banks", {}).get("union_all", {})
    n_rules = int(bank.get("n_rules_total", 1715))
    recall = float(bank.get("test_upper_bound_recall", 0.3758))
    return n_rules, recall


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_TRIPLES_CLEAN.exists():
        raise FileNotFoundError(f"Missing clean train triples: {TRAIN_TRIPLES_CLEAN}")
    if not TEST_TRIPLES_CLEAN.exists():
        raise FileNotFoundError(f"Missing clean test triples: {TEST_TRIPLES_CLEAN}")

    positive_rows, mol_cache = load_clean_positive_pairs()
    training_substrates = sorted({sub for sub, _ in positive_rows})
    total_positive_pairs = len(positive_rows)

    start_time = time.perf_counter()
    pair_cache: Dict[Tuple[str, str], MiningOutcome] = {}
    stats: Counter[str] = Counter()
    raw_successes = 0
    unique_rules: Dict[str, RuleCatalogEntry] = {}
    target_limit = total_positive_pairs
    adjusted_target = False

    for index, pair in enumerate(positive_rows, start=1):
        elapsed = time.perf_counter() - start_time
        if elapsed > MINING_TIME_LIMIT_SECONDS:
            print(f"[mine] Time limit reached at pair {index - 1}; stopping early", flush=True)
            break
        if index > target_limit:
            print(f"[mine] Reached capped pair limit {target_limit}; stopping mining loop", flush=True)
            break

        if pair not in pair_cache:
            pair_cache[pair] = process_pair(*pair)
        outcome = pair_cache[pair]

        if outcome.smirks is not None and outcome.metadata is not None:
            stats["success"] += 1
            raw_successes += 1
            entry = unique_rules.setdefault(outcome.smirks, RuleCatalogEntry())
            entry.count += 1
            entry.source_pairs.append(pair)
            if len(entry.metadata_examples) < 5:
                entry.metadata_examples.append(dict(outcome.metadata))
        else:
            stats[outcome.reason or "unknown_error"] += 1

        if index == PAIR_PROGRESS_EVERY and not adjusted_target and total_positive_pairs > FALLBACK_PAIR_LIMIT:
            estimated_total_seconds = elapsed / index * total_positive_pairs if index else 0.0
            if estimated_total_seconds > MINING_TIME_LIMIT_SECONDS:
                target_limit = min(total_positive_pairs, FALLBACK_PAIR_LIMIT)
                adjusted_target = True
                print(
                    f"[mine] Estimated full runtime {estimated_total_seconds / 3600:.2f}h exceeds limit; "
                    f"capping processing to first {target_limit} pairs",
                    flush=True,
                )

        if index == 1 or index % PAIR_PROGRESS_EVERY == 0:
            failures = index - stats["success"]
            print(
                f"[mine] Pair {index}/{target_limit if adjusted_target else total_positive_pairs} "
                f"- {raw_successes} rules extracted so far, {failures} failures, {len(unique_rules)} unique raw rules",
                flush=True,
            )

    pairs_processed = min(target_limit, total_positive_pairs, sum(stats.values()))
    processing_time = time.perf_counter() - start_time

    for key in (
        "success",
        "parse_fail",
        "mcs_error",
        "mcs_timeout",
        "mcs_empty",
        "mcs_too_small",
        "mcs_parse_fail",
        "no_match",
        "no_center",
        "build_fail",
        "selftest_fail",
    ):
        stats.setdefault(key, 0)

    with open(RULE_CATALOG_PATH, "w") as handle:
        json.dump(
            {
                rule: {
                    "count": entry.count,
                    "source_pairs": entry.source_pairs,
                    "metadata_examples": entry.metadata_examples,
                }
                for rule, entry in unique_rules.items()
            },
            handle,
            indent=2,
        )

    kept_mined_rules, filter_details, filter_rejections = filter_rule_candidates(unique_rules, training_substrates)
    stats.update({f"filter_{key}": value for key, value in filter_rejections.items()})

    existing_rule_union: Dict[str, None] = {}
    existing_banks: Dict[str, List[str]] = {}
    for bank_name, path in RULE_BANK_PATHS.items():
        rules = read_rule_file(path)
        existing_banks[bank_name] = rules
        for rule in rules:
            existing_rule_union.setdefault(rule, None)

    novel_mined_rules = [rule for rule in kept_mined_rules if rule not in existing_rule_union]
    extended_union: Dict[str, None] = dict(existing_rule_union)
    for rule in kept_mined_rules:
        extended_union.setdefault(rule, None)

    save_rule_file(MINED_ONLY_PATH, kept_mined_rules)
    save_rule_file(EXTENDED_RULES_PATH, extended_union.keys())

    print(
        f"[mine] Saved {len(kept_mined_rules)} filtered mined rules to {MINED_ONLY_PATH} "
        f"and {len(extended_union)} extended rules to {EXTENDED_RULES_PATH}",
        flush=True,
    )

    test_split = load_positive_split("test_clean", TEST_SDF, TEST_TRIPLES_CLEAN)
    bank_specs = {
        "mined_only": (MINED_ONLY_PATH, kept_mined_rules),
        "notebooks_rules": (RULE_BANK_PATHS["notebooks_rules"], existing_banks["notebooks_rules"]),
        "extended_all": (EXTENDED_RULES_PATH, list(extended_union.keys())),
    }
    bank_results, recovered_pairs = measure_rule_banks(test_split, bank_specs)

    previous_union_rules, previous_union_recall = load_previous_union_baseline()
    previous_gap_pairs = load_previous_gap_pairs()
    gap_targets = Counter(item.get("type") for item in previous_gap_pairs if item.get("type"))
    gap_targets = {
        "glucuronidation": int(gap_targets.get("glucuronidation", 278)),
        "hydroxylation": int(gap_targets.get("hydroxylation", 168)),
        "sulfation": int(gap_targets.get("sulfation", 102)),
    }
    gap_closure_counts = {key: 0 for key in gap_targets}
    extended_recovered = recovered_pairs.get("extended_all", set())
    for item in previous_gap_pairs:
        gap_type = item.get("type")
        if gap_type not in gap_targets:
            continue
        pair = (item.get("substrate_smi"), item.get("metabolite_smi"))
        if pair in extended_recovered:
            gap_closure_counts[gap_type] += 1

    leakage_payload = {}
    if LEAKAGE_FIX_REPORT.exists():
        with open(LEAKAGE_FIX_REPORT) as handle:
            leakage_report = json.load(handle)
        leakage_payload = {
            "train_removed_substrates": leakage_report["cleaning"]["train"]["removed_substrates"],
            "train_removed_triples": leakage_report["cleaning"]["train"]["removed_triples"],
            "val_removed_substrates": leakage_report["cleaning"]["val"]["removed_substrates"],
            "val_removed_triples": leakage_report["cleaning"]["val"]["removed_triples"],
            "clean_train_substrates": leakage_report["cleaning"]["train"]["remaining_substrates"],
            "clean_train_positive_pairs": leakage_report["cleaning"]["train"]["remaining_positive_pairs"],
            "clean_val_substrates": leakage_report["cleaning"]["val"]["remaining_substrates"],
            "clean_test_substrates": leakage_report["cleaning"]["test"]["remaining_substrates"],
            "zero_substrate_overlap_verified": leakage_report["verification"]["zero_substrate_overlap_between_clean_splits"],
        }

    report = {
        "leakage_fix": leakage_payload,
        "mining": {
            "total_positive_pairs": total_positive_pairs,
            "pairs_processed": pairs_processed,
            "unique_pairs_evaluated": len(pair_cache),
            "stats": dict(stats),
            "raw_rules": raw_successes,
            "after_dedup": len(unique_rules),
            "after_filtering": len(kept_mined_rules),
            "novel_rules": len(novel_mined_rules),
            "processing_time_seconds": round(processing_time, 4),
            "target_limit": target_limit,
            "filter_rejections": dict(filter_rejections),
        },
        "banks": {
            "mined_only": {
                "n_rules": bank_results["mined_only"]["n_rules"],
                "test_recall_ceiling": bank_results["mined_only"]["test_recall_ceiling"],
            },
            "notebooks_rules": {
                "n_rules": bank_results["notebooks_rules"]["n_rules"],
                "test_recall_ceiling": bank_results["notebooks_rules"]["test_recall_ceiling"],
            },
            "extended_all": {
                "n_rules": bank_results["extended_all"]["n_rules"],
                "test_recall_ceiling": bank_results["extended_all"]["test_recall_ceiling"],
            },
            "previous_union_all": {
                "n_rules": previous_union_rules,
                "test_recall_ceiling": previous_union_recall,
            },
        },
        "gap_closure": {
            "glucuronidation_previously_uncovered": gap_targets["glucuronidation"],
            "glucuronidation_now_covered": gap_closure_counts["glucuronidation"],
            "hydroxylation_previously_uncovered": gap_targets["hydroxylation"],
            "hydroxylation_now_covered": gap_closure_counts["hydroxylation"],
            "sulfation_previously_uncovered": gap_targets["sulfation"],
            "sulfation_now_covered": gap_closure_counts["sulfation"],
        },
        "artifacts": {
            "mined_only_path": str(MINED_ONLY_PATH),
            "extended_rules_path": str(EXTENDED_RULES_PATH),
            "rule_catalog_path": str(RULE_CATALOG_PATH),
        },
        "filter_details_path": str(RESULTS_DIR / "mined_filter_details.json"),
    }

    with open(RESULTS_DIR / "mined_filter_details.json", "w") as handle:
        json.dump(filter_details, handle, indent=2)
    with open(RULE_MINING_REPORT, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\nSummary table", flush=True)
    print("| Bank | Rules | Test Recall Ceiling | Δ vs union_all |", flush=True)
    print("|------|-------|--------------------:|---------------:|", flush=True)
    print(f"| previous union_all | {previous_union_rules} | {previous_union_recall:.4f} | baseline |", flush=True)
    print(
        f"| mined_only | {bank_results['mined_only']['n_rules']} | "
        f"{bank_results['mined_only']['test_recall_ceiling']:.4f} | "
        f"{bank_results['mined_only']['test_recall_ceiling'] - previous_union_recall:+.4f} |",
        flush=True,
    )
    print(
        f"| extended (union+mined) | {bank_results['extended_all']['n_rules']} | "
        f"{bank_results['extended_all']['test_recall_ceiling']:.4f} | "
        f"{bank_results['extended_all']['test_recall_ceiling'] - previous_union_recall:+.4f} |",
        flush=True,
    )
    print(f"Wrote {RULE_MINING_REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

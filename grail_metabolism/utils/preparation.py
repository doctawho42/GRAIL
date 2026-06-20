from __future__ import annotations

import os
import re
import signal
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, DefaultDict, Dict, FrozenSet, Iterable, Iterator, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from .transform import EDGE_DIM, FINGERPRINT_DIM, PAIR_NODE_DIM, SINGLE_NODE_DIM, apply_feature_projection, from_pair, from_rdmol

warnings.filterwarnings("ignore")
tqdm.pandas()
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

_UNCHARGER = rdMolStandardize.Uncharger()
_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    includeChirality=True,
    fpSize=1024,
    countSimulation=False,
)
_REACTION_PRODUCT_CAP = 500
_REACTION_TIMEOUT_SECONDS = 5.0
_PAIR_GRAPH_TIMEOUT_SECONDS = 3.0


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _local_data_dir() -> Path:
    return _package_root() / "data"


def _resources_dir() -> Path:
    return _package_root() / "resources"


def _normalize_cache_path(path: Union[str, Path, None]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path)


def _atomic_torch_save(payload: Any, path: Union[str, Path]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def _load_cached_mapping(path: Union[str, Path, None]) -> Dict[str, Any]:
    cache_path = _normalize_cache_path(path)
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def carbon_counter(smiles: str) -> int:
    return smiles.count("C") + smiles.count("c") - sum(
        smiles.count(token) for token in ["Cl", "Ca", "Co", "Sc", "Cr", "Cd", "Cs"]
    )


# Minimum heavy-atom count a fragment must have to be retained. The previous
# heuristic required >= 3 carbons, which silently discarded entire classes of
# real small metabolites (formaldehyde, formate, acetaldehyde, glycine/acetyl
# conjugate fragments, small N-/O-dealkylation products). A heavy-atom floor of 2
# keeps every realistic small metabolite while still dropping lone-atom inorganic
# leaving groups (water, halide, ammonia).
_MIN_FRAGMENT_HEAVY_ATOMS = 2


def iscorrect(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumHeavyAtoms() >= _MIN_FRAGMENT_HEAVY_ATOMS


def atom_counter(smiles: str) -> int:
    return len(re.findall(r"((?<=\[)[A-Z][a-z])|((?!\[)[A-GI-Za-z])", smiles))


def cpunum(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def extract(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    fragment = max(smiles.split("."), key=atom_counter)
    return fragment if iscorrect(fragment) else None


def _normalize_return_type(mol: Chem.Mol, return_smiles: bool) -> Union[Chem.Mol, str]:
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
    if return_smiles:
        return canonical
    return Chem.MolFromSmiles(canonical)


def standardize_mol(mol: Union[Chem.Mol, str], ph: Optional[float] = None) -> Union[Chem.Mol, str]:
    return_smiles = isinstance(mol, str)
    if return_smiles:
        parsed = Chem.MolFromSmiles(mol)
        if parsed is None:
            raise ValueError(f"Failed to parse SMILES: {mol}")
        mol = parsed

    cleaned = rdMolStandardize.Cleanup(mol)
    parent = rdMolStandardize.FragmentParent(cleaned)

    if ph is None:
        neutral = _UNCHARGER.uncharge(parent)
    else:
        try:
            from dimorphite_dl import DimorphiteDL
        except ImportError as exc:
            raise RuntimeError("dimorphite_dl is required for pH-specific standardization") from exc
        protonator = DimorphiteDL(
            min_ph=ph,
            max_ph=ph,
            max_variants=128,
            label_states=False,
            pka_precision=1.0,
        )
        candidates = protonator.protonate(Chem.MolToSmiles(parent))
        neutral = Chem.MolFromSmiles(candidates[0]) if candidates else parent

    tautomer = _TAUTOMER_ENUMERATOR.Canonicalize(neutral)
    if tautomer is None:
        raise ValueError("Failed to canonicalize molecule")
    return _normalize_return_type(tautomer, return_smiles)


@lru_cache(maxsize=262144)
def _standardize_smiles_cached(smiles: str) -> str:
    return str(standardize_mol(smiles))


@lru_cache(maxsize=262144)
def _canonicalize_smiles_cached(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _normalize_smiles_cached(smiles: str, mode: Literal["standardize", "canonical"]) -> str:
    if mode == "standardize":
        return _standardize_smiles_cached(smiles)
    return _canonicalize_smiles_cached(smiles)


def _smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


@lru_cache(maxsize=32768)
def _compile_rule_reaction(rule: str):
    try:
        return AllChem.ReactionFromSmarts(rule)
    except Exception:
        return None


@lru_cache(maxsize=32768)
def _compile_rule_pattern(rule: str) -> Optional[Chem.Mol]:
    try:
        reactant = rule.split(">>", 1)[0].strip()
        if reactant.startswith("(") and reactant.endswith(")"):
            reactant = reactant[1:-1]
        return Chem.MolFromSmarts(reactant)
    except Exception:
        return None


class ReactionTimeoutError(Exception):
    pass


class PairGraphTimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise ReactionTimeoutError()


def _pair_timeout_handler(signum: int, frame: object) -> None:
    raise PairGraphTimeoutError()


def safe_run_reactants(
    rxn,
    substrate: Chem.Mol,
    timeout: float = _REACTION_TIMEOUT_SECONDS,
    max_products: int = _REACTION_PRODUCT_CAP,
):
    if rxn is None or substrate is None:
        return ()
    if not hasattr(signal, "setitimer"):
        try:
            return rxn.RunReactants((substrate,), maxProducts=max_products)
        except Exception:
            return ()

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        return rxn.RunReactants((substrate,), maxProducts=max_products)
    except ReactionTimeoutError:
        return ()
    except Exception:
        return ()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _pair_graph_without_cross_edges(mol1: Chem.Mol, mol2: Chem.Mol) -> Optional[Data]:
    graph1 = from_rdmol(mol1)
    graph2 = from_rdmol(mol2)
    if graph1 is None or graph2 is None:
        return None

    x1 = torch.cat(
        (
            graph1.x,
            torch.zeros((graph1.x.size(0), 1), dtype=torch.float32),
            torch.zeros((graph1.x.size(0), 1), dtype=torch.float32),
        ),
        dim=1,
    )
    x2 = torch.cat(
        (
            graph2.x,
            torch.ones((graph2.x.size(0), 1), dtype=torch.float32),
            torch.zeros((graph2.x.size(0), 1), dtype=torch.float32),
        ),
        dim=1,
    )
    x = torch.cat((x1, x2), dim=0).view(-1, PAIR_NODE_DIM)

    edge_indices: List[Tensor] = []
    edge_attrs: List[Tensor] = []
    if graph1.edge_index.numel():
        edge_indices.append(graph1.edge_index.clone())
        edge_attrs.append(graph1.edge_attr.clone())
    if graph2.edge_index.numel():
        offset = graph1.x.size(0)
        edge_indices.append(graph2.edge_index.clone() + offset)
        edge_attrs.append(graph2.edge_attr.clone())

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float32)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.fp = torch.cat((graph1.fp.float(), graph2.fp.float()), dim=1)
    graph.fallback_pair = 1
    return graph


def build_pair_graph(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    timeout: float = _PAIR_GRAPH_TIMEOUT_SECONDS,
) -> Optional[Data]:
    if mol1 is None or mol2 is None:
        return None

    if not hasattr(signal, "setitimer"):
        try:
            graph = from_pair(mol1, mol2)
        except Exception:
            graph = None
        if graph is None:
            graph = _pair_graph_without_cross_edges(mol1, mol2)
        if graph is not None and not hasattr(graph, "fallback_pair"):
            graph.fallback_pair = 0
        return graph

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _pair_timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        graph = from_pair(mol1, mol2)
    except PairGraphTimeoutError:
        graph = None
    except Exception:
        graph = None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

    if graph is None:
        graph = _pair_graph_without_cross_edges(mol1, mol2)
    if graph is not None and not hasattr(graph, "fallback_pair"):
        graph.fallback_pair = 0
    return graph


def _iter_reaction_products(substrate: Chem.Mol, rule: str) -> Iterator[Chem.Mol]:
    pattern = _compile_rule_pattern(rule)
    rxn = _compile_rule_reaction(rule)
    if pattern is None or rxn is None:
        return iter(())
    try:
        if not substrate.HasSubstructMatch(pattern):
            return iter(())
    except Exception:
        return iter(())
    outcomes = safe_run_reactants(rxn, substrate)
    flattened = []
    for product_tuple in outcomes:
        flattened.extend(product_tuple)
    return iter(flattened)


def _clean_product_smiles(smiles: str) -> List[str]:
    valid = []
    for fragment in smiles.split("."):
        fragment = fragment.strip()
        if fragment and iscorrect(fragment):
            valid.append(fragment)
    return valid


def apply_rules_to_molecule(
    mol: Chem.Mol,
    rules: Sequence[str],
    normalization_mode: Literal["standardize", "canonical"] = "standardize",
) -> DefaultDict[str, Set[int]]:
    products: DefaultDict[str, Set[int]] = defaultdict(set)
    if mol is None:
        return products

    substrate = Chem.AddHs(Chem.Mol(mol))
    for rule_index, rule in enumerate(rules):
        for product in _iter_reaction_products(substrate, rule):
            try:
                smiles = Chem.MolToSmiles(product)
            except Exception:
                continue
            for fragment in _clean_product_smiles(smiles):
                try:
                    normalized = _normalize_smiles_cached(fragment, normalization_mode)
                except Exception:
                    continue
                if normalized:
                    products[normalized].add(rule_index)
    return products


def metaboliser(
    mol: Chem.Mol,
    rules: Optional[Iterable[str]] = None,
    normalization_mode: Literal["standardize", "canonical"] = "standardize",
) -> DefaultDict[str, Set[int]]:
    selected_rules = list(rules) if rules is not None else load_default_rules()
    return apply_rules_to_molecule(mol, selected_rules, normalization_mode=normalization_mode)


def generate_vectors(
    reaction_dict: Dict[str, Dict[str, Iterable[int]]],
    real_products_dict: Dict[str, Iterable[str]],
    num_rules: int,
    normalization_mode: Literal["standardize", "canonical"] = "standardize",
) -> Dict[str, List[int]]:
    def normalize_label(value: str) -> str:
        try:
            return _normalize_smiles_cached(str(value), normalization_mode)
        except Exception:
            return value

    vectors: Dict[str, List[int]] = {}
    normalized_real = {
        substrate: {normalize_label(product) for product in products}
        for substrate, products in real_products_dict.items()
    }
    for substrate, product_map in reaction_dict.items():
        vector = [0] * num_rules
        allowed = normalized_real.get(substrate, set())
        for product, indexes in product_map.items():
            normalized_product = normalize_label(product)
            if normalized_product not in allowed:
                continue
            for index in indexes:
                if 0 <= index < num_rules:
                    vector[index] = 1
        vectors[substrate] = vector
    return vectors


@lru_cache(maxsize=4)
def _load_pickle(path: str) -> Any:
    import pickle

    with open(path, "rb") as handle:
        return pickle.load(handle)


def _maybe_project_graph(graph: Data, node_path: Optional[Path], edge_path: Optional[Path]) -> Data:
    node_projector = _load_pickle(str(node_path)) if node_path and node_path.exists() else None
    edge_projector = _load_pickle(str(edge_path)) if edge_path and edge_path.exists() else None
    if node_projector is None and edge_projector is None:
        return graph
    return apply_feature_projection(graph, node_projector=node_projector, edge_projector=edge_projector)


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _default_rule_bank_candidates() -> List[Path]:
    """Canonical, ordered preference list for the default rule bank.

    Single source of truth so every entry point (presets, load_default_rules,
    PretrainedGrail) resolves to the SAME bank. Previously these disagreed: training
    used resources/extended_smirks.txt (7581 rules) while packaged inference loaded
    data/merged_smirks.txt (656), a train/inference label-space mismatch.
    """
    resources = _resources_dir()
    data = _local_data_dir()
    return [
        resources / "extended_smirks.txt",
        resources / "mined_only.txt",
        resources / "notebooks_rules.txt",
        data / "merged_smirks.txt",
        data / "smirks.txt",
        resources / "example_rules.txt",
    ]


def resolve_default_rule_bank() -> Optional[Path]:
    return _first_existing(_default_rule_bank_candidates())


def load_default_rules() -> List[str]:
    candidate = resolve_default_rule_bank()
    if candidate is None:
        return []
    with open(candidate) as handle:
        return [line.strip() for line in handle if line.strip()]


def load_phase2_rules() -> List[str]:
    """Curated phase II conjugation SMIRKS (glucuronidation, sulfation, methylation,
    N-acetylation, glycine/taurine conjugation, GSH conjugation), keyed to the correct
    acceptor functional groups. Each rule is validated to compile and fire under RDKit.
    Closes the phase II coverage gap flagged as the main rule-based SOTA limitation."""
    path = _resources_dir() / "phase2_conjugation.smarts"
    if not path.exists():
        return []
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


class MolFrame:
    def __init__(
        self,
        data: Union[pd.DataFrame, Dict[str, Iterable[str]]],
        sub_name: str = "sub",
        prod_name: str = "prod",
        real_name: str = "real",
        gen_map: Optional[DefaultDict[str, Set[str]]] = None,
        mol_structs: Optional[Dict[str, Chem.Mol]] = None,
        standartize: bool = True,
        normalization_mode: Optional[Literal["standardize", "canonical"]] = None,
    ) -> None:
        self.map: DefaultDict[str, Set[str]] = defaultdict(set)
        self.gen_map: DefaultDict[str, Set[str]] = defaultdict(set)
        self.negs: DefaultDict[str, Set[str]] = defaultdict(set)
        self.graphs: DefaultDict[str, List[Data]] = defaultdict(list)
        self.single: Dict[str, Data] = {}
        self.morgan: Dict[str, Tensor] = {}
        self.reaction_labels: Dict[str, List[int]] = {}
        self.mol_structs: Dict[str, Chem.Mol] = {}
        self.normalization_mode: Literal["standardize", "canonical"] = (
            normalization_mode if normalization_mode is not None else ("standardize" if standartize else "canonical")
        )

        if isinstance(data, pd.DataFrame):
            self._init_from_dataframe(
                data=data,
                sub_name=sub_name,
                prod_name=prod_name,
                real_name=real_name,
                mol_structs=mol_structs,
                standartize=standartize,
            )
        elif isinstance(data, dict):
            self._init_from_mapping(data=data, gen_map=gen_map, mol_structs=mol_structs, standartize=standartize)
        else:
            raise TypeError("MolFrame expects a pandas.DataFrame or a mapping of substrate to products")

        self.clean()

    def _normalize_smiles(self, smiles: str, standartize: bool) -> str:
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("Empty SMILES are not supported")
        if not standartize:
            # In canonical mode, map products must be CANONICALIZED so they match the
            # canonical generated products produced during reaction labeling (otherwise
            # raw SDF SMILES never match -> labeling marks nothing). Standardize mode
            # passes through SMILES that were already standardized upstream.
            if self.normalization_mode == "canonical":
                try:
                    return _canonicalize_smiles_cached(smiles)
                except Exception:
                    return smiles
            return smiles
        try:
            normalized = _standardize_smiles_cached(smiles)
        except Exception:
            warnings.warn(f"Failed to standardize {smiles}; using original representation")
            normalized = smiles
        return str(normalized)

    def _init_from_dataframe(
        self,
        data: pd.DataFrame,
        sub_name: str,
        prod_name: str,
        real_name: str,
        mol_structs: Optional[Dict[str, Chem.Mol]],
        standartize: bool,
    ) -> None:
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        required_columns = {sub_name, prod_name}
        if not required_columns.issubset(data.columns):
            missing = required_columns.difference(data.columns)
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        normalized = data.copy()
        # normalized.get(real_name, 1) returns a scalar int when the column is absent,
        # and ``int`` has no ``.astype`` -> crash. Default the whole column instead.
        if real_name in normalized.columns:
            normalized[real_name] = normalized[real_name].fillna(1).astype(int)
        else:
            normalized[real_name] = 1

        cache: Dict[str, str] = {}

        def normalize_cached(value: str) -> str:
            key = str(value)
            if key not in cache:
                cache[key] = self._normalize_smiles(key, standartize)
            return cache[key]

        normalized[sub_name] = normalized[sub_name].map(normalize_cached)
        normalized[prod_name] = normalized[prod_name].map(normalize_cached)

        for row in normalized[[sub_name, prod_name, real_name]].itertuples(index=False):
            substrate, product, is_real = row
            if is_real:
                self.map[substrate].add(product)
            else:
                self.gen_map[substrate].add(product)

        self._bootstrap_molecules(mol_structs)

    def _init_from_mapping(
        self,
        data: Dict[str, Iterable[str]],
        gen_map: Optional[DefaultDict[str, Set[str]]],
        mol_structs: Optional[Dict[str, Chem.Mol]],
        standartize: bool,
    ) -> None:
        if not data:
            raise ValueError("Input mapping is empty")
        for substrate, products in data.items():
            substrate_smiles = self._normalize_smiles(substrate, standartize)
            for product in products:
                self.map[substrate_smiles].add(self._normalize_smiles(str(product), standartize))
        if gen_map:
            for substrate, products in gen_map.items():
                substrate_smiles = self._normalize_smiles(substrate, standartize)
                for product in products:
                    self.gen_map[substrate_smiles].add(self._normalize_smiles(str(product), standartize))
        self._bootstrap_molecules(mol_structs)

    def _bootstrap_molecules(self, mol_structs: Optional[Dict[str, Chem.Mol]]) -> None:
        if mol_structs:
            self.mol_structs = {smiles: Chem.Mol(mol) for smiles, mol in mol_structs.items() if mol is not None}
            return

        smiles_set = set(self.map.keys()) | set(self.gen_map.keys())
        for products in self.map.values():
            smiles_set.update(products)
        for products in self.gen_map.values():
            smiles_set.update(products)
        self.mol_structs = {
            smiles: mol
            for smiles in smiles_set
            if (mol := _smiles_to_mol(smiles)) is not None
        }

    @staticmethod
    def read_triples(file_path: Union[str, Path]) -> List[Tuple[int, int, int]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        triples = []
        with open(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid triple at line {line_number}: {stripped}")
                triples.append(tuple(int(value) for value in parts))
        return triples

    @staticmethod
    def from_file(
        file_path: Union[str, Path],
        triples: List[Tuple[int, int, int]],
        standartize: bool = True,
    ) -> "MolFrame":
        sdf_path = Path(file_path)
        if not sdf_path.exists():
            raise FileNotFoundError(sdf_path)
        if not triples:
            raise ValueError("Triples are required for SDF loading")

        required_ids = {sub_idx for sub_idx, _, _ in triples} | {prod_idx for _, prod_idx, _ in triples}
        index_to_smiles: Dict[int, str] = {}
        index_to_mol: Dict[int, Chem.Mol] = {}
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        for fallback_index, mol in enumerate(supplier, start=1):
            if mol is None:
                continue
            try:
                index = int(mol.GetProp("Index")) if mol.HasProp("Index") else fallback_index
            except Exception:
                index = fallback_index
            if index not in required_ids:
                continue
            smiles = mol.GetProp("SMILES") if mol.HasProp("SMILES") else Chem.MolToSmiles(mol, isomericSmiles=False)
            index_to_smiles[index] = smiles
            index_to_mol[index] = mol
            if len(index_to_smiles) == len(required_ids):
                break

        if not index_to_smiles:
            raise ValueError(f"Failed to load molecules from {sdf_path}")

        normalized_index_to_smiles = dict(index_to_smiles)
        if standartize:
            for index, smiles in list(normalized_index_to_smiles.items()):
                try:
                    normalized_index_to_smiles[index] = _standardize_smiles_cached(smiles)
                except Exception:
                    warnings.warn(f"Failed to standardize {smiles}; using original representation")
                    normalized_index_to_smiles[index] = smiles

        records = []
        for sub_idx, prod_idx, is_real in triples:
            substrate = normalized_index_to_smiles.get(sub_idx)
            product = normalized_index_to_smiles.get(prod_idx)
            if substrate is None or product is None:
                continue
            records.append({"sub": substrate, "prod": product, "real": is_real})

        if not records:
            raise ValueError("No valid triples could be matched to SDF records")

        used_smiles = {record["sub"] for record in records} | {record["prod"] for record in records}
        if standartize:
            mol_structs = {
                smiles: mol
                for smiles in used_smiles
                if (mol := _smiles_to_mol(smiles)) is not None
            }
        else:
            mol_structs = {
                smiles: index_to_mol[index]
                for index, smiles in index_to_smiles.items()
                if smiles in used_smiles and index_to_mol.get(index) is not None
            }
        return MolFrame(
            pd.DataFrame.from_records(records),
            mol_structs=mol_structs,
            standartize=False if standartize else standartize,
            normalization_mode="standardize" if standartize else "canonical",
        )

    def _subset_mapping(self, keys: Iterable[str], mapping: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        return {key: set(mapping.get(key, set())) for key in keys if key in mapping}

    def train_val_test_split(self, frac: float, seed: int = 42, enforce_disjoint: bool = False) -> List["MolFrame"]:
        if not 0.0 < frac < 0.5:
            raise ValueError("frac must be in (0, 0.5)")
        substrates = np.array(sorted(self.map.keys()))
        rng = np.random.default_rng(seed)
        rng.shuffle(substrates)
        size = int(len(substrates) * frac)
        val_keys = list(substrates[:size])
        test_keys = list(substrates[size : 2 * size])
        train_keys = list(substrates[2 * size :])

        if enforce_disjoint:
            # Substrate-level disjointness is not enough: the SAME molecule can be a
            # train product and a test substrate/product, leaking structures across
            # splits. Drop any val/test substrate whose molecule set (substrate + its
            # products) intersects the train molecule set, so the three splits share no
            # molecule at all.
            def molecule_set(keys: Iterable[str]) -> Set[str]:
                molecules: Set[str] = set()
                for key in keys:
                    molecules.add(key)
                    molecules.update(self.map.get(key, set()))
                    molecules.update(self.gen_map.get(key, set()))
                return molecules

            train_molecules = molecule_set(train_keys)
            val_keys = [
                key for key in val_keys
                if not ({key} | self.map.get(key, set()) | self.gen_map.get(key, set())) & train_molecules
            ]
            kept_molecules = train_molecules | molecule_set(val_keys)
            test_keys = [
                key for key in test_keys
                if not ({key} | self.map.get(key, set()) | self.gen_map.get(key, set())) & kept_molecules
            ]

        out = []
        for subset in (train_keys, val_keys, test_keys):
            subset_map = self._subset_mapping(subset, self.map)
            subset_gen = defaultdict(set, self._subset_mapping(subset, self.gen_map))
            subset_mols = {smiles: self.mol_structs[smiles] for smiles in self.mol_structs if smiles in subset_map or any(smiles in products for products in subset_map.values()) or any(smiles in products for products in subset_gen.values())}
            out.append(
                MolFrame(
                    subset_map,
                    gen_map=subset_gen,
                    mol_structs=subset_mols,
                    standartize=False,
                    normalization_mode=self.normalization_mode,
                )
            )
        return out

    def subset(self, substrates: Iterable[str]) -> "MolFrame":
        selected = {substrate for substrate in substrates if substrate in self.map}
        subset_map = self._subset_mapping(selected, self.map)
        subset_gen = defaultdict(set, self._subset_mapping(selected, self.gen_map))
        included_smiles: Set[str] = set(selected)
        for mapping in (subset_map, subset_gen):
            for products in mapping.values():
                included_smiles.update(products)
        subset_mols = {
            smiles: self.mol_structs[smiles]
            for smiles in included_smiles
            if smiles in self.mol_structs
        }
        return MolFrame(
            subset_map,
            gen_map=subset_gen,
            mol_structs=subset_mols,
            standartize=False,
            normalization_mode=self.normalization_mode,
        )

    def sample_substrates(self, max_substrates: Optional[int], seed: int = 42) -> "MolFrame":
        if max_substrates is None or max_substrates <= 0 or max_substrates >= len(self.map):
            return self
        rng = np.random.default_rng(seed)
        substrates = np.array(sorted(self.map.keys()))
        rng.shuffle(substrates)
        return self.subset(substrates[:max_substrates].tolist())

    def exclude_substrates(self, excluded: Iterable[str]) -> "MolFrame":
        excluded = {str(substrate) for substrate in excluded}
        if not excluded:
            return self
        kept = [substrate for substrate in self.map.keys() if substrate not in excluded]
        return self.subset(kept)

    def __or__(self, other: "MolFrame") -> "MolFrame":
        merged_map: DefaultDict[str, Set[str]] = defaultdict(set)
        merged_gen: DefaultDict[str, Set[str]] = defaultdict(set)

        for mapping, target in ((self.map, merged_map), (other.map, merged_map), (self.gen_map, merged_gen), (other.gen_map, merged_gen)):
            for substrate, products in mapping.items():
                target[substrate].update(products)

        merged_mols = dict(self.mol_structs)
        merged_mols.update(other.mol_structs)
        return MolFrame(
            dict(merged_map),
            gen_map=merged_gen,
            mol_structs=merged_mols,
            standartize=False,
            normalization_mode=self.normalization_mode,
        )

    def clean(self) -> None:
        cleaned_map: DefaultDict[str, Set[str]] = defaultdict(set)
        cleaned_gen: DefaultDict[str, Set[str]] = defaultdict(set)
        valid_smiles: Set[str] = set()

        for mapping, target in ((self.map, cleaned_map), (self.gen_map, cleaned_gen)):
            for substrate, products in mapping.items():
                substrate_mol = self.mol_structs.get(substrate) or _smiles_to_mol(substrate)
                if substrate_mol is None or not products:
                    continue
                target_products = {product for product in products if (self.mol_structs.get(product) or _smiles_to_mol(product)) is not None}
                if not target_products:
                    continue
                target[substrate].update(target_products)
                valid_smiles.add(substrate)
                valid_smiles.update(target_products)

        self.map = cleaned_map
        self.gen_map = cleaned_gen
        self.mol_structs = {
            smiles: self.mol_structs.get(smiles) or _smiles_to_mol(smiles)
            for smiles in valid_smiles
            if (self.mol_structs.get(smiles) or _smiles_to_mol(smiles)) is not None
        }
        self.negatives()

    def negatives(self) -> None:
        self.negs = defaultdict(set)
        for substrate, generated in self.gen_map.items():
            self.negs[substrate].update(product for product in generated if product not in self.map.get(substrate, set()))

    def metabolize(
        self,
        rules: Sequence[str],
        mode: Literal["opt", "gen"] = "opt",
    ) -> Optional[Set[FrozenSet[int]]]:
        equivalence: Set[FrozenSet[int]] = set()
        product_rule_matrix: Dict[str, np.ndarray] = {}

        for substrate, mol in tqdm(self.mol_structs.items(), desc="Applying rules"):
            if substrate not in self.map:
                continue
            generated = apply_rules_to_molecule(mol, list(rules), normalization_mode=self.normalization_mode)
            for product, indexes in generated.items():
                self.gen_map[substrate].add(product)
                self.mol_structs.setdefault(product, _smiles_to_mol(product))
                if mode == "opt":
                    vec = product_rule_matrix.setdefault(product, np.zeros(len(rules), dtype=int))
                    for index in indexes:
                        vec[index] = 1

        self.negatives()
        if mode != "opt":
            return None

        matrix = np.array(list(product_rule_matrix.values()), dtype=int).T if product_rule_matrix else np.zeros((len(rules), 0), dtype=int)
        for left in range(matrix.shape[0]):
            for right in range(left + 1, matrix.shape[0]):
                if np.array_equal(matrix[left], matrix[right]):
                    equivalence.add(frozenset({left, right}))
        return equivalence

    def augment(self, rules: Sequence[str]) -> None:
        self.metabolize(rules, mode="gen")
        self.clean()

    def morganize(
        self,
        size: int = 1024,
        cache_path: Union[str, Path, None] = None,
        save_interval: int = 1000,
    ) -> None:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            includeChirality=True,
            fpSize=size,
            countSimulation=False,
        )
        self.morgan = {}
        cache_file = _normalize_cache_path(cache_path)
        if cache_file is not None:
            cached = _load_cached_mapping(cache_file)
            self.morgan = {
                smiles: tensor
                for smiles, tensor in cached.items()
                if smiles in self.mol_structs and isinstance(tensor, Tensor)
            }
            if self.morgan:
                print(
                    f"Loaded {len(self.morgan)} Morgan fingerprints from cache: {cache_file}",
                    flush=True,
                )

        pending = [smiles for smiles in self.mol_structs.keys() if smiles not in self.morgan]
        for index, smiles in enumerate(tqdm(pending, desc="Building Morgan fingerprints"), start=1):
            mol = self.mol_structs.get(smiles)
            if mol is None:
                continue
            self.morgan[smiles] = torch.tensor(
                np.asarray(generator.GetFingerprint(mol), dtype=np.float32),
                dtype=torch.float32,
            ).view(1, -1)
            if cache_file is not None and (index % save_interval == 0 or index == len(pending)):
                _atomic_torch_save(self.morgan, cache_file)

    def singlegraphs(
        self,
        pca: bool = False,
        smiles: Optional[Iterable[str]] = None,
        cache_path: Union[str, Path, None] = None,
        save_interval: int = 250,
    ) -> None:
        node_path = _local_data_dir() / "pca_ats_single.pkl" if pca else None
        edge_path = _local_data_dir() / "pca_bonds_single.pkl" if pca else None
        self.single = {}
        cache_file = _normalize_cache_path(cache_path)
        selected_smiles = list(smiles) if smiles is not None else list(self.mol_structs.keys())
        if cache_file is not None:
            cached = _load_cached_mapping(cache_file)
            self.single = {
                key: graph
                for key, graph in cached.items()
                if key in self.mol_structs and isinstance(graph, Data)
            }
            if self.single:
                print(
                    f"Loaded {len(self.single)} single graphs from cache: {cache_file}",
                    flush=True,
                )

        pending = [key for key in selected_smiles if key not in self.single]
        for index, smiles_value in enumerate(tqdm(pending, desc="Building single graphs"), start=1):
            mol = self.mol_structs.get(smiles_value)
            if mol is None:
                continue
            graph = from_rdmol(mol)
            if graph is None:
                continue
            graph = _maybe_project_graph(graph, node_path=node_path, edge_path=edge_path)
            self.single[smiles_value] = graph
            if cache_file is not None and (index % save_interval == 0 or index == len(pending)):
                _atomic_torch_save(self.single, cache_file)

    def pairgraphs(self, pca: bool = False) -> None:
        if not self.single:
            self.singlegraphs(pca=False)

        node_path = _local_data_dir() / "pca_ats.pkl" if pca else None
        edge_path = _local_data_dir() / "pca_bonds.pkl" if pca else None
        self.graphs = defaultdict(list)

        for label, mapping in ((1.0, self.map), (0.0, self.negs)):
            total_substrates = len(mapping)
            pair_count = 0
            fallback_count = 0
            split_name = "positive" if label else "negative"
            for substrate_index, (substrate, products) in enumerate(
                tqdm(mapping.items(), desc=f"Building {split_name} pair graphs"),
                start=1,
            ):
                if substrate_index == 1 or substrate_index % 100 == 0 or substrate_index == total_substrates:
                    print(
                        f"pairgraphs {split_name} progress {substrate_index}/{total_substrates} "
                        f"substrates, {pair_count} graphs, {fallback_count} fallback",
                        flush=True,
                    )
                sub_mol = self.mol_structs.get(substrate)
                if sub_mol is None:
                    continue
                for product in products:
                    prod_mol = self.mol_structs.get(product)
                    if prod_mol is None:
                        continue
                    graph = build_pair_graph(sub_mol, prod_mol)
                    if graph is None:
                        continue
                    fallback_count += int(getattr(graph, "fallback_pair", 0))
                    pair_count += 1
                    graph = _maybe_project_graph(graph, node_path=node_path, edge_path=edge_path)
                    graph.y = torch.tensor([label], dtype=torch.float32)
                    graph.smiles = product
                    self.graphs[substrate].append(graph)

    def pair_records(self) -> List[Tuple[str, str, float]]:
        records: List[Tuple[str, str, float]] = []
        for label, mapping in ((1.0, self.map), (0.0, self.negs)):
            for substrate, products in mapping.items():
                if substrate not in self.mol_structs:
                    continue
                for product in products:
                    if product not in self.mol_structs:
                        continue
                    records.append((substrate, product, label))
        return records

    def pair_loader(self, batch_size: int, shuffle: bool = True, pca: bool = False) -> DataLoader:
        node_path = _local_data_dir() / "pca_ats.pkl" if pca else None
        edge_path = _local_data_dir() / "pca_bonds.pkl" if pca else None
        dataset = _LazyPairDataset(self, self.pair_records(), node_path=node_path, edge_path=edge_path)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def label_reactions(
        self,
        rules: Sequence[str],
        cache_path: Union[str, Path, None] = None,
        save_interval: int = 100,
    ) -> None:
        selected_rules = list(rules)
        cache_file = _normalize_cache_path(cache_path)
        self.reaction_labels = {}
        if cache_file is not None:
            cached = _load_cached_mapping(cache_file)
            self.reaction_labels = {
                substrate: list(vector)
                for substrate, vector in cached.items()
                if substrate in self.map and isinstance(vector, list) and len(vector) == len(selected_rules)
            }
            if self.reaction_labels:
                print(
                    f"Loaded {len(self.reaction_labels)} reaction label vectors from cache: {cache_file}",
                    flush=True,
                )

        pending = [substrate for substrate in self.map.keys() if substrate not in self.reaction_labels]
        total = len(self.map)
        for pending_index, substrate in enumerate(tqdm(pending, desc="Labeling reactions"), start=1):
            completed = total - len(pending) + pending_index
            if pending_index == 1 or completed % 100 == 0 or completed == total:
                print(f"label_reactions progress {completed}/{total}", flush=True)
            mol = self.mol_structs.get(substrate)
            if mol is None:
                continue
            generated = apply_rules_to_molecule(mol, selected_rules, normalization_mode=self.normalization_mode)
            vector = [0] * len(selected_rules)
            allowed = self.map.get(substrate, set())
            for product, indexes in generated.items():
                if product not in allowed:
                    continue
                for rule_index in indexes:
                    if 0 <= rule_index < len(vector):
                        vector[rule_index] = 1
            self.reaction_labels[substrate] = vector
            if cache_file is not None and (pending_index % save_interval == 0 or pending_index == len(pending)):
                _atomic_torch_save(self.reaction_labels, cache_file)

    def full_setup(
        self,
        pca: bool = False,
        rules: Optional[Sequence[str]] = None,
        include_reaction_labels: bool = True,
        include_pair_graphs: bool = True,
        include_morgan: bool = False,
        single_smiles: Optional[Iterable[str]] = None,
        include_single_graphs: bool = True,
        morgan_cache_path: Union[str, Path, None] = None,
        single_cache_path: Union[str, Path, None] = None,
        reaction_label_cache_path: Union[str, Path, None] = None,
        cache_save_interval: int = 100,
    ) -> None:
        self.clean()
        if include_morgan:
            self.morganize(cache_path=morgan_cache_path, save_interval=max(cache_save_interval, 100))
        else:
            self.morgan = {}
        if include_single_graphs:
            self.singlegraphs(
                pca=pca,
                smiles=single_smiles,
                cache_path=single_cache_path,
                save_interval=max(cache_save_interval, 50),
            )
        else:
            self.single = {}
        if include_pair_graphs:
            self.pairgraphs(pca=pca)
        else:
            self.graphs = defaultdict(list)
        selected_rules = list(rules) if rules is not None else load_default_rules()
        if selected_rules and include_reaction_labels:
            self.label_reactions(
                selected_rules,
                cache_path=reaction_label_cache_path,
                save_interval=max(cache_save_interval, 25),
            )
        elif not include_reaction_labels:
            self.reaction_labels = {}

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for substrate, products in self.map.items():
            rows.extend({"sub": substrate, "prod": product, "real": 1} for product in products)
        for substrate, products in self.negs.items():
            rows.extend({"sub": substrate, "prod": product, "real": 0} for product in products)
        return pd.DataFrame(rows)

    def passify(self, name: Union[str, Path]) -> List[Tuple[int, int, int]]:
        from rdkit.Chem import PandasTools

        output = Path(name)
        if output.suffix != ".sdf":
            output = output.with_suffix(".sdf")

        rows = []
        triples: List[Tuple[int, int, int]] = []
        index = 1
        index_map: Dict[str, int] = {}

        for substrate in self.map:
            index_map[substrate] = index
            rows.append({"Index": index, "SMILES": substrate, "Molecules": self.mol_structs[substrate], "Status": "Substrate"})
            index += 1

        for substrate, products in self.map.items():
            for product in products:
                index_map[product] = index
                rows.append({"Index": index, "SMILES": product, "Molecules": self.mol_structs[product], "Status": "Real"})
                triples.append((index_map[substrate], index, 1))
                index += 1

        for substrate, products in self.negs.items():
            for product in products:
                index_map[product] = index
                rows.append({"Index": index, "SMILES": product, "Molecules": self.mol_structs[product], "Status": "Generated"})
                triples.append((index_map[substrate], index, 0))
                index += 1

        PandasTools.WriteSDF(pd.DataFrame(rows), str(output), molColName="Molecules", properties=["Index", "SMILES", "Status"])
        return triples

    def create_rules(self) -> Optional[Set[str]]:
        try:
            from .reaction_mapper import combine_reaction
        except ImportError:
            warnings.warn("reaction_mapper is not available in this environment")
            return None

        rules = set()
        for substrate, products in tqdm(self.map.items(), desc="Generating SMARTS rules"):
            substrate_mol = self.mol_structs.get(substrate)
            if substrate_mol is None:
                continue
            for product in products:
                product_mol = self.mol_structs.get(product)
                if product_mol is None:
                    continue
                try:
                    rule = combine_reaction(substrate_mol, product_mol)
                except Exception:
                    continue
                if rule:
                    rules.add(rule)
        return rules

    def plot_coverage(self) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        self.negatives()
        coverages = []
        for substrate, products in self.map.items():
            generated = self.gen_map.get(substrate, set())
            if products:
                coverages.append(len(products & generated) / len(products))
        sns.boxplot(coverages)
        plt.show()

    def _pair_dataset(self) -> List[Data]:
        dataset = []
        for graphs in self.graphs.values():
            dataset.extend(graph for graph in graphs if graph is not None)
        return dataset

    def _single_dataset(self) -> List[Tuple[Data, Data]]:
        dataset = []
        for label, mapping in ((1.0, self.map), (0.0, self.negs)):
            for substrate, products in mapping.items():
                sub_graph = self.single.get(substrate)
                if sub_graph is None:
                    continue
                for product in products:
                    prod_graph = self.single.get(product)
                    if prod_graph is None:
                        continue
                    pair = (sub_graph.clone(), prod_graph.clone())
                    pair[1].y = torch.tensor([label], dtype=torch.float32)
                    dataset.append(pair)
        return dataset

    def train_pairs(
        self,
        model: Module,
        test_set: Optional["MolFrame"] = None,
        lr: float = 1e-5,
        eps: int = 100,
        decay: float = 1e-8,
        verbose: bool = True,
        prior: float = 0.75,
        nnPU: bool = True,
    ) -> Module:
        del test_set, decay
        model.fit(self, lr=lr, eps=eps, verbose=verbose, prior=prior, nnPU=nnPU)
        return model

    def train_singles(
        self,
        model: Module,
        test_set: Optional["MolFrame"] = None,
        lr: float = 1e-5,
        eps: int = 100,
        decay: float = 1e-8,
        verbose: bool = True,
        prior: float = 0.75,
        nnPU: bool = True,
    ) -> Module:
        del test_set, decay
        model.fit(self, lr=lr, eps=eps, verbose=verbose, prior=prior, nnPU=nnPU)
        return model

    def train_generator(
        self,
        model: Module,
        lr: float = 1e-5,
        eps: int = 100,
        decay: float = 1e-10,
        verbose: bool = True,
    ) -> Tuple[Module, float]:
        del decay
        model.fit(self, lr=lr, eps=eps, verbose=verbose)
        return model, float("nan")

    def test(self, model: Module, mode: Literal["single", "pair"] = "pair") -> Tuple[float, float]:
        from sklearn.metrics import matthews_corrcoef, roc_auc_score

        model.eval()
        preds: List[float] = []
        binary: List[int] = []
        targets: List[int] = []
        threshold = float(getattr(model, "calibrated_threshold", 0.5) or 0.5)

        if mode == "pair":
            if self.graphs:
                dataset = self._pair_dataset()
                if not dataset:
                    raise ValueError("Pair graphs are not prepared")
                loader = DataLoader(dataset, batch_size=64)
            else:
                loader = self.pair_loader(batch_size=64, shuffle=False)
            for batch in loader:
                output = model(batch).view(-1)
                preds.extend(cpunum(output).tolist())
                binary.extend((cpunum(output) > threshold).astype(int).tolist())
                targets.extend(cpunum(batch.y.view(-1)).astype(int).tolist())
        else:
            dataset = self._single_dataset()
            if not dataset:
                raise ValueError("Single graphs are not prepared")

            def collate(items: List[Tuple[Data, Data]]) -> Tuple[Batch, Batch]:
                left = Batch.from_data_list([item[0] for item in items])
                right = Batch.from_data_list([item[1] for item in items])
                return left, right

            loader = DataLoader(dataset, batch_size=64, collate_fn=collate)
            for sub_batch, prod_batch in loader:
                output = model(sub_batch, prod_batch).view(-1)
                preds.extend(cpunum(output).tolist())
                binary.extend((cpunum(output) > threshold).astype(int).tolist())
                targets.extend(cpunum(prod_batch.y.view(-1)).astype(int).tolist())

        if len(set(targets)) < 2:
            return 0.0, 0.0
        return matthews_corrcoef(targets, binary), roc_auc_score(targets, preds)

    def __repr__(self) -> str:
        return f"MolFrame(substrates={len(self.map)}, positives={sum(len(products) for products in self.map.values())}, negatives={sum(len(products) for products in self.negs.values())})"

    __str__ = __repr__


def _invalid_pair_graph(label: float, product: str) -> Data:
    graph = Data(
        x=torch.zeros((1, PAIR_NODE_DIM), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, EDGE_DIM), dtype=torch.float32),
    )
    graph.fp = torch.zeros((1, 2 * FINGERPRINT_DIM), dtype=torch.float32)
    graph.y = torch.tensor([label], dtype=torch.float32)
    graph.smiles = product
    graph.fallback_pair = 1
    graph.invalid_pair = 1
    return graph


class _LazyPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame: MolFrame,
        records: Sequence[Tuple[str, str, float]],
        node_path: Optional[Path] = None,
        edge_path: Optional[Path] = None,
    ) -> None:
        self.frame = frame
        self.records = list(records)
        self.node_path = node_path
        self.edge_path = edge_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Data:
        substrate, product, label = self.records[index]
        sub_mol = self.frame.mol_structs.get(substrate)
        prod_mol = self.frame.mol_structs.get(product)
        if sub_mol is None or prod_mol is None:
            return _invalid_pair_graph(label, product)

        graph = build_pair_graph(sub_mol, prod_mol)
        if graph is None:
            return _invalid_pair_graph(label, product)

        graph = _maybe_project_graph(graph, node_path=self.node_path, edge_path=self.edge_path)
        graph.y = torch.tensor([label], dtype=torch.float32)
        graph.smiles = product
        return graph

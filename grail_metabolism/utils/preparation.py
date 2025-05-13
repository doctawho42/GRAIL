import os
import re
import gc
import errno
import signal
import warnings
import pickle as pkl
import random
from tqdm.auto import tqdm
import pandas as pd
try:
    from grail_metabolism.utils.reaction_mapper import combine_reaction
except ImportError:
    print('ATTENTION: you use incorrect for rxnmapper version of rdkit')
    def combine_reaction(*args, **kwargs) -> None:
        return None
from grail_metabolism.utils.transform import from_pair, from_rdmol
from sklearn.impute import SimpleImputer
from sklearn.metrics import matthews_corrcoef as mcc, roc_auc_score as roc_auc, jaccard_score as jac
from torch import tensor
from torch.nn import Module, BCELoss
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from multipledispatch import dispatch
from aizynthfinder.aizynthfinder import AiZynthExpander
import faiss

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from typing import Dict, Union, Literal, Optional, Set, List, Tuple, FrozenSet, Any, DefaultDict, Iterable
from itertools import chain
from collections import defaultdict as dd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator
from dimorphite_dl import DimorphiteDL

from functools import wraps
from threading import Timer

import numpy as np
from rdkit.Chem.PandasTools import WriteSDF, LoadSDF

def handler(signum, frame):
    raise TimeoutError

def get_reactions(expander: AiZynthExpander, smiles: str) -> list[list[str]]:
    reactions = expander.do_expansion(smiles)
    reactants_smiles = []
    for reaction_tuple in reactions:
        reactants_smiles.append([mol.smiles for mol in reaction_tuple[0].reactants[0]])
    return reactants_smiles


warnings.filterwarnings('ignore')
tqdm.pandas()

with open('/Users/nikitapolomosnov/PycharmProjects/GRAIL/grail_metabolism/data/smirks.txt') as rulefile:
    rules = tuple(x.rstrip() for x in rulefile)

uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
te = rdMolStandardize.TautomerEnumerator()  # canonicalize tautomer form

def carbon_counter(x):
    return x.count('C') + x.count('c') - sum(x.count(e) for e in ['Cl', 'Ca', 'Co', 'Sc', 'Cr', 'Cd', 'Cs'])
def iscorrect(x):
    return carbon_counter(x) >= 3
atom_counter = lambda x: len(re.findall('((?<=\[)[A-Z][a-z])|((?!\[)[A-GI-Za-z])', x))


def cpunum(tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


def extract(smiles: str) -> Optional[str]:
    """Extract the biggest correct molecule from a given SMILES string"""
    xtr = max(smiles.split('.'), key=atom_counter)
    if iscorrect(xtr):
        return xtr
    else:
        return None

import time
'''
def timeout(seconds: int = 30):
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            old_time_left = signal.alarm(seconds)
            if 0 < old_time_left < seconds: # never lengthen existing timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
            except TimeoutError:
                result = None
            finally:
                if old_time_left > 0: # deduct f's run time from the saved timer
                    old_time_left -= time.time() - start_time
                signal.signal(signal.SIGALRM, old)
                signal.alarm(old_time_left)
            return result
        return new_f
    return decorate
'''
def timeout(seconds: float = 30, error_message: str = os.strerror(errno.ETIME)):

    def decorator(func):

        @wraps(func)
        def _handle_timeout(*args, **kwargs):

            def _raise_timeout():
                raise TimeoutError

            timer = Timer(seconds, _raise_timeout)
            timer.start()
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                print('timeout')
                gc.collect()
                return None
            finally:
                timer.cancel()
            return result

        return _handle_timeout

    return decorator

def generate_vectors(reaction_dict, real_products_dict):
    vectors = {}
    for substrate in reaction_dict:
        # Initialize a vector of 474 zeros
        vector = [0] * 474
        # Get the real products for this substrate, default to empty set if not present
        real_products = real_products_dict.get(substrate, set())
        # Iterate over each product and its indexes in the reaction_dict
        for product, indexes in reaction_dict[substrate].items():
            if product in real_products:
                for idx in indexes:
                    # Ensure the index is within the valid range
                    if 0 <= idx < 474:
                        vector[idx] = 1
        vectors[substrate] = vector
    return vectors

@timeout(seconds=30)
def standardize_mol(mol: Union[rdkit.Chem.rdchem.Mol, str], ph: float = None) -> Union[rdkit.Chem.rdchem.Mol, str]:
    """Standardize the :class:`rdkit` molecule, select its parent molecule, uncharge it,
       then enumerate all tautomers."""

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
        is_mol = False
    else:
        is_mol = True

    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    clean_mol = rdMolStandardize.Cleanup(mol)
    # if many fragments, get the "parent" (the actual mol we are interested in)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    # try to neutralize molecule and make one resonance structure

    # if ph is None - try to uncharge molecule
    if ph is None:
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    else:
        dimorphite_dl = DimorphiteDL(min_ph=ph,
                                     max_ph=ph,
                                     max_variants=128,
                                     label_states=False,
                                     pka_precision=1.0) # protonate with specified pH
        uncharged_parent_clean_mol = Chem.MolFromSmiles(
            dimorphite_dl.protonate(Chem.MolToSmiles(parent_clean_mol))[0])
    taut_uncharged_parent_clean_mol = te.Canonicalize(
        uncharged_parent_clean_mol)
    if taut_uncharged_parent_clean_mol is None: raise ValueError('Invalid molecule state')

    if is_mol:
        return Chem.MolFromSmiles(Chem.MolToSmiles(taut_uncharged_parent_clean_mol, isomericSmiles=False))
    else:
        return Chem.MolToSmiles(taut_uncharged_parent_clean_mol, isomericSmiles=False)

@timeout(seconds=30)
def metaboliser(mol: rdkit.Chem.rdchem.Mol, rules: Iterable[str] = rules) -> Optional[dd[Any, set]]:
    r"""
    Applies all known biotransformation rules to the given substrate and returns
    products with corresponding SMARTS

    Args:
        mol (rdkit.Chem.rdchem.Mol): the class :class:`rdkit` molecule
        rules (Iterable[str]): an iterable of transformation rules in SMARTS format
    :return: (Optional[dd[Any, Set]]): :class:`collections.defaultdict` with :class:`set` of product :class:`str` SMILES
    """

    dict_rules = dd(set)
    if mol is not None:
        mol = Chem.AddHs(mol)
    else:
        return None

    for i, rule in enumerate(rules):
        rxn = AllChem.ReactionFromSmarts(rule)
        try:
            mols_prebuild = chain.from_iterable(rxn.RunReactants((mol,)))
        except ValueError:
            continue
        if not mols_prebuild:
            continue
        else:
            mols_splitted = []
            for preb in mols_prebuild:
                mols_splitted += Chem.MolToSmiles(preb).split('.')
            mols_splitted = [x for x in mols_splitted if iscorrect(x)]
            mols_splitted = list(map(Chem.MolFromSmiles, mols_splitted))
            mols_splitted = [x for x in mols_splitted if x is not None]
            if not mols_splitted:
                continue
            try:
                mols_standart = list(map(standardize_mol, mols_splitted))
            except Chem.KekulizeException:
                continue
            except RuntimeError:
                continue
            except Chem.AtomValenceException:
                continue
            for stand in mols_standart:
                if stand != '':
                    dict_rules[stand].add(i)

    for key in dict_rules:
        dict_rules[key] = set(dict_rules[key])

    return dict_rules


class MolFrame:
    r"""
    An extension providing more precious pipelines'
    form to work with GRAIL.

    Initialization:
        From :class:`pd.DataFrame`:
            **value** (pd.DataFrame): :class:`pd.DataFrame` with substrate-product pairs

            **sub_name** (str): name of column with substrates SMILES

            **prod_name** (str): name of column with products SMILES

            **real_name** (str): name of column indicating if this pair is real

            **mol_structs** (Dict[str, Chem.Mol]): optional :class:`dict`-like object with all needed :class:Chem.Mol objects
        From :class:`dict`:
            **card** (Dict): :class:`dict`-like object with substrates as keys and products as values

            **gen_card** (Dict): :class:`dict`-like object with substrates as keys and generated products as values

            **mol_structs** (Dict[str, Chem.Mol]): optional :class:`dict`-like object with all needed Chem.Mol objects
    """

    @dispatch(pd.DataFrame)
    def __init__(self, card: pd.DataFrame, sub_name: str = 'sub', prod_name: str = 'prod', real_name: str = 'real',
                 mol_structs: Optional[Dict[str, Chem.Mol]] = None, standartize: bool = True) -> None:
        # Standardize molecules if required
        if standartize:

            subs_std = card[sub_name].progress_apply(standardize_mol)
            prods_std = card[prod_name].progress_apply(standardize_mol)
        else:
            subs_std = card[sub_name]
            prods_std = card[prod_name]

        # Ensure real column exists and is integer type
        if real_name not in card.columns:
            card[real_name] = 1
        card[real_name] = card[real_name].astype(int)

        # Create optimized processing dataframe
        processed_df = pd.DataFrame({
            'sub_std': subs_std,
            'prod_std': prods_std,
            'real': card[real_name]
        })

        # Group by standardized substrates
        grouped = processed_df.groupby('sub_std')

        # Initialize data containers
        self.card = dd(set)
        self.gen_card = dd(set)
        self.negs = dd(set)
        self.graphs = dd(list)
        self.single = {}
        self.morgan = {}

        # Process groups efficiently
        for sub, group in tqdm(grouped, desc="Processing reactions"):
            # Handle real products
            real_products = group[group['real'] == 1]['prod_std']
            self.card[sub].update(real_products)

            # Handle generated products
            gen_products = group[group['real'] == 0]['prod_std']
            if not gen_products.empty:
                self.gen_card[sub].update(gen_products)

        # Handle molecular structures

        @timeout(seconds=30)
        def from_smiles(smiles: str) -> Chem.Mol:
            return Chem.MolFromSmiles(smiles)

        if mol_structs is None:
            unique_smiles = set(processed_df['sub_std']).union(processed_df['prod_std'])
            self.mol_structs = {}
            for smile in tqdm(unique_smiles, desc="Creating molecule objects"):
                try:
                    self.mol_structs[smile] = from_smiles(smile)
                except TimeoutError:
                    pass
            #self.mol_structs = {smile: Chem.MolFromSmiles(smile) for smile in
                                #tqdm(unique_smiles, desc="Creating molecules")}
        else:
            self.mol_structs = mol_structs

    @dispatch(dict)
    def __init__(self, card: Dict, gen_card: Optional[DefaultDict] = None, mol_structs: Optional[Dict] = None) -> None:
        self.card = card
        self.gen_card = dd(set) if gen_card is None else gen_card
        self.negs = dd(set)
        self.graphs = dd(list)
        self.single = {}
        self.morgan = {}
        if mol_structs is not None:
            self.mol_structs = mol_structs
        else:
            for sub in card:
                self.mol_structs[sub] = Chem.MolFromSmiles(sub)
                for prod in card[sub]:
                    self.mol_structs[prod] = Chem.MolFromSmiles(prod)
            if gen_card:
                for sub in gen_card:
                    self.mol_structs[sub] = Chem.MolFromSmiles(sub)
                    for prod in gen_card[sub]:
                        self.mol_structs[prod] = Chem.MolFromSmiles(prod)

    @staticmethod
    def from_file(file_path: str, triples: List[Tuple[int, int, int]], standartize: bool = True) -> 'MolFrame':
        r"""
        Read MolFrame from file and list of triples.
        :param file_path: (str) path to SDF file
        :param triples: (List[Tuple[int, int, int]]) list of triples (sub, prod, real)
        :param standartize: (bool, optional) to standardize or not
        :return: MolFrame
        """
        # Load SDF file into a dataframe
        data = LoadSDF(file_path, molColName='Molecules', smilesName='SMILES')
        if 'Index' not in data.columns:
            raise ValueError('No Index attribute in an SDF file')
        # Change columns types in the dataframe
        data['Index'] = data['Index'].apply(int)
        triple_data = pd.DataFrame(triples, columns=['sub', 'prod', 'real'])
        triple_data['sub'] = triple_data['sub'].apply(lambda x: data[data['Index'] == x]['SMILES'].item())
        triple_data['prod'] = triple_data['prod'].apply(lambda x: data[data['Index'] == x]['SMILES'].item())
        mol_structs = dict(zip(data.SMILES, data.Molecules))

        return MolFrame(triple_data, mol_structs=mol_structs, standartize=standartize)

    @staticmethod
    def read_triples(file_path: str) -> List[Tuple[int, int, int]]:
        r"""
        Read triples from file.
        :param file_path: path to txt file
        :return: list of triples (sub, prod, real)
        """
        with open(file_path) as file:
            return [tuple(int(x) for x in file.readline().split()) for line in file]

    def metabolize(self, rules: List[str], mode: Literal['opt', 'gen'] = 'opt') -> Optional[Set[FrozenSet[int]]]:
        r"""
        Produce pseudo-metabolized molecules. If mode is 'opt', generate set of pairs of equal rules.
        :param rules: list of SMARTS
        :param mode: (Literal['opt', 'gen']) 'opt' - generate set of pairs of equal rules, 'gen' - only generate molecules
        :return: set of pairs of equal rules
        """
        signal.signal(signal.SIGALRM, handler)
        if mode == 'opt':
            equivalent = set()

            def _zeros():
                return np.zeros(len(rules))

            opt_matrix = dd(_zeros)
            for substrate in tqdm(self.card, desc='Metabolize substrates'):
                try:
                    #signal.alarm(10)
                    gen_stat = metaboliser(self.mol_structs[substrate], rules=rules)
                    for product in gen_stat:
                        product_smiles = Chem.MolToSmiles(product)
                        np.put(opt_matrix[product_smiles], list(gen_stat[product]), 1)
                        self.gen_card[substrate].add(product_smiles)
                        self.mol_structs[product_smiles] = product
                except TimeoutError:
                    continue
                except TypeError:
                    print('TypeError')
                    continue
            opt_matrix = np.array(list(opt_matrix.values()), dtype=int).T
            for i, col1 in enumerate(opt_matrix):
                for j, col2 in enumerate(opt_matrix):
                    if np.equal(col1, col2).all() and i != j:
                        equivalent.add(frozenset([i, j]))
            return equivalent
        elif mode == 'gen':
            for substrate in tqdm(self.card):
                if True:  # todo - correct work with substrate not in gen_card
                    gen_stat = metaboliser(self.mol_structs[substrate], rules=rules)
                    for product in gen_stat:
                        product_smiles = Chem.MolToSmiles(product)
                        self.gen_card[substrate].add(product_smiles)
                        self.mol_structs[product_smiles] = product
        else:
            raise ValueError

    def __or__(self, other: 'MolFrame') -> 'MolFrame':
        r"""
        Combine two MolFrames.
        :param other: :class:`MolFrame`
        :return: :class:`MolFrame`
        """
        new_card = self.card.copy()
        new_card.update(other.card)
        new_gencard = self.gen_card.copy()
        new_gencard.update(other.gen_card)
        new_mols = self.mol_structs.copy()
        new_mols.update(other.mol_structs)
        new_molframe = MolFrame(new_card, gen_card=new_gencard, mol_structs=new_mols)
        return new_molframe

    def clean(self) -> None:
        to_del_card = []
        for key in self.card:
            if len(self.card[key]) == 0:
                to_del_card.append(key)
        for key in to_del_card:
            del self.card[key]
        to_del_gencard = []
        for key in self.gen_card:
            if len(self.gen_card[key]) == 0:
                to_del_gencard.append(key)
        for key in to_del_gencard:
            del self.gen_card[key]
        self.__reconstruct(self.card)
        self.__reconstruct(self.gen_card)

    def __reconstruct(self, card: Dict[str, Set[Chem.Mol]]) -> None:
        for sub in card:
            if sub not in self.mol_structs.keys():
                self.mol_structs[sub] = Chem.MolFromSmiles(sub)
            for prod in card[sub]:
                if prod not in self.mol_structs.keys():
                    self.mol_structs[prod] = Chem.MolFromSmiles(prod)

    def train_val_test_split(self, frac: float) -> List['MolFrame']:
        x = np.array(list(self.card.keys()))
        np.random.shuffle(x)
        idx = int(frac * len(x))
        train, val, test = x[idx * 2:], x[:idx], x[idx:idx * 2]
        tables = []
        for sub_set in [train, val, test]:
            cards = {}
            gen_cards = {}
            mol_structs = {}
            for key in self.card:
                if key in sub_set:
                    cards[key] = self.card[key]
                    gen_cards[key] = self.gen_card[key]
                    mol_structs[key] = self.mol_structs[key]
            molframe = MolFrame(cards, gen_card=dd(set, gen_cards), mol_structs=mol_structs)
            tables.append(molframe)
        return tables

    def negatives(self) -> None:
        r"""
        Generate negative subclass.
        :return: :class:None
        """
        for sub in self.card:
            for prod in self.gen_card[sub]:
                if prod not in self.card[sub]:
                    self.negs[sub].add(prod)

    def passify(self, name: str) -> List[Tuple[int, int, int]]:
        r"""
        Transform dataset to work with PASS or USSR programs
        :param name: Name of SDF file to write into
        :return: list of triples (sub, prod, real)
        """
        self.negatives()
        smiles = []
        molecules = []
        indexes = []
        status = []
        triples = []
        for key in self.card:
            smiles.append(key)
            molecules.append(self.mol_structs[key])
            i = 1 if not indexes else indexes[-1] + 1
            indexes.append(i)
            status.append('Substrate')
            for val in self.card[key]:
                smiles.append(val)
                molecules.append(self.mol_structs[val])
                j = indexes[-1] + 1
                indexes.append(j)
                status.append('Real')
                triples.append((i, j, 1))
            for val in self.negs[key]:
                smiles.append(val)
                molecules.append(self.mol_structs[val])
                j = indexes[-1] + 1
                indexes.append(j)
                status.append('Generated')
                triples.append((i, j, 0))

        data = pd.DataFrame({
            'Index': indexes,
            'SMILES': smiles,
            'Molecules': molecules,
            'Status': status
        })
        WriteSDF(data, f'{name}.sdf', molColName='Molecules', properties=list(data.columns))

        return triples

    def create_rules(self) -> Optional[Set[str]]:
        r"""
        Augment rule set with all "real" substrate-metabolite pairs.
        :return: (Set[str]) set of SMARTS rules
        """
        try:
            from rxnmapper import RXNMapper
        except ImportError:
            print('ATTENTION: rxnmapper is not installed')
            return None

        rules = set()
        for substrate in tqdm(self.card, desc='Generating rules'):
            for product in self.card[substrate]:
                rules.add(combine_reaction(self.mol_structs[substrate], self.mol_structs[product]))
        return rules

    def augment(self, rules: List[str]) -> None:
        r"""
        Augment dataset with the given rules.
        :param rules: (List[str]) list of SMARTS rules
        :return: :class:None
        """
        self.metabolize(rules)
        self.clean()
        self.negatives()

    def plot_coverage(self) -> None:
        coverages = []
        for substrate in tqdm(self.card):
            coverages.append(len(self.card[substrate] & self.gen_card[substrate]) / len(self.card[substrate]))
        sns.boxplot(coverages)
        plt.show()

    def __repr__(self) -> str:
        self.clean()
        return f'MolFrame: {len(self.card)} substrates'

    def __str__(self) -> str:
        self.clean()
        return f'MolFrame: {len(self.card)} substrates'

    def morganize(self, size: int = 256) -> None:
        r"""
        Generate Morgan fingerprints for each molecule.
        :param size: size of Morgan fingerprints
        :return: :class:None
        """
        morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            includeChirality=True, fpSize=size, countSimulation=False)
        for mol in tqdm(self.mol_structs):
            self.morgan[mol] = tensor([morgan_fp_gen.GetFingerprint(self.mol_structs[mol])], dtype=torch.double)

    def pairgraphs(self, pca: bool = True) -> None:
        r"""
        Generate molecule pair graphs.
        :return: :class:None
        """
        if pca:
            with open('/Users/nikitapolomosnov/PycharmProjects/GRAIL/notebooks/pca_ats.pkl', 'rb') as file:
                pca_x = pkl.load(file)
            with open('/Users/nikitapolomosnov/PycharmProjects/GRAIL/notebooks/pca_bonds.pkl', 'rb') as file:
                pca_b = pkl.load(file)
        mols = self.mol_structs
        for substrate in tqdm(self.card):
            for product in self.card[substrate]:
                self.graphs[substrate].append(from_pair(mols[substrate], mols[product]))
                if self.graphs[substrate][-1] is not None:
                    self.graphs[substrate][-1].y = tensor([1.], dtype=torch.double)
                    self.graphs[substrate][-1].fp = torch.cat([self.morgan[substrate], self.morgan[product]], dim=1)
                    self.graphs[substrate][-1].smiles = product
        for substrate in tqdm(self.negs):
            for product in self.negs[substrate]:
                try:
                    self.graphs[substrate].append(from_pair(mols[substrate], mols[product]))
                except TimeoutError:
                    self.graphs[substrate].append(None)
                    continue
                finally:
                    if self.graphs[substrate][-1] is not None:
                        self.graphs[substrate][-1].y = tensor([0.], dtype=torch.double)
                        self.graphs[substrate][-1].fp = torch.cat([self.morgan[substrate], self.morgan[product]], dim=1)
                        self.graphs[substrate][-1].smiles = product

        for graph_card in tqdm(self.graphs.values()):
            for pair in graph_card:
                if pair is not None:
                    for i in range(len(pair.x)):
                        for j in range(len(pair.x[i])):
                            if pair.x[i][j] == float('inf'):
                                pair.x[i][j] = 0
                    pair.x = tensor(SimpleImputer(missing_values=np.nan,
                                                  strategy='constant',
                                                  fill_value=0).fit_transform(pair.x))
                    if pca:
                        pair.x = tensor(pca_x.transform(pair.x))
                        pair.edge_attr = tensor(pca_b.transform(pair.edge_attr))

    def singlegraphs(self, pca: bool = True) -> None:
        r"""
        Generate molecule graphs.
        :return: :class:None
        """
        if pca:
            with open('/Users/nikitapolomosnov/PycharmProjects/GRAIL/notebooks/pca_ats_single.pkl', 'rb') as file:
                pca_x = pkl.load(file)
            with open('/Users/nikitapolomosnov/PycharmProjects/GRAIL/notebooks/pca_bonds_single.pkl', 'rb') as file:
                pca_b = pkl.load(file)
        mols = self.mol_structs
        for mol in tqdm(self.card):
            try:
                self.single[mol] = from_rdmol(mols[mol])
            except ValueError:
                self.single[mol] = None
                continue
            self.single[mol].fp = self.morgan[mol]
            self.single[mol].y = tensor([1.], dtype=torch.double)
            for prod in self.card[mol]:
                try:
                    self.single[prod] = from_rdmol(mols[prod])
                    self.single[prod].y = tensor([1.], dtype=torch.double)
                    self.single[prod].fp = self.morgan[prod]
                except ValueError:
                    self.single[prod] = None
        for mol in tqdm(self.negs):
            try:
                self.single[mol] = from_rdmol(mols[mol])
            except ValueError:
                self.single[mol] = None
                continue
            self.single[mol].fp = self.morgan[mol]
            self.single[mol].y = tensor([1.], dtype=torch.double)
            for prod in self.negs[mol]:
                try:
                    self.single[prod] = from_rdmol(mols[prod])
                    self.single[prod].y = tensor([0.], dtype=torch.double)
                    self.single[prod].fp = self.morgan[prod]
                except ValueError:
                    self.single[prod] = None

        for key, pair in tqdm(self.single.items()):
            if pair is not None:
                for i in range(len(pair.x)):
                    for j in range(len(pair.x[i])):
                        if pair.x[i][j] == float('inf'):
                            pair.x[i][j] = 0
                pair.x = tensor(SimpleImputer(missing_values=np.nan,
                                              strategy='constant',
                                              fill_value=0).fit_transform(pair.x))
                if pca:
                    pair.x = tensor(pca_x.transform(pair.x))
                    try:
                        pair.edge_attr = tensor(pca_b.transform(pair.edge_attr))
                    except ValueError:
                        print('Some issue happened with this molecule:')
                        print(key, pair.edge_attr, pair.x)
                        self.single[key] = None

    def full_setup(self) -> None:
        self.clean()
        self.negatives()
        print('Morgan fingerprints generation')
        self.morganize()
        print('Pair graphs generation')
        self.pairgraphs()
        print('Single graphs generation')
        self.singlegraphs()

    def make_everything_good(self):
        del self

    def train_pairs(self, model: Module,
                    test: 'MolFrame',
                    lr: float = 1e-5,
                    eps: int = 100,
                    decay: float = 1e-8,
                    verbose: bool = True) -> Module:
        r"""
        Train the given model on pairgraphs.
        :param model: Model to train
        :param test: MolFrame object to train on
        :param lr: learning rate
        :param eps: number of epochs
        :param verbose: toggle verbose mode
        :param decay: weight decay factor
        :return: :class:`Module` - trained model
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        criterion = BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

        def cpunum(tensor) -> np.ndarray:
            return tensor.cpu().detach().numpy()

        if verbose:
            print('Starting DataLoaders generation')

        train_loader = []
        for pairs in self.graphs.values():
            for pair in pairs:
                if pair is not None:
                    train_loader.append(pair)
        train_loader = DataLoader(train_loader, batch_size=128, shuffle=True)

        test_loader = []
        for pairs in test.graphs.values():
            for pair in pairs:
                if pair is not None:
                    test_loader.append(pair)
        test_loader = DataLoader(test_loader, batch_size=128)

        @torch.no_grad()
        def test(loader: DataLoader) -> Tuple[float, float]:
            model.eval()
            pred = []
            bin_pred = []
            real = []
            for data in tqdm(loader):
                out = model(data)
                pred.extend(list(cpunum(out)))
                bin_pred.extend(list((cpunum(out) > 0.5).astype(int)))
                real.extend(list(cpunum(data.y)))
            mat = mcc(real, bin_pred)
            roc = roc_auc(real, pred)
            return mat, roc

        history = []
        for epoch in tqdm(range(eps)):
            model.train()
            for batch in train_loader:
                out = model(batch)
                loss = criterion(out, batch.y.unsqueeze(1).float())
                history.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            if (epoch+1) % 10 == 0:
                mat, roc = test(test_loader)
                plt.plot(history)
                if verbose:
                    print(f'Epoch {epoch+1}: MCC = {mat:.4f}, ROC = {roc:.4f}')

        return model

    def train_singles(self, model: Module, test: 'MolFrame', lr: float = 1e-5, eps: int = 100, decay: float = 1e-8, verbose: bool = True) -> Module:
        r"""
        Train the given model on singlegraphs.
        :param model: Model to train
        :param test: MolFrame object to train on
        :param lr: learning rate
        :param eps: number of epochs
        :param verbose: toggle verbose mode
        :return: :class:`Module` - trained model
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        criterion = BCELoss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

        def collate(data_list: List) -> Tuple[Batch, Batch]:
            batchA = Batch.from_data_list([data[0] for data in data_list])
            batchB = Batch.from_data_list([data[1] for data in data_list])
            return batchA, batchB

        if verbose:
            print('Starting DataLoaders generation')

        train_loader = []
        for mol in self.card.keys():
            sub = self.single[mol]
            if sub is not None:
                for met in self.card[mol]:
                    if self.single[met] is not None:
                        train_loader.append((sub, self.single[met]))
        for mol in self.negs.keys():
            sub = self.single[mol]
            if sub is not None:
                for met in self.negs[mol]:
                    if self.single[met] is not None:
                        train_loader.append((sub, self.single[met]))
        train_loader = DataLoader(train_loader, batch_size=128, shuffle=True, collate_fn=collate)

        test_loader = []
        for mol in test.card.keys():
            sub = test.single[mol]
            if sub is not None:
                for met in test.card[mol]:
                    if test.single[met] is not None:
                        test_loader.append((sub, test.single[met]))
        for mol in test.negs.keys():
            sub = test.single[mol]
            if sub is not None:
                for met in test.negs[mol]:
                    if test.single[met] is not None:
                        test_loader.append((sub, test.single[met]))
        test_loader = DataLoader(test_loader, batch_size=128, collate_fn=collate)

        @torch.no_grad()
        def test(loader: DataLoader) -> Tuple[float, float]:
            model.eval()
            pred = []
            bin_pred = []
            real = []
            for data in tqdm(loader):
                met_batch = data[1].to(device)
                sub_batch = data[0].to(device)
                out = model(sub_batch, met_batch)
                pred.extend(list(cpunum(out)))
                bin_pred.extend(list((cpunum(out) > 0.5).astype(int)))
                real.extend(list(cpunum(met_batch.y)))
            mat = mcc(real, bin_pred)
            roc = roc_auc(real, pred)
            return mat, roc

        history = []
        for epoch in tqdm(range(eps)):
            model.train()
            for batch in train_loader:
                met_batch = batch[1].to(device)
                sub_batch = batch[0].to(device)
                out = model(sub_batch, met_batch)
                loss = criterion(out, met_batch.y.unsqueeze(1).float())
                history.append(loss.item())
                if verbose: plt.plot(history)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            if (epoch+1) % 10 == 0:
                mat, roc = test(test_loader)
                if verbose:
                    print(f'Epoch {epoch+1}: MCC = {mat:.4f}, ROC = {roc:.4f}')

        return model

    @torch.no_grad()
    def test(self, model: Module, mode: Literal['single', 'pair']) -> Tuple[float, float]:
        r"""
        Test the given model and return its metrics.
        :param model: :class:`Module` model to train
        :param mode: if 'single' tests on singlegraphs, if 'pair' tests on pairgraphs
        :return: MCC and ROC AUC values of the model on this MolFrame
        """
        model.eval()
        pred = []
        bin_pred = []
        real = []

        def collate(data_list: List) -> Tuple[Batch, Batch]:
            batchA = Batch.from_data_list([data[0] for data in data_list])
            batchB = Batch.from_data_list([data[1] for data in data_list])
            return batchA, batchB

        loader = []
        if mode == 'single':
            for mol in self.card.keys():
                sub = self.single[mol]
                if sub is not None:
                    for met in self.card[mol]:
                        if self.single[met] is not None:
                            loader.append((sub, self.single[met]))
            for mol in self.negs.keys():
                sub = self.single[mol]
                if sub is not None:
                    for met in self.negs[mol]:
                        if self.single[met] is not None:
                            loader.append((sub, self.single[met]))
            loader = DataLoader(loader, batch_size=128, shuffle=True, collate_fn=collate)

        elif mode == 'pair':
            for pairs in self.graphs.values():
                for pair in pairs:
                    if pair is not None:
                        loader.append(pair)
            loader = DataLoader(loader, batch_size=128, shuffle=True)

        else:
            raise AttributeError

        for data in tqdm(loader):
            if mode == 'single':
                out = model(*data)
            elif mode == 'pair':
                out = model(data)
            else:
                raise AttributeError
            pred.extend(list(cpunum(out)))
            bin_pred.extend(list((cpunum(out) > 0.5).astype(int)))
            if mode == 'single':
                real.extend(list(cpunum(data[1].y)))
            else:
                real.extend(list(cpunum(data.y)))
        mat = mcc(real, bin_pred)
        roc = roc_auc(real, pred)
        return mat, roc

    @torch.no_grad()
    def visualize(self, model: Module, mode: Literal['single', 'pair']) -> Optional[tuple[list[float], list[float],
    list[float], list[float]]]:
        r"""
        Visualize the given model metrics on each metabolic map and return lists of them.
        :param model: model to visualize
        :param mode: if 'single' - tests on singlegraphs, if 'pair' - tests on pairgraphs
        :return: Jaccard, F1, Precision and Recall of the given model on each metabolic map
        """
        jac = []
        f1 = []
        precision = []
        recall = []
        model.eval()
        if mode == 'single':
            for mol, prods in tqdm(self.card.items(), total=len(self.card)):
                if self.single[mol] is not None:
                    tp = 0
                    fp = 0
                    fn = 0
                    tn = 0
                    for prod in prods:
                        if self.single[prod] is not None:
                            out = model(self.single[mol], self.single[prod]).item()
                            if out >= 0.5:
                                if prod in self.gen_card[mol]:
                                    tp += 1
                                else:
                                    fn +=1
                            else:
                                fn += 1
                    for genprod in self.negs[mol]:
                        if self.single[genprod] is not None:
                            out = model(self.single[mol], self.single[genprod]).item()
                            if out >= 0.5:
                                fp += 1
                            else:
                                tn += 1
                    precision.append(tp / (tp + fp))
                    recall.append(tp / (tp + fn))
                    jac.append(tp / (tp + fp + fn))
                    f1.append(2 * tp / (2 * tp + fn + tn))

        elif mode == 'pair':
            #print('In process :)')
            for sub, pairs in tqdm(self.graphs.items()):
                tp = 0
                fp = 0
                fn = 0
                tn = 0
                for pair in pairs:
                    if pair is not None:
                        out = model(pair).item()
                        if out >= 0.5:
                            if pair.smiles in self.gen_card[sub] and pair.y.item == 1:
                                tp += 1
                            elif pair.y.item == 0:
                                fp += 1
                            else:
                                fn += 1
                        else:
                            if pair.y.item == 0:
                                tn += 1
                            elif pair.y.item == 1:
                                fn += 1
                precision.append(tp / (tp + fp))
                recall.append(tp / (tp + fn))
                jac.append(tp / (tp + fn + tn))
                f1.append(2 * tp / (2 * tp + fn + tn))

        else:
            raise AttributeError

        nested_list = [precision, recall, f1, jac]
        fig, ax = plt.subplots()
        flierprops = dict(marker='x', markerfacecolor='orange', markersize=1,
                          markeredgecolor='none')
        meanpointprops = dict(marker='D', markeredgecolor='black',
                              markerfacecolor='firebrick')
        ax.boxplot(nested_list,
                   showmeans=True,
                   flierprops=flierprops,
                   meanprops=meanpointprops,
                   bootstrap=10_000)
        ax.set_xticks([y + 1 for y in range(len(nested_list))],
                      labels=['Precision', 'Recall', 'F1', 'Jaccard index'])
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Observed values')
        plt.show()

        return precision, recall, f1, jac

    def permute_augmentation(self, cutoff: Optional[float] = 0.85) -> None:
        dim = 256
        bits = 256
        index = faiss.IndexLSH(dim, bits)
        morgans = []
        reversed_morgans = {}
        for key, tens in tqdm(self.morgan.items()):
            if key not in self.card.keys():
                morgans.append(tens.squeeze(0).numpy())
                reversed_morgans[tuple(tens.squeeze(0).numpy())] = key
        morgans = np.array(morgans)
        index.add(morgans)
        D, I = index.search(morgans, 10)

        transposed = {}
        for key, prods in tqdm(self.card.items()):
            for prod in prods:
                transposed[prod] = key

        morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            includeChirality=True, fpSize=256, countSimulation=False)

        if not os.path.exists('uspto_model.onnx'):
            os.system('download_public_data .')

        expander = AiZynthExpander(configfile='config.yml')
        expander.expansion_policy.select("uspto")
        expander.filter_policy.select("uspto")

        def real_in_generated(frac: float, I) -> list[str]:
            reals = []
            for subset in tqdm(I):
                for idx in subset[1:]:
                    sub = reversed_morgans[tuple(morgans[idx])]
                    real = False
                    for pair in get_reactions(expander, reversed_morgans[tuple(morgans[idx])]):
                        for mol in pair:
                            if jac(self.morgan[transposed[sub]].squeeze(0).numpy(),
                                   np.array(morgan_fp_gen.GetFingerprint(Chem.MolFromSmiles(mol)))) >= frac:
                                real = True
                                reals.append(mol)
                                break
                        if real:
                            break
            return reals

        if cutoff is not None:
            reals = real_in_generated(cutoff, I)
        for subset in tqdm(I):
            for idx in subset[1:]:
                prod = reversed_morgans[tuple(morgans[idx])]
                sub = transposed[prod]
                if cutoff is not None:
                    if prod in reals:
                        self.card[sub].add(prod)
                    else:
                        self.gen_card[sub].add(prod)
                else:
                    self.gen_card[sub].add(prod)

    @torch.no_grad()
    def metabolize_filter(self, filter_model: nn.Module, rules: List[str]) -> DefaultDict[str, Set[str]]:
        out_card = dd(set)
        filter_model.eval()
        for substrate in tqdm(self.card.keys()):
            for product in metaboliser(self.mol_structs[substrate], rules):
                if product is not None:
                    input = from_pair(substrate, product)
                    out = cpunum(filter_model(input))
                    if all(out >= 0.5):
                        out_card[substrate].add(product)
        return out_card

    def metabolic_mapping(self, rules: List[str]) -> DefaultDict[str, Dict[str, Set[int]]]:
        mapped_card = dd(dict)
        for substrate in tqdm(self.card):
            try:
                gen_stat = metaboliser(self.mol_structs[substrate], rules=rules)
            except TimeoutError:
                continue
            finally:
                for product in gen_stat:
                    product_smiles = Chem.MolToSmiles(product)
                    mapped_card[substrate][product_smiles] = gen_stat[product]
        return mapped_card

    def sample_cards(self, frac: float) -> 'MolFrame':
        subs = list(self.card.keys())
        sampled = random.sample(subs, int(len(subs) * frac))
        new_card = {}
        gen_card = dd(set)
        mol_structs = self.mol_structs
        for substrate in sampled:
            new_card[substrate] = self.card[substrate]
            gen_card[substrate] = self.gen_card[substrate]
        return MolFrame(new_card, gen_card=gen_card, mol_structs=mol_structs)

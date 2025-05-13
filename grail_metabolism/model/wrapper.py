from torch import nn
from torch_geometric.data import Data
import typing as tp
from abc import ABC, abstractmethod
from grail_metabolism.utils.preparation import MolFrame, iscorrect, standardize_mol
from ..utils.transform import from_rule, from_rdmol
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.AllChem import ReactionFromSmarts
from itertools import chain

class GFilter(nn.Module, ABC):
    def __init__(self):
        nn.Module.__init__(self)
        self.mode: tp.Optional[tp.Literal['single', 'pair']] = None

    @abstractmethod
    def fit(self, data: MolFrame, lr: float = 1e-5, verbose: bool = True) -> 'GFilter':
        r"""
        Learn the model from the given MolFrame
        :param data:
        :param lr: learning rate
        :param verbose: verbose training process
        :return: self
        """

    @abstractmethod
    def predict(self, sub: str, prod: str) -> int:
        r"""
        Predict whether the given data or data pair is correct substrate-metabolite pair or not
        :param sub: substrate SMILES
        :param prod: product SMILES
        :return: pair class (correct or not)
        """

class GGenerator(nn.Module, ABC):
    def __init__(self):
        nn.Module.__init__(self)

    @abstractmethod
    def fit(self, data: MolFrame, lr:float = 1e-5, verbose: bool = True) -> 'GGenerator':
        r"""
        Learn the model from the given MolFrame
        :param data: MolFrame
        :param lr: learning rate
        :param verbose: verbose training process
        :return: self
        """

    @abstractmethod
    def generate(self, sub: str) -> tp.List[str]:
        r"""
        Generate metabolites of the given substrate
        :param sub: substrate SMILES
        :return: list of product SMILES
        """


class ModelWrapper:
    def __init__(self, filter: GFilter, generator: tp.Union[tp.Literal['simple'], GGenerator], rules: tp.Optional[tp.List[str]] = None) -> None:
        self.filter = filter
        self.generator = generator
        if generator == 'simple':
            self.rules = rules
            self.generator = SimpleGenerator(rules)
        else:
            self.rules = list(generator.rules.keys())

    def fit(self, data: MolFrame) -> 'ModelWrapper':
        r"""
        Learn the model from the given MolFrame
        :param data: MolFrame
        :return: self
        """
        if not data.graphs:
            data.full_setup()
        print('Filter learning')
        self.filter.fit(data)
        print('Generator learning')
        self.generator.fit(data)
        return self

    def generate(self, sub: str) -> tp.List[str]:
        to_check = self.generator.generate(sub)
        to_return = []
        for mol in to_check:
            is_real = bool(self.filter.predict(sub, mol))
            if is_real:
                to_return.append(mol)
        return to_return

class SimpleGenerator(GGenerator):
    def __init__(self, rules: tp.List[str]):
        self.rules = rules

    def fit(self, data: MolFrame):
        pass

    def generate(self, sub: str) -> tp.List[str]:
        mol = Chem.MolFromSmiles(sub)
        out = []
        for i, rule in enumerate(tqdm(self.rules)):
            rxn = ReactionFromSmarts(rule)
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
                    out.append(stand)
        return out
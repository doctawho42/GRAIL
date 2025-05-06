from torch import nn
from torch_geometric.data import Data
import typing as tp
from abc import ABC, abstractmethod
from grail_metabolism.utils.preparation import MolFrame

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
    def __init__(self, filter: GFilter, generator: tp.Union[tp.Literal['simple'], GGenerator]) -> None:
        self.filter = filter
        self.generator = generator

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
        if self.generator == 'simple':
            pass
        else:
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
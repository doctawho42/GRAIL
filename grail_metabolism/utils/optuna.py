import optuna
from typing import Optional
import pickle as pkl

from .preparation import MolFrame
from ..model.filter import Filter
from ..model.generator import Generator
from typing import Literal, Dict

class OptunaWrapper:
    r"""
    Optuna wrapper class
    """
    def __init__(self, study: Optional[optuna.study.Study] = None, mode: Literal['pair', 'single'] = 'pair') -> None:
        self.study = study
        self.lr = 0
        self.decay = 0
        self.arg_vec = []
        self.filter = None
        self.generator = None
        self.mode = mode

    @staticmethod
    def from_pickle(file_path: str) -> 'OptunaWrapper':
        r"""
        Create Optuna wrapper from pickled study file
        :param file_path: path to pickled study file
        :return:
        """
        with open(file_path, 'rb') as f:
            return OptunaWrapper(pkl.load(f))

    def make_study(self, train_set: MolFrame, test_set: MolFrame, direction: Literal['filter', 'generator'], rule_dict: Optional[Dict] = None) -> None:
        study = optuna.create_study(study_name=f'{direction}_{self.mode}' if direction == 'filter' else f'{direction}',
                                    direction='maximize' if direction == 'filter' else 'minimize',
                                    sampler=optuna.samplers.TPESampler(),
                                    pruner=optuna.pruners.HyperbandPruner())

        def objective(trial: optuna.trial.Trial) -> float:
            lr = trial.suggest_float('lr', 1e-7, 1e-2, log=True)
            decay = trial.suggest_float('decay', 1e-10, 1e-2, log=True)
            arg_vec = []
            if direction == 'filter':
                for i in range(1, 7):
                    arg_vec.append(trial.suggest_int(f"x{i}", 50, 1000))
                if self.mode == 'pair':
                    model = Filter(12, 6, arg_vec, self.mode)
                    train_set.train_pairs(model, test_set, lr=lr, eps=2, decay=decay)
                elif self.mode == 'single':
                    model = Filter(12, 6, arg_vec, self.mode)
                    train_set.train_singles(model, test_set, lr=lr, eps=2, decay=decay)
                else:
                    raise ValueError
                model.eval()
                mcc, roc = test_set.test(model, mode=self.mode)
                return mcc
            elif direction == 'generator':
                arg_vec.append(trial.suggest_int(f"x1", 50, 1000))
                arg_vec.append(trial.suggest_int(f"x2", 50, 1000))
                model = Generator(rule_dict, 10, 6, arg_vec=arg_vec)
                _, loss = train_set.train_generator(model, lr, eps=10, decay=decay)
                return loss

        study.optimize(objective, n_trials=100)
        self.study = study

    def create_optimal_filter(self) -> None:
        r"""
        Create model with optimal hyperparameters
        :return:
        """
        if self.study is None:
            raise ValueError('Study is None')
        self.lr = self.study.best_params['lr']
        self.decay = self.study.best_params['decay']
        for key in self.study.best_params.keys():
            if key.startswith('x'):
                self.arg_vec.append(self.study.best_params[key])
        if self.mode == 'pair':
            self.model = Filter(12, 6, self.arg_vec, self.mode)
        elif self.mode == 'single':
            self.model = Filter(12, 6, self.arg_vec, self.mode)
        else:
            raise ValueError

    def train_on(self, train_set: MolFrame, test_set: MolFrame) -> None:
        if self.study is None or self.model is None:
            raise ValueError('Nothing to train')
        self.create_optimal_filter()
        train_set.train_pairs(self.model, test_set, lr=self.lr, eps=100, decay=self.decay)

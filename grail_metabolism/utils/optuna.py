import optuna
from typing import Optional
import pickle as pkl

from grail_metabolism.utils.preparation import MolFrame
from grail_metabolism.model.filter import create_filter_pairs, create_filter_singles
from typing import Literal

class OptunaWrapper:

    def __init__(self, study: Optional[optuna.study.Study], mode: Literal['pair', 'single'] = 'pair') -> None:
        self.study = study
        self.lr = 0
        self.decay = 0
        self.arg_vec = []
        self.model = None
        self.mode = mode

    @staticmethod
    def from_pickle(file_path: str) -> 'OptunaWrapper':
        with open(file_path, 'rb') as f:
            return OptunaWrapper(pkl.load(f))

    def make_study(self, train_set: MolFrame, test_set: MolFrame) -> None:
        study = optuna.create_study(study_name=self.mode,
                                    direction='maximize',
                                    sampler=optuna.samplers.TPESampler(),
                                    pruner=optuna.pruners.HyperbandPruner())

        def objective(trial: optuna.trial.Trial) -> float:
            lr = trial.suggest_float('lr', 1e-7, 1e-2, log=True)
            decay = trial.suggest_float('decay', 1e-10, 1e-2, log=True)
            arg_vec = []
            for i in range(1, 7):
                arg_vec.append(trial.suggest_int(f"x{i}", 50, 1000))
            if self.mode == 'pair':
                model = create_filter_pairs(arg_vec)
                train_set.train_pairs(model, test_set, lr=lr, eps=2, decay=decay)
            elif self.mode == 'single':
                model = create_filter_singles(arg_vec)
                train_set.train_singles(model, test_set, lr=lr, eps=2, decay=decay)
            else:
                raise ValueError
            model.eval()
            mcc, roc = test_set.test(model, mode=self.mode)
            return mcc

        study.optimize(objective, n_trials=100)
        self.study = study

    def create_optimal_model(self) -> None:
        if self.study is None:
            raise ValueError('Study is None')
        self.lr = self.study.best_params['lr']
        self.decay = self.study.best_params['decay']
        for key in self.study.best_params.keys():
            if key.startswith('x'):
                self.arg_vec.append(self.study.best_params[key])
        if self.mode == 'pair':
            self.model = create_filter_pairs(self.arg_vec)
        elif self.mode == 'single':
            self.model = create_filter_singles(self.arg_vec)
        else:
            raise ValueError

    def train_on(self, train_set: MolFrame, test_set: MolFrame) -> None:
        if self.study is None or self.model is None:
            raise ValueError('Nothing to train')
        self.create_optimal_model()
        train_set.train_pairs(self.model, test_set, lr=self.lr, eps=100, decay=self.decay)

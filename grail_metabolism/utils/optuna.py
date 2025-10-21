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
            self.filter = Filter(12, 6, self.arg_vec, self.mode)
        elif self.mode == 'single':
            self.filter = Filter(10, 6, self.arg_vec, self.mode)
        else:
            raise ValueError

    def create_optimal_generator(self, rule_dict) -> None:
        if self.study is None:
            raise ValueError('Study is None')
        self.lr = self.study.best_params['lr']
        self.decay = self.study.best_params['decay']
        for key in self.study.best_params.keys():
            if key.startswith('x'):
                self.arg_vec.append(self.study.best_params[key])
        self.generator = Generator(rule_dict, 10, 6, arg_vec=self.arg_vec)

    def train_on(self, train_set: MolFrame, test_set: MolFrame) -> None:
        if self.study is None or self.model is None:
            raise ValueError('Nothing to train')
        self.create_optimal_filter()
        train_set.train_pairs(self.model, test_set, lr=self.lr, eps=100, decay=self.decay)


import optuna
from typing import Optional, Dict, Literal, List, Union
import pickle as pkl
import torch

from .preparation import MolFrame
from ..model.filter import Filter, MolPathFilter
from ..model.generator import Generator
from ..model.train_model import test
from torch_geometric.loader import DataLoader


class EnhancedOptunaWrapper:
    r"""
    Enhanced Optuna wrapper supporting MolPathFilter and updated Generator
    """

    def __init__(self,
                 study: Optional[optuna.study.Study] = None,
                 mode: Literal['pair', 'single'] = 'pair',
                 use_molpath: bool = False) -> None:
        self.study = study
        self.lr = 0
        self.decay = 0
        self.arg_vec = []
        self.filter = None
        self.generator = None
        self.mode = mode
        self.use_molpath = use_molpath
        self.molpath_params = {}

    @staticmethod
    def from_pickle(file_path: str) -> 'EnhancedOptunaWrapper':
        r"""
        Create Optuna wrapper from pickled study file
        """
        with open(file_path, 'rb') as f:
            study = pkl.load(f)
        return EnhancedOptunaWrapper(study)

    def make_study(self,
                   train_set: MolFrame,
                   test_set: MolFrame,
                   direction: Literal['filter', 'generator'],
                   rule_dict: Optional[Dict] = None,
                   n_trials: int = 100) -> None:

        study = optuna.create_study(
            study_name=f'{direction}_{self.mode}_molpath' if self.use_molpath else f'{direction}_{self.mode}',
            direction='maximize' if direction == 'filter' else 'minimize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner()
        )

        def objective(trial: optuna.trial.Trial) -> float:
            lr = trial.suggest_float('lr', 1e-7, 1e-2, log=True)
            decay = trial.suggest_float('decay', 1e-10, 1e-2, log=True)

            if direction == 'filter':
                if self.use_molpath:
                    # Parameters for MolPathFilter
                    molpath_hidden = trial.suggest_int('molpath_hidden', 64, 512)
                    molpath_cutoff = trial.suggest_int('molpath_cutoff', 3, 8)
                    molpath_y = trial.suggest_float('molpath_y', 0.1, 0.9)

                    if self.mode == 'pair':
                        arg_vec = [
                            trial.suggest_int(f"x{i}", 50, 500) for i in range(1, 5)
                        ]
                        model = MolPathFilter(
                            12, 6, arg_vec, self.mode,
                            molpath_hidden=molpath_hidden,
                            molpath_cutoff=molpath_cutoff,
                            molpath_y=molpath_y
                        )
                    else:  # single mode
                        arg_vec = [
                            trial.suggest_int(f"x{i}", 50, 500) for i in range(1, 6)
                        ]
                        model = MolPathFilter(
                            10, 6, arg_vec, self.mode,
                            molpath_hidden=molpath_hidden,
                            molpath_cutoff=molpath_cutoff,
                            molpath_y=molpath_y
                        )
                else:
                    # Standard Filter parameters
                    arg_vec = [trial.suggest_int(f"x{i}", 50, 1000) for i in range(1, 7)]

                    if self.mode == 'pair':
                        model = Filter(12, 6, arg_vec, self.mode)
                    else:  # single mode
                        model = Filter(10, 6, arg_vec, self.mode)

                # Train and evaluate
                if self.mode == 'pair':
                    train_set.train_pairs(model, test_set, lr=lr, eps=5, decay=decay, verbose=False)
                else:
                    train_set.train_singles(model, test_set, lr=lr, eps=5, decay=decay, verbose=False)

                model.eval()
                mcc, roc = test_set.test(model, mode=self.mode)
                return mcc  # Maximize MCC

            elif direction == 'generator':
                if self.use_molpath:
                    # Parameters for Generator with MolPath
                    molpath_hidden = trial.suggest_int('molpath_hidden', 64, 512)
                    molpath_cutoff = trial.suggest_int('molpath_cutoff', 3, 8)
                    molpath_y = trial.suggest_float('molpath_y', 0.1, 0.9)
                    rp_arg_vec = [trial.suggest_int(f"rp_x{i}", 100, 500) for i in range(1, 6)]

                    arg_vec = [
                        trial.suggest_int(f"x{i}", 100, 800) for i in range(1, 3)
                    ]

                    model = Generator(
                        rule_dict, 10, 6,
                        arg_vec=arg_vec,
                        rp_arg_vec=rp_arg_vec,
                        use_molpath=True,
                        molpath_hidden=molpath_hidden,
                        molpath_cutoff=molpath_cutoff,
                        molpath_y=molpath_y
                    )
                else:
                    # Standard Generator parameters
                    arg_vec = [
                        trial.suggest_int(f"x{i}", 100, 800) for i in range(1, 3)
                    ]
                    rp_arg_vec = [trial.suggest_int(f"rp_x{i}", 100, 500) for i in range(1, 6)]

                    model = Generator(
                        rule_dict, 10, 6,
                        arg_vec=arg_vec,
                        rp_arg_vec=rp_arg_vec
                    )

                # Train generator (simplified training for hyperparameter search)
                model.fit(train_set, lr=lr, verbose=False)

                # Evaluate using Jaccard score
                jaccard_scores = model.jaccard(test_set)
                avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
                return -avg_jaccard  # Minimize negative Jaccard (equivalent to maximizing Jaccard)

        study.optimize(objective, n_trials=n_trials)
        self.study = study

    def create_optimal_filter(self) -> None:
        r"""
        Create optimal filter model with best hyperparameters
        """
        if self.study is None:
            raise ValueError('Study is None')

        self.lr = self.study.best_params['lr']
        self.decay = self.study.best_params['decay']

        if self.use_molpath:
            # Extract MolPath parameters
            self.molpath_params = {
                'molpath_hidden': self.study.best_params['molpath_hidden'],
                'molpath_cutoff': self.study.best_params['molpath_cutoff'],
                'molpath_y': self.study.best_params['molpath_y']
            }

            # Extract architecture parameters
            if self.mode == 'pair':
                self.arg_vec = [self.study.best_params[f'x{i}'] for i in range(1, 5)]
                self.filter = MolPathFilter(
                    12, 6, self.arg_vec, self.mode, **self.molpath_params
                )
            else:
                self.arg_vec = [self.study.best_params[f'x{i}'] for i in range(1, 6)]
                self.filter = MolPathFilter(
                    10, 6, self.arg_vec, self.mode, **self.molpath_params
                )
        else:
            # Standard filter
            self.arg_vec = [self.study.best_params[f'x{i}'] for i in range(1, 7)]

            if self.mode == 'pair':
                self.filter = Filter(12, 6, self.arg_vec, self.mode)
            else:
                self.filter = Filter(10, 6, self.arg_vec, self.mode)

    def create_optimal_generator(self, rule_dict: Dict) -> None:
        r"""
        Create optimal generator model with best hyperparameters
        """
        if self.study is None:
            raise ValueError('Study is None')

        self.lr = self.study.best_params['lr']
        self.decay = self.study.best_params['decay']

        if self.use_molpath:
            # Extract MolPath parameters
            self.molpath_params = {
                'molpath_hidden': self.study.best_params['molpath_hidden'],
                'molpath_cutoff': self.study.best_params['molpath_cutoff'],
                'molpath_y': self.study.best_params['molpath_y']
            }

            # Extract architecture parameters
            self.arg_vec = [self.study.best_params[f'x{i}'] for i in range(1, 3)]
            rp_arg_vec = [self.study.best_params[f'rp_x{i}'] for i in range(1, 6)]

            self.generator = Generator(
                rule_dict, 10, 6,
                arg_vec=self.arg_vec,
                rp_arg_vec=rp_arg_vec,
                use_molpath=True,
                **self.molpath_params
            )
        else:
            # Standard generator
            self.arg_vec = [self.study.best_params[f'x{i}'] for i in range(1, 3)]
            rp_arg_vec = [self.study.best_params[f'rp_x{i}'] for i in range(1, 6)]

            self.generator = Generator(
                rule_dict, 10, 6,
                arg_vec=self.arg_vec,
                rp_arg_vec=rp_arg_vec
            )

    def train_on(self, train_set: MolFrame, test_set: MolFrame, direction: str) -> None:
        r"""
        Train optimal model on full dataset
        """
        if direction == 'filter':
            if self.filter is None:
                self.create_optimal_filter()

            if self.mode == 'pair':
                train_set.train_pairs(self.filter, test_set, lr=self.lr, eps=100, decay=self.decay)
            else:
                train_set.train_singles(self.filter, test_set, lr=self.lr, eps=100, decay=self.decay)

        elif direction == 'generator':
            if self.generator is None:
                raise ValueError('Generator not created. Call create_optimal_generator first.')

            self.generator.fit(train_set, lr=self.lr, verbose=True)


class MorganOnlyFilter:
    r"""
    Filter using only Morgan fingerprints (baseline)
    """

    def __init__(self, input_dim: int = 2048, hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.extend([
            torch.nn.Linear(prev_dim, 1),
            torch.nn.Sigmoid()
        ])

        self.model = torch.nn.Sequential(*layers)
        self.mode = 'pair'

    def forward(self, data):
        return self.model(data.fp)

    def fit(self, data: MolFrame, lr: float = 1e-5, eps: int = 100, verbose: bool = True,
            prior: float = 0.75, nnPU: bool = True) -> 'MorganOnlyFilter':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-8)

        # Prepare training data
        train_loader = []
        for pairs in data.graphs.values():
            for pair in pairs:
                if pair is not None:
                    train_loader.append(pair)
        train_loader = DataLoader(train_loader, batch_size=128, shuffle=True)

        for epoch in range(eps):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                out = self(batch)
                loss = criterion(out, batch.y.unsqueeze(1).float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose and (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')

        return self

    def predict(self, sub: str, prod: str, pca: bool = True) -> int:
        # For Morgan-only filter, we don't need molecular graphs
        # This would need to be implemented based on your specific fingerprint generation
        # Placeholder implementation
        return 0

    @torch.no_grad()
    def mcc(self, test_frame):
        self.model.eval()
        device = next(self.model.parameters()).device

        mccs = []
        for sub in test_frame.map:
            reals = []
            bins = []
            for prod in test_frame.map[sub]:
                bins.append(self.predict(sub, prod))
                reals.append(1)
            for prod in test_frame.negs[sub]:
                bins.append(self.predict(sub, prod))
                reals.append(0)
            mccs.append(matthews_corrcoef(reals, bins))
        return mccs
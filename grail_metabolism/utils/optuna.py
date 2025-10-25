import optuna
from typing import Optional
import pickle as pkl

from .preparation import MolFrame
from ..model.filter import Filter, GATv2Filter
from ..model.generator import Generator
from typing import Literal, Dict
from torch.nn import Module
from sklearn.metrics import matthews_corrcoef

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
                    molpath_cutoff = trial.suggest_int('molpath_cutoff', 3, 8)
                    molpath_y = trial.suggest_float('molpath_y', 0.1, 0.9)

                    if self.mode == 'pair':
                        arg_vec = [
                            trial.suggest_int(f"x{i}", 50, 500) for i in range(1, 5)
                        ]
                        model = MolPathFilter(
                            12, 6, arg_vec, self.mode,
                            molpath_cutoff=molpath_cutoff,
                            molpath_y=molpath_y
                        )
                    else:  # single mode
                        arg_vec = [
                            trial.suggest_int(f"x{i}", 50, 500) for i in range(1, 6)
                        ]
                        model = MolPathFilter(
                            10, 6, arg_vec, self.mode,
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
                self.filter = GATv2Filter(12, 6, self.arg_vec, self.mode)
            else:
                self.filter = GATv2Filter(10, 6, self.arg_vec, self.mode)

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


import optuna
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle as pkl
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

from .preparation import MolFrame
from ..model.filter import Filter, MolPathFilter, GINFilter, GCNFilter, MorganOnlyFilter
from ..model.generator import Generator
from ..model.train_model import test
from torch_geometric.loader import DataLoader
from rdkit import Chem


class ComprehensiveOptunaWrapper:
    r"""
    Comprehensive Optuna wrapper for all filter models with hyperparameter optimization
    """

    def __init__(self,
                 study: Optional[optuna.study.Study] = None,
                 mode: str = 'pair',
                 model_type: str = 'Filter',
                 use_molpath: bool = False) -> None:
        self.study = study
        self.mode = mode
        self.model_type = model_type
        self.use_molpath = use_molpath
        self.lr = 0
        self.decay = 0
        self.arg_vec = []
        self.filter = None
        self.generator = None
        self.molpath_params = {}
        self.best_score = -1
        self.best_roc_auc = -1

    @staticmethod
    def from_pickle(file_path: str) -> 'ComprehensiveOptunaWrapper':
        r"""
        Create Optuna wrapper from pickled study file
        """
        with open(file_path, 'rb') as f:
            study = pkl.load(f)
        return ComprehensiveOptunaWrapper(study)

    def _get_filter_model(self, trial: optuna.trial.Trial, in_channels: int, edge_dim: int) -> Tuple[
        torch.nn.Module, float, float]:
        """Create filter model based on model_type with suggested hyperparameters"""
        lr = trial.suggest_float('lr', 1e-7, 1e-2, log=True)
        decay = trial.suggest_float('decay', 1e-10, 1e-2, log=True)

        if self.model_type == 'Filter':
            arg_vec = [trial.suggest_int(f"x{i}", 50, 1000) for i in range(1, 7)]
            return Filter(in_channels, edge_dim, arg_vec, self.mode), lr, decay

        elif self.model_type == 'GATv2Filter':
            arg_vec = [trial.suggest_int(f"x{i}", 50, 1000) for i in range(1, 7)]
            return Filter(in_channels, edge_dim, arg_vec, self.mode), lr, decay

        elif self.model_type == 'MolPathFilter':
            molpath_cutoff = trial.suggest_int('molpath_cutoff', 3, 8)
            molpath_y = trial.suggest_float('molpath_y', 0.1, 0.9)

            if self.mode == 'pair':
                arg_vec = [trial.suggest_int(f"x{i}", 50, 500) for i in range(1, 5)]
            else:
                arg_vec = [trial.suggest_int(f"x{i}", 50, 500) for i in range(1, 6)]

            model = MolPathFilter(
                in_channels, edge_dim, arg_vec, self.mode,
                molpath_cutoff=molpath_cutoff,
                molpath_y=molpath_y
            )
            return model, lr, decay

        elif self.model_type == 'GINFilter':
            arg_vec = [trial.suggest_int(f"x{i}", 50, 1000) for i in range(1, 7)]
            return GINFilter(in_channels, edge_dim, arg_vec, self.mode), lr, decay

        elif self.model_type == 'GCNFilter':
            arg_vec = [trial.suggest_int(f"x{i}", 50, 1000) for i in range(1, 7)]
            return GCNFilter(in_channels, edge_dim, arg_vec, self.mode), lr, decay

        elif self.model_type == 'MorganOnlyFilter':
            hidden_dims = []
            for i in range(3):
                hidden_dims.append(trial.suggest_int(f"hidden_dim_{i}", 64, 512))
            return MorganOnlyFilter(input_dim=2048, hidden_dims=hidden_dims), lr, decay

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def make_study(self,
                   train_set: MolFrame,
                   test_set: MolFrame,
                   direction: str = 'filter',
                   rule_dict: Optional[Dict] = None,
                   n_trials: int = 50) -> None:
        r"""
        Create Optuna study for hyperparameter optimization
        """

        def objective(trial: optuna.trial.Trial) -> float:
            try:
                # Get model dimensions from sample data
                if self.mode == 'pair':
                    sample_key = list(train_set.graphs.keys())[0]
                    in_channels = train_set.graphs[sample_key][0].x.shape[1]
                    edge_dim = train_set.graphs[sample_key][0].edge_attr.shape[1]
                else:
                    sample_key = list(train_set.single.keys())[0]
                    in_channels = train_set.single[sample_key].x.shape[1]
                    edge_dim = train_set.single[sample_key].edge_attr.shape[1]

                # Create model with suggested hyperparameters
                model, lr, decay = self._get_filter_model(trial, in_channels, edge_dim)

                if direction == 'filter':
                    # Train and evaluate filter
                    if self.model_type == 'MorganOnlyFilter':
                        # MorganOnlyFilter has different training interface
                        model.fit(train_set, lr=lr, eps=3, verbose=False)
                    else:
                        if self.mode == 'pair':
                            train_set.train_pairs(model, test_set, lr=lr, eps=3,
                                                  decay=decay, verbose=False)
                        else:
                            train_set.train_singles(model, test_set, lr=lr, eps=3,
                                                    decay=decay, verbose=False)

                    # Evaluate model
                    model.eval()
                    mcc, roc_auc = test_set.test(model, mode=self.mode)

                    # Store both metrics in trial user attributes
                    trial.set_user_attr("roc_auc", roc_auc)
                    return mcc  # Maximize MCC

                elif direction == 'generator':
                    # Generator optimization (placeholder - implement as needed)
                    if rule_dict is None:
                        raise ValueError("rule_dict required for generator optimization")

                    # Simplified generator training for hyperparameter search
                    arg_vec = [trial.suggest_int(f"x{i}", 100, 800) for i in range(1, 3)]
                    rp_arg_vec = [trial.suggest_int(f"rp_x{i}", 100, 500) for i in range(1, 6)]

                    generator = Generator(
                        rule_dict, in_channels, edge_dim,
                        arg_vec=arg_vec,
                        rp_arg_vec=rp_arg_vec
                    )

                    generator.fit(train_set, lr=lr, verbose=False)
                    jaccard_scores = generator.jaccard(test_set)
                    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
                    return avg_jaccard  # Maximize Jaccard

                else:
                    raise ValueError(f"Unsupported direction: {direction}")

            except Exception as e:
                # Return low score for failed trials
                print(f"Trial failed: {e}")
                return -1.0 if direction == 'filter' else 0.0

        study = optuna.create_study(
            study_name=f'{self.model_type}_{self.mode}_{direction}',
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner()
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.study = study
        self.best_score = study.best_value
        # Get ROC AUC from best trial
        if study.best_trial:
            self.best_roc_auc = study.best_trial.user_attrs.get("roc_auc", -1)

    def create_optimal_filter(self, train_set: MolFrame, best_params: Dict = None) -> None:
        r"""
        Create optimal filter model with best hyperparameters
        """
        if best_params is None:
            if self.study is None:
                raise ValueError('No study found. Run make_study first.')
            best_params = self.study.best_params

        # Get model dimensions
        if self.mode == 'pair':
            sample_key = list(train_set.graphs.keys())[0]
            in_channels = train_set.graphs[sample_key][0].x.shape[1]
            edge_dim = train_set.graphs[sample_key][0].edge_attr.shape[1]
        else:
            sample_key = list(train_set.single.keys())[0]
            in_channels = train_set.single[sample_key].x.shape[1]
            edge_dim = train_set.single[sample_key].edge_attr.shape[1]

        # Extract best parameters
        self.lr = best_params['lr']
        self.decay = best_params['decay']

        if self.model_type == 'Filter':
            self.arg_vec = [best_params[f'x{i}'] for i in range(1, 7)]
            self.filter = Filter(in_channels, edge_dim, self.arg_vec, self.mode)

        elif self.model_type == 'GATv2Filter':
            self.arg_vec = [best_params[f'x{i}'] for i in range(1, 7)]
            self.filter = Filter(in_channels, edge_dim, self.arg_vec, self.mode)

        elif self.model_type == 'MolPathFilter':
            self.molpath_params = {
                'molpath_cutoff': best_params['molpath_cutoff'],
                'molpath_y': best_params['molpath_y']
            }
            if self.mode == 'pair':
                self.arg_vec = [best_params[f'x{i}'] for i in range(1, 5)]
            else:
                self.arg_vec = [best_params[f'x{i}'] for i in range(1, 6)]
            self.filter = MolPathFilter(
                in_channels, edge_dim, self.arg_vec, self.mode, **self.molpath_params
            )

        elif self.model_type == 'GINFilter':
            self.arg_vec = [best_params[f'x{i}'] for i in range(1, 7)]
            self.filter = GINFilter(in_channels, edge_dim, self.arg_vec, self.mode)

        elif self.model_type == 'GCNFilter':
            self.arg_vec = [best_params[f'x{i}'] for i in range(1, 7)]
            self.filter = GCNFilter(in_channels, edge_dim, self.arg_vec, self.mode)

        elif self.model_type == 'MorganOnlyFilter':
            hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(3)]
            self.filter = MorganOnlyFilter(input_dim=2048, hidden_dims=hidden_dims)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train_optimal_filter(self, train_set: MolFrame, test_set: MolFrame,
                             epochs: int = 100) -> None:
        r"""
        Train optimal filter on full dataset
        """
        if self.filter is None:
            raise ValueError('Filter not created. Call create_optimal_filter first.')

        print(f"Training optimal {self.model_type} ({self.mode}) with LR: {self.lr:.2e}")

        if self.model_type == 'MorganOnlyFilter':
            self.filter.fit(train_set, lr=self.lr, eps=epochs, verbose=True)
        else:
            if self.mode == 'pair':
                train_set.train_pairs(self.filter, test_set, lr=self.lr,
                                      eps=epochs, decay=self.decay, verbose=True)
            else:
                train_set.train_singles(self.filter, test_set, lr=self.lr,
                                        eps=epochs, decay=self.decay, verbose=True)


class TwoStageOptunaFilterComparator:
    r"""
    Two-stage comprehensive filter comparison using Optuna:
    - Stage 1: Hyperparameter optimization with 3-fold CV
    - Stage 2: Final evaluation with 10-fold CV using optimized parameters
    """

    def __init__(self, optimization_folds: int = 3, evaluation_folds: int = 10,
                 n_trials: int = 30, random_state: int = 42):
        self.optimization_folds = optimization_folds
        self.evaluation_folds = evaluation_folds
        self.n_trials = n_trials
        self.random_state = random_state

        # Define all models to compare (including original Filter)
        self.all_models = [
            'Filter',  # Original Filter (GAT-based)
            'GATv2Filter',  # GAT-based filter
            'MolPathFilter',  # Molecular path-based filter
            'GINFilter',  # GIN-based filter
            'GCNFilter',  # GCN-based filter
            'MorganOnlyFilter'  # Morgan fingerprint baseline
        ]

        # Models and their supported modes
        self.model_modes = {
            'Filter': ['pair', 'single'],
            'GATv2Filter': ['pair', 'single'],
            'MolPathFilter': ['pair', 'single'],
            'GINFilter': ['pair', 'single'],
            'GCNFilter': ['pair', 'single'],
            'MorganOnlyFilter': ['pair']  # Only supports pair mode
        }

    def k_fold_split(self, molframe: MolFrame, k_folds: int) -> List[Tuple[MolFrame, MolFrame]]:
        """
        K-fold split by scaffolds using stratified approach
        """
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from collections import defaultdict
        import random
        import numpy as np

        random.seed(self.random_state)
        np.random.seed(self.random_state)

        def get_scaffold(smiles: str) -> str:
            try:
                mol = molframe.mol_structs.get(smiles)
                if mol is None:
                    mol = Chem.MolFromSmiles(smiles)
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                return Chem.MolToSmiles(scaffold)
            except:
                return smiles

        # Group substrates by scaffolds and count reactions
        scaffold_to_substrates = defaultdict(list)
        scaffold_reaction_counts = defaultdict(int)

        for substrate in molframe.map.keys():
            scaffold = get_scaffold(substrate)
            scaffold_to_substrates[scaffold].append(substrate)
            scaffold_reaction_counts[scaffold] += len(molframe.map[substrate])

        # Convert to list and sort by reaction count
        scaffolds = sorted(scaffold_reaction_counts.keys(),
                           key=lambda x: scaffold_reaction_counts[x],
                           reverse=True)

        # Initialize folds
        folds = [[] for _ in range(k_folds)]
        fold_reaction_counts = [0] * k_folds

        # Assign scaffolds to folds using stratified approach
        for scaffold in scaffolds:
            min_fold_idx = np.argmin(fold_reaction_counts)
            folds[min_fold_idx].append(scaffold)
            fold_reaction_counts[min_fold_idx] += scaffold_reaction_counts[scaffold]

        # Create train-val splits
        splits = []
        for i in range(k_folds):
            # Validation scaffolds
            val_scaffolds = folds[i]

            # Validation substrates
            val_subs = set()
            for scaffold in val_scaffolds:
                val_subs.update(scaffold_to_substrates[scaffold])

            # Training substrates
            train_subs = set(molframe.map.keys()) - val_subs

            # Create MolFrames
            train_molframe = molframe.subset_from_substrates(list(train_subs))
            val_molframe = molframe.subset_from_substrates(list(val_subs))

            splits.append((train_molframe, val_molframe))

        return splits

    def run_two_stage_comparison(self, molframe: MolFrame) -> Dict[str, Any]:
        """
        Run two-stage comparison:
        Stage 1: Hyperparameter optimization with 3-fold CV
        Stage 2: Final evaluation with 10-fold CV using optimized parameters
        """
        print("Starting Two-Stage Model Comparison Pipeline")
        print("=" * 80)
        print(f"Stage 1: Hyperparameter optimization with {self.optimization_folds}-fold CV")
        print(f"Stage 2: Final evaluation with {self.evaluation_folds}-fold CV")
        print(f"Models: {self.all_models}")
        print(f"Trials per model: {self.n_trials}")

        # Precompute everything once for the entire dataset
        print("\nPrecomputing graphs and fingerprints for the entire dataset...")
        molframe.full_setup(pca=True)
        print("Precomputation completed!")

        # Stage 1: Hyperparameter optimization with 3-fold CV
        print("\n" + "=" * 50)
        print("STAGE 1: HYPERPARAMETER OPTIMIZATION")
        print("=" * 50)

        optimization_splits = self.k_fold_split(molframe, self.optimization_folds)
        optimized_params = {}

        for model_type in tqdm(self.all_models, desc="Optimizing models"):
            supported_modes = self.model_modes[model_type]

            for mode in supported_modes:
                config_key = f"{model_type}_{mode}"
                print(f"\nOptimizing {config_key}...")

                # Use first fold for optimization (could use all, but this is faster)
                train_molframe_opt, val_molframe_opt = optimization_splits[0]
                # No need to call full_setup here - already precomputed

                try:
                    # Create Optuna wrapper
                    optuna_wrapper = ComprehensiveOptunaWrapper(
                        mode=mode,
                        model_type=model_type
                    )

                    # Run hyperparameter optimization
                    optuna_wrapper.make_study(
                        train_molframe_opt,
                        val_molframe_opt,
                        direction='filter',
                        n_trials=self.n_trials
                    )

                    if optuna_wrapper.study is not None:
                        # Store best parameters
                        best_params = optuna_wrapper.study.best_params
                        best_mcc = optuna_wrapper.best_score
                        best_roc_auc = optuna_wrapper.best_roc_auc

                        optimized_params[config_key] = {
                            'params': best_params,
                            'mcc': best_mcc,
                            'roc_auc': best_roc_auc
                        }

                        print(f"  {config_key}: Best MCC = {best_mcc:.4f}, ROC AUC = {best_roc_auc:.4f}")

                except Exception as e:
                    print(f"  Error optimizing {config_key}: {e}")
                    # Use default parameters if optimization fails
                    optimized_params[config_key] = {
                        'params': {'lr': 1e-5, 'decay': 1e-8},
                        'mcc': -1.0,
                        'roc_auc': -1.0
                    }

        # Stage 2: Final evaluation with 10-fold CV
        print("\n" + "=" * 50)
        print("STAGE 2: FINAL EVALUATION")
        print("=" * 50)

        evaluation_splits = self.k_fold_split(molframe, self.evaluation_folds)

        # Initialize results storage
        results = {
            'optimized_parameters': optimized_params,
            'model_performance_mcc': defaultdict(list),
            'model_performance_roc_auc': defaultdict(list),
            'training_history': [],
            'fold_details': defaultdict(list),
            'stage1_folds': self.optimization_folds,
            'stage2_folds': self.evaluation_folds
        }

        for fold_idx, (train_molframe, val_molframe) in enumerate(tqdm(evaluation_splits, desc="Evaluation folds")):
            print(f"\n--- Evaluation Fold {fold_idx + 1}/{self.evaluation_folds} ---")

            # No need to call full_setup here - already precomputed

            for model_type in tqdm(self.all_models, desc="Models", leave=False):
                supported_modes = self.model_modes[model_type]

                for mode in supported_modes:
                    config_key = f"{model_type}_{mode}"

                    if config_key not in optimized_params:
                        print(f"  No optimized parameters for {config_key}, skipping")
                        continue

                    print(f"  Evaluating {config_key}...")

                    try:
                        best_params = optimized_params[config_key]['params']

                        # Create and train model with optimized parameters
                        optuna_wrapper = ComprehensiveOptunaWrapper(
                            mode=mode,
                            model_type=model_type
                        )

                        # Create model with optimized parameters
                        optuna_wrapper.create_optimal_filter(train_molframe, best_params)

                        # Train model on current fold
                        if model_type == 'MorganOnlyFilter':
                            optuna_wrapper.filter.fit(train_molframe, lr=best_params['lr'],
                                                      eps=3, verbose=False)
                        else:
                            if mode == 'pair':
                                train_molframe.train_pairs(optuna_wrapper.filter, val_molframe,
                                                           lr=best_params['lr'], eps=3,
                                                           decay=best_params['decay'], verbose=False)
                            else:
                                train_molframe.train_singles(optuna_wrapper.filter, val_molframe,
                                                             lr=best_params['lr'], eps=3,
                                                             decay=best_params['decay'], verbose=False)

                        # Evaluate model
                        optuna_wrapper.filter.eval()
                        mcc, roc_auc = val_molframe.test(optuna_wrapper.filter, mode=mode)

                        # Store results
                        results['model_performance_mcc'][config_key].append(mcc)
                        results['model_performance_roc_auc'][config_key].append(roc_auc)
                        results['training_history'].append({
                            'fold': fold_idx + 1,
                            'model': model_type,
                            'mode': mode,
                            'mcc': mcc,
                            'roc_auc': roc_auc,
                            'parameters': best_params
                        })

                        # Store fold details for statistical analysis
                        results['fold_details'][config_key].append({
                            'fold': fold_idx + 1,
                            'mcc': mcc,
                            'roc_auc': roc_auc
                        })

                        print(f"    {config_key}: MCC = {mcc:.4f}, ROC AUC = {roc_auc:.4f}")

                    except Exception as e:
                        print(f"    Error evaluating {config_key} on fold {fold_idx + 1}: {e}")
                        results['model_performance_mcc'][config_key].append(-1.0)
                        results['model_performance_roc_auc'][config_key].append(-1.0)

        # Analyze and plot results
        self._analyze_results(results)

        return results

    def _analyze_results(self, results: Dict[str, Any]):
        """Analyze and visualize comparison results with statistical tests"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        # Prepare data for plotting
        mcc_data = []
        roc_data = []

        for model_mode, mcc_scores in results['model_performance_mcc'].items():
            roc_scores = results['model_performance_roc_auc'][model_mode]
            for fold_idx, (mcc, roc) in enumerate(zip(mcc_scores, roc_scores)):
                mcc_data.append({
                    'Model': model_mode,
                    'MCC': mcc,
                    'Fold': fold_idx + 1
                })
                roc_data.append({
                    'Model': model_mode,
                    'ROC_AUC': roc,
                    'Fold': fold_idx + 1
                })

        df_mcc = pd.DataFrame(mcc_data)
        df_roc = pd.DataFrame(roc_data)

        # Find best model based on mean MCC
        model_means = df_mcc.groupby('Model')['MCC'].mean()
        best_model = model_means.idxmax()
        best_model_mean_mcc = model_means.max()

        print(f"\nBest model: {best_model} (Mean MCC: {best_model_mean_mcc:.4f})")
        print(f"Optimization: {results['stage1_folds']}-fold CV, Evaluation: {results['stage2_folds']}-fold CV")

        # Perform Mann-Whitney U tests
        significance_results = self._perform_statistical_tests(results, best_model)

        # Create plots with statistical annotations
        self._create_comparison_plots(df_mcc, df_roc, best_model, significance_results, results)

        # Print statistical summary
        self._print_statistical_summary(results, best_model, significance_results)

    def _perform_statistical_tests(self, results: Dict[str, Any], best_model: str) -> Dict[str, Dict[str, float]]:
        """Perform statistical tests between best model and all others"""
        significance_results = {}

        # Get best model scores
        best_model_mcc = results['model_performance_mcc'].get(best_model, [])
        best_model_roc = results['model_performance_roc_auc'].get(best_model, [])

        # Filter out invalid scores for best model
        best_mcc_valid = [s for s in best_model_mcc if s != -1.0]
        best_roc_valid = [s for s in best_model_roc if s != -1.0]

        print(f"\nPerforming statistical tests for best model: {best_model}")
        print(f"Best model valid MCC scores: {best_mcc_valid}")
        print(f"Best model valid ROC AUC scores: {best_roc_valid}")

        # Skip if best model doesn't have enough valid data
        if len(best_mcc_valid) < 2 or len(best_roc_valid) < 2:
            print(f"Warning: Best model {best_model} doesn't have enough valid data for statistical tests")
            return significance_results

        for model in results['model_performance_mcc'].keys():
            if model == best_model:
                continue

            current_mcc = results['model_performance_mcc'].get(model, [])
            current_roc = results['model_performance_roc_auc'].get(model, [])

            # Filter out invalid scores
            valid_mcc = [s for s in current_mcc if s != -1.0]
            valid_roc = [s for s in current_roc if s != -1.0]

            print(f"\nComparing {model}:")
            print(f"  Valid MCC: {valid_mcc}")
            print(f"  Valid ROC: {valid_roc}")

            # Skip if not enough valid data for comparison
            if len(valid_mcc) < 2 or len(valid_roc) < 2:
                print(f"  Skipping {model}: insufficient valid data")
                continue

            # Mann-Whitney U test for MCC
            try:
                mcc_stat, mcc_p = stats.mannwhitneyu(best_mcc_valid, valid_mcc,
                                                     alternative='two-sided')
                print(f"  MCC test: p = {mcc_p:.6f}")
            except Exception as e:
                print(f"  Error in MCC test: {e}")
                mcc_p = 1.0

            # Mann-Whitney U test for ROC AUC
            try:
                roc_stat, roc_p = stats.mannwhitneyu(best_roc_valid, valid_roc,
                                                     alternative='two-sided')
                print(f"  ROC test: p = {roc_p:.6f}")
            except Exception as e:
                print(f"  Error in ROC test: {e}")
                roc_p = 1.0

            mcc_significant = mcc_p < 0.05
            roc_significant = roc_p < 0.05

            significance_results[model] = {
                'mcc_p_value': mcc_p,
                'roc_p_value': roc_p,
                'mcc_significant': mcc_significant,
                'roc_significant': roc_significant
            }

            if mcc_significant or roc_significant:
                print(f"  *** SIGNIFICANT DIFFERENCES FOUND for {model} ***")

        print(
            f"\nSummary: Found {sum(1 for m in significance_results.values() if m['mcc_significant'] or m['roc_significant'])} models with significant differences")
        return significance_results

    def _create_comparison_plots(self, df_mcc: pd.DataFrame, df_roc: pd.DataFrame,
                                 best_model: str, significance_results: Dict[str, Dict[str, float]],
                                 results: Dict[str, Any]):
        """Create comparison plots with statistical annotations"""
        # Create a larger figure to accommodate the stars
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))  # Увеличили высоту

        # MCC Violin plot
        sns.violinplot(data=df_mcc, x='Model', y='MCC', ax=ax1, palette='Set3')
        ax1.set_title(
            f'Model Comparison - MCC Scores\n(Best: {best_model}, {results["stage1_folds"]}-fold Opt + {results["stage2_folds"]}-fold Eval)',
            fontsize=14, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Увеличиваем верхний предел для оси Y
        y_min1, y_max1 = ax1.get_ylim()
        ax1.set_ylim(y_min1, y_max1 * 1.2)  # +20% сверху

        # Highlight the best model
        best_model_idx = np.where(df_mcc['Model'].unique() == best_model)[0][0]
        for idx, label in enumerate(ax1.get_xticklabels()):
            if idx == best_model_idx:
                label.set_fontweight('bold')
                label.set_color('green')
                label.set_fontsize(12)

        # ROC AUC Violin plot
        sns.violinplot(data=df_roc, x='Model', y='ROC_AUC', ax=ax2, palette='Set3')
        ax2.set_title(
            f'Model Comparison - ROC AUC Scores\n(Best: {best_model}, {results["stage1_folds"]}-fold Opt + {results["stage2_folds"]}-fold Eval)',
            fontsize=14, fontweight='bold')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Увеличиваем верхний предел для оси Y
        y_min2, y_max2 = ax2.get_ylim()
        ax2.set_ylim(y_min2, y_max2 * 1.2)  # +20% сверху

        # Highlight the best model
        for idx, label in enumerate(ax2.get_xticklabels()):
            if idx == best_model_idx:
                label.set_fontweight('bold')
                label.set_color('green')
                label.set_fontsize(12)

        # Add statistical significance annotations with stars
        print("Adding significance annotations for MCC:")
        self._add_significance_annotations(ax1, df_mcc, best_model, significance_results, 'MCC')
        print("\nAdding significance annotations for ROC AUC:")
        self._add_significance_annotations(ax2, df_roc, best_model, significance_results, 'ROC_AUC')

        # Add legend for significance stars
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='red', markerfacecolor='red',
                       markersize=10, label='p < 0.05', linestyle='None'),
            plt.Line2D([0], [0], marker='*', color='red', markerfacecolor='red',
                       markersize=15, label='p < 0.01', linestyle='None'),
            plt.Line2D([0], [0], marker='*', color='red', markerfacecolor='red',
                       markersize=20, label='p < 0.001', linestyle='None')
        ]

        ax1.legend(handles=legend_elements, loc='upper right', title='Significance')
        ax2.legend(handles=legend_elements, loc='upper right', title='Significance')

        plt.tight_layout()
        plt.show()

    def _add_significance_annotations(self, ax, df: pd.DataFrame, best_model: str,
                                      significance_results: Dict[str, Dict[str, float]], metric: str):
        """Add significance stars to the plot - alternative approach with fixed positions"""
        models = df['Model'].unique()

        # Получаем текущие пределы оси Y
        y_min, y_max = ax.get_ylim()

        # Устанавливаем фиксированную позицию для всех звёздочек (90% от верхнего предела)
        y_star_position = y_min + 0.85 * (y_max - y_min)

        for i, model in enumerate(models):
            if model == best_model:
                continue  # Skip best model

            # Get statistical significance for this model vs best model
            if model in significance_results:
                if metric == 'MCC':
                    p_value = significance_results[model].get('mcc_p_value', 1.0)
                    significant = significance_results[model].get('mcc_significant', False)
                else:  # ROC_AUC
                    p_value = significance_results[model].get('roc_p_value', 1.0)
                    significant = significance_results[model].get('roc_significant', False)

                if significant:
                    # Determine star level based on p-value
                    if p_value < 0.001:
                        stars = '***'
                    elif p_value < 0.01:
                        stars = '**'
                    elif p_value < 0.05:
                        stars = '*'
                    else:
                        continue

                    # Add annotation with star(s) at fixed position
                    ax.text(i, y_star_position, stars,
                            ha='center', va='center',  # Изменили на 'center' для лучшего позиционирования
                            fontweight='bold',
                            fontsize=18,  # Ещё больше увеличили размер
                            color='red',
                            bbox=dict(boxstyle="circle,pad=0.3",
                                      facecolor="gray",
                                      alpha=0.9,
                                      edgecolor='black',
                                      linewidth=2))

                    print(f"Added {stars} for {model} at fixed position ({i}, {y_star_position:.3f})")

    def _print_statistical_summary(self, results: Dict[str, Any], best_model: str,
                                   significance_results: Dict[str, Dict[str, float]]):
        """Print detailed statistical summary"""
        print("\n" + "=" * 80)
        print("DETAILED STATISTICAL ANALYSIS")
        print("=" * 80)
        print(f"Optimization: {results['stage1_folds']}-fold CV")
        print(f"Evaluation: {results['stage2_folds']}-fold CV")

        # Model performance summary
        summary_data = []
        for model in results['model_performance_mcc'].keys():
            mcc_scores = results['model_performance_mcc'][model]
            roc_scores = results['model_performance_roc_auc'][model]

            valid_mcc = [s for s in mcc_scores if s >= 0]
            valid_roc = [s for s in roc_scores if s >= 0]

            if valid_mcc:
                summary_data.append({
                    'Model': model,
                    'Mean MCC': np.mean(valid_mcc),
                    'Std MCC': np.std(valid_mcc),
                    'Mean ROC AUC': np.mean(valid_roc),
                    'Std ROC AUC': np.std(valid_roc),
                    'Folds': len(valid_mcc),
                    'Is Best': model == best_model
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Mean MCC', ascending=False)

        print("\nModel Performance Summary (sorted by Mean MCC):")
        print(summary_df.to_string(index=False, float_format='%.4f'))

        # Statistical significance results
        print("\nStatistical Significance (Mann-Whitney U test vs Best Model):")
        print("-" * 60)
        print(f"Best Model: {best_model}")

        if not significance_results:
            print("No statistical tests performed - insufficient data for comparison")
            print("-" * 60)
        else:
            print(f"{'Model':<20} {'MCC p-value':<12} {'ROC AUC p-value':<15} {'Significant'}")
            print("-" * 60)

            for model, stats in significance_results.items():
                mcc_sig = "✓" if stats.get('mcc_significant', False) else "✗"
                roc_sig = "✓" if stats.get('roc_significant', False) else "✗"
                mcc_p = stats.get('mcc_p_value', 1.0)
                roc_p = stats.get('roc_p_value', 1.0)
                print(f"{model:<20} {mcc_p:<12.4f} {roc_p:<15.4f} "
                      f"MCC:{mcc_sig} ROC:{roc_sig}")

        # Best parameters from optimization stage
        print("\nBest Hyperparameters from Optimization Stage:")
        print("-" * 60)
        best_config = best_model
        if best_config in results.get('optimized_parameters', {}):
            best_params = results['optimized_parameters'][best_config]['params']
            for param, value in best_params.items():
                print(f"  {param}: {value}")
        else:
            print("  No optimized parameters found for best model")

    def get_best_model_configuration(self, results: Dict[str, Any]) -> Tuple[str, Dict]:
        """Get the best model configuration across all trials"""
        best_mcc = -1
        best_config = None
        best_params = None

        for config_key, best_info in results['optimized_parameters'].items():
            if best_info['mcc'] > best_mcc:
                best_mcc = best_info['mcc']
                best_config = config_key
                best_params = best_info['params']

        return best_config, best_params

    def train_final_model(self, molframe: MolFrame, best_config: str,
                          best_params: Dict, epochs: int = 100) -> torch.nn.Module:
        """Train final model with the best hyperparameters on full dataset"""
        # Split model_type and mode from config key
        model_type, mode = best_config.split('_', 1)

        print(f"Training final model: {best_config}")
        print(f"Best parameters: {best_params}")

        # Create and train optimal model
        optuna_wrapper = ComprehensiveOptunaWrapper(
            mode=mode,
            model_type=model_type
        )

        # Create model with best parameters
        optuna_wrapper.create_optimal_filter(molframe, best_params)
        optuna_wrapper.train_optimal_filter(molframe, molframe, epochs=epochs)

        return optuna_wrapper.filter


# Main function for two-stage comparison
def run_two_stage_optuna_comparison(molframe: MolFrame,
                                    optimization_folds: int = 3,
                                    evaluation_folds: int = 10,
                                    n_trials: int = 30) -> Tuple[Dict[str, Any], str, torch.nn.Module]:
    """
    Complete two-stage pipeline for model comparison:
    - Stage 1: Hyperparameter optimization with 3-fold CV
    - Stage 2: Final evaluation with 10-fold CV
    """
    print("Starting Two-Stage Optuna Model Comparison Pipeline")
    print("=" * 80)
    print(f"Stage 1: {optimization_folds}-fold CV for hyperparameter optimization")
    print(f"Stage 2: {evaluation_folds}-fold CV for final evaluation")
    print(f"Number of trials per model: {n_trials}")

    # Initialize comparator
    comparator = TwoStageOptunaFilterComparator(
        optimization_folds=optimization_folds,
        evaluation_folds=evaluation_folds,
        n_trials=n_trials,
        random_state=42
    )

    # Run two-stage comparison
    results = comparator.run_two_stage_comparison(molframe)

    # Get best configuration
    best_config, best_params = comparator.get_best_model_configuration(results)

    print("\n" + "=" * 80)
    print("FINAL TWO-STAGE OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Best Configuration: {best_config}")

    # Calculate mean performance across evaluation folds
    mean_mcc = np.mean(results['model_performance_mcc'][best_config])
    mean_roc_auc = np.mean(results['model_performance_roc_auc'][best_config])
    std_mcc = np.std(results['model_performance_mcc'][best_config])
    std_roc_auc = np.std(results['model_performance_roc_auc'][best_config])

    print(f"Mean MCC ({evaluation_folds}-fold): {mean_mcc:.4f} ± {std_mcc:.4f}")
    print(f"Mean ROC AUC ({evaluation_folds}-fold): {mean_roc_auc:.4f} ± {std_roc_auc:.4f}")
    print(f"Best Parameters: {best_params}")

    # Train final model with best configuration on full dataset
    final_model = comparator.train_final_model(
        molframe, best_config, best_params, epochs=100
    )

    return results, best_config, final_model


# Additional analysis function for two-stage results
def plot_two_stage_metric_correlation(results: Dict[str, Any]):
    """Plot correlation between MCC and ROC AUC scores for two-stage results"""
    mcc_scores = []
    roc_scores = []
    models = []

    for model in results['model_performance_mcc'].keys():
        mcc_scores.extend(results['model_performance_mcc'][model])
        roc_scores.extend(results['model_performance_roc_auc'][model])
        models.extend([model] * len(results['model_performance_mcc'][model]))

    df = pd.DataFrame({
        'Model': models,
        'MCC': mcc_scores,
        'ROC_AUC': roc_scores
    })

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='MCC', y='ROC_AUC', hue='Model', s=100, alpha=0.7)
    plt.title(
        f'Correlation between MCC and ROC AUC Scores\n({results["stage1_folds"]}-fold Opt + {results["stage2_folds"]}-fold Eval)')
    plt.grid(True, alpha=0.3)

    # Calculate correlation
    correlation = np.corrcoef(df['MCC'], df['ROC_AUC'])[0, 1]
    plt.text(0.05, 0.95, f'Pearson r = {correlation:.3f}',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()

    return correlation
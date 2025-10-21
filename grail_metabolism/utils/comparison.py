import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from pathlib import Path
import json
import torch
from tqdm.auto import tqdm
import logging
import os
import optuna
from optuna.trial import Trial
import pickle

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from grail_metabolism.utils.preparation import MolFrame
from grail_metabolism.model.filter import Filter, MolPathFilter
from grail_metabolism.model.generator import Generator
from grail_metabolism.utils.transform import from_rule
from grail_metabolism.utils.optuna import EnhancedOptunaWrapper


class ComprehensiveModelComparator:
    """Комплексный компаратор моделей с полным подбором гиперпараметров через Optuna"""

    def __init__(self, data: MolFrame, rules: List[str], n_folds: int = 5):
        self.data = data
        self.rules = rules
        self.rule_dict = self._create_rule_dict(rules)
        self.n_folds = min(n_folds, len(list(data.map.keys())))

        self.results = {
            'filter': {
                'Standard_GNN': [],
                'MolPath_GNN': [],
                'Morgan_Only': []
            },
            'generator': {
                'Standard_GNN': [],
                'MolPath_GNN': []
            }
        }

        self.best_params = {
            'filter': {},
            'generator': {}
        }

        logger.info(
            f"Initialized comprehensive comparator with {len(self.data.map)} substrates and {self.n_folds} folds")

    def _create_rule_dict(self, rules: List[str]) -> Dict:
        """Создание rule_dict с обработкой ошибок"""
        rule_dict = {}
        successful = 0
        for rule in rules:
            try:
                rule_graph = from_rule(rule)
                if rule_graph is not None:
                    rule_dict[rule] = rule_graph
                    successful += 1
                else:
                    logger.warning(f"Failed to create graph for rule: {rule}")
            except Exception as e:
                logger.warning(f"Error creating rule {rule}: {e}")
                continue

        logger.info(f"Successfully created {successful}/{len(rules)} rule graphs")
        return rule_dict

    def create_stratified_folds(self) -> List[Tuple[MolFrame, MolFrame]]:
        """Создание стратифицированных фолдов с сохранением распределения метаболитов"""
        substrates = list(self.data.map.keys())

        if len(substrates) < self.n_folds:
            logger.warning(f"Too few substrates ({len(substrates)}) for {self.n_folds} folds")
            self.n_folds = max(2, len(substrates) // 2)
            logger.info(f"Reduced number of folds to {self.n_folds}")

        if len(substrates) < 2:
            raise ValueError(f"Not enough substrates ({len(substrates)}) for cross-validation")

        # Стратификация по количеству метаболитов
        substrate_sizes = [(sub, len(self.data.map[sub])) for sub in substrates]
        substrate_sizes.sort(key=lambda x: x[1])

        folds = [[] for _ in range(self.n_folds)]
        for i, (sub, size) in enumerate(substrate_sizes):
            folds[i % self.n_folds].append(sub)

        result_folds = []
        for i, fold_subs in enumerate(folds):
            test_subs = fold_subs
            train_subs = [s for s in substrates if s not in test_subs]

            logger.info(f"Fold {i + 1}: {len(train_subs)} train, {len(test_subs)} test substrates")

            # Создаем train_frame
            train_map = {}
            train_gen_map = {}
            for sub in train_subs:
                if sub in self.data.map:
                    train_map[sub] = self.data.map[sub].copy()
                if sub in self.data.gen_map:
                    train_gen_map[sub] = self.data.gen_map[sub].copy()

            train_molecules = set(train_subs)
            for sub in train_subs:
                if sub in self.data.map:
                    train_molecules.update(self.data.map[sub])
                if sub in self.data.gen_map:
                    train_molecules.update(self.data.gen_map[sub])

            train_mol_structs = {k: v for k, v in self.data.mol_structs.items()
                                 if k in train_molecules and v is not None}

            train_frame = MolFrame(train_map, gen_map=train_gen_map, mol_structs=train_mol_structs)

            # Создаем test_frame
            test_map = {}
            test_gen_map = {}
            for sub in test_subs:
                if sub in self.data.map:
                    test_map[sub] = self.data.map[sub].copy()
                if sub in self.data.gen_map:
                    test_gen_map[sub] = self.data.gen_map[sub].copy()

            test_molecules = set(test_subs)
            for sub in test_subs:
                if sub in self.data.map:
                    test_molecules.update(self.data.map[sub])
                if sub in self.data.gen_map:
                    test_molecules.update(self.data.gen_map[sub])

            test_mol_structs = {k: v for k, v in self.data.mol_structs.items()
                                if k in test_molecules and v is not None}

            test_frame = MolFrame(test_map, gen_map=test_gen_map, mol_structs=test_mol_structs)

            result_folds.append((train_frame, test_frame))

        return result_folds

    def _prepare_data_comprehensive(self, train_frame: MolFrame, test_frame: MolFrame):
        """Полная подготовка данных для обучения"""
        logger.info("  Comprehensive data preparation...")

        try:
            # Очистка данных
            train_frame.clean()
            test_frame.clean()

            if len(train_frame.map) == 0 or len(test_frame.map) == 0:
                logger.warning("No data after cleaning")
                return False

            # Генерация негативных примеров
            for frame in [train_frame, test_frame]:
                for sub in frame.map:
                    if sub not in frame.gen_map:
                        frame.gen_map[sub] = set()
                    frame.negs[sub] = set()
                    if sub in frame.gen_map:
                        for prod in frame.gen_map[sub]:
                            if prod not in frame.map[sub]:
                                frame.negs[sub].add(prod)

            # Полная подготовка данных
            logger.info("  Full data setup...")
            train_frame.full_setup(pca=True)
            test_frame.full_setup(pca=True)

            return True

        except Exception as e:
            logger.error(f"Error in comprehensive data preparation: {e}")
            return False

    def _optimize_filter_hyperparameters(self, train_frame: MolFrame, test_frame: MolFrame,
                                         model_type: str, n_trials: int = 50) -> Dict[str, Any]:
        """Оптимизация гиперпараметров для фильтров через Optuna"""
        logger.info(f"  Optimizing {model_type} filter hyperparameters...")

        def objective(trial: Trial) -> float:
            # Параметры архитектуры
            if model_type == 'Standard_GNN':
                arg_vec = [
                    trial.suggest_int('hidden1', 64, 512),
                    trial.suggest_int('hidden2', 64, 512),
                    trial.suggest_int('hidden3', 64, 512),
                    trial.suggest_int('hidden4', 64, 256),
                    trial.suggest_int('hidden5', 32, 256),
                    trial.suggest_int('hidden6', 16, 128)
                ]

                model = Filter(12, 6, arg_vec, 'pair')

            elif model_type == 'MolPath_GNN':
                arg_vec = [
                    trial.suggest_int('hidden1', 64, 512),
                    trial.suggest_int('hidden2', 32, 256),
                    trial.suggest_int('hidden3', 16, 128)
                ]

                molpath_hidden = trial.suggest_int('molpath_hidden', 64, 256)
                molpath_cutoff = trial.suggest_int('molpath_cutoff', 3, 6)
                molpath_y = trial.suggest_float('molpath_y', 0.1, 0.9)

                model = MolPathFilter(
                    12, 6, arg_vec, 'pair',
                    molpath_hidden=molpath_hidden,
                    molpath_cutoff=molpath_cutoff,
                    molpath_y=molpath_y
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Параметры обучения
            lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
            decay = trial.suggest_float('decay', 1e-10, 1e-5, log=True)
            prior = trial.suggest_float('prior', 0.1, 0.9)

            try:
                # Обучение с текущими гиперпараметрами
                model.fit(train_frame, lr=lr, eps=10, verbose=False, prior=prior, nnPU=True)

                # Оценка на validation set
                mcc_scores = model.mcc(test_frame)
                score = np.mean(mcc_scores) if mcc_scores else 0.0

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        # Создание и запуск исследования Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner()
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best {model_type} filter score: {study.best_value:.4f}")
        return study.best_params

    def _optimize_generator_hyperparameters(self, train_frame: MolFrame, test_frame: MolFrame,
                                            model_type: str, n_trials: int = 50) -> Dict[str, Any]:
        """Оптимизация гиперпараметров для генераторов через Optuna"""
        logger.info(f"  Optimizing {model_type} generator hyperparameters...")

        def objective(trial: Trial) -> float:
            # Параметры архитектуры
            arg_vec = [
                trial.suggest_int('gnn_hidden1', 100, 800),
                trial.suggest_int('gnn_hidden2', 50, 400)
            ]

            rp_arg_vec = [
                trial.suggest_int('rp_hidden1', 100, 500),
                trial.suggest_int('rp_hidden2', 100, 500),
                trial.suggest_int('rp_hidden3', 100, 500),
                trial.suggest_int('rp_hidden4', 50, 300),
                trial.suggest_int('rp_hidden5', 50, 200)
            ]

            projection_dim = trial.suggest_int('projection_dim', 128, 512)

            if model_type == 'MolPath_GNN':
                use_molpath = True
                molpath_hidden = trial.suggest_int('molpath_hidden', 64, 256)
                molpath_cutoff = trial.suggest_int('molpath_cutoff', 3, 6)
                molpath_y = trial.suggest_float('molpath_y', 0.1, 0.9)
            else:
                use_molpath = False
                molpath_hidden = None
                molpath_cutoff = None
                molpath_y = None

            # Параметры обучения
            lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
            gamma = trial.suggest_float('gamma', 1.0, 3.0)
            freeze_pretrained = trial.suggest_categorical('freeze_pretrained', [True, False])

            try:
                model = Generator(
                    self.rule_dict, 10, 6,
                    arg_vec=arg_vec,
                    rp_arg_vec=rp_arg_vec,
                    projection_dim=projection_dim,
                    use_molpath=use_molpath,
                    molpath_hidden=molpath_hidden,
                    molpath_cutoff=molpath_cutoff,
                    molpath_y=molpath_y
                )

                # Обучение с текущими гиперпараметрами
                model.fit(train_frame, lr=lr, verbose=False, gamma=gamma,
                          freeze_pretrained=freeze_pretrained)

                # Оценка на validation set
                jaccard_scores = model.jaccard(test_frame)
                score = np.mean(jaccard_scores) if jaccard_scores else 0.0

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        # Создание и запуск исследования Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner()
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best {model_type} generator score: {study.best_value:.4f}")
        return study.best_params

    def _create_model_from_params(self, params: Dict[str, Any], model_type: str, component: str):
        """Создание модели на основе оптимизированных параметров"""
        if component == 'filter':
            if model_type == 'Standard_GNN':
                arg_vec = [
                    params['hidden1'], params['hidden2'], params['hidden3'],
                    params['hidden4'], params['hidden5'], params['hidden6']
                ]
                return Filter(12, 6, arg_vec, 'pair')

            elif model_type == 'MolPath_GNN':
                arg_vec = [params['hidden1'], params['hidden2'], params['hidden3']]
                return MolPathFilter(
                    12, 6, arg_vec, 'pair',
                    molpath_hidden=params['molpath_hidden'],
                    molpath_cutoff=params['molpath_cutoff'],
                    molpath_y=params['molpath_y']
                )

        elif component == 'generator':
            arg_vec = [params['gnn_hidden1'], params['gnn_hidden2']]
            rp_arg_vec = [
                params['rp_hidden1'], params['rp_hidden2'], params['rp_hidden3'],
                params['rp_hidden4'], params['rp_hidden5']
            ]

            if model_type == 'MolPath_GNN':
                return Generator(
                    self.rule_dict, 10, 6,
                    arg_vec=arg_vec,
                    rp_arg_vec=rp_arg_vec,
                    projection_dim=params['projection_dim'],
                    use_molpath=True,
                    molpath_hidden=params['molpath_hidden'],
                    molpath_cutoff=params['molpath_cutoff'],
                    molpath_y=params['molpath_y']
                )
            else:
                return Generator(
                    self.rule_dict, 10, 6,
                    arg_vec=arg_vec,
                    rp_arg_vec=rp_arg_vec,
                    projection_dim=params['projection_dim'],
                    use_molpath=False
                )

        raise ValueError(f"Unknown model type: {model_type}")

    def run_comprehensive_filter_comparison(self, folds: List[Tuple[MolFrame, MolFrame]],
                                            n_trials: int = 50):
        """Полное сравнение фильтров с подбором гиперпараметров"""
        logger.info("Starting comprehensive filter comparison with hyperparameter optimization...")

        for fold_idx, (train_frame, test_frame) in enumerate(folds):
            logger.info(f"Processing filter fold {fold_idx + 1}/{len(folds)}")

            try:
                # Подготовка данных
                if not self._prepare_data_comprehensive(train_frame, test_frame):
                    logger.warning(f"Skipping filter fold {fold_idx + 1} due to data preparation issues")
                    continue

                # Разделение train_frame на train и validation для оптимизации гиперпараметров
                train_substrates = list(train_frame.map.keys())
                np.random.shuffle(train_substrates)
                split_idx = int(0.8 * len(train_substrates))

                opt_train_subs = train_substrates[:split_idx]
                opt_val_subs = train_substrates[split_idx:]

                # Создание оптимизационных фолдов
                opt_train_map = {k: train_frame.map[k] for k in opt_train_subs}
                opt_train_gen_map = {k: train_frame.gen_map[k] for k in opt_train_subs}
                opt_train_mols = set(opt_train_subs)
                for sub in opt_train_subs:
                    opt_train_mols.update(train_frame.map.get(sub, set()))
                    opt_train_mols.update(train_frame.gen_map.get(sub, set()))
                opt_train_mol_structs = {k: v for k, v in train_frame.mol_structs.items()
                                         if k in opt_train_mols}

                opt_val_map = {k: train_frame.map[k] for k in opt_val_subs}
                opt_val_gen_map = {k: train_frame.gen_map[k] for k in opt_val_subs}
                opt_val_mols = set(opt_val_subs)
                for sub in opt_val_subs:
                    opt_val_mols.update(train_frame.map.get(sub, set()))
                    opt_val_mols.update(train_frame.gen_map.get(sub, set()))
                opt_val_mol_structs = {k: v for k, v in train_frame.mol_structs.items()
                                       if k in opt_val_mols}

                opt_train_frame = MolFrame(opt_train_map, gen_map=opt_train_gen_map,
                                           mol_structs=opt_train_mol_structs)
                opt_val_frame = MolFrame(opt_val_map, gen_map=opt_val_gen_map,
                                         mol_structs=opt_val_mol_structs)

                # Подготовка оптимизационных данных
                opt_train_frame.full_setup(pca=True)
                opt_val_frame.full_setup(pca=True)

                # Оптимизация гиперпараметров для каждого типа фильтра
                for model_type in ['Standard_GNN', 'MolPath_GNN']:
                    if fold_idx == 0 or model_type not in self.best_params['filter']:
                        # Оптимизируем гиперпараметры только на первом фолде или если их еще нет
                        best_params = self._optimize_filter_hyperparameters(
                            opt_train_frame, opt_val_frame, model_type, n_trials
                        )
                        self.best_params['filter'][model_type] = best_params

                    # Создание и обучение модели с лучшими параметрами
                    best_params = self.best_params['filter'][model_type]
                    model = self._create_model_from_params(best_params, model_type, 'filter')

                    # Обучение на полном train_frame
                    model.fit(train_frame, lr=best_params['lr'], eps=50, verbose=False,
                              prior=best_params.get('prior', 0.75), nnPU=True)

                    # Оценка на test_frame
                    mcc_scores = model.mcc(test_frame)
                    score = np.mean(mcc_scores) if mcc_scores else 0.0
                    self.results['filter'][model_type].append(score)
                    logger.info(f"    {model_type} MCC: {score:.4f}")

                # Morgan Only Baseline
                logger.info("  Testing Morgan Only Baseline...")
                try:
                    morgan_scores = self._morgan_baseline(test_frame)
                    morgan_score = np.mean(morgan_scores) if morgan_scores else 0.0
                    self.results['filter']['Morgan_Only'].append(morgan_score)
                    logger.info(f"    Morgan Only MCC: {morgan_score:.4f}")
                except Exception as e:
                    logger.error(f"    Morgan Only error: {e}")
                    self.results['filter']['Morgan_Only'].append(0.0)

            except Exception as e:
                logger.error(f"Error in filter fold {fold_idx + 1}: {e}")
                for model_type in ['Standard_GNN', 'MolPath_GNN', 'Morgan_Only']:
                    self.results['filter'][model_type].append(0.0)

    def run_comprehensive_generator_comparison(self, folds: List[Tuple[MolFrame, MolFrame]],
                                               n_trials: int = 50):
        """Полное сравнение генераторов с подбором гиперпараметров"""
        logger.info("Starting comprehensive generator comparison with hyperparameter optimization...")

        for fold_idx, (train_frame, test_frame) in enumerate(folds):
            logger.info(f"Processing generator fold {fold_idx + 1}/{len(folds)}")

            try:
                # Подготовка данных
                if not self._prepare_data_comprehensive(train_frame, test_frame):
                    logger.warning(f"Skipping generator fold {fold_idx + 1} due to data preparation issues")
                    continue

                # Разделение train_frame на train и validation для оптимизации гиперпараметров
                train_substrates = list(train_frame.map.keys())
                np.random.shuffle(train_substrates)
                split_idx = int(0.8 * len(train_substrates))

                opt_train_subs = train_substrates[:split_idx]
                opt_val_subs = train_substrates[split_idx:]

                # Создание оптимизационных фолдов
                opt_train_map = {k: train_frame.map[k] for k in opt_train_subs}
                opt_train_gen_map = {k: train_frame.gen_map[k] for k in opt_train_subs}
                opt_train_mols = set(opt_train_subs)
                for sub in opt_train_subs:
                    opt_train_mols.update(train_frame.map.get(sub, set()))
                    opt_train_mols.update(train_frame.gen_map.get(sub, set()))
                opt_train_mol_structs = {k: v for k, v in train_frame.mol_structs.items()
                                         if k in opt_train_mols}

                opt_val_map = {k: train_frame.map[k] for k in opt_val_subs}
                opt_val_gen_map = {k: train_frame.gen_map[k] for k in opt_val_subs}
                opt_val_mols = set(opt_val_subs)
                for sub in opt_val_subs:
                    opt_val_mols.update(train_frame.map.get(sub, set()))
                    opt_val_mols.update(train_frame.gen_map.get(sub, set()))
                opt_val_mol_structs = {k: v for k, v in train_frame.mol_structs.items()
                                       if k in opt_val_mols}

                opt_train_frame = MolFrame(opt_train_map, gen_map=opt_train_gen_map,
                                           mol_structs=opt_train_mol_structs)
                opt_val_frame = MolFrame(opt_val_map, gen_map=opt_val_gen_map,
                                         mol_structs=opt_val_mol_structs)

                # Подготовка оптимизационных данных
                opt_train_frame.full_setup(pca=True)
                opt_val_frame.full_setup(pca=True)

                # Оптимизация гиперпараметров для каждого типа генератора
                for model_type in ['Standard_GNN', 'MolPath_GNN']:
                    if fold_idx == 0 or model_type not in self.best_params['generator']:
                        # Оптимизируем гиперпараметры только на первом фолде или если их еще нет
                        best_params = self._optimize_generator_hyperparameters(
                            opt_train_frame, opt_val_frame, model_type, n_trials
                        )
                        self.best_params['generator'][model_type] = best_params

                    # Создание и обучение модели с лучшими параметрами
                    best_params = self.best_params['generator'][model_type]
                    model = self._create_model_from_params(best_params, model_type, 'generator')

                    # Обучение на полном train_frame
                    model.fit(train_frame, lr=best_params['lr'], verbose=False,
                              gamma=best_params.get('gamma', 2.0),
                              freeze_pretrained=best_params.get('freeze_pretrained', False))

                    # Оценка на test_frame
                    jaccard_scores = model.jaccard(test_frame)
                    score = np.mean(jaccard_scores) if jaccard_scores else 0.0
                    self.results['generator'][model_type].append(score)
                    logger.info(f"    {model_type} Jaccard: {score:.4f}")

            except Exception as e:
                logger.error(f"Error in generator fold {fold_idx + 1}: {e}")
                for model_type in ['Standard_GNN', 'MolPath_GNN']:
                    self.results['generator'][model_type].append(0.0)

    def _morgan_baseline(self, test_frame) -> List[float]:
        """Morgan fingerprint similarity baseline"""
        mcc_scores = []

        for sub in test_frame.map:
            if sub not in test_frame.morgan:
                continue

            sub_fp = test_frame.morgan[sub]
            predictions = []
            reals = []

            # Positive examples
            for prod in test_frame.map[sub]:
                if prod in test_frame.morgan:
                    try:
                        prod_fp = test_frame.morgan[prod]
                        similarity = torch.cosine_similarity(sub_fp, prod_fp, dim=0).item()
                        predictions.append(1 if similarity > 0.7 else 0)
                        reals.append(1)
                    except Exception:
                        continue

            # Negative examples
            if sub in test_frame.negs:
                for prod in test_frame.negs[sub]:
                    if prod in test_frame.morgan:
                        try:
                            prod_fp = test_frame.morgan[prod]
                            similarity = torch.cosine_similarity(sub_fp, prod_fp, dim=0).item()
                            predictions.append(1 if similarity > 0.7 else 0)
                            reals.append(0)
                        except Exception:
                            continue

            if len(reals) > 1 and len(predictions) == len(reals):
                try:
                    from sklearn.metrics import matthews_corrcoef
                    mcc_val = matthews_corrcoef(reals, predictions)
                    mcc_scores.append(mcc_val)
                except Exception:
                    mcc_scores.append(0.0)

        return mcc_scores if mcc_scores else [0.0]

    def create_detailed_plots(self, output_dir: str = "comprehensive_comparison_results"):
        """Создание детализированных графиков сравнения"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Сохранение лучших параметров
        with open(output_path / 'best_hyperparameters.json', 'w') as f:
            json.dump(self.best_params, f, indent=2, default=str)

        # Создание сравнительных графиков
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # Filter results - violin plot
        filter_data = []
        for model_type, scores in self.results['filter'].items():
            for score in scores:
                filter_data.append({'Model': model_type, 'Score': score, 'Type': 'Filter'})

        filter_df = pd.DataFrame(filter_data)

        if not filter_df.empty:
            sns.violinplot(data=filter_df, x='Model', y='Score', ax=ax1, inner='box', palette='viridis')
            ax1.set_title('Filter Models Comparison (MCC Score)', fontsize=14, fontweight='bold')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax1.set_ylabel('MCC Score', fontsize=12)

            # Добавление средних значений
            means = filter_df.groupby('Model')['Score'].mean()
            for i, (model, mean_val) in enumerate(means.items()):
                ax1.text(i, mean_val + 0.02, f'{mean_val:.3f}',
                         ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Generator results - violin plot
        generator_data = []
        for model_type, scores in self.results['generator'].items():
            for score in scores:
                generator_data.append({'Model': model_type, 'Score': score, 'Type': 'Generator'})

        generator_df = pd.DataFrame(generator_data)

        if not generator_df.empty:
            sns.violinplot(data=generator_df, x='Model', y='Score', ax=ax2, inner='box', palette='viridis')
            ax2.set_title('Generator Models Comparison (Jaccard Score)', fontsize=14, fontweight='bold')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax2.set_ylabel('Jaccard Score', fontsize=12)

            # Добавление средних значений
            means = generator_df.groupby('Model')['Score'].mean()
            for i, (model, mean_val) in enumerate(means.items()):
                ax2.text(i, mean_val + 0.02, f'{mean_val:.3f}',
                         ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Filter results - line plot across folds
        if not filter_df.empty:
            fold_means = filter_df.groupby(['Model', filter_df.groupby('Model').cumcount()])['Score'].mean().unstack(0)
            fold_means.plot(ax=ax3, marker='o', linewidth=2)
            ax3.set_title('Filter Performance Across Folds', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Fold Number')
            ax3.set_ylabel('MCC Score')
            ax3.legend(title='Model')
            ax3.grid(True, alpha=0.3)

        # Generator results - line plot across folds
        if not generator_df.empty:
            fold_means = generator_df.groupby(['Model', generator_df.groupby('Model').cumcount()])[
                'Score'].mean().unstack(0)
            fold_means.plot(ax=ax4, marker='o', linewidth=2)
            ax4.set_title('Generator Performance Across Folds', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Fold Number')
            ax4.set_ylabel('Jaccard Score')
            ax4.legend(title='Model')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Сохранение детализированных результатов
        summary = {
            'Filter_Models': {
                model: {
                    'mean': np.mean(scores) if scores else 0.0,
                    'std': np.std(scores) if scores else 0.0,
                    'min': np.min(scores) if scores else 0.0,
                    'max': np.max(scores) if scores else 0.0,
                    'scores': scores,
                    'n_folds': len(scores),
                    'best_hyperparameters': self.best_params['filter'].get(model, {})
                } for model, scores in self.results['filter'].items()
            },
            'Generator_Models': {
                model: {
                    'mean': np.mean(scores) if scores else 0.0,
                    'std': np.std(scores) if scores else 0.0,
                    'min': np.min(scores) if scores else 0.0,
                    'max': np.max(scores) if scores else 0.0,
                    'scores': scores,
                    'n_folds': len(scores),
                    'best_hyperparameters': self.best_params['generator'].get(model, {})
                } for model, scores in self.results['generator'].items()
            }
        }

        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Вывод сводки
        print("\n" + "=" * 60)
        print("COMPREHENSIVE COMPARISON RESULTS SUMMARY")
        print("=" * 60)
        print("\nFILTER MODELS:")
        for model, stats in summary['Filter_Models'].items():
            print(f"  {model:20} MCC: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['n_folds']})")

        print("\nGENERATOR MODELS:")
        for model, stats in summary['Generator_Models'].items():
            print(f"  {model:20} Jaccard: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['n_folds']})")

        print("\nBest hyperparameters saved to:", output_path / 'best_hyperparameters.json')
        print("Detailed results saved to:", output_path / 'detailed_results.json')
        print("=" * 60)

    def run_complete_comprehensive_comparison(self, n_trials: int = 50):
        """Запуск полного комплексного сравнения с оптимизацией гиперпараметров"""
        logger.info("Starting complete comprehensive model comparison...")

        # Создание стратифицированных фолдов
        folds = self.create_stratified_folds()
        logger.info(f"Created {len(folds)} stratified folds")

        # Запуск сравнения фильтров
        self.run_comprehensive_filter_comparison(folds, n_trials=n_trials)

        # Запуск сравнения генераторов
        self.run_comprehensive_generator_comparison(folds, n_trials=n_trials)

        # Создание детализированных отчетов
        self.create_detailed_plots()

        return self.results, self.best_params


# Функции для удобного запуска
def run_comprehensive_comparison(sdf_path: str, triples_path: str, rules_path: str,
                                 n_folds: int = 5, n_trials: int = 50):
    """
    Запуск полного комплексного сравнения моделей

    Args:
        sdf_path: путь к SDF файлу
        triples_path: путь к файлу с triples
        rules_path: путь к файлу с правилами
        n_folds: количество фолдов для кросс-валидации
        n_trials: количество trials для оптимизации гиперпараметров
    """
    try:
        # Load data
        logger.info("Loading data...")
        triples = MolFrame.read_triples(triples_path)
        data = MolFrame.from_file(sdf_path, triples)

        # Load rules
        logger.info("Loading rules...")
        with open(rules_path, 'r') as f:
            rules = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(rules)} rules")

        # Run comprehensive comparison
        comparator = ComprehensiveModelComparator(data, rules, n_folds=n_folds)
        results, best_params = comparator.run_complete_comprehensive_comparison(n_trials=n_trials)

        return results, best_params

    except Exception as e:
        logger.error(f"Comprehensive comparison failed: {e}")
        raise


if __name__ == "__main__":
    results, best_params = run_comprehensive_comparison(
        sdf_path='../grail_metabolism/data/train.sdf',
        triples_path='../grail_metabolism/data/train_triples.txt',
        rules_path='../grail_metabolism/data/smirks.txt',
        n_folds=5,
        n_trials=30  # Можно уменьшить для более быстрого выполнения
    )
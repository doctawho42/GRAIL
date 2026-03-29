from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Sequence

from .preparation import MolFrame
from ..model.filter import Filter, GATv2Filter, GCNFilter, GINFilter, MolPathFilter, MorganOnlyFilter
from ..model.generator import Generator


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is not installed; install the 'tuning' extra to use OptunaWrapper") from exc
    return optuna


FILTER_REGISTRY = {
    "Filter": Filter,
    "GATv2Filter": GATv2Filter,
    "GCNFilter": GCNFilter,
    "GINFilter": GINFilter,
    "MolPathFilter": MolPathFilter,
    "MorganOnlyFilter": MorganOnlyFilter,
}


@dataclass
class OptunaWrapper:
    study: Optional[object] = None
    mode: Literal["pair", "single"] = "pair"
    model_type: str = "GATv2Filter"
    lr: float = 0.0
    decay: float = 0.0
    arg_vec: list[int] = field(default_factory=list)
    filter: Optional[Filter] = None
    generator: Optional[Generator] = None

    @staticmethod
    def from_pickle(file_path: str) -> "OptunaWrapper":
        with open(file_path, "rb") as handle:
            study = pickle.load(handle)
        return OptunaWrapper(study=study)

    def _build_filter(self, trial, in_channels: int, edge_dim: int):
        model_cls = FILTER_REGISTRY[self.model_type]
        arg_vec = [trial.suggest_int(f"x{i}", 32, 256) for i in range(1, 7)]
        kwargs = {}
        if model_cls is MolPathFilter:
            kwargs = {
                "molpath_cutoff": trial.suggest_int("molpath_cutoff", 2, 8),
                "molpath_y": trial.suggest_float("molpath_y", 0.1, 0.9),
                "molpath_hidden": trial.suggest_int("molpath_hidden", 32, 256),
            }
        return model_cls(in_channels, edge_dim, arg_vec, self.mode, **kwargs), arg_vec

    def make_study(
        self,
        train_set: MolFrame,
        test_set: MolFrame,
        direction: Literal["filter", "generator"],
        rule_dict: Optional[Dict] = None,
        n_trials: int = 20,
    ) -> None:
        optuna = _require_optuna()
        study = optuna.create_study(
            study_name=f"{direction}_{self.mode}",
            direction="maximize" if direction == "filter" else "minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        def objective(trial) -> float:
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            decay = trial.suggest_float("decay", 1e-8, 1e-4, log=True)

            if direction == "filter":
                in_channels = 18 if self.mode == "pair" else 16
                model, _ = self._build_filter(trial, in_channels=in_channels, edge_dim=18)
                model.fit(train_set, lr=lr, eps=3, verbose=False, weight_decay=decay)
                mcc, _ = test_set.test(model, mode=self.mode)
                return mcc

            arg_vec = [trial.suggest_int(f"x{i}", 64, 256) for i in range(1, 3)]
            rp_arg_vec = [trial.suggest_int(f"rp_x{i}", 64, 256) for i in range(1, 4)]
            model = Generator(rule_dict or {}, 16, 18, arg_vec=arg_vec, rp_arg_vec=rp_arg_vec)
            model.fit(train_set, lr=lr, eps=3, verbose=False, weight_decay=decay)
            jaccard = model.jaccard(test_set)
            return -(sum(jaccard) / len(jaccard) if jaccard else 0.0)

        study.optimize(objective, n_trials=n_trials)
        self.study = study

    def create_optimal_filter(self) -> None:
        if self.study is None:
            raise ValueError("study is not initialized")
        params = self.study.best_params
        self.lr = params["lr"]
        self.decay = params["decay"]
        self.arg_vec = [params[key] for key in sorted(params) if key.startswith("x")]
        in_channels = 18 if self.mode == "pair" else 16
        model_cls = FILTER_REGISTRY[self.model_type]
        kwargs = {}
        if model_cls is MolPathFilter:
            kwargs = {
                "molpath_cutoff": params["molpath_cutoff"],
                "molpath_y": params["molpath_y"],
                "molpath_hidden": params["molpath_hidden"],
            }
        self.filter = model_cls(in_channels, 18, self.arg_vec, self.mode, **kwargs)

    def create_optimal_generator(self, rule_dict: Dict) -> None:
        if self.study is None:
            raise ValueError("study is not initialized")
        params = self.study.best_params
        self.lr = params["lr"]
        self.decay = params["decay"]
        arg_vec = [params[key] for key in sorted(params) if key.startswith("x") and not key.startswith("rp_")]
        rp_arg_vec = [params[key] for key in sorted(params) if key.startswith("rp_x")]
        self.generator = Generator(rule_dict, 16, 18, arg_vec=arg_vec, rp_arg_vec=rp_arg_vec)

    def train_on(self, train_set: MolFrame, test_set: MolFrame) -> None:
        del test_set
        if self.filter is None:
            self.create_optimal_filter()
        if self.filter is None:
            raise ValueError("filter could not be created")
        self.filter.fit(train_set, lr=self.lr or 1e-4, eps=20, verbose=True, weight_decay=self.decay or 1e-6)


EnhancedOptunaWrapper = OptunaWrapper

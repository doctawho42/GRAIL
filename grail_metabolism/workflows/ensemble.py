from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict

import pandas as pd
from rdkit import Chem

from ..artifacts import ArtifactStore
from ..config import ExperimentConfig
from ..model.wrapper import ModelWrapper
from ..utils.preparation import MolFrame
from .data import DatasetBundle, load_dataset_bundle
from .evaluation import collect_ensemble_predictions, evaluate_ensemble, evaluate_filter, evaluate_generator
from .factory import build_filter, build_generator
from .pretraining import PretrainingWorkflow
from .training import FilterTrainingWorkflow, GeneratorTrainingWorkflow


def generate_filter_training_data(
    generator,
    train_data: MolFrame,
    rules,
    top_k: int = 200,
    verbose: bool = True,
) -> MolFrame:
    del rules
    rows = []
    n_substrates = 0
    n_candidates = 0
    n_positives = 0
    n_missed_positives = 0
    generator_threshold = getattr(generator, "calibrated_threshold", None)

    substrates = list(train_data.map.keys())
    for index, sub_smi in enumerate(substrates):
        if verbose and (index == 0 or index % 100 == 0 or index == len(substrates) - 1):
            print(
                f"  Generating filter candidates: {index}/{len(substrates)} "
                f"({n_candidates} candidates, {n_positives} positives)",
                flush=True,
            )

        true_mets = set()
        for metabolite in train_data.map.get(sub_smi, set()):
            try:
                mol = Chem.MolFromSmiles(metabolite)
                canon = Chem.MolToSmiles(mol) if mol is not None else None
            except Exception:
                canon = None
            if canon:
                true_mets.add(canon)
        if not true_mets:
            continue

        try:
            scored = generator.generate_scored(sub_smi, top_k=top_k, threshold=generator_threshold)
        except Exception:
            continue

        gen_products = set()
        for prod_smi, _ in scored:
            try:
                mol = Chem.MolFromSmiles(prod_smi)
                canon = Chem.MolToSmiles(mol) if mol is not None else None
            except Exception:
                canon = None
            if not canon or canon in gen_products:
                continue
            gen_products.add(canon)
            label = 1 if canon in true_mets else 0
            rows.append({"sub": sub_smi, "prod": canon, "real": label})
            n_candidates += 1
            if label == 1:
                n_positives += 1

        for met_smi in sorted(true_mets):
            if met_smi in gen_products:
                continue
            rows.append({"sub": sub_smi, "prod": met_smi, "real": 1})
            n_missed_positives += 1
            n_positives += 1

        n_substrates += 1

    if not rows:
        raise ValueError("Generator-produced filter training data is empty")

    if verbose:
        print(
            f"  Filter training data: {n_substrates} substrates, "
            f"{n_candidates} gen candidates, {n_positives} positives "
            f"({n_missed_positives} missed by generator)",
            flush=True,
        )

    frame = MolFrame(pd.DataFrame(rows), standartize=False)
    frame.generated_candidate_stats = {
        "substrates": n_substrates,
        "rows": len(rows),
        "generator_candidates": n_candidates,
        "positives": n_positives,
        "negatives": max(0, len(rows) - n_positives),
        "missed_positives_added": n_missed_positives,
        "generator_top_k": top_k,
        "generator_threshold": generator_threshold,
    }
    return frame


@dataclass
class EnsembleWorkflow:
    config: ExperimentConfig
    artifacts: ArtifactStore

    def run_bundle(self, bundle: DatasetBundle) -> Dict[str, Dict[str, float]]:
        runtime: Dict[str, float] = {}
        if (
            not bundle.train.single
            or not bundle.train.reaction_labels
            or not bundle.val.single
            or not bundle.val.reaction_labels
        ):
            prepare_started = time.perf_counter()
            bundle.prepare(
                rules=bundle.rules,
                include_val=True,
                include_test=False,
                include_pair_graphs=False,
                include_morgan=False,
                single_substrates_only=self.config.filter.mode == "pair",
            )
            runtime["data_prepare_seconds"] = time.perf_counter() - prepare_started
        generator = build_generator(self.config.generator, bundle.rules)
        filter_model = build_filter(self.config.filter)

        pretrain_started = time.perf_counter()
        generator = PretrainingWorkflow(self.config, self.artifacts).run(generator)
        runtime["pretrain_seconds"] = time.perf_counter() - pretrain_started

        generator_train_started = time.perf_counter()
        generator = GeneratorTrainingWorkflow(self.config, self.artifacts).run(generator, bundle)
        runtime["generator_train_seconds"] = time.perf_counter() - generator_train_started

        filter_train_data = bundle.train
        candidate_stats: Dict[str, Any] = {"enabled": False}
        if self.config.filter.train_on_candidates:
            candidate_started = time.perf_counter()
            print("Generating filter training candidates from trained generator...", flush=True)
            filter_train_data = generate_filter_training_data(
                generator,
                bundle.train,
                bundle.rules,
                top_k=self.config.filter.candidate_generation_top_k,
                verbose=True,
            )
            filter_train_data.full_setup(
                pca=self.config.dataset.pca,
                rules=bundle.rules,
                include_reaction_labels=False,
                include_pair_graphs=False,
                include_morgan=False,
                include_single_graphs=self.config.filter.mode != "pair",
            )
            runtime["filter_candidate_generation_seconds"] = time.perf_counter() - candidate_started
            candidate_stats = {"enabled": True, **dict(getattr(filter_train_data, "generated_candidate_stats", {}))}
            self.artifacts.save_json("reports/filter_candidate_training_data.json", candidate_stats)

        filter_train_started = time.perf_counter()
        filter_model = FilterTrainingWorkflow(self.config, self.artifacts).run(filter_model, bundle, train_data=filter_train_data)
        runtime["filter_train_seconds"] = time.perf_counter() - filter_train_started

        model = ModelWrapper(filter_model, generator)
        evaluation_started = time.perf_counter()
        generator_metrics = evaluate_generator(generator, bundle, self.config.evaluation)
        filter_metrics = evaluate_filter(filter_model, bundle)
        ensemble_metrics = evaluate_ensemble(model, bundle, self.config.evaluation)
        predictions = collect_ensemble_predictions(model, bundle, self.config.evaluation)
        runtime["evaluation_seconds"] = time.perf_counter() - evaluation_started

        metrics = {
            "generator": generator_metrics,
            "filter": filter_metrics,
            "ensemble": ensemble_metrics,
            "runtime": runtime,
        }
        if candidate_stats.get("enabled"):
            metrics["filter_candidates"] = candidate_stats
        self.artifacts.save_json("reports/metrics.json", metrics)
        self.artifacts.save_json("reports/runtime.json", runtime)
        if self.config.evaluation.export_predictions:
            rows = [
                {
                    "substrate": row["substrate"],
                    "predicted": "|".join(row["predicted"]),
                    "real": "|".join(row["real"]),
                }
                for row in predictions
            ]
            self.artifacts.save_csv("predictions/test_predictions.csv", rows)
        return metrics

    def run(self) -> Dict[str, Dict[str, float]]:
        bundle = load_dataset_bundle(self.config.dataset)
        return self.run_bundle(bundle)

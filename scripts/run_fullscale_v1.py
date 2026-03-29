from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

sys.path.insert(0, ".")


class StageTimeoutError(RuntimeError):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise StageTimeoutError()


def run_with_timeout(name: str, fn: Callable[[], Any], timeout_seconds: int | None = None) -> Tuple[Any, float]:
    started = time.perf_counter()
    if timeout_seconds is None or not hasattr(signal, "setitimer"):
        return fn(), time.perf_counter() - started

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
        return fn(), time.perf_counter() - started
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


def read_timeout_seconds(env_name: str, default: int | None) -> int | None:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    text = raw_value.strip().lower()
    if text in {"", "none", "off", "disable", "disabled"}:
        return None
    value = int(text)
    return value if value > 0 else None


def configure_experiment():
    from grail_metabolism.experiments.presets import get_experiment_preset

    config = get_experiment_preset("paper_full_ensemble")
    config.name = "fullscale_v1"
    config.description = "First full-scale experiment on clean splits with the extended 7581-rule bank"
    config.output_dir = "artifacts"
    config.tags = sorted(set(list(config.tags) + ["fullscale", "v1", "extended-bank"]))

    config.dataset.use_clean_splits = True
    config.dataset.standardize = False
    config.dataset.cache_preprocessed = True
    config.dataset.cache_dir = "artifacts/preprocessed"
    config.dataset.max_train_substrates = None
    config.dataset.max_val_substrates = None
    config.dataset.max_test_substrates = None
    config.dataset.rules_path = "grail_metabolism/resources/extended_smirks.txt"
    config.dataset.max_uspto_rows = 10000

    config.pretrain.enabled = True
    config.pretrain.epochs = 15

    config.generator.scoring = "retrieval"
    config.generator.top_k = 200
    config.generator.candidate_aggregation = "noisy_or"
    config.generator.use_applicability_mask = True
    config.generator.prior_strength = 0.4
    config.generator.rank_weight = 0.25
    config.generator.use_fingerprint = True
    config.generator_optim.epochs = 30
    config.generator_optim.batch_size = 64
    config.generator_optim.lr = 1e-4
    config.generator_optim.patience = 7

    config.filter.mode = "pair"
    config.filter.model_type = "GATv2Filter"
    config.filter.use_graph = True
    config.filter.use_fingerprint = True
    config.filter.train_on_candidates = False
    config.filter.candidate_generation_top_k = 200
    config.filter_optim.epochs = 20
    config.filter_optim.lr = 1e-4
    config.filter_optim.patience = 7

    config.evaluation.candidate_top_k = 200
    config.evaluation.generator_top_k = [1, 3, 5, 10, 20, 50]

    return config


def main() -> int:
    from grail_metabolism.artifacts import ArtifactStore
    from grail_metabolism.model.wrapper import ModelWrapper
    from grail_metabolism.workflows.data import load_dataset_bundle
    from grail_metabolism.workflows.ensemble import generate_filter_training_data
    from grail_metabolism.workflows.evaluation import (
        collect_ensemble_predictions,
        evaluate_ensemble,
        evaluate_filter,
        evaluate_generator,
    )
    from grail_metabolism.workflows.factory import build_filter, build_generator
    from grail_metabolism.workflows.pretraining import PretrainingWorkflow
    from grail_metabolism.workflows.training import FilterTrainingWorkflow, GeneratorTrainingWorkflow

    total_started = time.perf_counter()
    config = configure_experiment()
    artifacts = ArtifactStore.create("artifacts", "fullscale_v1")
    generator_train_timeout_seconds = read_timeout_seconds("GRAIL_GENERATOR_TRAIN_TIMEOUT_SECONDS", 43200)
    filter_train_timeout_seconds = read_timeout_seconds("GRAIL_FILTER_TRAIN_TIMEOUT_SECONDS", 21600)
    config.dump_yaml(artifacts.path("config.yaml"))
    artifacts.save_json("config.json", config.to_dict())

    print("=" * 60)
    print("FULL-SCALE EXPERIMENT V1")
    print(f"Rules: {config.dataset.rules_path}")
    print(f"Clean splits: {config.dataset.use_clean_splits}")
    print(f"Train-on-candidates: {config.filter.train_on_candidates}")
    print(f"Generator train timeout: {generator_train_timeout_seconds or 'disabled'}")
    print(f"Filter train timeout: {filter_train_timeout_seconds or 'disabled'}")
    print("=" * 60)

    stage_runtimes: Dict[str, float] = {}
    notes: list[str] = []
    results: Dict[str, Any] = {
        "config": config.to_dict(),
        "stage_runtimes": stage_runtimes,
        "notes": notes,
    }

    try:
        bundle, stage_runtimes["load_data_seconds"] = run_with_timeout(
            "load_data",
            lambda: load_dataset_bundle(config.dataset),
            timeout_seconds=1800,
        )

        try:
            _, stage_runtimes["data_prepare_seconds"] = run_with_timeout(
                "data_prepare",
                lambda: bundle.prepare(
                    rules=bundle.rules,
                    include_val=True,
                    include_test=False,
                    include_pair_graphs=False,
                    include_morgan=False,
                    single_substrates_only=config.filter.mode == "pair",
                ),
                timeout_seconds=7200,
            )
        except StageTimeoutError:
            notes.append("Data preparation exceeded 7200s; retrying with max_train_substrates=4000.")
            print(notes[-1], flush=True)
            config.dataset.max_train_substrates = 4000
            results["config"] = config.to_dict()
            config.dump_yaml(artifacts.path("config.yaml"))
            artifacts.save_json("config.json", config.to_dict())
            bundle, stage_runtimes["load_data_seconds_resampled"] = run_with_timeout(
                "load_data_resampled",
                lambda: load_dataset_bundle(config.dataset),
                timeout_seconds=1800,
            )
            _, stage_runtimes["data_prepare_seconds"] = run_with_timeout(
                "data_prepare_resampled",
                lambda: bundle.prepare(
                    rules=bundle.rules,
                    include_val=True,
                    include_test=False,
                    include_pair_graphs=False,
                    include_morgan=False,
                    single_substrates_only=config.filter.mode == "pair",
                ),
                timeout_seconds=7200,
            )

        (generator, filter_model), stage_runtimes["build_models_seconds"] = run_with_timeout(
            "build_models",
            lambda: (build_generator(config.generator, bundle.rules), build_filter(config.filter)),
            timeout_seconds=600,
        )

        generator, stage_runtimes["pretrain_seconds"] = run_with_timeout(
            "pretrain",
            lambda: PretrainingWorkflow(config, artifacts).run(generator),
            timeout_seconds=10800,
        )

        generator_train_started = time.perf_counter()
        generator = GeneratorTrainingWorkflow(config, artifacts).run(
            generator,
            bundle,
            timeout_seconds=generator_train_timeout_seconds,
        )
        stage_runtimes["generator_train_seconds"] = time.perf_counter() - generator_train_started
        if getattr(generator, "timed_out_", False):
            note = (
                f"Generator training reached its cooperative budget after "
                f"{getattr(generator, 'epochs_trained_', 0)} epochs; continuing with best completed epoch."
            )
            notes.append(note)
            print(note, flush=True)

        filter_train_data = bundle.train
        if config.filter.train_on_candidates:
            def _build_filter_candidates():
                candidate_frame = generate_filter_training_data(
                    generator,
                    bundle.train,
                    bundle.rules,
                    top_k=config.filter.candidate_generation_top_k,
                    verbose=True,
                )
                candidate_frame.full_setup(
                    pca=config.dataset.pca,
                    rules=bundle.rules,
                    include_reaction_labels=False,
                    include_pair_graphs=False,
                    include_morgan=False,
                    include_single_graphs=config.filter.mode != "pair",
                )
                return candidate_frame

            filter_train_data, stage_runtimes["filter_candidate_generation_seconds"] = run_with_timeout(
                "filter_candidate_generation",
                _build_filter_candidates,
                timeout_seconds=10800,
            )
            artifacts.save_json(
                "reports/filter_candidate_training_data.json",
                dict(getattr(filter_train_data, "generated_candidate_stats", {})),
            )

        filter_train_started = time.perf_counter()
        filter_model = FilterTrainingWorkflow(config, artifacts).run(
            filter_model,
            bundle,
            train_data=filter_train_data,
            timeout_seconds=filter_train_timeout_seconds,
        )
        stage_runtimes["filter_train_seconds"] = time.perf_counter() - filter_train_started
        if getattr(filter_model, "timed_out_", False):
            note = (
                f"Filter training reached its cooperative budget after "
                f"{getattr(filter_model, 'epochs_trained_', 0)} epochs; continuing with best completed epoch."
            )
            notes.append(note)
            print(note, flush=True)

        model = ModelWrapper(filter_model, generator)

        generator_metrics, stage_runtimes["generator_eval_seconds"] = run_with_timeout(
            "generator_eval",
            lambda: evaluate_generator(generator, bundle, config.evaluation),
            timeout_seconds=10800,
        )
        filter_metrics, stage_runtimes["filter_eval_seconds"] = run_with_timeout(
            "filter_eval",
            lambda: evaluate_filter(filter_model, bundle),
            timeout_seconds=10800,
        )
        ensemble_metrics, stage_runtimes["ensemble_eval_seconds"] = run_with_timeout(
            "ensemble_eval",
            lambda: evaluate_ensemble(model, bundle, config.evaluation),
            timeout_seconds=10800,
        )
        predictions, stage_runtimes["prediction_export_seconds"] = run_with_timeout(
            "prediction_export",
            lambda: collect_ensemble_predictions(model, bundle, config.evaluation),
            timeout_seconds=10800,
        )

        if config.evaluation.export_predictions:
            artifacts.save_csv(
                "predictions/test_predictions.csv",
                [
                    {
                        "substrate": row["substrate"],
                        "predicted": "|".join(row["predicted"]),
                        "real": "|".join(row["real"]),
                    }
                    for row in predictions
                ],
            )

        results.update(
            {
                "generator": generator_metrics,
                "filter": filter_metrics,
                "ensemble": ensemble_metrics,
                "generator_training": load_json(artifacts.root / "reports" / "generator_training.json"),
                "filter_training": load_json(artifacts.root / "reports" / "filter_training.json"),
                "generator_calibration": load_json(artifacts.root / "reports" / "generator_calibration.json"),
                "filter_calibration": load_json(artifacts.root / "reports" / "filter_calibration.json"),
            }
        )
        if (artifacts.root / "reports" / "filter_candidate_training_data.json").exists():
            results["filter_candidate_training_data"] = load_json(
                artifacts.root / "reports" / "filter_candidate_training_data.json"
            )
    except Exception as exc:
        results["error"] = {"message": str(exc), "traceback": traceback.format_exc()}
        results["total_runtime_seconds"] = time.perf_counter() - total_started
        with open(artifacts.path("error.txt"), "w") as handle:
            handle.write(results["error"]["traceback"])
        with open(artifacts.path("results.json"), "w") as handle:
            json.dump(results, handle, indent=2, default=str)
        print(f"\nEXPERIMENT FAILED: {exc}", flush=True)
        print(results["error"]["traceback"], flush=True)
        return 1

    results["total_runtime_seconds"] = time.perf_counter() - total_started
    artifacts.save_json("reports/stage_runtimes.json", stage_runtimes)
    with open(artifacts.path("results.json"), "w") as handle:
        json.dump(results, handle, indent=2, default=str)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for section in ("generator", "filter", "ensemble"):
        print(f"\n--- {section.capitalize()} ---")
        for key, value in results.get(section, {}).items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\n--- Thresholds ---")
    print(f"  generator: {results.get('generator_calibration', {}).get('calibrated_threshold')}")
    print(f"  filter: {results.get('filter_calibration', {}).get('calibrated_threshold')}")

    print("\n--- Early stopping ---")
    print(f"  generator: {results.get('generator_training', {}).get('early_stopped_epoch')}")
    print(f"  filter: {results.get('filter_training', {}).get('early_stopped_epoch')}")

    print("\n--- Stage runtimes (s) ---")
    for key, value in stage_runtimes.items():
        print(f"  {key}: {value:.1f}")
    print(f"\nTotal runtime: {results['total_runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

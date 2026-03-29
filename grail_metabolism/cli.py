from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from .artifacts import ArtifactStore
from .config import ExperimentConfig, load_experiment_config
from .experiments.presets import export_presets, get_experiment_preset, list_experiment_presets
from .experiments.runner import ExperimentRunner
from .model.wrapper import SimpleGenerator
from .utils.preparation import load_default_rules
from .workflows.inference import InferenceService


def _load_rules(path: str | None) -> List[str]:
    if path:
        with open(path) as handle:
            return [line.strip() for line in handle if line.strip()]
    return load_default_rules()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grail", description="Research workflows for GRAIL metabolism prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Generate candidate metabolites with the simple rule engine")
    predict.add_argument("smiles")
    predict.add_argument("--rules", default=None)
    predict.add_argument("--top-k", type=int, default=10)
    predict.add_argument("--threshold", type=float, default=None)
    predict.add_argument("--json", action="store_true")

    rules = subparsers.add_parser("rules", help="Print the active ruleset")
    rules.add_argument("--rules", default=None)

    presets = subparsers.add_parser("presets", help="List or export preset experiment configs")
    presets.add_argument("--export-dir", default=None)

    run_preset = subparsers.add_parser("run-preset", help="Run a named preset experiment")
    run_preset.add_argument("name", choices=list_experiment_presets())

    run_config = subparsers.add_parser("run-config", help="Run an experiment from YAML/JSON config")
    run_config.add_argument("config")

    ablate = subparsers.add_parser("ablate", help="Run multiple named presets and print a comparison table")
    ablate.add_argument("names", nargs="+", choices=list_experiment_presets())

    infer = subparsers.add_parser("infer", help="Load a saved experiment directory and run ensemble inference")
    infer.add_argument("smiles")
    infer.add_argument("--experiment-dir", required=True)
    infer.add_argument("--config", required=True)
    infer.add_argument("--rules", required=True)
    infer.add_argument("--top-k", type=int, default=None)
    infer.add_argument("--threshold", type=float, default=None)
    infer.add_argument("--json", action="store_true")

    return parser


def _print_metrics(metrics):
    print(json.dumps(metrics, indent=2))


def _simple_predict(args) -> int:
    selected_rules = _load_rules(args.rules)
    if not selected_rules:
        raise SystemExit("No rules are available. Provide --rules or install package resources.")
    generator = SimpleGenerator(selected_rules)
    products = generator.generate(args.smiles, top_k=args.top_k, threshold=args.threshold)
    if args.json:
        print(json.dumps({"substrate": args.smiles, "products": products}, indent=2))
    else:
        for product in products:
            print(product)
    return 0


def _run_preset(name: str) -> int:
    result = ExperimentRunner().run_preset(name)
    _print_metrics(result.metrics)
    return 0


def _run_config(path: str) -> int:
    config = load_experiment_config(path)
    result = ExperimentRunner().run_config(config)
    _print_metrics(result.metrics)
    return 0


def _run_ablation(names: Iterable[str]) -> int:
    runner = ExperimentRunner()
    results = [runner.run_preset(name) for name in names]
    print(json.dumps(runner.compare(results), indent=2))
    return 0


def _run_inference(args) -> int:
    rules = _load_rules(args.rules)
    service = InferenceService.from_experiment_dir(
        experiment_dir=args.experiment_dir,
        config_path=args.config,
        rules=rules,
    )
    products = service.predict(args.smiles, top_k=args.top_k, threshold=args.threshold)
    if args.json:
        print(json.dumps({"substrate": args.smiles, "products": products}, indent=2))
    else:
        for product in products:
            print(product)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "rules":
        for rule in _load_rules(args.rules):
            print(rule)
        return 0
    if args.command == "predict":
        return _simple_predict(args)
    if args.command == "presets":
        if args.export_dir:
            export_presets(args.export_dir)
        for name in list_experiment_presets():
            print(name)
        return 0
    if args.command == "run-preset":
        return _run_preset(args.name)
    if args.command == "run-config":
        return _run_config(args.config)
    if args.command == "ablate":
        return _run_ablation(args.names)
    if args.command == "infer":
        return _run_inference(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from ..artifacts import ArtifactStore
from ..config import ExperimentConfig
from ..workflows.ensemble import EnsembleWorkflow
from .presets import get_experiment_preset


@dataclass
class ExperimentResult:
    name: str
    artifact_dir: Path
    metrics: Dict[str, Dict[str, float]]


class ExperimentRunner:
    def __init__(self, output_dir: str = "artifacts") -> None:
        self.output_dir = output_dir

    def run_config(self, config: ExperimentConfig) -> ExperimentResult:
        artifacts = ArtifactStore.create(config.output_dir or self.output_dir, config.name)
        config.dump_yaml(artifacts.path("config.yaml"))
        metrics = EnsembleWorkflow(config, artifacts).run()
        return ExperimentResult(name=config.name, artifact_dir=artifacts.root, metrics=metrics)

    def run_preset(self, name: str) -> ExperimentResult:
        return self.run_config(get_experiment_preset(name))

    def run_many(self, configs: Sequence[ExperimentConfig]) -> List[ExperimentResult]:
        return [self.run_config(config) for config in configs]

    def compare(self, results: Sequence[ExperimentResult]) -> List[Dict[str, object]]:
        rows = []
        for result in results:
            row: Dict[str, object] = {"experiment": result.name}
            for section, metrics in result.metrics.items():
                for metric_name, value in metrics.items():
                    row[f"{section}.{metric_name}"] = value
            rows.append(row)
        return rows

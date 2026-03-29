from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from ..config import ExperimentConfig, load_experiment_config
from ..model.wrapper import ModelWrapper
from .factory import build_filter, build_generator


def _load_model_checkpoint(module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        module.load_state_dict(payload["state_dict"], strict=False)
        if "calibrated_threshold" in payload:
            module.calibrated_threshold = payload["calibrated_threshold"]
        return
    module.load_state_dict(payload, strict=False)


@dataclass
class InferenceService:
    model: ModelWrapper
    config: ExperimentConfig

    @classmethod
    def from_experiment_dir(
        cls,
        experiment_dir: str | Path,
        config_path: str | Path,
        rules: Sequence[str],
    ) -> "InferenceService":
        config = load_experiment_config(config_path)
        generator = build_generator(config.generator, list(rules))
        filter_model = build_filter(config.filter)
        root = Path(experiment_dir)
        generator_path = root / "checkpoints" / "generator.pt"
        filter_path = root / "checkpoints" / "filter.pt"
        if generator_path.exists():
            _load_model_checkpoint(generator, generator_path)
        if filter_path.exists():
            _load_model_checkpoint(filter_model, filter_path)
        return cls(ModelWrapper(filter_model, generator), config)

    def predict(self, smiles: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[str]:
        return self.model.generate(
            smiles,
            top_k=top_k or self.config.evaluation.candidate_top_k,
            threshold=self.config.evaluation.threshold if threshold is None else threshold,
        )

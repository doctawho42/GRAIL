from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from ..artifacts import ArtifactStore
from ..config import ExperimentConfig
from ..model.generator import Generator


@dataclass
class PretrainingWorkflow:
    config: ExperimentConfig
    artifacts: ArtifactStore

    @staticmethod
    def _split_reaction_smiles(value: str) -> Iterable[str]:
        text = str(value).strip()
        if not text:
            return []
        if ">>" in text:
            left, right = text.split(">>", 1)
            fields = [left, right]
        elif ">" in text:
            parts = text.split(">")
            if len(parts) >= 2:
                fields = [parts[0], parts[-1]]
            else:
                fields = parts
        else:
            fields = [text]
        fragments: List[str] = []
        for field in fields:
            for fragment in field.split("."):
                fragment = fragment.strip()
                if fragment:
                    fragments.append(fragment)
        return fragments

    def _load_uspto_smiles(self) -> List[str]:
        path = self.config.dataset.uspto_csv
        if not path:
            return []
        nrows = self.config.dataset.max_uspto_rows if self.config.dataset.max_uspto_rows else None
        frame = pd.read_csv(path, nrows=nrows)

        columns = [
            column
            for column in frame.columns
            if column.lower() in {"reactants", "products", "reaction_smiles", "rxn_smiles", "reaction", "reactions"}
        ]
        smiles: List[str] = []
        for column in columns:
            for value in frame[column].dropna().astype(str):
                smiles.extend(self._split_reaction_smiles(value))
        return smiles

    def run(self, generator: Generator) -> Generator:
        if not self.config.pretrain.enabled:
            return generator
        smiles = self._load_uspto_smiles()
        if not smiles:
            return generator
        generator.comprehensive_pretrain(
            smiles_list=smiles,
            epochs=self.config.pretrain.epochs,
            batch_size=self.config.pretrain.batch_size,
            lr=self.config.pretrain.lr,
            contrastive_ratio=self.config.pretrain.contrastive_ratio,
            maccs_ratio=self.config.pretrain.maccs_ratio,
            masked_ratio=self.config.pretrain.masked_ratio,
        )
        self.artifacts.save_checkpoint("checkpoints/generator_pretrain.pt", generator.state_dict())
        return generator

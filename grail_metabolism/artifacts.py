from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch


@dataclass
class ArtifactStore:
    root: Path

    @classmethod
    def create(cls, root: str | Path, experiment_name: str) -> "ArtifactStore":
        destination = Path(root) / experiment_name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "checkpoints").mkdir(exist_ok=True)
        (destination / "reports").mkdir(exist_ok=True)
        (destination / "predictions").mkdir(exist_ok=True)
        return cls(destination)

    def path(self, *parts: str) -> Path:
        destination = self.root.joinpath(*parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def save_json(self, relative_path: str, payload: Dict[str, Any]) -> Path:
        destination = self.path(relative_path)
        with open(destination, "w") as handle:
            json.dump(payload, handle, indent=2)
        return destination

    def save_csv(self, relative_path: str, rows: Sequence[Dict[str, Any]]) -> Path:
        destination = self.path(relative_path)
        rows = list(rows)
        with open(destination, "w", newline="") as handle:
            if not rows:
                handle.write("")
                return destination
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return destination

    def save_text(self, relative_path: str, content: str) -> Path:
        destination = self.path(relative_path)
        destination.write_text(content)
        return destination

    def save_checkpoint(self, relative_path: str, state: Any) -> Path:
        destination = self.path(relative_path)
        torch.save(state, destination)
        return destination

    def latest_checkpoint(self, prefix: str) -> Path | None:
        candidates = sorted(self.root.glob(f"checkpoints/{prefix}*.pt"))
        return candidates[-1] if candidates else None

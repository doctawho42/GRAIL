#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.experiments.presets import get_experiment_preset
from grail_metabolism.model.generator import Generator
from grail_metabolism.utils.transform import from_rule
from grail_metabolism.workflows.factory import build_generator


def main() -> int:
    config = get_experiment_preset("paper_full_ensemble")
    print(f"Rules path: {config.dataset.rules_path}")
    assert "extended" in config.dataset.rules_path or "notebooks" in config.dataset.rules_path
    assert config.dataset.use_clean_splits is True

    rules_path = ROOT / config.dataset.rules_path
    with open(rules_path) as handle:
        rules = [line.strip() for line in handle if line.strip()]
    print(f"Rules loaded: {len(rules)}")
    assert rules

    for rule in rules[:10]:
        graph = from_rule(rule)
        assert graph.x.shape[1] == 16

    generator = build_generator(config.generator, rules[:32])
    assert isinstance(generator, Generator)
    cached_graphs = generator._get_rule_graphs(rules[:5])
    assert len(cached_graphs) == min(5, len(rules))
    assert all(graph.x.shape[1] == 16 for graph in cached_graphs)

    print("✓ All smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

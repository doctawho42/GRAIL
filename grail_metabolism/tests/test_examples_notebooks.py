from __future__ import annotations

import json
from pathlib import Path


def test_curated_example_notebooks_are_present_and_small():
    examples_dir = Path("examples/notebooks")
    notebooks = sorted(examples_dir.glob("*.ipynb"))

    assert [path.name for path in notebooks] == [
        "01_inference_demo.ipynb",
        "02_run_preset.ipynb",
        "03_workflow_smoke.ipynb",
        "04_ablation_analysis.ipynb",
    ]

    for notebook_path in notebooks:
        payload = json.loads(notebook_path.read_text())
        assert payload["nbformat"] == 4
        assert any(cell["cell_type"] == "code" for cell in payload["cells"])

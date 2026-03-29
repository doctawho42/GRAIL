# Curated Notebooks

The historical `notebooks/` directory accumulated exploratory material,
duplicate experiments and heavyweight local artefacts. That content is not part
of the supported repository surface anymore.

The repository now keeps only lightweight, publication-friendly notebooks under
`examples/notebooks/`.

## Included notebooks

- `01_inference_demo.ipynb`
  minimal metabolite generation demo using packaged example rules
- `02_run_preset.ipynb`
  inspect and launch a preset-based experiment
- `03_workflow_smoke.ipynb`
  tiny end-to-end training/inference smoke example on an in-memory dataset
- `04_ablation_analysis.ipynb`
  compare preset outputs with the experiment runner utilities

## Principles

Curated notebooks should be:

- executable in a clean developer environment
- independent of private local paths
- small enough for code review
- aligned with the tested package API

Notebooks are examples, not the primary workflow interface. Reproducible runs
should go through:

- CLI commands
- YAML configs
- `grail_metabolism.workflows`

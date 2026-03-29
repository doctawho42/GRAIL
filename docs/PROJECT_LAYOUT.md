# Project Layout

This repository now separates stable package code from generated research output.

## Tracked source

- `grail_metabolism/`
  - importable package code
  - workflows, models, configs, metrics, packaged resources
- `scripts/`
  - supported command-line entrypoints and maintenance utilities
- `configs/`
  - exported experiment presets
- `docs/`
  - architecture and reproducibility notes
- `examples/notebooks/`
  - curated lightweight walkthrough notebooks

## Local-only output

These paths are intentionally ignored and should not be committed:

- `artifacts/`
  - experiment runs, checkpoints, logs, cached preprocessing
- `results/`
  - one-off audit outputs, measurement logs, smoke-test output
- `notebooks/`
  - exploratory notebooks and heavyweight local assets
- `grail_metabolism/data/`
  - local research datasets

## Legacy helpers

Legacy one-off preprocessing scripts live in `scripts/legacy/`.

They are kept for reference, but they are not part of the supported package API,
the release artifact surface, or the automated test matrix.

## Practical rule

If a file is required to:

- import the package
- run the CLI
- execute presets/workflows
- reproduce supported tests

it should live in tracked source directories.

If a file is large, generated, machine-specific, or a one-off research artifact,
it should live in ignored output directories instead.

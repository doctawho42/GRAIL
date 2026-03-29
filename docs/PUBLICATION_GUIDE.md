# Publication Guide

This repository is structured to support a journal-style release of a
xenobiotic metabolism prediction model.

## Minimum publishable package surface

The publishable source should include:

- package code in `grail_metabolism/`
- experiment configs in `configs/`
- workflow scripts in `scripts/`
- documentation in `docs/`
- lightweight example notebooks in `examples/notebooks/`

## Recommended supplementary artefacts

For a manuscript submission or archival release, attach:

- the exact main-model config
- the exact rule bank used for the main model
- clean split definitions
- aggregate metrics JSON
- prediction export on the test set
- environment specification (`pyproject.toml`, `requirements.txt`, `environment.yml`)

## What should stay out of the source release

Do not mix the repository with:

- local virtual environments
- transient logs
- exploratory notebooks
- heavyweight intermediate pickles
- generated HTML visualizations
- stale wheel/tarball build products

Those should live in ignored local output directories or in separate archival
bundles if they are scientifically relevant.

## Reproducibility checklist

Before tagging a release:

1. Run the package test suite
2. Export preset YAMLs
3. Verify the main preset resolves to the intended rule bank
4. Archive metrics and prediction outputs from the final run
5. Build source and wheel distributions from a clean staging directory

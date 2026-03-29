# Reproducibility

## Required inputs

The repo expects the already prepared research artefacts mentioned in the manuscript:

- `train.sdf`, `val.sdf`, `test.sdf`
- `*_triples.txt`
- a rules file such as `grail_metabolism/resources/extended_smirks.txt`
- optionally `USPTO_FULL.csv` for pretraining

Primary raw-data extraction before SDF creation is intentionally out of scope.

## Environment notes

- Use `numpy<2` with the current RDKit / PyG stack.
- Run in the repository venv or recreate the environment from `pyproject.toml`.
- For large experiments, prefer a dedicated clean environment without the legacy nested `grail_metabolism/venv`.

## Reproducible output

Every experiment run writes:

- immutable config snapshot
- checkpoints
- metrics report
- prediction export
- reusable split-level preprocessing cache

This is the minimum needed for a manuscript appendix / supplementary archive.

## Build and release checks

Use a clean staging build before tagging or uploading:

```bash
bash scripts/build_release.sh
```

This avoids contaminating the sdist/wheel with local virtual environments, large research data or legacy artefacts. The script runs both `python -m build` and `twine check`, then copies the verified distributions back into the repository `dist/` directory.

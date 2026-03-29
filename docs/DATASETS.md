# Datasets and Splits

## Required research inputs

The full training/evaluation workflow expects the following prepared files:

- `grail_metabolism/data/train.sdf`
- `grail_metabolism/data/val.sdf`
- `grail_metabolism/data/test.sdf`
- `grail_metabolism/data/train_triples.txt`
- `grail_metabolism/data/val_triples.txt`
- `grail_metabolism/data/test_triples.txt`
- optionally `grail_metabolism/data/USPTO_FULL.csv` for generator pretraining

The repository treats these as external research data. They are not packaged
into source distributions or wheels.

## Clean split policy

The project defaults to canonicalized clean splits:

- `train_triples_clean.txt`
- `val_triples_clean.txt`
- `test_triples_clean.txt`

These files remove substrate-level leakage between train, validation and test.
`DatasetConfig.use_clean_splits=True` is the default, and the loader will fall
back to the original triples only if a clean file is unavailable.

## Rule banks

Packaged rules/resources live in `grail_metabolism/resources/`.

Current important files:

- `extended_smirks.txt`
  publication-facing merged rule bank
- `notebooks_rules.txt`
  preserved snapshot of the historical large rule bank
- `mined_only.txt`
  mined rules only
- `example_rules.txt`
  tiny smoke-test ruleset

## Preprocessing cache

Large runs can reuse preprocessing through:

- `DatasetConfig.cache_preprocessed`
- `DatasetConfig.cache_dir`

The split-level cache stores:

- single-molecule graphs
- reaction label vectors
- Morgan fingerprints when requested

Cache keys include source file signatures, sampling settings, standardization
mode and the active rule bank hash. This prevents accidental cross-run reuse.

## Publication recommendation

For a paper supplement or public artifact bundle, ship:

- exact experiment configs
- clean split triples
- the rule bank used for the main run
- metrics reports and prediction exports
- a short checksum manifest for data-dependent files

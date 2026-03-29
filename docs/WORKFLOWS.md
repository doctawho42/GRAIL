# Workflows

## Main execution paths

Supported entrypoints:

- `grail run-preset <preset>`
- `grail run-config <config.yaml>`
- `grail infer <smiles> --experiment-dir ... --config ... --rules ...`
- `grail predict <smiles> --rules ...`

Thin wrapper scripts in `scripts/` forward to the same package CLI for
backward compatibility and automation.

## Training flow

The full ensemble workflow is:

1. Load dataset bundle and rules
2. Prepare cached split artifacts
3. Optional generator pretraining on USPTO-like reaction strings
4. Generator training with early stopping
5. Generator threshold calibration on validation data
6. Filter training with early stopping
7. Filter threshold calibration on validation data
8. Test-set evaluation and prediction export

Optional two-stage filter training replaces step 6 with:

1. train generator
2. generate top-k candidates on train substrates
3. train filter on generator-produced negatives plus missed positives

## Runtime outputs

Every experiment writes to `artifacts/<experiment>/`:

- `config.yaml`
- `config.json`
- `checkpoints/`
- `reports/`
- `predictions/`

Large preprocessing caches live separately under `artifacts/preprocessed/`.

## Recommended modes

For publication-quality experiments:

- use `paper_full_ensemble`
- use `extended_smirks.txt`
- keep `use_clean_splits=True`
- keep validation-based calibration enabled
- preserve the generated `config.yaml` and metrics JSON verbatim

For lightweight local sanity checks:

- use `paper_minimal_baseline`
- keep a tiny ruleset or reduced substrate budget

# Architecture

## Target research pipeline

The codebase is structured around the three-stage architecture described in the manuscript:

1. `Generator`
   Selects reaction SMARTS / SMIRKS rules with a multi-label classifier over substrate and rule graphs.
2. `Rule application`
   Applies all selected rules to the target substrate with RDKit and keeps provenance of generated candidates.
3. `Filter`
   Scores substrate-metabolite pairs with graph encoders and Morgan fingerprint branches.

The current publication-oriented default uses:

- clean train/val/test substrate splits
- the extended packaged rule bank
- validation-based early stopping
- validation threshold calibration
- preprocessing cache for expensive graph/label stages

## Code layout

- `grail_metabolism/utils/preparation.py`
  Dataset assembly, `MolFrame`, SDF/triples loading, negative generation, rule labeling.
- `grail_metabolism/utils/transform.py`
  Featurizers for substrate graphs, pair graphs and rule graphs.
- `grail_metabolism/model/generator.py`
  Generator model, pretraining tasks and rule scoring heads.
- `grail_metabolism/model/filter.py`
  Pair/single filtering models and graph/fingerprint ablations.
- `grail_metabolism/workflows/`
  Reproducible pipelines for loading data, pretraining, training, evaluation and inference.
- `grail_metabolism/experiments/`
  Preset registry and experiment runner.
- `configs/`
  YAML entrypoints for reproducible experiments.

## Generator design

The generator now explicitly supports substrate-rule scoring ablations:

- `bilinear`
  Scalar attention via bilinear form, matching the intended paper architecture.
- `dot`
  Simpler similarity baseline.
- `mlp`
  Learned pair scorer baseline over concatenated substrate/rule embeddings.

The generator also includes optional pretraining routines:

- contrastive graph pretraining
- masked graph modeling
- MACCS auxiliary supervision

## Filter design

The filter supports:

- `pair` mode over merged substrate-product graphs
- `single` mode over independently encoded substrate and product graphs
- graph + fingerprint fusion
- graph-only and fingerprint-only ablations
- GATv2 / GCN / GIN backbone variants

## Experiment shell

The intended top-level execution paths are:

- `grail run-config configs/paper_full_ensemble.yaml`
- `grail run-preset paper_full_ensemble`
- `grail ablate paper_no_pretrain paper_filter_graph_only paper_generator_dot`
- `grail infer ...`

Artifacts are written to:

- `artifacts/<experiment>/config.yaml`
- `artifacts/<experiment>/checkpoints/*.pt`
- `artifacts/<experiment>/reports/metrics.json`
- `artifacts/<experiment>/predictions/test_predictions.csv`

Expensive split preparation is cached separately under:

- `artifacts/preprocessed/<split>/<signature>/`

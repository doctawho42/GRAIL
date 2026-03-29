# Experiments

## Preset experiment families

The repository now ships named presets for the main paper model and common ablations:

- `paper_full_ensemble`
- `paper_full_ensemble_two_stage_filter`
- `paper_no_pretrain`
- `paper_filter_graph_only`
- `paper_filter_morgan_only`
- `paper_filter_single`
- `paper_generator_dot`
- `paper_generator_mlp`
- `paper_filter_gcn`
- `paper_filter_gin`
- `paper_minimal_baseline`

## Recommended study order

1. Sanity / smoke
   Run `paper_minimal_baseline`
2. Main model
   Run `paper_full_ensemble`
   Compare with `paper_full_ensemble_two_stage_filter` if you want filter
   training on generator-produced candidates.
3. Generator ablations
   Run `paper_no_pretrain`, `paper_generator_dot`, `paper_generator_mlp`
4. Filter ablations
   Run `paper_filter_graph_only`, `paper_filter_morgan_only`, `paper_filter_single`
5. Backbone ablations
   Run `paper_filter_gcn`, `paper_filter_gin`

## Metrics

Generator / ensemble:

- Jaccard
- Precision
- Recall
- F1
- Exact match
- Top-k recall

Filter:

- MCC
- ROC-AUC

Generator and filter decision thresholds are calibrated on validation data after
training.

## Example commands

```bash
grail run-config configs/paper_full_ensemble.yaml
grail ablate paper_no_pretrain paper_filter_graph_only paper_generator_dot
grail run-preset paper_minimal_baseline
```

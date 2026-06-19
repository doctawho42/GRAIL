# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

GRAIL predicts xenobiotic/drug **metabolite structures** with a three-stage, rule-based-plus-learned pipeline. It is research code targeting a publication and a SOTA claim; favor correctness, reproducibility, and apples-to-apples evaluation over speed.

## Commands

```bash
make install                      # pip install -r requirements.txt + pip install -e .[tuning]
make test                         # pytest grail_metabolism/tests -q   (33 smoke/unit + 10 regression)
python -m pytest grail_metabolism/tests/test_audit_fixes.py::test_puloss_trains_on_logits -q   # single test
make smoke                        # CLI predict + preset export, no dataset needed

# CLI (entry point `grail` == `python -m grail_metabolism`)
grail predict "CCO" --rules grail_metabolism/resources/example_rules.txt
grail run-preset paper_full_ensemble
grail run-config configs/paper_full_ensemble.yaml
grail ablate paper_no_pretrain paper_filter_graph_only paper_generator_dot
grail presets --export-dir configs/generated

# Research scripts (need the external dataset under grail_metabolism/data/)
python scripts/measure_coverage.py                 # rule-bank recall ceiling + leakage report -> results/
python scripts/run_benchmark.py [--sample N] [--with-phase2]   # GRAIL ceiling + SyGMa baseline, InChIKey matching
python scripts/run_multiseed.py --preset paper_minimal_baseline --seeds 0 1 2   # mean±std headline numbers
python scripts/fix_splits.py --molecule-disjoint   # regenerate leakage-free clean split triples
```

Environment constraint: **`numpy<2`** is required with the RDKit / torch-geometric stack. `optuna` (tuning) and `streamlit` (app) are optional extras.

## Dataset (external, not in git)

Large training data lives in `grail_metabolism/data/` (`{train,val,test}.sdf`, `*_triples.txt`, optional `USPTO_FULL.csv`) and is **gitignored** — the test suite runs without it. Triples format: `substrate_id metabolite_id is_real`. `DatasetConfig.use_clean_splits=True` prefers `*_triples_clean.txt`. To exercise full workflows in a worktree, symlink the files from the main checkout into `grail_metabolism/data/` (they stay gitignored).

## Architecture

Three stages (see `docs/ARCHITECTURE.md`):
1. **Generator** (`model/generator.py`) — multi-label scorer over (substrate graph, rule graph) pairs; selects which SMARTS/SMIRKS rules to apply. Default `scoring="retrieval"` (attention + similarity + MLP head). Trained as "which rules yield a *true* metabolite for this substrate".
2. **Rule application** — RDKit applies selected rules (`utils/preparation.safe_run_reactants`) to enumerate candidate products.
3. **Filter** (`model/filter.py`) — binary classifier scoring (substrate, product) pairs; `mode="pair"` (merged MCS-aware graph) or `mode="single"` (independent encoders). `model/wrapper.py:ModelWrapper.generate` runs all three and ranks by `filter_score * generator_score`.

Data & featurization flow:
- `utils/preparation.py:MolFrame` is the central data object: assembles substrate→product maps from DataFrame/dict/SDF, derives negatives, builds graphs/labels, holds `map` (positives), `negs`, `single`/`graphs`, `reaction_labels`. Heavy stages cache to `.pt` files.
- `utils/transform.py` featurizes molecules (`from_rdmol`), substrate–product pairs (`from_pair`), and rule graphs (`from_rule`). Fixed dims: `SINGLE_NODE_DIM=16`, `PAIR_NODE_DIM=18`, `EDGE_DIM=18`, `FINGERPRINT_DIM=1024`. The generator encoder takes 16-dim single-graph nodes; the pair filter takes 18-dim pair nodes.
- `model/_graph.py:GraphEncoder` is the shared GNN backbone (`gatv2`/`gcn`/`gin`).

Orchestration:
- `config.py` — dataclass config tree (`ExperimentConfig` → dataset/generator/filter/pretrain/optim/evaluation); YAML-serializable.
- `experiments/presets.py` — named presets (`paper_full_ensemble`, ablations); `experiments/runner.py` + `experiments/study.py` drive runs.
- `workflows/` — `data.py` (`load_dataset_bundle`, `DatasetBundle.prepare`, preprocessing cache), `pretraining.py`, `training.py`, `ensemble.py`. **`EnsembleWorkflow.run_bundle` is the single chokepoint** that wires the whole pipeline and seeds RNGs; route cross-cutting changes through it.
- `workflows/factory.py:build_generator/build_filter` construct models from config — keep these the single construction path.

## Invariants & gotchas (don't regress these)

- **PU data, not clean labels.** Negatives = rule-applicable products that aren't annotated (`MolFrame.negs`), i.e. positive-unlabeled. The filter trains in the **logit domain** via `Filter.forward(..., return_logits=True)` so `PULoss`/nnPU isn't fed a double sigmoid (BCE path uses probabilities). The generator down-weights unobserved-applicable rules via `GeneratorConfig.unlabeled_weight`. Never reintroduce probability-input to PULoss or treat `negs` as ground-truth negatives.
- **Select on validation, never test.** Use `evaluate_ensemble_val`, `OptunaWrapper.make_study(val_set=...)`, and `ensemble_val.f1` for model/preset/HP selection; touch the test split once for the final report.
- **Reproducibility.** Set `ExperimentConfig.seed` (applied by `utils/seed.seed_everything` in `run_bundle`, recorded in `reports/metrics.json`). `DatasetConfig.sampling_seed` only controls data subsampling. Report headline numbers as mean±std over seeds (`run_multiseed.py`).
- **Rule bank consistency.** All entry points resolve the default bank through `preparation.resolve_default_rule_bank()` (default `resources/extended_smirks.txt`, ~7581 rules). Checkpoints persist `arch` + `rules`; `grail.py:_load_checkpoint_payload` verifies the match. The generator re-encodes **all** rule graphs each forward — large banks are the dominant cost.
- **Standardization is lossy & slow.** `standardize_mol` runs tautomer canonicalization and uses `isomericSmiles=False` (stereo stripped) — so Morgan chirality/bond-stereo features are inert by design. Small fragments are kept down to **≥2 heavy atoms** (not the old ≥3-carbon filter). Route SMILES normalization through `_standardize_smiles_cached`.
- **Pair filter depends on MCS alignment.** `from_pair` adds cross-edges between substrate/product atoms by the *element-aware MCS atom correspondence* (positional zip of `GetSubstructMatch`), not sorted index. Don't reorder to sets/sorted indices.
- **Metrics.** `metrics.py` is set-based macro P/R/F1/Jaccard over **annotated positives only**, so precision is a pessimistic lower bound — lead with recall@k and `mean_output_size`. `EvaluationConfig.match="inchikey"` enables literature-standard structure matching; `EvaluationConfig.max_output` caps the ensemble output set.
- Public quick-start `model/grail.py:summon_the_grail` hardcodes dims `(16,18)/(18,18)` and filter hidden `[128,256,128,128,64,32]`; trained checkpoints with other dims are reconstructed from their saved `arch`.

## Testing notes

Tests are smoke/unit and run without the dataset. `tests/test_audit_fixes.py` guards the invariants above (MCS alignment, logit-domain filter, PU weighting, seeding determinism, rule-bank consistency, InChIKey matching). When changing model/loss/metric behavior, update or add a guard there and keep `make test` green.

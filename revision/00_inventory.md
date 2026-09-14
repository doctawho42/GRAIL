# Phase 0 inventory

Every number here was computed in this session from files in this repository by
`revision/00_inventory.py`, which writes `revision/00_inventory_facts.json`. That JSON is the
authoritative form; this document is its reading. Re-verify with:

    python revision/00_inventory.py

No prose in the manuscript was read for values and none was edited. Nothing below is estimated.

---

## 1. Populations

| population | defined by | n | note |
|---|---|---|---|
| `evaluated1170` | `results/test_references.json` | 1,170 | all 1,170 carry at least one reference |
| `comparison291` | see below | 291 | verified to be a strict subset of the 1,170 |

Two facts about the 291 that bear on the revision:

- **How the 291 were drawn is not recorded.** `results/population_definition.json` states it
  outright: *"not recorded: no script in this repository selects them and the submission directory
  documents the file format rather than the draw"*. The same file gives the intent: *"the
  intersection of the substrates each method has an entry for; the binding constraint is the
  291-substrate submission list sent to the web-service comparator, which returned all 291. No
  substrate was removed for emitting nothing."*
- **The artifact that names the members cannot be joined to the prediction files.**
  `results/comparison_set_members.json` keys its 291 members by tautomer InChIKey
  (e.g. `NVENYTWRUWFLAR-UHFFFAOYSA-N`), not by SMILES. The member SMILES are instead recovered
  from the four prediction files that carry exactly them, and those four agree exactly: pairwise
  identical, symmetric difference 0. The probe refuses if they ever disagree.

`results/population_definition.json` also records `deployed_arm_covers_the_whole_test_set: true`,
and that the exhaustive arm was built on the comparison set only, so the whole-test-set row carries
the deployed interactive arm and not the exhaustive one.

---

## 2. Prediction files, by computed coverage

Coverage is the size of the intersection with a declared population. Nothing is inferred from a
filename.

### Covering the whole evaluated set (5 files or shard sets)

| file | keys | of 1,170 | of 291 | system |
|---|---|---|---|---|
| `results/sygma_fulltest_predictions.json` | 1,170 | 1,170 | 291 | SyGMa |
| `results/biotransformer_fulltest_preds.json` | 1,170 | 1,170 | 291 | BioTransformer |
| `artifacts/tier2_1170/metapredictor_preds.json` | 1,170 | 1,170 | 291 | MetaPredictor |
| `results/widepools_fulltest/` (6 shards) | 1,170 | 1,170 | 291 | GRAIL, whole bank |
| `results/widepools_k30_fulltest/` (6 shards) | 1,170 | 1,170 | 291 | GRAIL, trained budget |

### Covering the comparison set only

| file | keys | of 1,170 | of 291 | system |
|---|---|---|---|---|
| `results/gloryx_service_preds.json` | 291 | 291 | 291 | GLORYx, web service |
| `results/metatox_smirks_preds.json` | 291 | 291 | 291 | MetaTox |
| `results/sygma_standardised_predictions.json` | 291 | 291 | 291 | SyGMa, standardised drawing |
| `results/biotransformer_allhuman_one_step_preds.json` | 291 | 291 | 291 | BioTransformer, one step |
| `results/biotransformer_allhuman_one_step_natural_drawing_preds.json` | 291 | 291 | 291 | BioTransformer, natural drawing |
| `results/metapredictor_wide_beam_preds.json` | 291 | 291 | 291 | MetaPredictor, wide beam |
| `results/wide_pools.json` | 291 | 291 | 291 | GRAIL |
| `results/widepools/`, `widepools_implicit/`, `widepools_ship/` (6 shards each) | 291 | 291 | 291 | GRAIL |
| `results/widepools_k30/`, `widepools_k30_std/` (1 shard each) | 291 | 291 | 291 | GRAIL |

### Partial, and all of it inside the 291

| file | keys | of 1,170 | of 291 |
|---|---|---|---|
| `results/metatox_preds.json` | 248 | 248 | 248 |
| `results/widepools_std_fine/` (14 shards) | 146 | 146 | 146 |
| `results/widepools_std/` (2 shards) | 145 | 145 | 145 |
| `results/metapredictor_natural_drawing_preds.json` | 81 | 81 | 81 |

### Elsewhere, and on other populations

| file | keys | of 1,170 |
|---|---|---|
| `artifacts/tier2/biotransformer_preds.json` | 150 | 150 |
| `artifacts/tier2/metapredictor_preds.json` | 150 | 150 |
| `artifacts/tier2/metapredictor_preds_wide64.json` | 150 | 150 |
| `artifacts/tier2/metatrans_preds.json` | 150 | 150 |
| `docs/benchmark/data/metapredictor_gloryx.json` | 37 | 0 |

Validation-population pools, which are not the test set and are listed so they are not mistaken for
it: `results/valpools_k10/ k20/ k30/ k50/` 294 each, `valpools/` and `valpool_shards/` and
`valseedpools/` and `valpools_k7581/` 293 each, `trainpools/` 349, `seedpools/` 291 (the comparison
set). None intersects the 1,170.

---

## 3. Which file each reported comparator column reads

From `scripts/typed_edit/deployment_table.py`, lines 43-57:

| column | file it reads | that file's coverage |
|---|---|---|
| metatox | `results/metatox_smirks_preds.json` | 291 |
| sygma | `results/sygma_fulltest_predictions.json` | 1,170 |
| metapredictor | `artifacts/tier2_1170/metapredictor_preds.json` | 1,170 |
| biotransformer | `results/biotransformer_allhuman_one_step_preds.json` | 291 |
| gloryx | `results/gloryx_service_preds.json` | 291 |

Two of the five columns are already read from files that cover the whole evaluated set, and the
table is nevertheless computed on the 291. `results/deployment_table.json` records
`population: {"n": 291, "n_references": 665.0, "source": "the 291 of results/four_method_291.json"}`
and `comparators_absent: []`.

`scripts/four_method_291.py` reads a different set again: `artifacts/tier2_1170/metapredictor_preds.json`,
`results/metatox_smirks_preds.json` (291) and `results/sygma_fulltest_predictions.json` (1,170).

`results/deployment_table.json`'s own `inputs` block lists only GRAIL pools
(`widepools_implicit/w0..w5`, `widepools_k30/all`); the comparator files above are opened by the
producer but do not appear in that block.

---

## 4. Matching criteria

Dispatch: `grail_metabolism/metrics.py:_match_keys`.

| criterion | implementation | what it compares |
|---|---|---|
| `exact` | none (fallback branch) | the raw strings |
| `canonical` | `_canonical_key` | RDKit canonical SMILES with `isomericSmiles=False` |
| `inchikey` | `_inchikey` | InChIKey, falling back to the raw string if RDKit cannot parse |
| `inchikey_tautomer` | `_tautomer_inchikey` | Cleanup, FragmentParent, uncharge, `TautomerEnumerator.Canonicalize`, then InChIKey; per-input fallback to plain InChIKey |
| `inchi_no_stereo` | `_inchikey_skeleton` | the first block of the InChIKey, so stereoisomers and charge variants collide |
| `tanimoto1` | `_morgan_key` | a Morgan fingerprint used as an identity key |

**Discrepancy, unresolved.** `grail_metabolism/config.py` declares
`["exact", "inchi_no_stereo", "inchikey", "inchikey_tautomer", "tanimoto1"]` while
`grail_metabolism/metrics.py` declares those plus `canonical`. A configuration cannot request
`canonical` through the config type, but the metric implements it and the sweeps use it.

Precomputed key maps, SMILES to key, in `results/key_tables/`:

| file | entries |
|---|---|
| `canonical.json` | 108,967 |
| `inchikey.json` | 108,967 |
| `inchi_no_stereo.json` | 108,967 |
| `tanimoto1.json` | 108,967 |
| `inchikey_tautomer.json` | 116,882 |

Written by `scripts/bank_without_selection.py`, `scripts/ceiling_gap_by_similarity.py`,
`scripts/four_method_291.py`, `scripts/set_metrics_by_criterion.py`, `scripts/tier2_key_shard.py`.

---

## 5. recall@k, intervals, family-wise correction

| quantity | where | on which population |
|---|---|---|
| recall@k | `grail_metabolism/metrics.py:top_k_recall`, the size of top-k intersected with the truth set over the truth set | whatever the caller passes |
| paired interval | `scripts/_contrast.py:paired_contrast`, paired bootstrap over items, `n_boot=10000`, `seed=0` | stated per caller; refuses with `EmptyComparator` when the comparator covers nothing, and flags a gap within 0.95 of the arm's own score |
| the comparison table | `results/deployment_table.json`, micro as the ratio of sums, `n_boot=10000`, `seed=0`, parent dropped before the budget for every arm alike | 291, `n_references` 665 |

`scripts/_contrast.py` is imported by `typed_edit/deployment_table.py`,
`typed_edit/multiplicity.py`, `typed_edit/emission_leaderboard.py`,
`typed_edit/retraining_spread.py`, `typed_edit/h13_verdict.py`, `typed_edit/h15_verdict.py`,
`scripts/paper2_numbers.py`.

### The correction the manuscript quotes

`scripts/paper2_numbers.py` reads `results/multiplicity.json`.

| property | value |
|---|---|
| family, as declared in the artifact | "every GRAIL-arm-against-comparator contrast in the sweep: 2 arms by 3 comparators by 9 budgets" |
| tests | 54 |
| arms | whole bank 27, trained budget 27 |
| comparators | metatox 18, sygma 18, metapredictor 18 |
| budgets | 1, 3, 5, 8, 10, 15, 20, 30, 50 |
| alpha | 0.05 |
| procedure | Holm step-down on two-sided bootstrap p-values, B and seed as the intervals |
| separating per comparison | 33 |
| separating after Holm | 22 |
| cells whose verdict the correction changes | 9 |
| leads the correction removes | 4 |

**BioTransformer and GLORYx are not in the declared family.** The artifact's own note: the family
was fixed before the BioTransformer arm existed and is not widened after the fact. The wider
sensitivity, `over_every_contrast_the_paper_prints`, has 90 tests and 29 surviving.

**`results/multiplicity.json` has no `population` field.** Its cells are named for the arms of the
291 table, but the artifact does not record the population it was computed on. Establishing that is
Phase 1 work, not something to assume here.

Two further correction artifacts exist and are **not** what the manuscript quotes:

| artifact | families |
|---|---|
| `results/multiplicity_holm.json` | external 6, external_grid 12, internal_as_tested 2, internal_conservative 10, internal_grid 20, selection_confirmatory 1; alpha 0.05; p by normal approximation from the paired-bootstrap CI |
| `results/union_multiplicity.json` | union family 15,752 tests over 24 boards, 8,367 rejected |

---

## 6. What produced the existing tables and figures

| target | producer |
|---|---|
| `paper2/table_sweep.tex` | `scripts/paper2_tables.py` |
| `paper2/table_modes.tex`, `table_grain.tex`, `table_hypotheses.tex`, `table_case.tex` | `scripts/paper2_tables_more.py` |
| 24 `paper2/si_table_*.tex` | `scripts/paper2_si_tables.py` (on a generator's refusal it deletes the stale `.tex` so the build fails loudly rather than compiling old numbers) |
| all 9 `paper2/fig_*.eps` | `scripts/paper2_figures.py` |

What each document pulls:

| document | `\input` |
|---|---|
| `grail_jcim.tex` | body, markers, numbers |
| `body.tex` | si_table_chemistry, si_table_matched, table_case, table_grain, table_hypotheses, table_modes, table_sweep |
| `si.tex` | markers, numbers, and 22 `si_table_*` |

Figures referenced: fig_toc, fig_sweep, fig_criterion, fig_case, fig_cost, fig_ceiling,
fig_si_budget, fig_si_rarefaction, fig_si_external. All nine `.eps` files are present.

`paper2/claim.tex` exists and is pulled by no document.

---

## 7. What cannot be computed from this repository

| missing | consequence |
|---|---|
| GLORYx predictions on the 1,170 | no file exists. The only GLORYx predictions are the 291 from the web service, plus `docs/benchmark/data/metapredictor_gloryx.json`, which holds 37 substrates and intersects the 1,170 in 0 |
| MetaTox predictions on the 1,170 | no file exists. The widest MetaTox file is the 291 |
| the draw that produced the 291 | not recorded in any script; only the intent is recorded |
| the population `results/multiplicity.json` was computed on | the artifact has no population field |
| provenance stamps for four artifacts computed on the 1,170 | `budget_curves.json`, `criterion_within_method.json`, `decompose_sygma.json`, `coverage_gap_types.json` carry no provenance block; their producers were located by name search (`budget_curves.py`, `criterion_within_method.py`, `decompose_sygma.py`, and for the stamped one `ceiling_gap_ci.py`) rather than read from a stamp |

Already computed on the 1,170, and usable in Phase 1 without new predictions:
`budget_curves.json` (`inchikey_tautomer`, k_max 32, n_boot 10000, seed 0),
`criterion_within_method.json` (methods include GRAIL with its emitted-by-k curve),
`decompose_sygma.json` (SyGMa, `inchikey_tautomer`, k 15), `ceiling_gap_ci.json`,
`coverage_gap_types.json`.

---

## 8. Two errors made during this discovery, and the method that replaced them

Recorded because both produced plausible wrong numbers that a reader would not have caught.

1. Counting substrate keys by the look of the string. A filter requiring one of `()=#@` dropped
   short SMILES such as `Br` and `CCO`, reporting 1,144 substrates in the full-test prediction
   files; a looser filter reported 1,167. The exact figure is 1,170 in both files, with 0 evaluated
   substrates absent. Coverage in this inventory is only ever the size of an intersection with a
   declared population.
2. Reading the comparison set from the artifact that names it. `comparison_set_members.json` is
   keyed by tautomer InChIKey, so intersecting it with SMILES-keyed prediction files produced a
   membership of 7 and a whole column of meaningless counts. The 291 SMILES are now recovered from
   the four prediction files that carry exactly them, with a refusal if they disagree.

---

## Phase 1 readiness

| Phase 1 item | blocked? |
|---|---|
| `T_main.csv` on `comparison291` | no |
| `T_main.csv` on `evaluated1170` | partly: SyGMa, BioTransformer, MetaPredictor and both GRAIL arms can be computed; MetaTox and GLORYx cells cannot exist until Phase 2 |
| `F_budget.pdf` | no; emitted-candidate counts exist per system in `criterion_within_method.json` and in the pools |
| `T_criteria.md` | no for the implementations; the literature user of each criterion has not been searched yet and will be cited from the repository's bibliography or marked as having none |
| the family-wise recomputation | no, but note that the declared family excludes two of the five comparators and the artifact does not record its population |

Waiting for your go-ahead before Phase 1.

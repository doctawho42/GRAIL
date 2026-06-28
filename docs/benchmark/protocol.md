# MetaBench — a standardized, leakage-audited benchmark and evaluation protocol for metabolite structure prediction

Stage-1 benchmark datasheet + protocol spec. The field has no shared benchmark and matches
predicted vs reference structures inconsistently (GLORYx: InChI without stereo; MetaTrans:
Tanimoto=1; LAGOM: canonical SMILES), so headline rankings depend on the match definition.
This benchmark fixes a single protocol, a leakage-audited split, and a match-sensitivity
analysis. We are *not* the first comparison (cf. Scholz 2023, Gao 2026) — we are the first
*standardized, leakage-audited, match-sensitivity* protocol.

## 1. Task

Given a parent (substrate) SMILES, predict the set of its metabolite **structures** (SMILES)
as a ranked list. This is de-novo structure prediction from a parent — **not** site-of-
metabolism (FAME/SMARTCyp), **not** spectral annotation (CASMI), **not** retrosynthesis.

## 2. Composition (clean, molecule-disjoint splits)

Substrate→metabolite annotations as triples `substrate_id\tmetabolite_id\tis_real` over an
SDF of indexed molecules; positives are annotated metabolites, negatives are rule-applicable
but unannotated products (positive-unlabeled).

| split | substrates | annotated (positive) substrate–metabolite pairs | total rows (incl. PU negatives) |
|---|---|---|---|
| train | 9,339 | 22,061 | 345,031 |
| val | 1,028 | 2,451 | 37,959 |
| test | 1,246 | 3,700 | 43,886 |

Files: `data/{train,val,test}.sdf` + `*_triples_clean.txt`. (The rule-bank coverage ceiling
was measured on the parseable test subset: 1865/2597 reachable = **0.718**.)

## 3. Leakage control

Splits are **molecule-disjoint**: no substrate in test/val appears in train, and no test
substrate appears in val. Built and audited by `scripts/fix_splits.py`, which canonicalizes
SMILES, removes cross-split substrates, and verifies `zero_substrate_overlap_between_clean_
splits` and `zero_positive_pair_overlap_train_test` (→ `leakage_fix_report.json`). Use
`DatasetConfig.use_clean_splits=True` (auto-selects `*_clean.txt`).

**Fairness caveat (must be reported).** Rule-based tools (SyGMa, GLORYx, BioTransformer) have
no ML "training set" — their fixed expert rules may encode knowledge overlapping any split,
while learned models are held to molecule-disjoint splits. No prior benchmark controls this
asymmetry; we surface it explicitly rather than hide it. Shared DrugBank/MetXBioDB provenance
across tools is a further uncontrolled leakage source.

## 4. Standardized matching protocol

A predicted structure matches a reference iff their **keys** are equal. We expose every
protocol the literature uses, as set keys (`grail_metabolism/metrics.py:_match_keys`):

| mode | key | used by | blind to |
|---|---|---|---|
| `exact` | canonical isomeric-free SMILES | LAGOM/MetaSense | nothing beyond canonicalization |
| `inchikey` | full InChIKey | strict | — |
| `inchi_no_stereo` | InChIKey skeleton (1st block) | GLORYx | stereo, charge, isotope |
| `tanimoto1` | identical Morgan(r2,2048) fingerprint | MetaTrans | stereo, charge; lenient to fp collisions |
| `inchikey_tautomer` | InChIKey after full tautomer canonicalization | **ours (recommended)** | stereo, charge, tautomer |

**Recommended default: `inchikey_tautomer`.** Rules routinely emit a different tautomer of
the reference than standard InChI normalizes; plain InChIKey loses ~4.5× recall on this
engine. Tautomer-canonicalizing both sides is the most defensible structure-identity match.
The **match-sensitivity experiment** (`scripts/run_match_sensitivity.py`) re-scores every
method's fixed predictions under all five modes and reports the rank-flip.

## 5. Metrics

Lead with, per substrate, macro-averaged over the test set:
- **recall@k** (k ∈ {5,10,12,15}) — primary; matches the literature's reporting.
- **mean_output_size** — always reported beside precision (precision is a pessimistic lower
  bound under incomplete annotation: an unannotated prediction is a false positive but may be
  a real, unrecorded metabolite).
- **coverage** — fraction of substrates with ≥1 true hit.
Precision@k and F1 are secondary. Report mean±std over ≥3 seeds for learned methods.

## 6. Cross-paper shared set

For comparability with published numbers we additionally evaluate on the **GLORYx external
set (37 drugs / 136 metabolites)** reused by GLORYx, MetaTrans, MetaPredictor, LAGOM. *To be
added* (source the structures + each method's published predictions); drops into the
match-sensitivity engine as prediction files without changing it.

## 7. Baselines & leaderboard

Each method contributes a ranked-prediction file `{substrate_smiles: [pred_smiles…]}`;
scored uniformly (canonical dedup → top-k → match mode). Current: GRAIL (from its exported
CSV), SyGMa (generated via the `sygma` package). Planned: GLORYx, BioTransformer, MetaTrans,
MetaPredictor, LAGOM. GRAIL is one fair entry, not the headline — the contribution is the
protocol + the rank-flip + the limit decomposition (see `diagnostics.md`).

## 8. Reproducibility

Code: this repo (rule bank `resources/extended_smirks.txt`, 7,581 SMIRKS). Seeds via
`ExperimentConfig.seed` (recorded in `reports/metrics.json`); data subsampling via
`sampling_seed`. Match/ceiling/gap reproducible from `scripts/run_benchmark.py` and
`scripts/measure_coverage.py`.

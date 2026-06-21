# Site-of-Metabolism (SoM) regioselectivity ranking prior — design

Date: 2026-06-21
Status: approved (brainstorming) → implementation

## Problem

GRAIL's learned generator converts only a small fraction of the rule bank's recall
ceiling into top-k recall. Controlled measurement (245 test, max_output=15, canonical
generation, 2026-06-20): generator-only recall@15 ≈ 0.10, rank-only ensemble ≈ 0.141,
vs a rule-bank ceiling of 0.718 and SyGMa 0.558. Gap analysis attributes much of the
miss to **regioselectivity**: the right reaction applied at the wrong site yields a
product whose InChIKey does not match the annotated metabolite. The generator scores
*rules per substrate* but does not distinguish *where* on the substrate a rule fires, so
two products of the same rule at different atoms are ranked identically.

A **site-of-metabolism (SoM) prior** that scores how likely each substrate atom is the
reaction site lets us reweight candidates by site plausibility, directly attacking this
gap, without touching the generator.

## Decisions (brainstorming)

1. **Signal source:** learned from our own annotated pairs (no external dependency).
2. **Integration:** standalone lightweight head now, structured so it can later fold into
   joint generator training (joint-ready hook).
3. **Labels:** reacting atoms (MCS-diff between substrate and metabolite) **+ 1-hop
   neighbors**.
4. **Injection (Approach A):** per-product reweight; reacting atoms at inference are
   recovered by `MCS(substrate, product)` — the *same* routine that produced the labels,
   so there is no train/inference skew. Soft multiplicative reweight, never a hard gate.

## Components

New module `grail_metabolism/model/som.py`:

- `derive_som_labels(sub_smiles, met_smiles) -> set[int]`
  `rdFMCS.FindMCS([sub, met], atomCompare=CompareElements, bondCompare=CompareAny,
  ...)` (same params as `transform.from_pair`). Substrate atoms **not** in the MCS
  substructure match = reacting atoms; union their 1-hop neighbors. Return substrate atom
  indices. Degenerate cases (no/empty MCS, or MCS == whole substrate) → empty set (skip).
  Multiple metabolites for one substrate → union of per-pair reacting sets.

- `SoMPredictor(nn.Module)`
  Wraps a `GraphEncoder` over single-graph atom features (`SINGLE_NODE_DIM = 16`) and a
  node head `Linear(hidden, 1)` on `encoder.forward_nodes(data)` → per-atom logit.
  `score_atoms(smiles) -> np.ndarray` returns per-atom sigmoid probabilities (cached per
  substrate). Persists `arch` (encoder config + hidden dim) like other checkpoints so it
  can be reconstructed.

- `build_som_dataset(molframe) -> list[(Data, label_vector)]`
  For each substrate in `molframe.map`: `transform.from_rdmol` single-graph + a 0/1 node
  label vector from `derive_som_labels` over its metabolites. Atoms with no label = 0.

## Training

`scripts/train_som.py` (standalone):
- Build dataset from the **train** split MolFrame.
- Train `SoMPredictor` with node-level **BCE** for a few epochs. (PU caveat: a "0" atom
  may be the site of an *unannotated* metabolite — same positive-unlabeled nature as the
  rest of GRAIL. The prior only needs to *rank*, not be calibrated, so plain BCE is
  acceptable; revisit PU-weighting only if it underperforms.)
- Validate per-atom **ROC-AUC** on the val split.
- Save `artifacts/<run>/checkpoints/som.pt` with `state_dict` + `arch`.
- Cost: node-level, one forward per substrate per epoch → seconds/epoch on the 400
  subset, minutes/epoch on full 9125. CPU-friendly.

## Inference injection (ranking layer — generator untouched)

`ModelWrapper` gains an optional `som` predictor (default `None` → off).
In `generate(...)`, after candidates have generator + filter scores:
- For each candidate, recover reacting substrate atoms via `MCS(sub, candidate)`.
- `som_score = aggregate(som_atoms[reacting])`, `aggregate ∈ {max, mean}` (default max);
  if no reacting atoms → neutral `1.0` (no reweight, no penalty).
- `combined = filter_score × generator_score × som_score**β`.
- Ranking, dedup, `max_output` cap unchanged (still rank-only; SoM is a reweight, never a
  gate).

`β` is the strength knob. **β = 0 ⇒ `som_score**0 = 1` ⇒ byte-identical to current
behavior** — clean ablation + back-compat. MCS at inference is bounded to the candidates
actually ranked and cached by `(sub, prod)`.

## Config

- `SoMConfig{ enabled=False, beta=1.0, aggregation="max", checkpoint=None, hidden_dim }`.
- Attach as `EvaluationConfig.som`; round-trip in `experiment_from_dict`.
- `ModelWrapper.generate(..., som_beta=None)` explicit override (None → 0 / off).
- `factory.build_som(SoMConfig)` — single construction path (mirrors build_generator/filter).

## Evaluation

- `reeval_ranking.py` gains `--som-ckpt` / `--som-beta`.
- **Tune β on val** (small sweep, e.g. {0.5, 1, 2, 4}); select best by val recall@k.
- Then a single run on the same 245 test for the report.
- Success = recall@k lift over the rank-only baseline (0.141@15 tautomer). Select on val,
  touch test once (project invariant).

## Testing (dataset-free)

`grail_metabolism/tests/test_som.py`:
1. `derive_som_labels` on ethanol→acetaldehyde returns the oxidized carbon (+ neighbors).
2. `SoMPredictor.score_atoms` returns a vector of length = #atoms, values in [0, 1].
3. **β = 0 ⇒ identical ranking** to no-SoM (back-compat).
4. A high-SoM atom promotes its product above an equal-generator product at a low-SoM site.
5. Inference MCS reacting-set == label-derivation reacting-set on the same pair (no skew).

## Files

- New: `grail_metabolism/model/som.py`, `scripts/train_som.py`, `grail_metabolism/tests/test_som.py`.
- Modified: `grail_metabolism/config.py` (`SoMConfig`), `grail_metabolism/model/wrapper.py`
  (optional `som` + reweight), `grail_metabolism/workflows/factory.py` (`build_som`),
  `scripts/reeval_ranking.py` (`--som-ckpt/--som-beta`), `CLAUDE.md` (note).

## Out of scope (YAGNI)

- Joint generator+SoM training (only the structural hook is built now).
- PU-weighted SoM loss (plain BCE first; revisit only if it underperforms).
- External SMARTCyp/CYP heuristics.

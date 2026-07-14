# GRAIL factorized-generative redesign — design spec

> **Status:** design (2026-07-14). Approved architecture from the adversarial design panel
> (workflow `wfpc1fukk`, 11 agents) + user green-light. Feeds `writing-plans`. Replaces GRAIL's
> Stage-1 PU rule-selector with a dense maximum-likelihood factorized generator; keeps the pair
> filter; adds precision-calibrated abstention. Prototype gates A (GO) and C (GO, qualified)
> already validate the premise.

## Motivation (the diagnosed failure the redesign fixes)
GRAIL deployed recall@15 = 0.330 (SyGMa 0.572). The exact decomposition puts the dominant loss on
`selection_retention = 0.489`, and **Proposition 2** proves the learned rule-selector (0.266) loses
to a trivial frequency prior (0.410): the generator predicts over **7,581 rules with a ~99.9%-zero
per-substrate label**, so under PU the Bayes-competitive estimate IS the marginal `π(r)`. The rule
bank is 73% radius-1 singletons (memorization). **The generator asks an ill-posed question.**

Two no-training gates already confirm the fix direction on the existing data:
- **Gate A (GO):** radius-0 reaction TYPES are dense at the pair level — only **13.3%** of train
  pairs fall in single-pair types (vs 73% radius-1 rule-singletons); top types are enzyme
  mechanisms (C–O hydroxylation 3,782 pairs, sulfation, carbonyl reduction). `results/redesign_gate_a.json`.
- **Gate C (GO, qualified):** the existing trained SoM head puts the true reacting atom in **top-3
  for 80.7%** of test substrates vs a 0.649 random baseline (permissive labels; real lift +0.16).
  `results/redesign_gate_c.json`.

## Goal
Replace the 7,581-way PU rule-selector with a **factorized, densely-supervised generator**
`P(type | s) · P(site | type, s) · P(op | type, site, s)` trained by ordinary maximum likelihood,
apply the selected transformations **broadly** (not site-gated), rank the resulting products with
the **existing, unchanged pair filter**, and emit a **precision-calibrated, variable-size** output
(abstention). Target: **0.42–0.50 recall@15 at multiples-higher precision** than current GRAIL,
with a defensible dense-supervision architecture on which GFlowNet later becomes meaningful.

Non-goal: beating SyGMa's 0.572 / GLORYx's 0.77 — the 0.735 rule-bank ceiling + the out-of-bank
tail (glucuronides, GSH conjugates, multi-step) cap recall and require a broader rule bank (out of
scope). The win is a well-posed learner + a precision axis + interpretable, atom-localized outputs.

## Architecture

### Reaction-type vocabulary + rule→type map (offline)
- Extract a **radius-0 transformation type** per mined rule / train pair from the reaction center
  (bond-level broken/formed pattern, element-typed, periphery stripped) — reuse
  `scripts/mine_rules.py:find_reaction_center` and the signature logic already in
  `scripts/redesign_gate_a.py`. Collapse the crude per-bond signature into a **canonical ~100–200
  type vocabulary** (merge multi-bond/ring over-splits; cap the tail by pair-support).
- Emit `resources/coarse_type_vocab.json` (type_id → canonical signature + representative SMIRKS
  set) and a `rule_id → type_id` map, mirroring the `MINE_OUT_SUFFIX` pattern so the deployed
  `extended_smirks.txt` is never clobbered.

### Dense training targets (offline, per train pair)
- `y_type[t] = 1` iff a true metabolite is produced by any rule in type `t` (dense: ~10× the
  current per-column positive rate).
- `y_site`: per-atom 0/1 over the substrate = the reacting atoms of the true transformation for
  each positive type — reuse `som.derive_som_labels` / `som.build_som_dataset` / `som._reacting_atoms`.
- `y_op`: the concrete operation applied at the site (a small discrete set derived from the type),
  so the metabolite is reconstructed by applying `op` at `site` (scaffold copied by construction).

### Model (three heads on the shared GraphEncoder)
- Reuse `model/_graph.py:GraphEncoder` (16-dim single-graph nodes). Add:
  - `type_head`: graph-level softmax over the type vocabulary → `P(type | s)` (a ~100-way softmax
    with real per-class support — the 99.9%-zero PU vector is DELETED).
  - `site_head`: per-atom head conditioned on the chosen type → `P(site | type, s)` (dense BCE;
    the existing SoMPredictor head is the unconditioned special case).
  - `op_head`: small softmax → `P(op | type, site, s)`.
- Objective: ordinary **maximum-likelihood / cross-entropy**, NOT PU. Unobserved types contribute
  to the softmax normalizer, not a zero-binary — the precise condition Prop 2 says is needed to
  have a different asymptote than the marginal.
- Efficiency: dropping the per-forward re-encode of 7,581 rule graphs makes this **cheaper** than
  the current generator on CPU.

### Inference
- For a substrate: rank types by `P(type|s)` (keep those above a threshold / top-N types),
  enumerate products by applying each kept type's rules **broadly** at the substrate (RDKit), and
  rank the candidate pool by `P(type)·P(site)·P(op)·filter_score`. The filter is **reused frozen**
  (broad application + the learned filter already reaches 0.413 / the reranker 0.500 with NO
  generator help — the factorized selector only has to help, not carry).
- **Precision-calibrated abstention:** add `target="precision"` / `recall_at_precision` to
  `Filter.calibrate_threshold` (currently only `f1`/`mcc`) → a variable-size output that stops at a
  chosen precision, instead of a fixed top-15.

## Stages (feed writing-plans)
- **Stage 1 — vocabulary + labels (offline, no training):** canonical type vocab + `rule→type` map
  + per-pair `(type, site, op)` records. Reuses mine_rules + som. Emits the vocab/map/dataset.
- **Stage 2 — model + MLE training:** the three heads on `GraphEncoder`; train by CE/BCE.
  Validation gates (must pass to proceed): `P(type|s)` **beats the type-frequency prior** on val
  (this is the direct Prop-2-escape test — the number Gate D could only floor-test); `P(site|type,s)`
  holds the Gate-C top-3 bar on val.
- **Stage 3 — inference + precision + eval:** broad type-matched application → filter-ranked pool →
  precision-calibrated abstention. Wire through **`EnsembleWorkflow.run_bundle`** (the single
  chokepoint) behind a config flag; add a guard test beside `tests/test_audit_fixes.py`. Report on
  the FULL clean test (tautomer-InChIKey): **recall@15 vs 0.330 baseline / 0.413 broad-baseline /
  0.572 SyGMa**, AND the **recall-at-fixed-precision PR frontier vs SyGMa** (directly answers the
  "we ignored precision" critique).
- **Stage 4 — fallback (only if the site head under-delivers):** ship the coarse-type vocabulary as
  a **rule-bank consolidation** for the existing generator/filter (fewer, less-specialized rules,
  cheaper forward) + keep the precision-calibrated abstention win. Guarantees a positive return.

## Decision gate (go/no-go, inside Stage 2/3)
The single decisive number: **learned factorized selection beats the 0.413 broad+filter baseline on
the clean test with paired-bootstrap CI separation.** If it beats 0.413 (ideally trends toward the
0.500 reranker), build the full pipeline + report the recall lift. If it merely ties 0.413, the
recall story collapses to "broad application + existing filter" (already shipped) and the
contribution becomes **precision-only** (the abstention frontier) — still shipped, but reframed.

## Global constraints (binding)
1. **Existing data only** (~4,787 train substrates, no new labels). No external rule bank import.
2. Route all wiring through `EnsembleWorkflow.run_bundle`; construct models via `workflows/factory.py`.
3. Never clobber `resources/extended_smirks.txt` or the deployed checkpoints; new artifacts use
   suffixed paths.
4. **Select on val, touch test once** (CLAUDE.md). Report headline as the deployed protocol
   (tautomer-InChIKey, macro recall@15) + the PR frontier.
5. `make test` stays green; add guard tests for the new heads + the precision-calibration target.
6. Preserve interpretability: every prediction stays an RDKit-applied SMIRKS product with atom-level
   provenance ("aromatic hydroxylation at atom 7").
7. **No Claude/AI/Co-Authored-By attribution** in any commit or doc byline.

## Out of scope (later)
- Importing SyGMa/GLORYx SMIRKS to raise the 0.735 coverage ceiling.
- A GFlowNet set-generator over the factorized policy (only meaningful once the base is well-posed).
- Multi-step (depth-2) application (measured +0.012, not the lever).

## Verification
- Stage gates as above (type-head > type-prior on val; site top-3 on val; recall@15 > 0.413 on test
  with paired CI; PR frontier vs SyGMa).
- `make test` green; guard test for the factorized heads + precision calibration.
- Honest reporting: subset vs full-test clearly labeled; no recall-win claim vs SyGMa; the
  fallback (Stage 4) documented so a null result still ships value.

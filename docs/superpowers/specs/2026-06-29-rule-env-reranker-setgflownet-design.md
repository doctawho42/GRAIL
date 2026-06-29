# Stage 2 design: a regioselectivity-aware reranker over the rule pool, then a Set-GFlowNet

**Status:** design (brainstorming output) · **Date:** 2026-06-29 · **Author:** GRAIL team

## 1. Goal and evidence

Keep GRAIL's **rule environment** (RDKit applies SMIRKS; every candidate is a valid,
interpretable rule product) and reach **SOTA-competitive recall**, then add a **novel
GFlowNet set-generation method**. Two inference-only spikes settle the strategy
(`docs/benchmark/stage2_ranker_evidence.md`, `scripts/diagnose_ranker.py`,
`scripts/diagnose_rerank_ceiling.py`):

- **Rank-by-rule is capped ~0.38** and is *not* fixable within the paradigm: coverage
  (ceiling 0.718), bank size/noise (pruning only hurts), and learned-vs-prior (the GNN score
  adds ≈0 over the empirical prior) are all ruled out. The residual is **within-rule
  ranking** — one rule fires at many sites, the regioisomer products share a single per-rule
  score, so the score cannot pick the correct site.
- **The truth is in the pool, just mis-ranked.** Oracle recall@15 of a perfectly reranked
  top-N generator pool: N50 **0.516**, N100 **0.573** (> SyGMa 0.558; > MetaPredictor 0.504),
  N200 0.617. The entire 0.385→0.573 gap is ranking *within* a budget-100 pool.

**Therefore:** a strong **product-level reranker over a larger generation budget (~100)** is a
path to SOTA-competitive recall *without leaving the rule env*. A 12-agent design panel
converged on a **site-conditioned, regioisomer-contrastive reranker**, with the site signal
drawn from a stronger source than the already-neutral SoM head (contextual rule-match
embeddings + analogue retrieval + a contrastive loss defined *over the pool*).

## 2. Success criteria

- **Stage 2a (reranker, primary):** on the clean test split, recall@15 (tautomer-InChIKey,
  rule-env preserved) **≥ 0.50**, i.e. competitive with SyGMa (0.558) / MetaPredictor (0.504)
  and a large jump over the 0.385 rank-by-rule baseline. Selected on val, test touched once,
  mean±std over ≥2 seeds.
- **Stage 2b (Set-GFlowNet, follow-on):** a measurable **diversity/coverage advantage** at
  matched output budget over the reranker / temperature-sampling / beam, with recall no worse
  than the reranker — the novel-method contribution.
- Non-goals: beating every method on every protocol; leaving the rule env.

## 3. Stage 2a — the reranker (build first)

### 3.1 Candidate generation with provenance (small generator change)
`generate_scored` currently collapses a rule's regioisomer products into a per-SMILES dict and
aggregates `rule_score` (generator.py ~1160–1191) — discarding exactly the **rule-id + firing
site** the fix needs. Add a flagged path that returns, per candidate: `(smiles, gen_score,
rule_id, firing_site_atoms)`. The firing site = the rule template's matched substrate atoms
(fallback: `som._reacting_atoms(sub, prod)`). Raise the candidate budget to **~100** (the
oracle headroom needs it; current cap=32). **Invariant:** the public `generate()` signature and
the MCS-alignment in `from_pair` are unchanged; provenance is additive and behind a flag.

### 3.2 Reranker model (new module `model/reranker.py`)
A site-conditioned cross-encoder scoring `r(substrate, product, site)`:
- **Backbone:** reuse the pair-filter representation `transform.from_pair` (the element-aware
  MCS substrate↔product cross-edge graph) + `model/_graph.GraphEncoder` — no new featurizer.
- **Site channel:** add one substrate-node input feature = firing-site indicator (PAIR_NODE_DIM
  18→19, or reuse existing per-node slack), so the encoder *sees* which site this sibling
  modifies. This is the key difference from the neutral additive SoM prior: the site enters as
  input to a discriminative model, not as a post-hoc multiplier.
- **Contextual rule-match embedding (CRX):** concatenate an embedding of the rule encoded *in
  context* of its match (matched atoms + local environment), not the generic rule graph.
- **Analogue-retrieval feature (RNS, optional/ablated):** a per-candidate feature from k-NN
  analogue substrates' observed sites (transferred via MCS) — a site signal independent of the
  GNN, the panel's hedge against "same neutral encoder."

### 3.3 Training objective (new `workflows/reranker.py`)
Per substrate, assemble candidate **groups** with provenance, then optimize:
- **Within-rule sibling-contrastive (the core term):** among one rule's regioisomer siblings,
  the annotated metabolite is the positive and its wrong-site siblings are *hard negatives*;
  softmax cross-entropy / InfoNCE over the sibling group → the model must pick the **site**.
- **Cross-substrate listwise:** a LambdaRank/ListNet recall@k surrogate over the full pool →
  calibrates inter-rule order, replacing the noisy empirical prior as the cross-rule arbiter
  (handles the cross-rule share of the headroom; see M0).
- **PU-aware:** unobserved-applicable products are positive-unlabeled, not certain negatives —
  reuse the nnPU treatment / `MolFrame.negs` weighting; never treat them as hard negatives in
  the listwise term.
- Auxiliary SoM-atom head shares the encoder (free labels from `derive_som_labels`) to
  regularize the site channel.

### 3.4 Inference
Generator emits the budget-100 pool (rule env intact) → reranker scores → top-15. The reranker
replaces the neutral filter as the *ranker*; the filter may stay as a coarse pre-gate. Route
through `ModelWrapper.generate` / `EnsembleWorkflow.run_bundle` (the single chokepoint).

### 3.5 Validation milestones (Stage 2a)
- **M0 (de-risk, first):** with provenance plumbing in place, decompose the 0.385→0.573
  headroom into **within-rule** (reorder only siblings, keep generator's rule order) vs
  **cross-rule** (reorder only across rules). Confirms how much the site-contrastive term vs the
  listwise term must carry. Go/no-go evidence before training the full model.
- **M1 (small):** train on ~200–500 substrates; reranker val recall@15 beats the 0.385 baseline
  and approaches the oracle headroom. CPU/short GPU.
- **M2 (scale-up):** one larger run (~2–5k substrates); headline recall@15 ≥ 0.50 on val,
  reported on test once, ≥2 seeds. Colab GPU.

## 4. Stage 2b — Set-GFlowNet (follow-on, after the reranker lands)
Cast the reranker as the forward policy of a **(rule×site)-factored Set-GFlowNet**: actions =
`apply rule r at site s` (so the policy is regioselectivity-aware by construction); terminal =
a metabolite **set**; reward = **PU-aware set-coverage** (TP − λ·|S|, `R=exp(β·score)`); loss =
Trajectory Balance with uniform backward `P_B = 1/|S_t|` (set-DAG orderings analytic) reusing
the existing `log_z`. This reuses the scaffolded `GFlowNetTrainer`, `action_distribution`,
`stop_head`. **Eval:** recall@k + set-coverage + diversity (#distinct InChIKey modes, pairwise
Tanimoto) + set-size calibration, vs reranker / temperature-sampling / beam at matched budget.
This is the novel-method/diversity contribution; the reranker (3) secures the recall first.

## 5. Components: reused vs new
- **Reused:** `generate_scored` + rule application (provenance added), `from_pair`/MCS pair
  graph, `GraphEncoder`, `model/filter.py` body, `som._reacting_atoms`/`derive_som_labels`,
  PU/`MolFrame.negs`, tautomer-InChIKey metrics, val-selection harness, rule-encoding cache,
  `GFlowNetTrainer`/`action_distribution`/`stop_head` (Stage 2b), `EnsembleWorkflow.run_bundle`.
- **New:** provenance path in `generate_scored`; `model/reranker.py`; `workflows/reranker.py`
  (group assembly + contrastive+listwise+PU loss); one pair-node input channel; `scripts/
  run_reranker.py`; reranker eval; (2b) `(rule×site)` set actions + set-coverage reward +
  `scripts/run_gflownet.py` + set eval. Guard tests in `tests/` for: provenance correctness,
  MCS-alignment preserved, sibling-group construction, PU weighting, within/cross decomposition.

## 6. Compute (validate-small → one scale-up; Colab Pro+)
M0/M1 are cheap (CPU + short GPU). M2 and Stage 2b run on Colab GPU via a self-contained
notebook (data symlinked/mounted; checkpoints + eval JSON saved and downloaded), mirroring the
MetaPredictor notebook. RDKit rule application is CPU-bound → precompute candidate pools +
provenance once and cache (pool generation is the bottleneck, as the oracle spike showed).

## 7. Risks and mitigations
- **"Same neutral encoder" risk** (the panel's top concern): the site signal must not be only
  the GNN that gave a neutral SoM. Mitigate with the **contextual rule-match embedding** + the
  **analogue-retrieval feature** + a **contrastive loss over the pool** (discriminative, against
  siblings) — and gate on **M0/M1** before the full build.
- **Cross-rule share of the headroom:** if M0 shows much of the gap is cross-rule (not
  sibling), lean the listwise term harder; the architecture already includes it.
- **Provenance plumbing must not regress** the MCS-alignment or `generate()` API — additive,
  flagged, guard-tested.
- **Stage 2b variance** (set-DAG): analytic uniform `P_B`, small pools, sub-trajectory balance
  if needed — and it only starts after the reranker secures recall.

## 8. Honest framing
The benchmark (Stage 1) is the standalone strong A*/Q1 result. Stage 2a's contribution is a
**diagnosis-plus-fix**: rank-by-rule over mined rules is provably tied across regioisomer
siblings (a citable negative result), and a regioselectivity-aware contrastive reranker recovers
what curated calibrated probabilities (SyGMa) and FAME priors (GLORYx) buy — reaching their
recall from 7,581 noisy mined rules, with full rule interpretability. Stage 2b adds the novel
Set-GFlowNet method (diversity/coverage). Recall claims are bounded by the measured oracle
ceiling (0.573@100), reported on val-selection with seeds.

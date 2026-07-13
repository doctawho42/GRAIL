# GRAIL: rule-based metabolite-structure prediction, a coverage×selection×ranking diagnosis, and the TAME evaluation protocol

> **Draft status (2026-07-13):** first assembled full draft. Numbers sourced from `docs/GRAIL_FRAMING.md` / `results/*.json`. Compute-gated values are marked `[PENDING: ...]`; unverified citations are marked `[cite: ...]`. Venue target: JCIM / J. Cheminformatics.

## Abstract
> _[STUB — Task 11]_

## 1. Introduction
> _[STUB — Task 11]_

## 2. Related Work
> _[STUB — Task 10]_

## 3. Methods — GRAIL architecture

GRAIL predicts xenobiotic metabolite structures with a three-stage, rule-based-plus-learned
pipeline. The first stage, the **generator**, is a learned multi-label rule selector that
approximates `P(r|s)` — the probability that rule `r` applies to substrate `s` — over a curated
bank of **7,581 SMIRKS** rules; the default scorer is retrieval-based, combining cross-attention
between substrate and rule graph embeddings, an embedding-similarity term, and an MLP head. The
second stage, **RDKit rule application**, mechanically applies every rule the generator selects
and enumerates the resulting candidate products, retaining provenance of which rule produced which
candidate. The third stage is a **PU-trained, MCS-aware pair filter**: a binary classifier scoring
each (substrate, product) pair. Its training data are positive-unlabeled — annotated true
metabolites are the only positives, while rule-applicable products lacking a positive annotation
are treated as *unlabeled*, not as confirmed negatives, since absence of an annotation does not
certify a transformation does not occur. The filter is accordingly trained in the logit domain
(`return_logits=True`) so a PULoss/nnPU surrogate operates on raw classifier outputs rather than
post-sigmoid probabilities, and the generator itself down-weights unobserved-but-applicable rules
rather than penalizing them as hard negatives. Featurization uses fixed-width graphs: single-molecule
graphs feed the generator encoder with 16-dim node features, while the pair filter operates on
merged substrate–product graphs with 18-dim nodes and 18-dim edges plus a 1024-dim Morgan
fingerprint branch; cross-edges linking substrate and product atoms come from an element-aware
maximum common substructure (MCS) atom correspondence rather than sorted or arbitrary indices,
preserving chemically meaningful atom mappings across the reaction. At deployment,
`ModelWrapper.generate` runs all three stages and ranks the candidate set by
`filter_score × generator_score`, combining the filter's pair-plausibility judgment with the
generator's rule-selection confidence into a single ranking signal. Throughout this paper, unless
stated otherwise, structure matching between predicted and reference metabolites uses
`inchikey_tautomer` as the default match mode. The three stages together form an interpretable
instrument — the selected rule, the enumerated product, and the filter's pair judgment are each
inspectable — and the contribution we claim is interpretable learned rule selection paired with a
PU-aware pair filter, not recall supremacy over other metabolite predictors.

## 4. Methods — Formal framework

The architecture of §3 and the coverage ceiling and recall decomposition reported later (§6, §8)
are unified by one compact generative model, which also gives the three propositions of §10 a
principled home. The framework is deliberately thin: one likelihood, one decomposition, one
lever→factor map — it is not a rewrite of the empirical results that follow.

**A generative latent-reaction mixture.** For a substrate `s`, a metabolite `m` arises by choosing
a transformation rule `r`, choosing a firing site, and applying the rule:

```
P(m | s) = Σ_r Σ_{site ∈ sites(r,s)} P(r | s) · P(site | r, s) · 𝟙[ apply(r, s, site) = m ]
```

The application term is deterministic given RDKit, so the inner indicator collapses to the
enumerable product support; the latent variables are *which rule fired* and *at which site*, and
the observation — the annotated metabolite set — is positive-unlabeled: a rule-applicable,
non-annotated product is unlabeled, not a certified negative. The three deployed stages of §3 are
one marginal-likelihood approximation of this model — the model the pipeline approximates, not a
claim that the trained weights are its MLE. The **generator** approximates `P(r | s)` (rule
selection); its persisted `rule_prior_logits` is the marginal `π(r) = P(r fires)`. **RDKit rule
application** realises the deterministic support `𝟙[apply(r,s,site)=m]`, enumerating candidate
products. The **filter** approximates a discriminative correction `P(true | s, m)` over the
enumerated candidates. Deployment ranks by `filter_score × generator_score`. PU training
approximates EM over the unobserved rule-firing indicator.

**A recall decomposition.** Fix top-k. For substrate `s`, let `T_s` be the true references and let
the candidate sets nest `R_s(k) ⊆ P_bud,s ⊆ P_full,s`, where `P_full,s` is the full-bank depth-1
rule-applicable pool (the coverage ceiling of §6), `P_bud,s` the deployed budget-limited pool after
generator selection, and `R_s(k)` the top-k ranked output. With `hit(A_s)` the number of references
matched by set `A_s` under the tautomer-InChIKey quotient (monotone under inclusion) and pooled
(micro) sums `U=Σ|T_s|`, `C_full=Σ hit(P_full,s)`, `C_bud=Σ hit(P_bud,s)`, `H=Σ hit(R_s(k))`:

```
recall@k = H/U = (C_full/U) · (C_bud/C_full) · (H/C_bud)
                = coverage_bank · selection_retention · ranking_conversion
```

Because the sets nest, each factor lies in [0,1] and the three multiply out exactly to the
realised recall on any numbers — it is an accounting identity, not a theorem; its only use is to
localise where recall is lost across the pipeline, and its exact cancellation is never offered as
evidence for anything beyond that bookkeeping. Qualitatively, `coverage_bank` coincides with the
rule-bank ceiling reported in §6.

**Lever → factor map.** Every diagnostic in §10 and each proposition attaches to exactly one
factor:

| factor | levers & propositions |
|---|---|
| `coverage_bank` | multi-step rule application (depth-2), the ΔMW coverage gap, the external-validity composition covariate (§7), Proposition 3 |
| `selection_retention` | the learned-vs-prior rule-selection probe, data-scaling saturation, Proposition 2 |
| `ranking_conversion` | the filter / listwise reranker, the oracle bound, Proposition 1 |

## 5. Methods — TAME evaluation protocol

We package the evaluation apparatus as **TAME** — the **T**automer-**A**ware
**M**etabolite-structure **E**valuation protocol: a tautomer-InChIKey match quotient, a
leakage-audited molecule-disjoint train/val/test split, and a frozen multi-method re-scoring
harness. TAME is a *protocol + audited split + re-scoring harness*, not a leaderboard service —
it exists so the numbers in §3–§4 and §6–§9 are apples-to-apples and regenerable from committed
artifacts, not to rank community submissions.

**Matching protocol.** The field has never agreed on what counts as a "match": GLORYx uses a
stereo-blind InChIKey skeleton, MetaTrans uses Morgan-fingerprint Tanimoto=1, and LAGOM uses
canonical SMILES equality. TAME exposes each as an explicit, named quotient rather than picking
one silently: `canonical` (stereo-free canonical SMILES equality), `inchikey` (the
literature-standard full InChIKey), `inchi_no_stereo` (the InChIKey skeleton block, stereo-blind,
as GLORYx uses), `tanimoto1` (identical Morgan fingerprint, as MetaTrans uses), and
`inchikey_tautomer` (tautomer-canonicalize both predicted and reference structures before taking
the InChIKey) — **our recommended default**. Plain InChI only normalizes a *subset* of tautomers,
so a rule-emitted tautomer of a reference routinely fails to match under plain `inchikey` even
though it is the same chemical entity; tautomer-canonicalizing both sides first closes that gap
and is the most defensible structure-identity criterion for a rule-driven generator. Unless
stated otherwise, §3–§4 use `inchikey_tautomer`; §11 quantifies how much the method ranking
depends on which of the five quotients is used.

**Split and leakage audit.** All splits are molecule-disjoint: no substrate in test or val
appears in train, and no test substrate appears in val. The clean triples are built and verified
by `scripts/fix_splits.py --molecule-disjoint`, which canonicalizes SMILES, removes cross-split
substrates, and checks zero substrate and zero positive-pair overlap, emitting its audit summary
to `results/leakage_fix_report.json` when run. Trustworthiness is corroborated empirically, not
merely asserted: recall@15 tracks closely between val and test (0.327 vs 0.330), inconsistent
with test-set overfitting.

**Metrics.** We lead with recall@k, co-reported with `mean_output_size` rather than precision
alone, since precision is a pessimistic lower bound under incomplete annotation — an unannotated
candidate is scored a false positive even though it may be a real, simply unrecorded, metabolite.
Selection of models, presets, and hyperparameters happens exclusively on the validation split;
the test split is touched once, for the final reported numbers.

**Worked example.** The quotient dependence is not hypothetical. Consider a predicted set
`{D-alanine, acetone (enol tautomer)}` scored against annotated references `{L-alanine, acetone
(keto tautomer)}`. Under strict `inchikey`, both predictions miss: the alanine pair differs at a
stereocenter, and the acetone pair differs only by a keto/enol tautomerization outside the subset
plain InChI normalizes, so it also misses under `inchi_no_stereo`. Once both sides are
tautomer-canonicalized under `inchikey_tautomer`, the acetone pair becomes a hit. The same fixed
predictions can therefore score as a near-total miss or a hit purely by changing the match
quotient, with no change to the underlying chemistry — the phenomenon §11 quantifies across
methods and the shared external set.

## 6. Results — Rule-bank coverage ceiling
> _[STUB — Task 5]_

## 7. Results — External validity of the ceiling
> _[STUB — Task 5]_

## 8. Results — Recall decomposition
> _[STUB — Task 6]_

## 9. Results — Honest-anchor certification
> _[STUB — Task 7]_

## 10. Results — Diagnosis: levers and three propositions
> _[STUB — Task 8]_

## 11. Results — Match-sensitivity and cross-method comparison
> _[STUB — Task 9]_

## 12. Limitations
> _[STUB — Task 12]_

## 13. Data & Code Availability
> _[STUB — Task 12]_

## 14. Conclusion
> _[STUB — Task 12]_

## Figure 1 — pipeline schematic
> _[FIGURE 1: GRAIL 3-stage pipeline schematic — TO BUILD]_ A left-to-right schematic: (i) substrate + 7,581-rule SMIRKS bank → learned retrieval-scored **generator** selecting rules; (ii) **RDKit rule application** enumerating candidate products; (iii) **PU-trained MCS-aware pair filter** scoring (substrate, product) pairs; deployment ranks by `filter_score × generator_score`. Real schematic is a post-draft task.

## Draft TODO / open items
> _[STUB — Task 12 seeds this; final content is the out-of-scope track-list]_

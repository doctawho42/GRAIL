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
it exists so the numbers in §6–§9 are apples-to-apples and regenerable from committed
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
plain InChI normalizes, so it also misses under `inchi_no_stereo`. `inchikey_tautomer`
standardization additionally strips stereochemistry (canonical SMILES generated with
`isomericSmiles=False`), so tautomer-canonicalizing both sides collapses both gaps at once: the
acetone keto/enol pair becomes a hit *and* the D-/L-alanine stereocenter pair becomes a hit,
scoring full recall on this example. The same fixed predictions can therefore score as a
near-total miss or a full hit purely by changing the match quotient, with no change to the
underlying chemistry — the phenomenon §11 quantifies across methods and the shared external set.

## 6. Results — Rule-bank coverage ceiling

The rule bank's **recall ceiling** — the best achievable recall if rule selection and ranking
were both perfect, i.e. the full-bank depth-1 `apply_rules` pool matched against the annotated
references — on the full clean molecule-disjoint test split (1170 substrates, 2597 true pairs) is
**0.718 (plain InChIKey) / 0.735 (tautomer-InChIKey)**, recovering **1865 / 1910 of 2597** true
pairs. This is far above the best learned end-to-end systems on this benchmark (recall@15 roughly
0.47–0.585) and comparable to other rule-based systems evaluated on their own sets. **Coverage is
high; the bank is powerful.** The practical consequence is that the GRAIL task is limited by
*coverage-conversion* — a dominant selection loss followed by a ranking loss, quantified in §8 —
not by rule expressiveness.

Both ceiling figures are reported under the **same tautomer-InChIKey protocol used for GRAIL and
SyGMa** throughout this paper (no mixed match modes across methods). The plain-InChIKey figure,
0.718, exactly reproduces the previously reported ceiling, which validates that this run of
`scripts/run_benchmark.py` is consistent with earlier measurements. The same run co-measures the
**SyGMa** baseline on the identical split under both protocols: recall@15 **0.558 plain / 0.572
tautomer** (n=1168 substrates with SyGMa-scoreable output) — so the rule-bank ceiling, SyGMa, and
GRAIL are now all reported under one shared tautomer-InChIKey standard.[^tautomer-audit]

| system | protocol | recall | n |
|---|---|---|---|
| **rule-bank ceiling** (full bank, depth-1) | tautomer-InChIKey | **0.735** (1910/2597) | 1170 |
| rule-bank ceiling (full bank, depth-1) | plain InChIKey | 0.718 (1865/2597) | 1170 |
| SyGMa (recall@15) | tautomer-InChIKey | 0.572 | 1168 |
| SyGMa (recall@15) | plain InChIKey | 0.558 | 1168 |
| learned end-to-end systems (recall@15, literature) | mixed / method-specific | ~0.47–0.585 | — |

*Source: `results/benchmark_report.json` / `scripts/run_benchmark.py`.*

[^tautomer-audit]: The tautomer match is computed via a heavy-atom-formula prefilter for
tractability; an audit against naive (unfiltered) tautomer keying on 50 substrates found the
prefilter sound — mismatch = 0.

## 7. Results — External validity of the ceiling

The internal ceiling `coverage_bank = 0.735` (the **micro**, pooled ratio-of-sums estimate `Σhit /
Σtrue` — not a per-substrate mean; within-substrate pairs are dependent so a mean-of-ratios would
misrepresent the uncertainty) carries a cluster-bootstrap 95% CI of **[0.709, 0.762]** (resampling
substrates, 10,000 resamples). Recomputed apples-to-apples on an **external** set — the 37 GLORYx
parent substrates, one uncapped full-bank depth-1 `apply_rules` pass, no pool cap, the same
tautomer match — the external ceiling is **0.633** (95% CI **[0.531, 0.733]**, n=37, wide by
design at this sample size). This is far above the previously committed figure of **0.3715**,
which is **pool-capped** — a generator-budget artifact inherited from `gloryx_oracle.json` — and
must **never** be read as "the external ceiling."

Internal (0.735) and external (0.633) both sit in the **micro** frame. A separate question is
whether the internal-external gap is compositional: GLORYx parents tend to be larger,
more-conjugated drugs than the internal test set, and the ΔMW long tail noted in §4 suggests
coverage should degrade with molecular complexity. One OLS regression of **per-substrate**
coverage on molecular descriptors (molecular weight, ring count, aromatic-atom count, heteroatom
count, a conjugation-site count, and `n_true`) fit across both populations predicts **macro**
(per-substrate-mean) coverages of **~0.79** internal and **~0.74** external, in-sample — the macro
frame, distinct from the micro ceilings quoted above. Note honestly that this ~0.74 in-sample
external prediction sits *above* the measured external macro coverage of 0.697, so the in-sample
fit alone overstates transferability. Holding the external points fully out of the fit
(fit-internal-only → predict-external) instead lands at **0.738 out-of-sample**, recovering
**~56%** of the internal→external macro-coverage gap
(`regression.predicted_external_mean_oos = 0.738`, `regression.gap_recovery_frac = 0.565`). This
out-of-sample number, not the in-sample one, is the transferable claim.

We frame this as a **suggestive partial composition effect at n=37** — consistent with, but not
proof of, molecular composition driving part of the internal-external coverage gap — and
explicitly *not* a fitted or transferable law, and *not* a defect in the rule bank or the matching
protocol.

> _[FIGURE: internal-vs-external coverage scatter — optional, post-draft]_

*Source: `results/ceiling_external_validity.json`.*

## 8. Results — Recall decomposition

We now populate the §4 identity `recall@k = coverage_bank · selection_retention ·
ranking_conversion` with measured values, on the common clean molecule-disjoint test split (n=1170
substrates, tautomer-InChIKey, k=15; pooled/micro ratio-of-sums; 10,000-resample substrate
block-bootstrap; `results/recall_factorization.json`). The deployed pipeline realises **recall@15
= 0.261** `[PENDING: multi-seed mean±std over ≥3 seeds]` — a single deployed checkpoint, not yet a
multi-seed estimate.

| factor | value | 95% CI | reading |
|---|---|---|---|
| `coverage_bank` (C_full/U) | **0.735** | [0.709, 0.762] | equals the §6 rule-bank ceiling |
| `selection_retention` (C_bud/C_full) | **0.489** | [0.458, 0.520] | **the dominant loss** |
| `ranking_conversion` (H/C_bud) | **0.726** | [0.687, 0.765] | top-k ordering of the retained pool |
| product = micro recall@15 | **0.261** | — | 0.735 · 0.489 · 0.726 |

The **oracle** recall — the deployed candidate pool ranked perfectly (`ranking_conversion = 1`) — is
`C_bud/U =` **0.359** (micro). The single largest leak in the pipeline is
**`selection_retention = 0.489`**: fewer than half of the metabolites the full rule bank could in
principle recover survive into the deployed, budget-limited candidate pool. That is a **selection**
failure — the generator's rule choice plus the generation budget — and it is upstream of, and
larger than, the ranking loss. The decomposition's main use is exactly this localisation: it points
at *which* stage to fix, not at ranking alone.

Restated as a conversion rate, the deployed pipeline turns the **0.735** rule-bank ceiling into
**0.261** realised recall@15, a **35.5% conversion** (0.261 / 0.735). Because conversion is the
product of the last two factors, `selection_retention × ranking_conversion = 0.489 × 0.726`, this
35.5% is not one ranking failure but **two losses in series**, with selection (0.489) the larger of
the two and ranking (0.726) a smaller second cut on top of it.

**Figure 2** draws this as a waterfall on the same n=1170 population:

![Recall decomposition waterfall](factorization_waterfall.svg)

**Figure 2.** Recall decomposition waterfall, one common n=1170 population (tautomer-InChIKey,
micro ratio-of-sums): `U (1.0) → coverage_bank (0.735) → coverage·selection = oracle recall (0.359)
→ deployed recall (0.261)`, with the oracle line marking the ceiling on the ranking bar; the
coverage bar carries a 95% CI whisker and all three factor CIs are listed in the accompanying table. Bar-1→Bar-2 is the selection loss; Bar-2→Bar-3 the ranking loss.

Throughout, cross-method recall@15 is the per-substrate mean (macro): GRAIL **0.330**, SyGMa
**0.572**; the decomposition uses the pooled (micro) frame — the only frame in which the three
factors multiply exactly to the realised recall and `coverage_bank` equals the 0.735 ceiling — in
which deployed recall is **0.261**.

As established in §4, this is a **decomposition**, not a theorem: the identity telescopes and
closes on any numbers, so its exact cancellation is never offered as evidence for anything beyond
bookkeeping. Its only role here is to localise the loss to `selection_retention`, which motivates
the §10 diagnostics (the learned-vs-prior rule-selection probe and data-scaling saturation) and
Proposition 2.

*Source: `results/recall_factorization.json`.*

## 9. Results — Honest-anchor certification

GRAIL's loss to SyGMa (§8: deployed **0.330** macro vs SyGMa **0.572**) is not a sampling artifact
of which substrates happened to be scored. On the common substrate set shared by both methods
(n=1168, tautomer-InChIKey; `results/anchor_certification.json`), the paired per-substrate
difference `recall_GRAIL − recall_SyGMa` is **−0.242**, 95% CI **[−0.271, −0.212]** — wholly below
zero — computed by a 10,000-resample substrate block-bootstrap. An independent, distribution-free
check on the binary any-hit@15 outcome, the exact McNemar test, agrees: **b = 87** substrates hit
by GRAIL but missed by SyGMa vs **c = 379** hit by SyGMa but missed by GRAIL, **p ≈ 1.7×10⁻⁴⁴**.
The common-subset ceiling (**0.736**) matches the full-1170 ceiling (**0.735**), so restricting to
the common set does not bias the comparison — it is representative of the full population.

Scope matters here. The paired bootstrap covers continuous recall; McNemar covers only the binary
any-hit outcome. Together they certify **evaluation** variance — the uncertainty from scoring one
fixed, already-trained checkpoint over resampled/discordant substrates — not **training** variance
across independently seeded runs. The split itself is not overfit (val ≈ test), but the variance
certified here is evaluation variance only: the deployed **0.330** headline is a single checkpoint,
not a seed average `[PENDING: multi-seed mean±std]`. Within that scope, the result is unambiguous:
**SyGMa > GRAIL is significant — the anchor holds.**

> _[FIGURE: paired-Δ / McNemar — optional, post-draft]_

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

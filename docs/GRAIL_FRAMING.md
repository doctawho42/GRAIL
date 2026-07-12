# GRAIL — primary framing: a rule-based metabolite-structure predictor and a diagnosis of what limits it

> **Status:** living framing doc for the GRAIL-primary paper (reframed 2026-07-10). The
> match-sensitivity benchmark (`docs/benchmark/DNB_FRAMING.md`) is now a **supporting**
> fair-evaluation section, not the headline. Angle chosen by the user: **"GRAIL + its diagnosis."**

## Thesis

Rule-based metabolite-structure prediction is limited **not by rule coverage but by the
conversion of that coverage into a ranked prediction set.** We present **GRAIL**, a rule-based
predictor that learns to *select* SMIRKS transformations over a large curated bank (7,581 rules)
and score the resulting (substrate, product) pairs, and we use it to **decompose where the
rule-based paradigm's headroom is lost** — coverage vs **selection** vs ranking (with regioselectivity
and multi-step entering as sub-levers) — a gap **dominated by rule selection, then ranking.**
The contribution is the architecture **plus** a rigorous, honest diagnosis; a standardized,
tautomer-aware, leakage-audited evaluation protocol (the benchmark) is the apparatus that makes
the diagnosis fair and comparable.

## Honest anchor (state this up front, everywhere)

**GRAIL does not win on recall.** On our clean molecule-disjoint split it reaches
**0.330 recall@15** (per-substrate macro; **0.261** pooled micro) — tautomer-InChIKey, deployed
pipeline, full 1170-substrate clean test; prior-independent (verified on the 291-substrate evaluation;
the full-1170 headline scores the single deployed checkpoint) — below
SyGMa (0.572 tautomer) and MetaPredictor (0.585). (The earlier "~0.334" anchor was a 291-substrate
evaluation, `max_test_substrates:300`; the honest full-1170 figure is **0.330 macro / 0.261 micro**,
and every headline number here reports its match mode, split, n, and source artifact.)
The paper's A*/Q1 value is therefore the **ceiling + the diagnosis + the protocol**, *not* a SOTA
claim. GRAIL enters its own comparison as one honest row. We never imply GRAIL is the best
predictor; we show it is an interpretable, rule-grounded instrument that exposes *why* the
paradigm plateaus.

## Section structure & evidence

### §1 — Method: GRAIL (the rule-based architecture)
Three stages: (1) a learned multi-label **rule selector** over the 7,581-SMIRKS bank
(retrieval-scored generator); (2) RDKit **rule application** enumerating candidate products;
(3) a **PU-trained, MCS-aware pair filter** scoring (substrate, product) pairs. Trained on
positive-unlabeled data (rule-applicable non-annotated products are unlabeled, not negative).
*Status: built, trained (full5000 single-mode). Contribution = interpretable learned rule
selection + PU-aware pair filter, not recall supremacy.*

### §1.5 — Formal framework
*One compact generative model unifies §2 (coverage), §3 (conversion), and §4 (diagnosis) and gives
the three propositions of §4 a principled home. The framework is deliberately thin: one likelihood,
one decomposition, one lever→factor map — not a rewrite of the empirical results.*

**A generative latent-reaction mixture.** For a substrate `s`, a metabolite `m` arises by choosing a
transformation rule `r`, choosing a firing site, and applying the rule:

```
P(m | s) = Σ_r Σ_{site ∈ sites(r,s)} P(r | s) · P(site | r, s) · 𝟙[ apply(r, s, site) = m ]
```

The application term is deterministic given RDKit, so the inner indicator collapses to the
enumerable product support. The latent variables are *which rule fired* and *at which site*; the
observation is the annotated metabolite set, which is **positive-unlabeled** (a rule-applicable,
non-annotated product is unlabeled, not a known negative). The three deployed stages are one
marginal-likelihood **approximation** of this model — *the model the pipeline approximates*, not a
claim that the trained weights are its MLE:

- the **generator** approximates `P(r | s)` (rule selection); its persisted `rule_prior_logits`
  (`model/generator.py`) is the marginal `π(r) = P(r fires)`;
- **RDKit application** realises the deterministic support `𝟙[apply(r,s,site)=m]`, enumerating
  candidate products;
- the **filter** approximates a discriminative correction `P(true | s, m)` over enumerated candidates;
- deployment ranks by `filter_score × generator_score` (`model/wrapper.py`);
- **PU training** approximates EM over the unobserved rule-firing indicator.

Its value is organisational: `π(r)` is the model's marginal (the home of **Proposition 2**), and the
filter's proper-scoring objective versus the deployment ranking metric is the target of
**Proposition 1**.

**A recall decomposition (an accounting identity, not a theorem).** Fix top-k. For substrate `s`, let
`T_s` be the true references and let the candidate sets nest as `R_s(k) ⊆ P_bud,s ⊆ P_full,s`, where
`P_full,s` is the full-bank depth-1 rule-applicable pool (the ceiling pool), `P_bud,s` the deployed
budget-limited pool, and `R_s(k)` the top-k output. With `hit(A_s)` the number of references matched
by set `A_s` under the tautomer-InChIKey quotient (monotone under inclusion) and pooled (micro) sums
`U=Σ|T_s|`, `C_full=Σ hit(P_full,s)`, `C_bud=Σ hit(P_bud,s)`, `H=Σ hit(R_s(k))`:

```
recall@k = H/U = (C_full/U) · (C_bud/C_full) · (H/C_bud)
              = coverage_bank · selection_retention · ranking_conversion
```

Each factor lies in [0,1] because the sets nest. This telescopes and closes on *any* numbers, so we
label it a **decomposition** — never a theorem, verified, or validated, and its cancellation is never
offered as evidence for anything. Its only use is to *localise* the loss. On the common clean
molecule-disjoint test split (n=1170 substrates, tautomer-InChIKey, k=15;
`results/recall_factorization.json`, 10 000-resample substrate block-bootstrap, ratio-of-sums):

| factor | value | 95% CI | reading |
|---|---|---|---|
| `coverage_bank` (C_full/U) | **0.735** | [0.709, 0.762] | equals the §2 rule-bank ceiling |
| `selection_retention` (C_bud/C_full) | **0.489** | [0.458, 0.520] | **the dominant loss** |
| `ranking_conversion` (H/C_bud) | **0.726** | [0.687, 0.765] | top-k ordering of the retained pool |
| product = micro recall@15 | **0.261** | — | 0.735 · 0.489 · 0.726 |

The oracle (perfect ranking of the deployed pool, `ranking_conversion = 1`) is `C_bud/U =` **0.359**
(micro). The single largest leak is **`selection_retention = 0.489`**: fewer than half the metabolites
the full bank *could* recover survive into the deployed budget pool — a **selection** failure (the
generator's rule choice + the generation budget), upstream of ranking. (These are the pooled *micro*
factors, the only frame in which the three multiply exactly to the realised recall and in which
`coverage_bank` equals the published 0.735 ceiling; the per-substrate *macro* recall@15 the rest of
the paper reports is 0.330 — see §3 for the reconciliation of the two frames.)

**Lever → factor map.** Every §4 diagnostic and each proposition attaches to exactly one factor:

| factor | levers & propositions |
|---|---|
| `coverage_bank` | multi-step (depth-2, +0.012), the ΔMW coverage gap, the external-validity composition covariate (§2), **Proposition 3** |
| `selection_retention` | the learned-vs-prior rule-selection probe, data-scaling saturation, **Proposition 2** |
| `ranking_conversion` | the filter / listwise reranker, the oracle bound, **Proposition 1** |

### §2 — The rule-bank coverage ceiling (the hero number)
The bank's **recall ceiling** — the best achievable recall if the selector/filter were perfect —
on the full clean molecule-disjoint test split (1170 substrates, 2597 true pairs) is
**0.718 (plain InChIKey) / 0.735 (tautomer-InChIKey)** — 1865 / 1910 of 2597 recovered. Far above
the best learned systems (~0.47–0.585) and comparable to rule systems on their own sets.
**Coverage is high; the bank is powerful.** So the task is limited by *coverage-conversion* — a
dominant selection loss then ranking — not rule expressiveness. The 0.735 is reported under the **same tautomer
protocol as GRAIL and SyGMa** (referee MAJOR-3 resolved — no more mixed match modes); the plain
0.718 reproduces the previously reported number exactly, validating the run. The tautomer path is
computed via a heavy-atom-formula prefilter **verified sound** against naive keying (audit
mismatch = 0 on 50 substrates). The same run co-measures the **SyGMa** baseline under both
protocols on the identical split: recall@15 **0.558 plain / 0.572 tautomer** (n=1168) — so the
ceiling, SyGMa, and GRAIL are now all reported under tautomer-InChIKey. *Source:
`results/benchmark_report.json` / `run_benchmark.py`.*

**External validity of the ceiling (composition, not a bug).** The internal ceiling
`coverage_bank = 0.735` carries a cluster-bootstrap 95% CI of **[0.709, 0.762]** (resample
substrates, ratio-of-sums estimator `Σhit/Σtrue` — *not* mean-of-ratios, since within-substrate
pairs are dependent; 10 000 resamples; `results/ceiling_external_validity.json`). Recomputed
**apples-to-apples on the external GLORYx set** — one uncapped full-bank depth-1 `apply_rules` pass
over the 37 GLORYx parents, no pool cap, same tautomer match — the external ceiling is **0.633** (95%
CI **[0.531, 0.733]**, n=37, wide by design). This is far above the previously committed **0.3715**,
which is **pool-capped** (a generator-budget artifact from `gloryx_oracle.json`) and must **never** be
read as "the external ceiling." One OLS composition regression against molecular descriptors (MW, ring
count, aromatic/heteroatom counts, a conjugation-site count, and `n_true`) fits per-substrate coverage
across both populations. Its predicted **macro (per-substrate-mean)** coverages are **~0.79** internal
and **~0.74** external in-sample — the macro frame, distinct from the **micro** rule-bank ceilings
(0.735 internal / 0.633 external) quoted above; note the ~0.74 in-sample external prediction sits
*above* the measured external macro coverage (0.697). Holding the external points out of the fit
(fit-internal → predict-external) still lands at **0.738** out-of-sample, recovering **~56%** of the
internal→external macro gap (`regression.predicted_external_mean_oos` = 0.738,
`regression.gap_recovery_frac` = 0.565). This is a **suggestive partial composition effect at n=37** —
GLORYx parents are larger, more-conjugated drugs, the §4 ΔMW long tail made quantitative — **not a
fitted or transferable law**, and *not* a defect in the bank or the protocol.
*Source: `results/ceiling_external_validity.json`.*

### §3 — The conversion gap (ceiling ≫ realized)
In the pooled (micro) frame of §1.5, the deployed pipeline converts the **0.735** rule-bank ceiling
into **0.261** realised recall@15 — a **35.5% conversion** (0.261 / 0.735). By the §1.5 decomposition
this conversion gap is the product of the last two factors,
`selection_retention × ranking_conversion = 0.489 × 0.726`, so it is **not one ranking failure but
two losses in series**, with selection (0.489) the larger. The gap is the paper's central diagnostic
object; §4 attaches a mechanism to each factor.

The waterfall figure **`docs/benchmark/factorization_waterfall.svg`** draws this on one common
n=1170 population (tautomer-InChIKey, micro ratio-of-sums): `U (1.0) → coverage_bank (0.735) →
coverage·selection = oracle recall (0.359) → deployed recall (0.261)`, with the **oracle line**
marking the ceiling on the ranking bar and each factor annotated with its 95% CI. Bar-1→Bar-2 is the
selection loss; Bar-2→Bar-3 the ranking loss. The figure is labeled a decomposition, and never
asserts equality with any macro number.

**Reconciliation of the two frames.** Throughout, cross-method recall@15 is the per-substrate mean
(macro): GRAIL **0.330**, SyGMa **0.572**. The coverage→conversion decomposition instead uses the
pooled (micro) frame — the only frame in which the three factors multiply exactly to the realised
recall and in which `coverage_bank` equals the 0.735 rule-bank ceiling; in that frame deployed recall
is **0.261** (0.330 macro).

**Certifying the honest anchor (evaluation variance).** GRAIL's loss to SyGMa is not a sampling
artifact. On the common substrate set (n=1168, tautomer-InChIKey; `results/anchor_certification.json`),
the paired per-substrate difference `recall_GRAIL − recall_SyGMa` is **−0.242** with 95% CI
**[−0.271, −0.212]** — wholly below zero — and an exact McNemar test on any-hit@15 gives b=87
(GRAIL-only hits) vs c=379 (SyGMa-only hits), **p ≈ 1.7×10⁻⁴⁴**. The common-subset ceiling (0.736)
matches the full-1170 ceiling (0.735), so the common set is representative. The paired bootstrap
covers continuous recall and McNemar is used **only** for the binary any-hit outcome; together they
certify *evaluation* variance (a single deployed checkpoint scored over resampled substrates).
The split is not overfit (val ≈ test); the variance certified here is **evaluation** variance only —
the deployed headline is a single checkpoint, not a seed average. SyGMa > GRAIL is significant — the
anchor holds.

> **Note (checkpoint / prior — resolved).** Prior-independence of the *deployed* pipeline was
> established on the earlier 291-substrate evaluation: the prior-populated checkpoint `full5000_priors`
> (byte-identical learned weights, trained prior present) and `full5000_single` gave essentially
> identical recall (0.335 vs 0.334), so the deployed pipeline applies rules broadly and its recall is
> coverage-limited, not prior-limited. On the honest full-1170 split that deployed figure is
> **0.330 macro / 0.261 micro**. The large prior effect (§4 row 1) appears only in a *top_k-limited
> rule-selection* probe, not in deployment — an earlier draft over-extrapolated that probe to claim a
> ~0.40 headline; that claim is withdrawn.

### §4 — Diagnosis: where the headroom goes (all rule-based, all measured)
Each lever below attaches to exactly one factor of the §1.5 decomposition (annotated in the Lever
column), and the three refutable **Propositions** that follow the table localise the loss further.

| Lever | Finding | Evidence |
|---|---|---|
| **Ranking: learned vs prior** *(top_k-limited rule-selection probe — not the deployed recall)* — → `selection_retention` | In a controlled probe (each mode picks its top-30 rules by its own score, same product loop + filter), a SyGMa-style **frequency prior significantly OUT-ranks the learned generator**: prior-only 0.410 vs learned-only 0.266 gen-only (Δ −0.144, 95% CI [−0.196, −0.095], paired bootstrap); with the filter 0.405 vs 0.300 (−0.105, CI [−0.152, −0.058]). Adding the prior to the learned scorer lifts **+0.130** gen / +0.099 filter (both significant) — the prior is the load-bearing rule-selection signal; the filter significantly helps only the *weak learned* ordering (+0.034, CI [+0.011,+0.061]), not the strong prior (n.s.). NB: these probe numbers isolate rule *ranking*; the **deployed** GRAIL (0.330) applies rules broadly and is prior-independent, so it neither inherits the +0.13 nor exposes the learned selector's weakness — the probe reveals it. | `scripts/prior_vs_learned.py`, `results/prior_vs_learned.json` (245 test subs, paired-bootstrap CI) |
| **Multi-step** — → `coverage_bank` | Breadth-capped depth-2 rule application lifts the ceiling by **only +0.012** (0.711→0.723) at **8.5× candidate cost** (194→1653). Multi-step is **not** the dominant coverage lever; most uncovered metabolites are genuinely out-of-bank. | `run_benchmark --depth 2`, `results/benchmark_report_depth2.json` (150 subs, beam 10) |
| **Coverage (ΔMW gap)** — → `coverage_bank` | **26.9%** of true metabolites are uncovered by the depth-1 bank (plain-InChIKey; ~1.7 pp lower, ~25%, under the tautomer protocol the ceiling uses). Misses are a **diverse long tail** (top class = hydroxylation, only 6% of uncovered) — one-off large conjugates (glucuronide, glutathione, sulfate, mercapturate) + unusual transforms. No single rule-family addition closes much. | `run_benchmark --gap-analysis`, `results/benchmark_report_gap.json` (500 subs) |
| **Data scaling** — → `selection_retention` | Recall saturates (2418→4787 substrates ≈ flat) — the plateau is **not** a data-quantity problem. | `results/full{2500,5000}_single.log` |
| **Regioselectivity (SoM)** — → `ranking_conversion` | Site-of-metabolism prior gives only a small lift — regioselective ranking is hard within the bank. | `results/train_som.log` |

**Net diagnosis:** the rule bank covers 0.735 (micro); the deployed GRAIL surfaces only **0.261**
(micro; 0.330 macro) of it in top-15 (a 35.5% conversion). The gap is a **coverage-conversion**
problem that §1.5 splits into a *dominant selection loss* (`selection_retention` 0.489 — the deployed
budget pool loses more than half the bank's recoverable hits before ranking) followed by a *ranking
loss* (`ranking_conversion` 0.726): from a candidate pool whose full-bank coverage approaches the
ceiling, GRAIL's selection + filter ranking promotes only 0.261 of the references into the top 15. A
controlled rule-selection probe sharpens *why the learned part does not help*: in a top_k-limited
setting the learned generator ranks rules **worse than a trivial frequency prior** (−0.14 gen-only,
significant; row 1). The deployed pipeline masks this by applying rules broadly (so the prior does
not change deployed recall — verified), but the probe shows the learned rule-selector is itself a
liability vs frequency. Neither multi-step nor any single rule-family addition moves coverage much
(misses are a diverse long tail). **The open problem is coverage-conversion — a dominant selection
loss then a ranking loss (§1.5): neither the learned generator nor the filter surfaces the rich
candidate pool well, and the learned rule-selector loses to a frequency baseline.** (An earlier draft
reported the opposite "learned beats prior"; that was an artifact of a checkpoint whose prior buffer
was un-persisted, caught by adversarial verification and corrected here.)

#### Three refutable propositions
The decomposition localises *where* recall is lost; these three labeled Propositions say *why*, each
making a prediction that could be false and each checked against committed data. They are refutable
theory imported-and-applied, not accounting identities and not a claim that GRAIL wins on recall.

**Proposition 1 — Surrogate mismatch (→ `ranking_conversion`).**
*Statement.* A filter trained by a strictly proper scoring rule (BCE / PU) learns a globally
calibrated posterior `η(s,m)=P(true|s,m)`, which is Bayes-optimal for AUC and calibration. Ranking
each substrate's candidate pool by `η` and taking top-k is recall@k-optimal only when pools are
homogeneous. When pools vary in size and positive-rate across substrates (empirically pool size
17–150, `n_true` 1–18), the per-substrate top-k operator is **not** a monotone functional of a single
global `η`, so a pointwise-calibrated scorer can be recall@k-suboptimal while a listwise /
ranking-consistent surrogate dominates.
*Basis.* An imported ranking-consistency result (different target metrics have different
Bayes-optimal scores; pointwise proper scoring is consistent for the posterior / AUC but not for top-k
selection across heterogeneous queries), applied here, plus a minimal 2-substrate counterexample in
which a globally calibrated `η` yields **strictly lower** recall@k than a within-pool reorder —
checked arithmetically in `grail_metabolism/tests/test_prop1_counterexample.py`, where the
recall-superior reorder is verified *not* globally calibrated, so a proper-scoring objective rejects
it: that objective-vs-metric gap is the surrogate mismatch.
*Confirmation (committed).* A listwise-InfoNCE reranker of similar capacity beats the pointwise BCE
filter *as a ranker*: 0.433 → **0.500** @15 (+0.067), 74% of the oracle 0.677
(`docs/benchmark/stage2_ranker_evidence.md`, Spike-3) — the falsifiable prediction, already confirmed.
*Guardrail.* This is a theorem about **objectives**, not a recall win: the reranker's 0.500 **still
loses to SyGMa (0.558)** and is reported only as a separate Stage-2 artifact. The theory is
imported-and-applied, not a from-scratch theorem.

**Proposition 2 — Propensity-PU identifiability (→ `selection_retention`).**
*Statement.* Under PU annotation with an approximately constant labeling propensity (SCAR), a learner
with constant unlabeled weighting recovers a propensity-distorted score whose dominant component is
the marginal rule-firing rate `π(r)` — the frequency prior. The prior is therefore Bayes-competitive
**by construction**, and the learned selector improves on it only if substrate-conditional prevalence
variation exceeds estimation noise; the observed data saturation (2418 → 4787 substrates ≈ flat)
indicates it does not at current scale. In §1.5 terms `π(r)` is the generative marginal `P(r fires)`,
so this is an identifiability statement about `P(r|s)` beyond its marginal under SCAR — and it names
*why* `selection_retention` (0.489) is the dominant loss: the selector cannot separate the recoverable
pool better than the frequency prior.
*Anchor (committed, significant).* learned-only 0.266 vs prior-only 0.410 (gen-only @15,
Δ = **−0.144**, 95% CI **[−0.196, −0.095]**, paired bootstrap; `results/prior_vs_learned.json`, n=245).
*Falsifiable prediction.* Reweighting the labeled loss by `1/ê(r)` (a SAR correction) should shrink
the prior's edge — an **open test, not a promised fix**.
*Guardrail.* The propensity model `e(r) ∝ π(r)` is an **unmeasured modeling assumption**, flagged as
such; this is an explanatory model plus a refutable prediction, not a proof that learning cannot win.
It is consistent with the *deployed* pipeline being prior-independent — the probe reveals the
selector's weakness; deployment, applying rules broadly, masks it.

**Proposition 3 — Paradigm limit (→ `coverage_bank`).**
*Statement.* Single-step rule-based recall ≤ single-step `coverage_bank` < 1, because a non-vanishing
fraction of references are multi-generation (e.g. oxidation → conjugation) and unreachable by any
single rule application. An **irreducible residual** therefore exists that no ranking improvement can
recover. The bound is dataset-conditional and, critically, **single-step-conditional** — a bounded,
honest limit, **not** a claim that rule-based prediction is futile.
*Witnesses (committed).* depth-2 rule application lifts the ceiling by only **+0.012** at 8.5×
candidate cost (`results/benchmark_report_depth2.json`); on the external GLORYx set the uncapped
single-step ceiling is **0.633** (§2; `results/ceiling_external_validity.json`), whose references
include single-step-unreachable multi-generation metabolites.
*Guardrail.* Stated single-step-conditional; multi-step and out-of-bank chemistry remain open
coverage levers (Proposition 3 bounds the *single-step* paradigm, not the problem).

### §Supplement — Fair evaluation: a standardized, tautomer-aware, leakage-audited protocol
The match-sensitivity ("rank-flip") analysis and the multi-method comparison move here. Their job
is to justify that the numbers above are apples-to-apples and that prior rule-vs-learned
comparisons were confounded (methods gain method-dependent, significant amounts from the match
protocol — differential sensitivity CI [+0.073, +0.171]; see `docs/benchmark/DNB_FRAMING.md`).
Now spans **5 methods** (GRAIL, SyGMa, BioTransformer, MetaPredictor, MetaTrans;
`results/match_sensitivity_5method.json`), with two independent rank-flips (GRAIL↔BioTransformer;
MetaTrans↔SyGMa) and a **non-monotonic** protocol response for MetaTrans (canon 0.523 > InChIKey
0.494 < no-stereo 0.561) — the protocol effect is method-idiosyncratic, not a uniform lift.
(LAGOM was scoped but is unusable: code available, trained weights never released — retraining a
Chemformer on DrugBank-licensed data is out of scope.)

## Referee-risk register (GRAIL-primary)
- **"GRAIL loses — why publish it?"** → we do not claim a recall win; the contribution is the
  ceiling + diagnosis + protocol. Lead with that; keep GRAIL one honest row.
- **"Ceiling is trivially high because InChIKey nesting."** → resolved: ceiling reported under
  the SAME tautomer protocol as GRAIL (§2 provenance), prefilter verified sound.
- **"Depth-2 is a lower bound — maybe multi-step helps more."** → acknowledged (breadth-capped,
  beam 10); even the lower bound + the 8.5× cost make the point; state it as a bound.
- **"Prior-vs-learned n=245, single split."** → report CIs; it is a diagnostic, not a selection.
- **"Is the architecture novel enough?"** → position vs SyGMa (fixed rules) and transformers
  (no rule grounding): learned selection over a large SMIRKS bank + MCS-aware PU pair filter.

## Open items before submission
1. ~~**Provenance**: full-set ceiling + SyGMa under tautomer → fill §2 hero number.~~ **DONE** — §2
   reports the full-set tautomer ceiling (0.735) and SyGMa (0.572) on the identical split.
2. ~~**prior-vs-learned CIs** (paired bootstrap on the learned−prior gap) for §4 rigor.~~ **DONE** —
   §4 row 1 carries the paired-bootstrap CIs.
3. **Tier-2 for the supplement** (LAGOM/MetaTrans) to widen the match-sensitivity spread.
4. Fold `DNB_FRAMING.md` explicitly into the §Supplement narrative.

## §Reproducibility & provenance — the TAME protocol

We package the evaluation as **TAME** — the **T**automer-**A**ware **M**etabolite-structure
**E**valuation protocol: a tautomer-InChIKey match quotient, a leakage-audited molecule-disjoint
train/val/test split, and a frozen multi-method re-scoring harness. TAME is a *protocol + audited
split + re-scoring harness*, **not** a leaderboard service — its purpose is to make the numbers in
§2–§4 apples-to-apples and one-command reproducible, not to rank submissions. Every headline stat
below ships with its full provenance and regenerates from committed checkpoints + the symlinked
dataset via `scripts/regen_headline.sh`.

### Provenance table

Every row is under **tautomer-InChIKey** match unless noted. "micro" = pooled ratio-of-sums
(`Σhit/Σtrue`; the §1.5 decomposition frame, in which `coverage_bank` = the 0.735 ceiling);
"macro" = per-substrate mean (the `metrics.py` cross-method frame). CIs are 95% (10 000 resamples,
seed 0) over the stated resampling unit.

| stat | value | match mode | split | n | resampling unit | seed | source file |
|---|---|---|---|---|---|---|---|
| Rule-bank coverage ceiling (**micro**) | 0.7355, CI [0.709, 0.762] | tautomer-InChIKey | clean test | 1170 | substrate (cluster) | 0 | `results/recall_factorization.json` (`factors.coverage_bank`); also `results/benchmark_report.json` (`grail_rule_bank_ceiling.recall_ceiling_tautomer`) |
| GRAIL deployed recall@15 | 0.330 (**macro**) / 0.261 (**micro**) | tautomer-InChIKey | clean test | 1170 | substrate | 0 | `results/recall_factorization.json` (`macro_recall` / `micro_recall`) |
| selection_retention (**micro** factor) | 0.489, CI [0.458, 0.520] | tautomer-InChIKey | clean test | 1170 | substrate | 0 | `results/recall_factorization.json` (`factors.selection_retention`) |
| ranking_conversion (**micro** factor) | 0.726, CI [0.687, 0.765] | tautomer-InChIKey | clean test | 1170 | substrate | 0 | `results/recall_factorization.json` (`factors.ranking_conversion`) |
| SyGMa recall@15 | 0.572 (**macro**) | tautomer-InChIKey | clean test | 1168 | substrate | 0 | `results/benchmark_report.json` (`sygma_baseline.recall_at_tautomer["15"]`); `results/anchor_certification.json` (`mean_recall_SyGMa`) |
| MetaPredictor recall@15 | 0.585 (**macro**) | tautomer-InChIKey | clean-test tier-2 subset | 150 | substrate | 0 | `results/match_sensitivity_5method.json` (`by_method.MetaPredictor.inchikey_tautomer`) |
| External uncapped GLORYx-37 ceiling (**micro**) | 0.633, CI [0.531, 0.733] | tautomer-InChIKey | GLORYx external set | 37 | parent (cluster) | 0 | `results/ceiling_external_validity.json` (`external_ceiling_uncapped`) |
| Anchor Δ(GRAIL − SyGMa) (**macro**) | −0.242, CI [−0.271, −0.212]; McNemar p ≈ 1.7e-44 (any-hit@15) | tautomer-InChIKey | clean-test common set | 1168 | substrate (paired) | 0 | `results/anchor_certification.json` (`delta_mean_recall`, `mcnemar`) |
| **Differential match-protocol sensitivity (interaction)** | +0.120, CI [+0.073, +0.171] | canonical vs tautomer-InChIKey (Δ of Δ) | clean-test tier-2 subset | 150 | substrate (paired) | 0 | `results/rank_flip_ci.json` (`interaction_B_extra_gain_from_normalization`) |

Guardrail restated in tabular form: GRAIL's deployed recall (0.330 macro / 0.261 micro) sits
**below** SyGMa (0.572) and MetaPredictor (0.585); the anchor row certifies that loss (Δ = −0.242,
CI wholly < 0). We claim **no recall win** — the contribution is the ceiling, the decomposition, and
the protocol.

### Declared primary endpoint

The **pre-declared primary endpoint of TAME is the differential match-protocol-sensitivity
interaction**, `Δ_B − Δ_GRAIL = +0.120, 95% CI [+0.073, +0.171]` (`results/rank_flip_ci.json`,
`interaction_B_extra_gain_from_normalization`): the extra recall a method gains purely from moving
canonical → tautomer-InChIKey matching, above what GRAIL gains — establishing that the match
protocol is a **method-dependent confounder**, which is the paper's methodological claim.
Multiplicity is controlled by **Holm** correction *within each declared family of tests*: (i) the
**per-method protocol-sensitivity family** (`protocol_sensitivity_per_method` — GRAIL, SyGMa,
BioTransformer, MetaPredictor), and (ii) the **rank-flip pairwise family** (the per-protocol
GRAIL↔BioTransformer and MetaTrans↔SyGMa Δ contrasts); Holm is applied *within*, not across,
families. Everything else — the recall factorization (§1.5/§3), the external-validity ceiling (§2),
and the anchor certification (§2) — is **secondary / descriptive** and reported with CIs but not
counted against the primary-endpoint error budget.

### Released artifact + one-command regen

TAME releases the **frozen per-substrate, 5-method × 5-protocol prediction set** so the
match-sensitivity numbers re-score without re-running any predictor:

- **Frozen predictions** — `artifacts/tier2/biotransformer_preds.json`,
  `artifacts/tier2/metapredictor_preds.json`, `artifacts/tier2/metatrans_preds.json` (the three
  external tools); GRAIL's deployed per-substrate ranking at
  `artifacts/full5000_single/predictions/test_predictions.csv`; SyGMa is re-derivable (not a frozen
  JSON) via `scripts/run_benchmark.py`.
- **Re-scoring harness** — `scripts/run_match_sensitivity.py` re-scores those frozen predictions
  under all five match quotients (canonical, inchikey, inchi_no_stereo, tanimoto1,
  inchikey_tautomer) → `results/match_sensitivity_5method.json`; `scripts/rank_flip_ci.py` bootstraps
  the primary-endpoint CI → `results/rank_flip_ci.json`.
- **Leakage audit** — the molecule-disjoint clean split (`*_triples_clean.txt`) is produced by
  `scripts/fix_splits.py --molecule-disjoint`, which enforces full molecule-set disjointness across
  train/val/test and emits its audit summary to `results/leakage_fix_report.json` when run (that
  report is regenerated by the script, not committed to this tree).
- **One-command regen** — `scripts/regen_headline.sh` regenerates every headline number above
  (`benchmark_report`, `recall_factorization`, `ceiling_external_validity`, `anchor_certification`)
  and the `docs/benchmark/factorization_waterfall.{png,svg}` figure, in order, from committed
  checkpoints + the symlinked dataset. The dominant cost is `factorize_recall.py` (~90 min: deployed
  pipeline + parallel full-bank ceiling over all 1170).

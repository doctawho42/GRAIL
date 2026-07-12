# GRAIL theory spine + diagnosis — design spec

> **Status:** design (2026-07-12). Feeds `writing-plans`. Turns the currently
> **entirely empirical** GRAIL-primary paper (`docs/GRAIL_FRAMING.md`) into a
> *principled* one by adding a formal framework, three refutable propositions, a
> carrying figure, an external-validity finding, and camera-ready packaging.
> Architecture **C (hybrid)**: a compact §1.5 formal block + retrofit of the
> existing section structure — NOT a full rewrite.

## Goal

Add the largest defensible A*/Q1 lift per unit effort to the GRAIL-primary paper,
**without any claim that GRAIL wins on recall**, mostly from artifacts already on
disk. Deliver: (1) a formal framework that unifies §2/§3/§4/§Supplement; (2) three
*refutable* theoretical results (not accounting identities); (3) one carrying
figure; (4) an external-validity result that disarms the paper's biggest honesty
risk; (5) camera-ready statistical packaging.

## Honest anchor (unchanged, load-bearing)

GRAIL does **not** win on recall: deployed ~0.334 recall@15 (tautomer-InChIKey,
clean molecule-disjoint split, n=1170), below SyGMa (0.572 tautomer) and
MetaPredictor (0.585). The paper's value = ceiling + diagnosis + protocol. Every
addition below is engineered to be true *even though GRAIL loses*.

## Global constraints (honesty guardrails — these are hard requirements, not style)

1. **No addition may require GRAIL to win on recall.** The anchor stays 0.334.
2. **The recall factorization is an accounting identity** (it telescopes and
   closes on *any* numbers). It is labeled a **decomposition** everywhere. It is
   **never** called "a theorem," "verified," or "validated." Its cancellation is
   never presented as evidence for anything.
3. **The refutable content lives only in the three Propositions**, each of which
   makes a prediction that could be false and is checked against committed data.
4. **The Stage-2 listwise reranker (0.500@15) never appears near a headline.** It
   is labeled a separate Stage-2 artifact that **still loses to SyGMa (0.558)**.
   It is used only as the *confirmation* of Proposition 1, never as a result.
5. **Proposition 2's `e(r) ∝ π(r)` and the `1/ê` reweighting are explicitly an
   unmeasured modeling assumption and an open test** — never a proof or a promised
   fix.
6. **The external GLORYx ceiling is recomputed *uncapped*, apples-to-apples with
   the internal one.** The committed `0.3715` is pool-capped and understates; it is
   never presented as "the external ceiling." `n=37` → report a wide CI.
7. **No Claude / AI attribution in any commit trailer or doc byline.** (Standing
   user constraint.)
8. Every new number ships with its **provenance** (value, match mode, split, n,
   resampling unit, seed count, source file) and every claim states which factor /
   population it belongs to.

## Context & feasibility (verified 2026-07-12)

- Dataset is **symlinked** into this worktree (`grail_metabolism/data/{train,val,test}.sdf`
  + `*_triples_clean.txt`), so coverage/inference passes run locally. No GPU / Modal needed.
- GRAIL per-substrate test predictions are exported:
  `artifacts/full5000_single/predictions/test_predictions.csv` (deployed ranking)
  and `artifacts/full5000_priors/predictions/test_predictions.csv`.
- Committed evidence the additions lean on:
  - `results/benchmark_report.json` — ceiling 0.718 plain / **0.735 tautomer**
    (1865/1910 of 2597), SyGMa 0.558 / 0.572, mean output 80.98.
  - `results/prior_vs_learned.json` — learned-only 0.266 vs prior-only 0.410
    gen-only @15, Δ=−0.144 CI[−0.196,−0.095] SIGNIFICANT (n=245).
  - `docs/benchmark/stage2_ranker_evidence.md` (Spike-3) — generator 0.433 →
    **listwise-InfoNCE reranker 0.500 ± 0.015** @15 (+0.067), oracle 0.677,
    val≈test; reranker **< SyGMa 0.558**.
  - `results/gloryx_oracle.json` — GLORYx-37 external oracle/coverage 0.3715
    (pool-capped) per-parent records with n_true / pool_size.
  - `results/benchmark_report_depth2.json` — depth-2 ceiling +0.012 at 8.5× cost.
  - `results/rank_flip_ci.json`, `results/match_sensitivity_5method.json` — the
    differential-sensitivity interaction CI [+0.073,+0.171] (the declared primary endpoint).
- **Not** on disk here: `reranker_gate_bi_test*.json` (per-substrate reranker
  preds). So Proposition 1's confirmation cites the committed *aggregate* +0.067;
  a paired CI on it is an out-of-scope optional strengthening.

---

## Component 1 — §1.5 "Formal framework" (new block; core of architecture C)

### 1.1 Generative latent-reaction mixture

State the pipeline as one likelihood over metabolites. For substrate `s`, a
metabolite `m` arises by choosing a rule `r` and a firing site, then applying it:

```
P(m | s) = Σ_r Σ_{site ∈ sites(r,s)} P(r | s) · P(site | r, s) · 𝟙[ apply(r, s, site) = m ]
```

The product term is (near-)deterministic given RDKit, so it collapses to an
indicator over the enumerable product support. Latent variables: which rule fired,
at which site; observed: the annotated metabolite set (positive-unlabeled).

The three deployed stages are one marginal-likelihood approximation:
- **generator** ≈ `P(r | s)` (rule selection); its persisted `rule_prior_logits`
  (`model/generator.py`) is the marginal `π(r) = P(r fires)`;
- **RDKit application** = the deterministic support `𝟙[apply(...)=m]` (enumerates
  candidate products);
- **filter** ≈ a discriminative correction `P(true | s, m)` over enumerated candidates;
- deployment ranks by `filter_score × generator_score` (`model/wrapper.py`).
- **PU training** ≈ EM over the unobserved rule-firing indicator (rule-applicable
  non-annotated products are unlabeled, not negative).

This framing is what gives Proposition 2 a principled home (`π(r)` = the model's
marginal) and Proposition 1 its target (the filter's objective vs. the deployment metric).

**Guardrail:** present as "the model the pipeline *approximates*," not a claim it is
the MLE.

### 1.2 Recall factorization (labeled a decomposition)

Per substrate `s`: `T_s` = true references; nested candidate sets
`R_s(k) ⊆ P_bud,s ⊆ P_full,s`, where `P_full,s` = full-bank depth-1 rule-applicable
products (ceiling pool), `P_bud,s` = deployed budget-limited pool, `R_s(k)` = top-k
ranked output. Under match quotient `q`, `hit_q(A_s) = |{t ∈ T_s : ∃ a∈A_s, a ≡_q t}|`
is monotone under inclusion. With pooled (micro) sums
`U=Σ|T_s|`, `C_full=Σ hit(P_full,s)`, `C_bud=Σ hit(P_bud,s)`, `H=Σ hit(R_s(k))`:

```
recall@k = H/U = (C_full/U) · (C_bud/C_full) · (H/C_bud)
              = coverage_bank · selection_retention · ranking_conversion
```

Each factor ∈ [0,1] (since `R_s(k) ⊆ P_bud,s ⊆ P_full,s`). The oracle recall
(perfect ranking of the pool) is the `ranking_conversion = 1` point, i.e.
`C_bud/U`, when per-substrate pool hits fit in top-k.

Numeric closure on Spike-3 (pool-150): `0.718·(0.677/0.718=0.943)·(0.500/0.677=0.739)=0.500`
(reranker) and `·(0.433/0.677=0.640)=0.433` (generator).

**Guardrail:** identity, always closes → "decomposition," never "verified." Report
the deployed factorization on the common n=1170 population; the closure on Spike-3
is illustrative, not evidence.

### 1.3 Lever → factor map (retrofit of existing §4)

| factor | levers |
|---|---|
| `coverage_bank` | multi-step (depth-2 +0.012), ΔMW gap, external-validity/covariate model (Component 4), **Prop 3** |
| `selection_retention` | learned-vs-prior probe, **Prop 2** |
| `ranking_conversion` | filter / reranker, oracle bound, **Prop 1** |

---

## Component 2 — Three refutable propositions

### Proposition 1 — Surrogate mismatch (→ `ranking_conversion`)

**Claim.** A filter trained by a strictly proper scoring rule (BCE/PU) learns a
calibrated posterior `η(s,m)=P(true|s,m)`, which is Bayes-optimal for
AUC/calibration. Ranking each substrate's pool by `η` and taking top-k is
recall@k-optimal only under homogeneous pools. When pools vary in **size and
positive-rate across substrates** (empirically pool 17–150, `n_true` 1–18), the
per-substrate top-k set operator is **not** a monotone functional of a single
global `η`, so a pointwise-calibrated scorer can be recall@k-suboptimal while a
listwise / ranking-consistent surrogate dominates.

**Method.** Import the standard ranking-consistency result (different target
metrics have different Bayes-optimal scores; pointwise proper scoring is
consistent for the posterior/AUC but not for top-k selection across heterogeneous
queries). Give a **minimal 2-substrate counterexample**: two pools with different
sizes/positive-rates where a calibrated `η` yields strictly lower recall@k than a
listwise reorder. State as a labeled **Proposition** with a proof sketch (imported
result + counterexample), not a from-scratch theorem.

**Confirmation (committed).** listwise-InfoNCE reranker of similar capacity beats
the pointwise BCE filter *as a ranker*: 0.433 → 0.500 @15 (+0.067), 74% of oracle
0.677 (`stage2_ranker_evidence.md` Spike-3). This is the falsifiable prediction,
already confirmed.

**Guardrail:** a theorem about **objectives**, not a claim GRAIL wins — reranker
0.500 < SyGMa 0.558; label it a separate Stage-2 artifact. Theory is
imported-and-applied; say so. Confirmation cites committed aggregate; the paired CI
is out of scope.

### Proposition 2 — Propensity-PU identifiability (→ `selection_retention`)

**Claim.** Under PU annotation with ~constant labeling propensity (SCAR), a learner
with constant unlabeled weighting recovers a propensity-distorted score whose
dominant component is the marginal rule-firing rate `π(r)` — i.e. the frequency
prior. Hence the prior is Bayes-competitive *by construction*, and the learned
selector improves on it only if substrate-conditional prevalence variation exceeds
estimation noise; data saturation (2418→4787 flat) indicates it does not at current
scale. Ties to §1.1: `π(r)` is the generative model's marginal `P(r fires)`, so this
is an identifiability statement about `P(r|s)` beyond its marginal under SCAR.

**Anchor (committed, significant).** learned-only 0.266 vs prior-only 0.410 gen-only
@15, Δ=−0.144 CI[−0.196,−0.095] (`prior_vs_learned.json`).

**Falsifiable prediction.** Reweighting the labeled loss by `1/ê(r)` (SAR
correction) should shrink the prior's edge. **Open test, not a promised fix.**

**Guardrail:** `e(r) ∝ π(r)` is an unmeasured modeling assumption — flag it.
Explanatory model + falsifiable prediction, not proof learning can't win. Consistent
with deployed pipeline being prior-independent (probe reveals the weakness;
deployment masks it).

### Proposition 3 — Paradigm limit, impossibility direction (→ `coverage_bank`)

**Claim.** Single-step rule-based recall ≤ single-step `coverage_bank` < 1, because
a non-vanishing fraction of references are multi-generation (e.g.
oxidation→conjugation) unreachable by any single rule application. Therefore an
**irreducible residual** exists that no ranking improvement can recover.
Dataset-conditional.

**Witnesses (committed).** depth-2 lifts the ceiling only +0.012 at 8.5× candidate
cost (`benchmark_report_depth2.json`); GLORYx external oracle ≈ 0.50 at pool-150
with references single-step-unreachable (`gloryx_oracle.json`).

**Guardrail:** single-step-conditional; **not** "rule-based prediction is futile."
A bounded, honest limit.

---

## Component 3 — §3 carrying figure: coverage→selection→ranking waterfall

**What it shows.** On one common population (n=1170, micro ratio-of-sums,
**all factors under the harmonized tautomer-InChIKey mode** matching the §2 ceiling
provenance, consistent top-k cap): a waterfall `U (=1.0) → coverage_bank (0.735) → coverage·selection
(= oracle recall = C_bud/U) → deployed recall (0.334)`, with the **oracle line**
drawn as the ceiling on the ranking bar. Bar1→Bar2 gap = selection loss;
Bar2→Bar3 gap = ranking loss. Each factor annotated with value + CI.

**Data.** Deployed ranking from `artifacts/full5000_single/predictions/test_predictions.csv`;
`C_full` (full-bank ceiling hits per substrate) and `C_bud` (deployed-pool hits per
substrate) from a single per-substrate logging pass through
`scripts/run_benchmark.py`'s ceiling path (new `--log-per-substrate` dump); oracle
from `scripts/diagnose_rerank_ceiling.py` (reuse or recompute). Paired substrate
**block-bootstrap** CI per factor — resample substrates, recompute each factor as a
**ratio-of-sums** (numerator and denominator jointly), percentile CI.

**Output.** `results/recall_factorization.json` (per-factor values + CIs + n) and
`docs/benchmark/factorization_waterfall.{png,svg}` (match the existing
`rankflip.svg` / `scaling_curve.svg` style via a `scripts/make_*_figure.py`).

**Guardrail:** label "decomposition"; the micro factorization is the reported one;
never assert it equals the macro 0.500.

---

## Component 4 — §2 external validity + covariate-transfer model of the ceiling

Turns the paper's #1 honesty risk (0.72 internal vs 0.37 external) into a finding.

1. **Cluster-bootstrap CI on the internal ceiling** (0.735 = 1910/2597): resample
   substrates, ratio-of-sums estimator `R* = Σh_i/Σt_i` (NOT mean-of-ratios —
   within-substrate pairs are dependent), percentile CI; macro variant as robustness.
   Add a per-substrate accumulation to `run_benchmark.py:grail_ceiling`.
2. **Uncapped external GLORYx-37 ceiling**: recompute coverage via one
   `apply_rules` pass over the 37 parents with no pool cap (apples-to-apples with the
   internal full-bank ceiling), not the pool-80/pool-150 capped 0.37/0.50. Report
   with a wide CI (n=37).
3. **One composition regression** predicting per-substrate coverage from molecular
   descriptors (MW, #rings, aromatic/heteroatom counts, a conjugation-site count,
   `n_true`), fit pooled over internal per-substrate coverage (from step 1's dump) +
   `gloryx_oracle.json` per-parent records; show it predicts **both** the ~0.72
   internal and ~0.37 external means → "coverage is governed by a transferable
   composition covariate — the §4 ΔMW long tail made quantitative."

**Output.** `results/ceiling_external_validity.json` (internal CI, external uncapped
CI, regression coefficients + in/out-of-sample predicted means).

**Guardrail:** never present capped 0.37 as "the external ceiling"; n=37 → wide CI,
"suggestive-but-transferable," not a fitted law; attribute the gap to composition
(GLORYx = larger, more-conjugated drugs), never a bug.

---

## Component 5 — anchor certification + camera-ready packaging

### 5.1 Certify the honest anchor (§3)

Paired bootstrap on per-substrate `d_i = recall_GRAIL − recall_SyGMa` over a
**common substrate set** (GRAIL exported test preds vs `run_benchmark.py:sygma_baseline`
re-run on the same substrates) + **McNemar** on any-hit@15. Certifies SyGMa > GRAIL
is not within noise. Output `results/anchor_certification.json`.

**Guardrail:** report that the common-subset ceiling matches the full-1170 ceiling
(representativeness); paired bootstrap for continuous recall, McNemar **only** for
the binary any-hit outcome; certifies *evaluation* variance (training-seed variance
separately bounded by val≈test 0.327 vs 0.330).

### 5.2 Packaging (§Reproducibility)

- **Provenance table**: every headline stat → value, match mode, split, n,
  resampling unit, seed count, source file (a markdown table in `GRAIL_FRAMING.md`).
- **Declared primary endpoint**: the differential-sensitivity interaction CI
  [+0.073,+0.171] (`rank_flip_ci.json`); **Holm** correction within each declared
  family of tests.
- **Released artifact**: the frozen per-substrate 5-method × 5-protocol prediction
  set (`artifacts/tier2/*_preds.json` + GRAIL/SyGMa) + the re-scoring harness +
  `leakage_fix_report.json`, with a **one-command regen** of every headline number.
- **A name** for the protocol/benchmark.

---

## Placement in `docs/GRAIL_FRAMING.md` (architecture C)

- New **§1.5 Formal framework** — generative model + factorization identity +
  stage-approximation + lever→factor map.
- **§2** — add external-validity covariate model + cluster-bootstrap CI on the ceiling.
- **§3** — the waterfall figure as centerpiece; conversion gap = product of the last
  two factors; anchor certification.
- **§4** — retrofit each lever to its factor; add Proposition 1 (ranking),
  Proposition 2 (selection), Proposition 3 (coverage) as labeled Propositions.
- **§Reproducibility** (new or existing) — the packaging block.

Existing section structure and already-harmonized numbers are preserved.

## Verification

- **Theory (Props 1–3, §1.5):** internal-consistency review + an **adversarial
  referee pass at plan-time and post-execution** (a workflow with diverse lenses +
  a refute stage — the standing lesson: green prose hides real blockers). Check:
  Prop 1's counterexample is arithmetically valid; Prop 2 stays within the
  unmeasured-assumption guardrail; Prop 3 is stated single-step-conditional.
- **Analyses:** each emits a JSON. Sanity gates: factorization closes on the
  deployed n=1170 numbers (product = micro recall to rounding); internal ceiling CI
  contains 0.735; anchor Δ sign is negative and its CI excludes 0; external uncapped
  ceiling ≥ capped 0.37.
- **Guardrails** become an explicit reviewer checklist appended to the plan; the
  final whole-branch review checks every Global Constraint above.
- `make test` stays green (analyses are scripts; no core-model behavior changes).

## Out of scope (record, do not build)

- **T2** — match-protocol refinement poset / monotonicity lemma (optional §Supplement
  formal spine; pure-writing but a 4th theory piece the user did not request).
- **Paired CI on the +0.067** filter-vs-listwise gap (needs a reranker re-run;
  Prop 1 cites the committed aggregate instead).
- **Budget-matched leaderboard** (equal mean output size across methods).
- Any new training / GPU / Modal run.

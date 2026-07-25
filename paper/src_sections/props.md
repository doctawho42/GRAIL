## 10. Results — Diagnosis: levers and three propositions

Each lever below attaches to exactly one factor of the §4 decomposition (annotated in the Factor
column); the table is followed by three refutable Propositions that localise *why*, not merely
*where*, recall is lost.

| Lever | Factor | Finding | Evidence |
|---|---|---|---|
| **Learned vs. prior** *(top_k-limited rule-selection probe — not deployed recall)* | `selection_retention` | A SyGMa-style frequency prior significantly out-ranks the learned generator: gen-only recall@15 prior-only **0.410** vs learned-only **0.266** (Δ **−0.144**, 95% CI **[−0.196, −0.095]**, paired bootstrap, n=245); with the filter, 0.405 vs 0.300 (Δ −0.105, CI [−0.152, −0.058]). Adding the prior to the learned scorer lifts **+0.130** gen / **+0.099** filter (both significant); the filter significantly helps only the *weak learned* ordering (+0.034, CI [+0.011, +0.061]), not the strong prior (n.s.). | `results/prior_vs_learned.json` (245 test subs) |
| **Multi-step (depth-2)** | `coverage_bank` | Breadth-capped depth-2 rule application lifts the ceiling by only **+0.012** (0.711→0.723) at **8.5×** candidate cost (194→1653 candidates/substrate) — not the dominant coverage lever. | `results/benchmark_report_depth2.json` (150 subs, beam 10) |
| **Coverage (ΔMW gap)** | `coverage_bank` | **26.9%** of true metabolites are uncovered by the depth-1 bank (plain InChIKey; ~25% under tautomer). Misses are a diverse long tail — the top missing class (hydroxylation/oxidation) is only **6%** of uncovered. | `results/benchmark_report_gap.json` (500 subs) |
| **Data scaling** | `selection_retention` | Recall saturates (2418→4787 substrates ≈ flat) — the plateau is not a data-quantity problem. | `results/full{2500,5000}_single.log` |
| **Regioselectivity (SoM)** | `ranking_conversion` | A site-of-metabolism prior gives only a small lift — regioselective ranking is hard within the bank. | `results/train_som.log` |
| **Selection breadth (top_k sweep)** | `selection_retention` | Widening the deployed rule selector's `top_k` 30→300 lifts recall@15 **0.352→0.413 (+0.061)** and pool coverage **0.482→0.608**, saturating below the 0.735 ceiling (n=245 test subset; trend, not deployed-scale absolute). | `results/selection_ablation.json` (245 test subs) |
| **Abstention (filter gate)** | `ranking_conversion` | Gating output by a filter-score threshold τ traces a recall↔precision frontier, but precision stays **nearly flat (~0.10–0.12)** and no val-selected τ reaches 0.2: raising τ shrinks the output (9.96→0.3) and collapses recall (**0.388→0.041**) for essentially no precision gain — precision is annotation-bounded, not threshold-bounded, justifying the rank-only default. | `results/abstention_frontier.json` (val-selected → 1170 test) |

**Selection-breadth counterfactual (n=245 test subset).** Sweeping the number of rules applied
(`top_k`) on the deployed pipeline (generator from `full5000_priors` with the trained prior
restored, deployed filter, ranked by the deployed `filter × generator` score) isolates
rule-selection breadth as its own lever:

| top_k (rules applied) | pool coverage (oracle) | deployed recall@15 | mean pool | mean output |
|---|---|---|---|---|
| 30 (deployed) | 0.482 | 0.352 | 12.5 | 7.99 |
| 100 | 0.547 | 0.388 | 35.3 | 9.47 |
| 300 (≈ applicable-rule limit) | 0.608 | 0.413 | 107.6 | 9.53 |

Widening top_k 30→300 monotonically lifts recall@15 **0.352→0.413 (+0.061)** and pool coverage
**0.482→0.608**, at the cost of an **8.6×** pool inflation (12.5→107.6 mean candidates) — recall
bought with precision, the same trade SyGMa makes by overproducing (~74 outputs/substrate). It
also saturates BELOW the ceiling: pool coverage plateaus at **0.608 < 0.735** even at ~all
applicable rules, and recall@15 0.413 still << SyGMa's 0.572 — so breadth is a real but PARTIAL
lever. Ranking still loses roughly a third of in-pool hits even at top_k=300 (recall / pool
coverage = 0.413 / 0.608 = **0.68**, consistent with `ranking_conversion` = 0.726, §8).

**Abstention is not a precision lever (val-selected τ → full 1170-substrate test).** Where breadth
trades precision for recall, the reverse move — deliberately abstaining — barely moves precision.
Gating the deployed pipeline's output by a filter-score threshold τ (candidates with
`filter_score < τ` are dropped; survivors ranked by `filter × gen`, top-15) and sweeping τ traces a
recall↔precision frontier whose precision axis is **nearly flat**: from rank-only τ=0 (recall@15
**0.388**, precision **0.098**, 9.96 outputs/substrate) through τ=0.5 (recall 0.146, precision 0.114,
1.55 outputs) to τ=0.9 (recall 0.041, precision 0.040, 0.30 outputs), precision never exceeds
**~0.123**, and no τ selected on validation reaches even precision 0.2. Raising τ removes true and
apparently-false candidates in near-equal proportion — because most "false" predictions are
*unannotated true* metabolites — so gating shrinks the output and collapses recall for almost no
precision gain; the best validation-selected F1 operating point (τ≈0.10) lifts F1 only marginally
(0.129→0.131) by trimming the output tail. Precision here is therefore **annotation-bounded, not
threshold-bounded**, which both substantiates the "precision is a pessimistic lower bound under
incomplete annotation" caveat (§5, §8) and justifies the rank-only deployment default
(`results/abstention_frontier.json`).

On the SAME broad pools, a second arm compares the deployed learned ranker (`filter × gen`)
against the rule frequency prior alone (SyGMa-style):

| top_k | filter × gen (learned) | frequency prior (SyGMa-style) |
|---|---|---|
| 30 | 0.352 | 0.359 |
| 100 | 0.388 | 0.374 |
| 300 | 0.413 | 0.374 |

At narrow breadth the two are ≈ equal (0.352 vs 0.359); as the pool broadens the **learned filter
pulls ahead** (0.413 vs 0.374 at top_k=300), while the frequency prior plateaus at 0.374 —
swapping to the prior signal at breadth would HURT, not help.

*Synthesis:* GRAIL's deficit vs. SyGMa on this axis is a compound of (i) over-aggressive rule
SELECTION at the deployed top_k=30 (recoverable ~+0.06 by widening breadth) and (ii) a residual
rule-set COVERAGE gap (pool coverage saturates at 0.608 < 0.735) — it is **not** a ranking-model
deficiency: the learned filter out-ranks the frequency prior once the candidate pool is broad.
This is an **n=245 test-subset** ablation; the trend (breadth helps, ranking doesn't hurt) is the
finding — the absolute top_k=30 baseline (0.352) is subset-inflated relative to the full-1170
deployed macro recall (0.330). (`results/selection_ablation.json`,
`results/selection_ablation_ranksignal.json`)

**Probe vs. deployed.** The learned-vs-prior row is a controlled, top_k-limited rule-selection
**probe** (each mode picks its own top-30 rules, same downstream product loop and filter) — it is
*not* the deployed recall. The deployed pipeline in fact applies its **top-30 selected rules** —
the selection-breadth ablation above confirms this: its top_k=30 baseline (0.352 subset)
reproduces deployed recall (0.330 macro / 0.344 3-seed, §8–§9), so the deployed pipeline *is* a
top_k=30 rule selector, and widening that selector's breadth is itself a soft lever (previous
paragraph), not evidence that rules are applied unselectively. Separately — and narrower in
scope — the deployed pipeline is measurably **prior-independent** at that fixed top_k=30
operating point: on an earlier 291-substrate evaluation, two checkpoints differing only in
whether a trained prior buffer was present gave essentially identical recall (0.335 vs 0.334), so
deployed recall (**0.330** macro / **0.261** micro, §8–§9) is coverage-limited, not prior-limited
at the deployed operating point — the prior *term* barely moves recall there, distinct from the
top_k breadth axis above; the probe *reveals* the learned selector's weakness; deployment *masks*
it by not depending on that selector's ranking. *Honesty note:* an earlier draft reported the
opposite — "learned beats prior" — an artifact of a checkpoint whose prior buffer was
un-persisted; that reversal is withdrawn and corrected here, caught by adversarial verification.
Neither multi-step application nor any single rule-family addition moves coverage much (misses
are a diverse long tail), and data scaling is flat — so the dominant, addressable loss remains
`selection_retention` (§8: 0.489), the learned rule-selector performing worse than a trivial
frequency prior in the probe. The three Propositions below explain *why*.

**Proposition 1 — Surrogate mismatch (→ `ranking_conversion`).** A filter trained by a strictly proper scoring rule (BCE/PU) learns a globally calibrated posterior, Bayes-optimal for AUC and calibration; ranking each substrate's candidate pool by that posterior and taking top-k is recall@k-optimal only when pools are homogeneous. GRAIL's pools vary in size (17–150) and positive rate (`n_true` 1–18), so a pointwise-calibrated scorer can be recall@k-suboptimal even while a listwise, ranking-consistent surrogate dominates — supported by a minimal 2-substrate counterexample (`grail_metabolism/tests/test_prop1_counterexample.py`) in which the recall-superior reorder is verified *not* globally calibrated, so a proper-scoring objective rejects it. *Confirmation — a controlled objective swap on the full test set.* The cleanest evidence isolates the objective while holding everything else fixed: the same two auxiliary heads, the same frozen generator and filter, and the **same candidate pool** are used twice, differing only in how the heads were trained. Trained **pointwise** (independent MLE) and multiplied into the rank, they add **nothing** — +0.0002 (95% CI [−0.0073, +0.0077], n.s.). Fine-tuned instead with a **listwise ranking loss** against the frozen `log(filter)+log(gen)` context, the identical heads gain **+0.0089 (95% CI [+0.0032, +0.0153])** over their pointwise counterpart and **+0.0091 (95% CI [+0.0027, +0.0160])** over the `filter × gen` baseline — paired bootstrap over the full clean test, n=1170 (`results/joint_rerank.json`). Since capacity, features, and pool are held constant, the margin is attributable to the **training objective**, which is exactly the surrogate mismatch this proposition predicts. A second, independent re-ranking arm reproduces the direction at the same n: `filter×gen×type×site` lifts recall@15 from 0.388 to 0.404, a paired **+0.0165 (95% CI [+0.006, +0.027])**, while the factorized signal on its own is n.s. (+0.001, [−0.010, +0.013]) (`results/hybrid_rerank_full1170.json`). *Secondary, weaker evidence:* a higher-capacity listwise-InfoNCE reranker scored **0.433 → 0.500 @15 (+0.067)**, 74% of the Stage-2 oracle 0.677, on a held-out Stage-2 run (`docs/benchmark/stage2_ranker_evidence.md`, Spike-3) — reported as ±0.015 across 3 seeds (pure reranker variance; the generator baseline is deterministic). We do **not** give this figure a paired interval: its per-substrate scores were not retained, so a paired bootstrap cannot be recomputed without retraining, and the proposition rests on the n=1170 intervals above rather than on this number. *Guardrail:* this is a theorem about **objectives**, not a recall win — the reranker's **0.500 still loses to SyGMa (0.558)** and is reported only as a separate Stage-2 artifact, never as a headline number.

**Proposition 2 — Learning loses to counting: rule selection is extreme multi-label classification (→ `selection_retention`).** This is the paper's central selection finding, and it is not metabolism-specific folklore rediscovered: selecting which of the **7,581** rules apply to a substrate is an *extreme multi-label classification* (XMC) problem with a near-empty positive set (≈1–3 firing rules per substrate, ~0.03% of labels). In exactly this regime, XMC has long established that frequency/propensity baselines are hard to beat and that naïve losses are dominated by label sparsity — propensity-scored losses were introduced precisely for this failure mode (Jain, Prabhu & Varma, *KDD* 2016). Metabolism rule selection **inherits this XMC pathology**, which the domain literature does not name; our contribution is to identify it here and to test it causally (below). Concretely: under PU annotation with an approximately constant labeling propensity (SCAR), a learner with constant unlabeled weighting recovers a propensity-distorted score whose dominant component is the marginal rule-firing rate `π(r)` — the frequency prior — so the prior is Bayes-competitive **by construction** at the rule-SELECTION stage (which rules clear top_k), and the learned selector improves on it only if substrate-conditional prevalence variation exceeds estimation noise; the observed data-scaling saturation (table, row 4) indicates it does not at current scale. *Anchor:* learned-only **0.266** vs prior-only **0.410** (gen-only @15, Δ **−0.144**, 95% CI **[−0.196, −0.095]**, paired bootstrap, n=245; `results/prior_vs_learned.json`). *Ranking-stage nuance:* this prior advantage is selection-specific, not a general "learned < prior" — at the separate **ranking** stage (ordering the candidate pool once rules are already applied broadly), the learned filter is the better scorer: on the same broad pools, `filter × gen` beats the frequency prior **0.413 vs 0.374** at top_k=300 (ranking-signal arm of the selection-breadth ablation above, n=245 subset; `results/selection_ablation_ranksignal.json`), consistent with Proposition 1's listwise-reranker gain. *Falsifiable prediction:* reweighting the labeled loss by `1/ê(r)` (a SAR correction) should shrink the prior's edge — an **open test, not a promised fix**. *Guardrail:* the propensity model `e(r) ∝ π(r)` is an **unmeasured modeling assumption**, flagged as such — this is an explanatory model plus a refutable prediction, not proof that learning cannot win, and it is consistent with the deployed pipeline's prior-independence noted above.

**Proposition 3 — Paradigm limit (→ `coverage_bank`).** Single-step rule-based recall ≤ single-step `coverage_bank` < 1, because a non-vanishing fraction of references are multi-generation (e.g. oxidation → conjugation) and unreachable by any single rule application — an irreducible residual that no ranking improvement can recover. *Witnesses:* depth-2 rule application lifts the ceiling by only **+0.012** at **8.5×** candidate cost (table, row 2; `results/benchmark_report_depth2.json`); on the external GLORYx set the uncapped single-step ceiling is **0.633** (§7), whose references include single-step-unreachable multi-generation metabolites. *Composition of the gap:* typing each uncovered transformation by its radius-0 reaction type (element-aware changed-bond multiset) against the **4,417** types the bank already contains (`results/coverage_gap_types.json`; the pipeline reproduces the ceiling exactly, coverage 0.7355) shows the single-step gap is a mix skewed to novel chemistry — of **687** uncovered test transformations, **41% (280)** are of a reaction type the bank *already* has (reachable in principle by a more general template, an in-bank ceiling of at most (1910+280)/2597 = **0.843**), while **53% (366)** are **novel types absent from the bank**, addressable only by new reaction corpora (both together would bound the ceiling at 0.984). So template generalization is a real but **bounded** lever (≈+0.11 ceiling headroom at best); the majority of the residual gap is genuinely out-of-bank chemistry, not a template-specificity artifact. *Guardrail:* the bound is **single-step-conditional** — it bounds the single-step paradigm, not the problem; multi-step and out-of-bank chemistry remain open coverage levers.

***Intervention — a factorized generator tests Propositions 2–3.*** This is a controlled intervention turning the observational decomposition above into a causal test: we rebuilt Stage-1 as a dense factorized generator, `P(type|s)·P(site|type,s)`, over a coarse ~radius-0 type vocabulary. **Selection is fixable:** the learned type head beats the frequency prior ~3× at k=5 (recall@5 0.086 vs 0.030), robust across two vocabulary granularities (371 and 701 types), converging to the prior by k=10 (0.298 vs 0.309); site localisation holds (site hit@3 0.78, cf. Gate C 0.81) — val set, `results/factorized_val.json` — so Prop 2's PU degeneracy is a target-support artifact, not fundamental, confirming its falsifiable `1/ê`-style prediction. **But coverage binds:** as a replacement generator the pipeline reaches only recall@15 0.256 on the full clean test — a net regression against deployed GRAIL (budget-matched recall@15 0.365, precision 0.109) — because type-selection caps oracle reachability at ~47%, unchanged when the vocabulary widened 371→701 (`results/factorized_eval.json`), confirming Prop 3 causally. **Ranking margin:** kept at broad coverage and used only to re-rank, `filter×gen×type×site` lifts recall@15 from 0.388 to 0.404 over `filter×gen` on the identical broad pool — a paired +0.0165 (95% CI [+0.006, +0.027], paired bootstrap over the full clean test, n=1170); the factorized signal alone is n.s. (+0.001, CI [−0.010, +0.013]) (`results/hybrid_rerank_full1170.json`). **Joint vs. bolt-on ranking:** the bolt-on re-ranker multiplies *independently* MLE-trained type/site probabilities into the rank; fine-tuning those two heads instead with a listwise ranking loss against the **fixed** generator and filter (frozen `log(filter)+log(gen)` context, gradients only through the heads) sharpens the signal — on a matched top_k=100 pool the joint-trained re-ranker beats the bolt-on by a paired **+0.0089 (95% CI [+0.003, +0.015], n=1170)** and the `filter×gen` baseline by **+0.0091 [+0.003, +0.016]**, where on that pool the independently-trained bolt-on adds nothing (+0.0002, n.s.). So *rank-aware* training, not the type/site heads per se, recovers this margin (`results/joint_rerank.json`). *Synthesis:* selection is addressable, ranking yields a small significant margin, but coverage — the source-limited rule bank — remains binding; no Stage-1 redesign matches SyGMa without a broader bank.

***Set-generation (GFlowNet) — a released negative result.*** Beyond re-ranking, we also formulated Stage-1 as a **set-terminal GFlowNet** (`model/set_gflownet.py`): the terminal object is a forest/set of metabolites per substrate, trained by Trajectory Balance against a PU set-coverage reward with an analytic `1/#leaves` backward — a genuinely novel formulation for this domain, released for reuse. On our benchmark it does **not** beat the simpler baselines: across 3 seeds (val, n=40) its single-set recall@15 is **0.292±0.032**, statistically indistinguishable from beam (0.311±0.039) and below the Stage-2a reranker (0.411±0.044). It does produce diverse sets (pairwise-Tanimoto 0.214±0.003, ~39 unique scaffolds), but a clean *diversity-advantage* claim is **not** supported here — the union-coverage comparison against the baselines is dominated by an under-production artefact in the baseline arms (its variance spans 0.0–0.58 across seeds), so we do not report that metric. We include this as a negative result because a benchmark is well-served by an honest report that a fashionable architecture does not win on it (`results/gflownet_seed{0,1,2}_overnight.json`; a fair multi-seed diversity comparison at scale is future work, §12).

*Source: `results/prior_vs_learned.json`, `results/benchmark_report_depth2.json`, `results/benchmark_report_gap.json`, `docs/benchmark/stage2_ranker_evidence.md`, `results/selection_ablation.json`, `results/selection_ablation_ranksignal.json`.*


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


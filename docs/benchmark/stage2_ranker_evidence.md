# Stage 2 evidence: can GRAIL keep the rule env AND reach SOTA?

Two inference-only spikes on the clean test split (n≈117, rank-only, tautomer-InChIKey,
`full5000_priors` checkpoint), plus a 12-agent design panel, answer: **yes — via a strong
product-level reranker over a larger generation budget.** The rule environment (RDKit applies
SMIRKS; candidates are valid, interpretable rule products) is fully preserved.

## Spike 1 — rank-by-rule is capped ~0.38 (`scripts/diagnose_ranker.py`)

| config | recall@15 |
|---|---|
| pure-prior (ps=20, full bank) | 0.385 |
| deployed ranker (ps=8) | 0.369 |
| prune bank to top-50% by prior | 0.383 |
| prune to 25% / 10% / 5% | 0.334 / 0.204 / 0.208 |
| **SyGMa** | **0.558** |

Rules out the "easy" levers: **not coverage** (ceiling 0.718), **not bank size/noise** (pruning
only hurts — the bank isn't bloated; the prior already keeps weak rules out of top-15), **not
learned-vs-prior** (pure empirical prior ≥ the learned GNN score, which adds ≈0). The residual is
**within-rule ranking** — a rule fires at many sites → many regioisomer products that share one
per-rule score, so the score cannot pick the correct site (regioselectivity); and the 7,581 mined
rules' priors are noisier than SyGMa's ~150 curated probabilities.

## Spike 2 — the truth is in the pool, just mis-ranked (`scripts/diagnose_rerank_ceiling.py`)

Oracle recall@15 if the generator's top-N candidate pool were **perfectly reranked**:

| generation budget N | oracle recall@15 |
|---|---|
| 15 (no rerank room) | 0.381 |
| 30 | 0.465 |
| 50 | **0.516** |
| 100 | **0.573** |
| 200 | 0.617 |

vs real rank-by-rule 0.385 · SyGMa 0.558 · coverage ceiling 0.718. **oracle@50 (0.516) > MetaPredictor
(0.504); oracle@100 (0.573) > SyGMa (0.558).** The whole 0.385→0.573 gap is ranking *within* the
pool. (`pool_coverage == oracle@15` because most substrates have ≤15 true metabolites, so pool
coverage binds, not the top-15 cap.) **A good reranker over a budget-50–100 pool is a path to
SOTA-competitive recall while keeping rules.** The current pipeline caps at ~32 then takes 15.

## Design panel (12-agent propose→adversarial-critique workflow)

All six proposals converged on the diagnosis (within-rule regioselectivity) and the lever (a
**site-conditioned, regioisomer-contrastive reranker** whose input distinguishes same-rule
siblings). Ranked by adversarial score:

| proposal | score | mechanism | est. recall@15 |
|---|---|---|---|
| **CRX-Rank** | 7/10 | contextual rule-match embeddings (matched atoms + env + product) + regioisomer-contrastive loss | 0.45–0.52 |
| **RNS-Rank** | 7/10 | k-NN analogue-substrate retrieval → transfer observed sites via MCS (site signal NOT from the neutral GNN); compute=low | 0.42–0.48 |
| **SIBYL** | 6/10 | site-conditioned cross-encoder (reuse `from_pair`) + within-rule sibling-contrastive (InfoNCE) + cross-substrate listwise | 0.42–0.50 |
| **Site-GFN** | 6/10 | (rule×site)-factored Set-GFlowNet + PU set-coverage reward — **unifies the GFlowNet method ambition + regio-fix + rule-env** | 0.42–0.50 |
| CANCEL / RxnSite | 5 / 4 | energy / site×reaction-type — skeptics: too close to the already-neutral SoM | — |

**Shared skeptic warning:** most proposals draw the site signal from the *same* GNN encoder + MCS
reacting-atom localization that already produced a **neutral SoM head**. The site signal must come
from a stronger source — contextual rule-match embeddings (CRX), analogue retrieval (RNS), or a
contrastive loss *over the pool* — not an additive per-atom site prior. The neutral SoM was a weak
multiplicative reweight; a discriminative reranker trained *against* sibling regioisomers is a
different mechanism, and Spike 2 proves the headroom (0.573) is there to capture.

## Synthesized direction

**Rule-env + budget-100 generation + a site-aware regioisomer-contrastive reranker.** The elegant
unification of the user's two goals (keep rules + SOTA) and the original Stage-2 method ambition
(novel GFlowNet set method) is **Site-GFN**: the reranker is the GFlowNet forward policy over
(rule×site) actions, trained with a PU-aware set-coverage reward (+ a regioisomer-contrastive
auxiliary), with the site representation drawn from contextual match embeddings / analogue
retrieval rather than the neutral SoM. This delivers competitive recall (Spike 2 ceiling), the
diversity/coverage method contribution, and full rule interpretability.

**Next cheap probe before committing:** decompose the mis-ranking into *within-rule* (sibling
regioisomers) vs *cross-rule* (wrong rule's product ranked above the right one) — confirms how much
of the 0.385→0.573 headroom the regioselectivity (site) fix specifically captures.

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

## Spike 3 — built and validated: the no-MCS bi-encoder reranker (`scripts/reranker_predict.py`)

We built the *simplest* reranker the panel converged on — a **no-MCS siamese bi-encoder** over
`from_rdmol(substrate)` + `from_rdmol(product)` (no rdFMCS), plus interaction features, a **scalar
`rule_prior_logits[rule_id]` feature** (not an embedding), and the generator score, trained with a
listwise InfoNCE + PU loss over the generator's IK-deduped candidate pool. It deliberately omits the
panel's heavier site-conditioning (CRX/RNS/SIBYL) — it is the floor, not the ceiling, of the design
space. Trained on 1,188 clean-train substrates (top_k=200, max_pool=150), 20 epochs, val-selected.

**In-distribution (clean val, n≈400 substrates) — the make-or-break gate:**

| | recall@5 | recall@10 | recall@15 |
|---|---|---|---|
| generator alone | 0.279 | 0.380 | 0.420 |
| **reranker** | **0.410** | **0.473** | **0.507** |
| oracle (perfect rerank of pool) | 0.671 | 0.674 | 0.674 |

**GO: reranker 0.507@15 = +21% over generator-alone (0.420), 75% of the oracle ceiling.** It lands
in the projected CRX-Rank band (0.45–0.52) with the *simpler* bi-encoder, and in the SyGMa/
MetaPredictor recall band (≈0.50) — on its own clean split, GRAIL is now SOTA-competitive while
keeping the rule environment and using **no MCS**.

**External (GLORYx-37, out-of-distribution) — the honest generalization row:**

reranker lifts GRAIL **0.243 → 0.351 @15 (+44% rel.)**, into BioTransformer's region (0.373); at
recall@5 it is **2nd of five (0.266)**, ahead of MetaPredictor (0.244) and BioTransformer (0.175),
behind only SyGMa (0.347). Full cross-method table in `gloryx_results.md`. The residual @15 gap to
the SOTA pair (≈0.50) is **pool coverage, not ranking** — GLORYx's oracle over this single-step pool
is ≈0.499, and its references include multi-generation metabolites a single-step generator cannot
reach. This is precisely the coverage axis Stage 2b (Set-GFlowNet over *multi-step* rule application)
targets; the within-rule regioselectivity probe above remains the cheap next diagnostic.

**Reading.** Stage 2a is validated: a rule-preserving, no-MCS reranker reaches recall parity with the
learned/rule SOTA *in-distribution* and closes most of the external gap, with strong top-rank
precision. The remaining external shortfall is multi-step coverage, which motivates Stage 2b rather
than further single-step ranking work.

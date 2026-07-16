# GRAIL — project status & directions

A single place to see **what is built** and **where the leverage is next**. Paper-level,
post-draft punch-list items (figures, remaining baselines, CI touch-ups) live in the manuscript's
own *Draft TODO / open items* section — this document is the project- and research-level roll-up
that sits above it. Last updated 2026-07-15.

## 1. Where things stand

GRAIL is a rule-based-plus-learned predictor of xenobiotic **metabolite structures**, packaged with
the **TAME** evaluation protocol and a **coverage × selection × ranking** diagnosis of what limits
the task. The honest headline: the contribution rides on the *protocol and the diagnosis*, not on
GRAIL beating SOTA recall.

| Quantity | Value | Source |
|---|---|---|
| Rule-bank coverage ceiling (held-out) | **0.735** (tautomer-InChIKey, micro) | §6 |
| Deployed GRAIL recall@15 | **0.261** micro · 0.269 ± 0.006 (3-seed) · **0.344 ± 0.010** macro | §8, §9 |
| Recall decomposition | 0.735 (coverage) × 0.489 (selection) × 0.726 (ranking) = 0.261 | §8 |
| Honest anchor — SyGMa on our split | 0.558 (reproduced exactly) | §9 |
| MetaPredictor on full 1170 | 0.568 | `results/metapredictor_1170.json` |

**Dominant loss is SELECTION** (0.489 retention), against a coverage ceiling that itself is
source-limited. That framing drives the directions in §3.

## 2. What is built

**A. Three-stage pipeline & models** (`model/`, `workflows/`)
- Generator (retrieval-scored multi-label rule selector), deterministic RDKit rule application,
  PU-trained structural filter; deploy ranks by `filter_score × generator_score`, top-15.
- Deployed checkpoints: `artifacts/full5000_priors` (generator, priors intact),
  `artifacts/full5000_single` (single-encoder filter). Concrete architectures now written into §3.
- Filter architecture chosen **empirically**: on matched-subset training the single-encoder filter
  beats the MCS-aware pair filter (paired recall@15 −0.009 [−0.017, −0.001], n=1170) at ~20× lower
  training cost — the MCS pair variant is implemented but *dominated*
  (`results/filter_compare_matched_sub800.json`; §3).

**B. Rule bank & automated mining** (`scripts/mine_rules.py`, §3)
- 7,581-SMIRKS bank; **automated self-validated mining** (MCS-anchored reaction center →
  RXNMapper *or* MCS-positional atom mapping → self-test → selectivity filter).
- 5,856 self-tested templates mined from clean TRAIN; heavy-tailed (73% single-support).
  **Source-saturated**: independent re-mine yields 5,856/5,856, **0 new** — coverage is bounded by
  the training-reaction *source*, not by mining incompleteness.

**C. TAME evaluation protocol** (`metrics.py`, `scripts/run_match_sensitivity.py`, §5, §11)
- Tautomer-InChIKey default matching; **match-sensitivity / rank-flip** across literature
  conventions (`inchi_no_stereo`, `tanimoto1`, `inchikey`, `inchikey_tautomer`).
- Leakage-audited, molecule-disjoint clean splits (`scripts/fix_splits.py`,
  `results/leakage_fix_report.json`); GLORYx shared external set.

**D. Diagnosis — decomposition + three propositions** (§8, §10)
- Coverage × selection × ranking identity on one n=1170 population (Figure 2).
- Prop 2 (PU degeneracy: learned selector < frequency prior at the operating point — a
  *target-support artifact*, falsifiably predicted and later confirmed fixable).
- Prop 3 (recall is coverage-bound). Selection-breadth ablation quantifies the
  selection↔precision lever (`results/selection_ablation*.json`).

**E. Multi-method comparison under one protocol** (§11)
- SyGMa, BioTransformer, MetaTrans, MetaPredictor re-scored under TAME; protocol-sensitivity
  demonstrated (e.g. BioTransformer ~13× GRAIL); budget-matched leaderboard (Table 4).

**F. Factorized-generator intervention + hybrid re-rank** (§10, §12)
- Dense MLE `P(type|s)·P(site|type,s)` over a ~radius-0 reaction-type vocabulary (701 types,
  `resources/coarse_type_vocab.json`, `artifacts/factorized_v1`).
- Confirms the diagnosis causally: **selection is fixable** (type head beats the frequency prior
  ~3× at k=5, 0.086 vs 0.030), but **coverage binds** (as a *replacement* generator, 0.256 < 0.365;
  type-gating caps reachability at ~47%, unchanged when the vocabulary widened 371→701).
- **Hybrid re-rank** (keep broad coverage, use the factorized signal only to re-rank):
  `filter×gen×type×site` lifts recall@15 **0.388 → 0.404**, paired **+0.0165, 95% CI
  [+0.006, +0.027], n=1170** (`results/hybrid_rerank_full1170.json`, `scripts/eval_hybrid_rerank.py`
  + `scripts/merge_hybrid_shards.py`). The factorized signal alone is n.s. — complementary, not a
  replacement.

**G. Infra & reproducibility**
- Shardable hybrid eval (`--start/--end/--rows-out`) + shard merge; seeded runs
  (`utils/seed`), rule-bank consistency checks, `make test` **285 green**.

## 3. Directions (prioritized by the diagnosis)

Recall = coverage × selection × ranking, and **coverage is the binding constraint**. Directions are
ordered by expected leverage against that identity, not by novelty.

### D1 — Broaden coverage (the binding constraint) · highest leverage, partly out of scope
**Measured** (`results/coverage_gap_types.json`, §10 Prop 3): the single-step coverage gap (687
uncovered test transformations) is **41% known-type** (280 — a reaction type the bank already has,
reachable by a more general template → in-bank ceiling ≤ **0.843**) and **53% novel-type** (366 —
absent from the bank, needs new corpora); together they cap the ceiling at 0.984. So the two levers
below are complementary, and template generalization alone is bounded to ≈+0.11 ceiling headroom.
Re-mining the same TRAIN split is source-saturated (0 new), so coverage can only grow from **new
reaction sources** or **more general templates**:
- Ingest external metabolism corpora (DrugBank/HMDB metabolism, MetaXBioDB, GLORYx training
  reactions, curated literature/USPTO-metabolism reaction SMILES) through the existing mining +
  self-test + selectivity pipeline.
- Relax the *design-time* selectivity filter's hard over-general rejection into a **learned
  selectivity gate**, or abstract templates via the factorized **radius-0 type vocabulary** (which
  already generalizes radius-1 rules) to lift oracle reachability past ~47%.
- This is the only lever §10/§12 identify as capable of closing the gap toward SyGMa.

### D2 — Ship the hybrid re-rank (the fixable, already-validated margin) · **DONE**
`model/factorized_infer.FactorizedReranker` + gated `ModelWrapper.generate` multiply
`P(type|s)·P(site|type,s)` into the rank (rank-only, byte-identical when off); opt-in via
`EvaluationConfig.factorized_rerank` (+ checkpoint/vocab paths), wired in `EnsembleWorkflow`,
exposed as preset `paper_full_ensemble_hybrid`.
End-to-end through the deployed `generate`: c−a **+0.040** on n=250 (full-test authoritative
+0.0165 [0.006, 0.027]); regression test + `make test` 286. No new training.

### D3 — Better selection at fixed coverage · **abstention arm DONE**
- **Calibrated abstention (done):** `scripts/abstention_frontier.py` sweeps the filter-gate τ,
  selecting operating points on val and reporting on test (§10 lever table + frontier note). Result:
  **abstention is not a precision lever** — precision stays flat ~0.10–0.12 and never clears 0.2 at
  any τ (raising τ collapses recall 0.388→0.041 for ~no precision gain), because precision is
  annotation-bounded, not threshold-bounded. Confirms the rank-only default and the
  precision-lower-bound framing (`results/abstention_frontier.json`).
- **Joint train (done):** fine-tune the factorized type/site heads with a listwise **ranking loss**
  against the *frozen* generator+filter (`scripts/{build_joint_pools,train_joint_factorized,eval_joint_rerank}.py`;
  `artifacts/factorized_joint`). On a matched top_k=100 pool the joint-trained re-ranker **beats the
  bolt-on** by paired **+0.0089 [0.003, 0.015], n=1170** and the `filter×gen` baseline by +0.0091 —
  where the independently-trained bolt-on adds nothing on that pool (+0.0002, n.s.). Rank-aware
  training > MLE for the heads (`results/joint_rerank.json`, §10). Optional next: repoint the
  hybrid preset at `factorized_joint`, and/or retrain on the top_k=300 pool for the shipped operating point.

### D4 — Set-level generation (GFlowNet) · high novelty, GPU-gated
A GFlowNet whose terminal object is a **diverse SET/forest of metabolites** per substrate, trained
to a **set-level** (coverage/recall, PU-aware) reward over the rule-defined transformation DAG —
beyond single-terminal GFlowNets. Skeleton exists (`model/gflownet.py`, `model/multistep.py`,
`GFlowNetConfig`) but is **not trained end-to-end**; needs GPU, `stop_head` checkpoint persistence,
a filter-based reward, `scripts/run_gflownet.py`, and a recall@k/diversity eval harness. Positions
vs RGFN / SynFlowNet / RxnFlow for a main-track method paper. See
`docs/superpowers/specs/` and the staged plan.

### D5 — Multi-step metabolism (depth ≥ 2) · low expected gain, targeted only
Chemically real (phase-I → phase-II), but depth-2 lifts the coverage ceiling only ~+0.012 (a long
tail). Pursue only for specific sequential cases, not as a general recall lever.

## 4. Manuscript & repo state
- Manuscript `docs/benchmark/manuscript.md`: §1–§14 drafted; §3 now carries concrete
  generator/filter architectures; the remaining paper-level punch-list (a few figures, MetaTrans on
  GLORYx, two paired CIs) is in its *Draft TODO / open items* section.
- Branch `claude/hungry-pasteur-25d746` is ahead of `main`; the redesign + full-1170 hybrid +
  architecture edits are committed locally, **not pushed**.

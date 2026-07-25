# GRAIL — project status & directions

A single place to see **what is built** and **where the leverage is next**. Paper-level,
post-draft punch-list items (figures, remaining baselines, CI touch-ups) live in the manuscript's
own *Draft TODO / open items* section — this document is the project- and research-level roll-up
that sits above it. Last updated 2026-07-25. **Headline result: §0b (GRAIL vs MetaTox — tie, and strong complementarity).**

## 0. Post-external-review reframe (2026-07-17) — venue = D&B track

An external conceptual review pushed a grand reframe: *"recall ranks banks/enumeration, not models;
the leaderboard is an artifact of two conventions (match protocol + no output-size normalization)."*
We ran its **decisive falsification test before writing the abstract** — the budget-matched frontier
(`results/budget_matched_frontier.json`): **SyGMa dominates GRAIL at EVERY output budget k (crossover
k=1; GRAIL@64 0.41 < SyGMa@64 0.52)**. So the reframe is **FALSIFIED** — SyGMa is genuinely a better
metabolite predictor at matched budget; GRAIL's bank covers more (0.701 > SyGMa's pool 0.520) but its
selection+ranking cannot convert it. That is our own **P1 (weak selector)**, not an unfair metric.

**Retraction (same over-reach, caught):** "P2 in cross-method form" (SyGMa's prior-ranking beats
GRAIL's learned ranking across the frontier) is **not clean** — different banks/pools, and at k=64
SyGMa isn't ranking at all (dumps its whole pool). The clean P2 is the **internal** same-pool result
(learned 0.266 vs prior 0.410 on GRAIL's own pool). Do **not** claim a cross-method P2 without a
same-pool ranker test (GRAIL pool × {learned, prior, random}).

**Chosen framing — A, a Datasets & Benchmarks paper (A* venue; Syntheseus→NeurIPS'23, GuacaMol/MOSES
precedent).** What survives and carries it: **TAME rank-flip** (+0.120 [0.073, 0.171], Syntheseus-grade),
audited split, 5-method comparison under one protocol, the **coverage×selection×ranking decomposition**
(mechanism), **coverage provably corpus-limited** (re-mining TRAIN → 0/5856 new), **P2 as an XMC-framed
section** (rule selection = extreme multi-label with ~0 positives/substrate; dense reformulation → 3×
prior), and a **self-measured below-SOTA ablator** — the benchmark's authors ran their own method first
and reported it loses at every budget (falsification as a design principle, not a scar).

**Upside = TAME's generality — measured, and it is domain-attenuated.** The plan was: re-score external
predictions under the 5 match modes; if rank-flip reproduces elsewhere, TAME becomes a measurement
phenomenon of molecular ML. The single-model precursor **ran** (`results/xdomain_retro_protocol.json`,
ReactionT5v2 on USPTO-50k, n=200): top-1 moves only **~0.02** across {canonical, no-stereo, InChIKey,
tautomer} — an order of magnitude below metabolism's +0.120. So a *robust multi-method rank-flip in
retrosynthesis is unlikely* (movements too small to reorder well-separated methods). The honest,
defensible cross-domain claim is therefore **not** "protocol choice universally reorders leaderboards"
but "its magnitude scales with how far a generator's outputs diverge tautomerically/stereochemically
from the references" — large in metabolism (rule engine emits tautomers), small in canonical-clean
retrosynthesis. Written up in manuscript §12; **do not over-invest expecting a large cross-domain flip.**

**Order:** (1) ~~cross-domain rank-flip pilot~~ — done, weak, reframed above; (2) 1170 run overnight as a
*supporting* table (gates nothing); (3) optional same-pool ranker test (only if a cross-method P2 is
wanted); (4) abstract under D&B — TAME center, P2 as XMC section, GRAIL as self-measured ablator.
**Discipline adopted:** name the falsifying run *with* the thesis, not after.

**Lit-check verdict (2026-07-19, external review + verified sweep).** An external conceptual review
endorsed the plan — *write the D&B paper, TAME center, GRAIL as self-measured ablator, Set-GFlowNet to
future work* — and its own literature check returned "solid, publishable D&B contribution; not a
landmark (the neighbourhood is crowded and standardizing), but better positioned than the alternatives:
D&B venues **reward** 'your eval is broken + here is the fix', so non-SOTA is a feature there, and
metabolite-structure prediction is less saturated than retrosynthesis/property prediction."
We then ran our own verification sweep (8 agents, web-verified) rather than trusting either side:
- **All 4 reviewer citations are real** (Zagribelnyy *arXiv*:2602.03554 ICML'26; *Molecules* 31(5):769
  2026; MADGEN ICLR'25 *arXiv*:2501.01950; GuacaMol JCIM'19) — zero hallucinations, but none sits on
  TAME's axis; they are background/adjacent, not competing claims.
- **Sharpest neighbors were already cited + differentiated**: Syntheseus (Maziarz, the closest on the
  *axis*), Scholz 2023 and JCIM-ADME 2026 (the same in-domain 5-tool leaderboard).
- **Two genuinely-new concurrent neighbors added** to §2 after independently verifying both arXiv IDs:
  Agarwal & Bisht 2026 (*arXiv*:2606.12639, closest on *claim-shape* — metric choice inverts the winner;
  differs by axis/task/no decomposition) and Liu, Bushuiev et al. 2026 (*arXiv*:2606.19624, MassSpecGym
  in the Wild — implementation-side audit; differs by fixed-implementation vs controlled-convention).
- **Verdict:** the "overlaps uncited prior work" exposure was real and is now **neutralized**. *No prior
  work makes TAME's exact claim* — convention × method interaction, quantified with CI, pre-registered,
  on a leakage-audited split, in metabolite structure. The field is crowded on the *general* worry and
  empty on the *specific quantified instance*.
- **Standing risk (reviewer, accepted):** a paper whose thesis is evaluation rigor gets judged hardest on
  its own eval hygiene. See §0a's test-peeked caveat — exploratory runs go to val, test stays frozen for
  the single final MetaTox row.

## 0a. Rule-granularity probe (2026-07-19) — measured, mostly negative

Tested whether **rule granularity** (bank breadth/redundancy) is a lever on the recall factors
`coverage(0.735) × selection(0.489) × ranking(0.726)`. Four measurements, each pre-registered with its
falsifier; three of my own hypotheses were tempered or killed by the numbers.

**Methodology caveat — test peeked.** These exploratory probes (`rule_prune_probe`,
`ablate_id_embedding`, and the earlier budget-matched / cross-method runs) scored on the **test** split
(`test_predictions.csv`) — a soft breach of *touch-test-once*. Consequence: the final GRAIL-vs-MetaTox
row is now slightly adaptively biased. Mitigation going forward: **test is frozen for that single final
row** (spent on §0b); any follow-up exploratory run goes on **val**, and the val GRAIL predictions
now exist (`artifacts/full5000_single/predictions/val_predictions.csv`, 994 substrates, produced by
`dump_val_predictions.py` through the same code path as the test dump).

Verdict per claim, at measured confidence:

1. **Merging functional duplicates → SELECTION.** Redundancy is real: of the 2,747 rules that fire on a
   1,000-substrate probe, 25% are exact functional duplicates (`rule_collapse.json`, held-out merge
   stability 0.975); a distribution-free canonical-SMIRKS dedup over all 7,581 removes ~7.5%
   (`rule_dedup_provable.json`; permutation variants are only ~43% of the redundancy, the rest is
   non-structural). **But the net selection benefit is unconfirmed:** `label_reactions` marks every
   producing rule positive and each rule has its own `id_embedding`, so duplicates split the training
   signal — yet `noisy_or` at inference pools the duplicate scores back (multiplicity ≈ frequency prior),
   partially healing the split. And the zero-id ablation (below) shows the split `id` is itself inert.
   **Verdict: weak lever; treat as deploy hygiene**, which folds into the resource+cache measurement
   (rule embeddings are substrate-independent and already cached once at startup — bank size hits
   startup cost/memory, not per-query latency).

2. **Pruning dead rules.** *The reducible-sparsity fact is solid; the selection mechanism moved, and the
   pruning scope shrank.* Training-wide (4,787 substrates, `rule_train_positives.json`): **56.3% of the
   bank is NEVER a positive label** and **20.1% is positive on exactly one substrate** → **~76% of the
   TARGET is train-dead-or-singleton**, i.e. *training* sparsity is reducible (P2-relevant, solid). Two
   cautions the self-audit forces, both previously over-stated:
   - **Reachability (train/test confounder, third occurrence).** "Prunable without coverage loss" holds
     only for the *training target* — all-negative columns are pure train ballast. It is **not proven for
     the deployed bank**: a rule useless on train can be a true label on test/novel chemistry (e.g. a
     curated SyGMa general rule that never matched train). So "76% of the *bank* prunes" is unmeasured and
     carries **test-coverage risk**; only "76% of the *target* is train-dead-or-singleton" is solid.
   - **Mechanism (undercut by the same shot as merging).** The zero-id ablation shows `id` is inert, so
     "dead rules inject noise via untrained `id`" is *also* inert — the target-sparsity→selection story is
     undermined. What survived was a **different, id-independent, inference-side** hypothesis: dead rules
     emit **distractor products** that dilute ranking, so cutting them could raise recall.
   - **RESOLVED — and the lever is FALSIFIED (2026-07-19, `results/prune_and_rerank_val.json`).** Pruning
     the 4,271 training-dead rules (bank 7,581 → 3,310) and re-ranking on **val** (n=994, tautomer,
     deployed operating point, prune set derived from TRAINING positives only) **hurts on both axes**:
     recall@15 **0.388 → 0.352** (Δ −0.036, 95% CI [−0.050, −0.024]) and pool coverage
     **0.439 → 0.378** (Δ −0.061, CI [−0.078, −0.046]); mean pool 46.9 → 40.0. Both CIs exclude zero.
     So the pre-registered **reachability-loss** branch won, not distractor-cut: rules that never earn a
     positive label across 4,787 training substrates **still produce true metabolites on unseen val
     chemistry**. The §0a caveat is now measured rather than hypothesised — *"76% of the TARGET is
     train-dead-or-singleton" stands; "76% of the BANK is prunable" is FALSE.* The bank's apparent dead
     weight is generalization capacity living in its tail, which also argues against bank-shrinking as a
     deployment tactic (provable dedup remains safe; pruning does not). Note the ranking partially
     absorbs the damage — 6.1 points of lost coverage cost only 3.6 points of realised recall — but the
     net is unambiguously negative.

3. **`id_embedding` dominance → mechanism of P2. FALSIFIED.** The rule embedding is
   `norm(graph_encoding + id_embedding + meta)`; the `id_embedding` carries **82% of cross-rule variance**
   vs 17% for the GATv2 structure (`rule_embed_decomp.json`) — seductively "the model is a lookup table,
   rare-rule `id` is untrained noise, *that* is P2." The **zero-id inference ablation kills it**
   (`ablate_id_embedding.json`): zeroing `id` moves recall@15 by **−0.007** and *improves* recall@1/@5
   (+0.037/+0.027). The 82% variance is **not load-bearing** — a geometric illusion; downstream scoring
   looks past `id`. P2 is not explained here.

**Byproduct — VERIFIED ON VAL, and it does NOT replicate.** The single test run had zeroing `id`
*raise* top-k precision (+0.037@1, +0.027@5, n=291), which I downgraded to "anomaly, verify before
believing". Re-run on **val** with paired bootstrap CIs (n=994, 10k resamples,
`results/ablate_id_embedding_val.json`): **every k is n.s.** — @1 +0.0011 [−0.0086, +0.0107],
@5 +0.0055 [−0.0074, +0.0181], @15 −0.0116 [−0.0266, +0.0036], @30 −0.0119 [−0.0282, +0.0044].
The +0.037@1 was **single-run noise**; the "free tweak" is dead, exactly as the downgrade anticipated.
What the val run *does* confirm, now with CIs the original test run never had, is the main result:
zeroing 82% of the embedding variance moves recall by nothing distinguishable from zero at any k —
`id` is **not load-bearing**, and the `id`→P2 mechanism stays falsified on an independent split.
*(Method note: the first val attempt crashed at the CI step on a paired-vector alignment bug — the
per-substrate append was skipped for substrates with no candidates while `n` still counted them, so
the two arms dropped different substrates (991 vs 990). It only crashed because the counts differed;
equal counts would have produced a silently wrong CI. Fixed by recording recall 0 for empty
candidates, plus an explicit alignment guard.)*

**Net — the track is CLOSED, and negatively.** All three candidate levers were measured and none
survived: **merging** = weak/deploy-hygiene (net effect unconfirmed; `noisy_or` heals the training
split), **pruning** = actively harmful (falsified on val: −0.036 recall, −0.061 coverage, both CIs
excluding zero), **`id`→P2** = falsified (zero-id ablation). What remains is one solid *fact* (training
sparsity is reducible — 76% of the **target** is train-dead-or-singleton, which is *not* the same as the
bank being prunable) and one structural insight (the bank's tail carries generalization capacity, so
coverage really is the physical limit — consistent with §0's P1 diagnosis). Three of these outcomes
falsified hypotheses I had argued for; that is the intended function of pre-registering the falsifier.
Both deadline levers are now **done**: resource+cache (`resource_cache_profile.json`) and
GRAIL-vs-MetaTox (§0b). Scripts: `prune_and_rerank_val.py`, `rule_collapse.py`,
`rule_dedup_provable.py`, `rule_prune_probe.py`, `probe_rule_embeddings.py`, `ablate_id_embedding.py`.

## 0b. GRAIL vs MetaTox (2026-07-25) — the grant number: a TIE, and strong COMPLEMENTARITY

The #1 deliverable. MetaTox is the way2drug incumbent; this is the first head-to-head under one
protocol (tautomer-InChIKey, TAME). Predictions supplied as an SDF; join verified before scoring —
`ID_all = <structure_ordinal>_<metabolite_number>`, ordinal → our `SUB####` submission id, and
**268 of 270 returned parents match our submitted structure exactly** by canonical SMILES (the 2
exceptions differ only by MetaTox's internal aromaticity/charge perception), so the ordinal join is
unshifted. Metabolites are ranked by `Values` (combined biotransformation × site probability, max
over producing reactions). Artifacts: `results/grail_vs_metatox.json`,
`results/metatox_complementarity.json`, `results/metatox_preds.json`; scripts `compare_metatox.py`,
`metatox_complementarity.py`.

**Result 1 — statistical tie at every k.** On the substrates MetaTox returned (n=270):

| k | GRAIL | MetaTox | Δ (GRAIL−MetaTox) | 95% CI |
|---|---|---|---|---|
| 1 | **0.149** | 0.141 | +0.008 | [−0.050, +0.065] tie |
| 5 | **0.299** | 0.293 | +0.006 | [−0.067, +0.079] tie |
| 10 | 0.333 | **0.347** | −0.015 | [−0.092, +0.062] tie |
| 15 | 0.340 | **0.363** | −0.023 | [−0.101, +0.054] tie |

Counting the 21 substrates MetaTox returned nothing for as failures (n=291): GRAIL 0.334 vs MetaTox
0.337, Δ −0.003 [−0.076, +0.070] — also a tie, with GRAIL ahead at k≤5 (+0.017@1, +0.023@5). GRAIL
is more precise per output (precision@15 **0.101 vs 0.092**) with a smaller set (**8.5 vs 10.3**),
and the crossover sits at k≈10 — GRAIL leads where a UI actually shows results.

**Result 2 — the deployment-decisive finding: the two methods are strongly COMPLEMENTARY.** A tie
invites the wrong question ("which replaces which?"); the overlap answers the right one. On the 248
substrates where MetaTox produced metabolites:

| | recall@15 |
|---|---|
| GRAIL alone | 0.317 |
| MetaTox alone | 0.395 |
| **union** | **0.611** |

Adding GRAIL to MetaTox gains **+0.216 [+0.171, +0.263]**; adding MetaTox to GRAIL gains +0.294
[+0.247, +0.343] — both far outside zero. Census of correctly-found metabolites: **95 only-GRAIL,
170 only-MetaTox, 55 both — just 17.2% shared**, i.e. **83% of true hits are found by exactly one
method**. So the positioning is *GRAIL alongside MetaTox*, not instead of it: an integration that
raises the incumbent's recall by ~55% relative, rather than a replacement that must win.

**Scope caveat, load-bearing.** This is MetaTox **layer 1 only, WITHOUT its SMIRKS-rule variant**
(supplier's note; the SMIRKS version follows). That is not MetaTox's ceiling, and it bears directly
on the headline: **if the SMIRKS rules overlap GRAIL's bank, complementarity should shrink.** That
is the pre-named falsifier — re-run `compare_metatox.py` + `metatox_complementarity.py` unchanged on
the new SDF when it arrives. Until then, +0.216 is an upper estimate of the integration gain.
Secondary caveats: MetaTox returned nothing for 21/291 substrates and empty metabolite lists for a
further 22; and the test split was peeked during exploration (§0a) — this is the single final row it
was frozen for.

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
  training > MLE for the heads (`results/joint_rerank.json`, §10). **Shipped:** preset
  `paper_full_ensemble_hybrid` now points at `artifacts/factorized_joint` (the joint reranker) by
  default. Optional next: retrain the joint heads on the top_k=300 pool to match the shipped
  operating point exactly (current joint was trained/eval'd on top_k=100).

### D4 — Set-level generation (GFlowNet) · machinery built + validated e2e; result compute-gated
A GFlowNet whose terminal object is a **diverse SET/forest of metabolites** per substrate, trained
to a **PU set-coverage** reward over the rule DAG — beyond single-terminal GFlowNets. **Fully built
+ unit-tested** (`model/set_gflownet.py`: `ForestState`, `set_coverage_logreward` β/λ, analytic
`P_B=1/#leaves`, `StopHead`, TB loss; `scripts/run_gflownet.py` = train + dual-eval matrix
{gflownet, reranker, beam} at recall@K + diversity {modes, pairwise-Tanimoto, unique-scaffolds,
circles}). Positions vs RGFN / SynFlowNet / RxnFlow.
- **First end-to-end run (this session):** validated at `top_k=40` — produces the full dual-eval
  matrix. Three blockers surfaced (none showed in unit tests):
  1. `--bootstrap` (depth-2 chain enumeration) **hangs** (combinatorial) → use `--no-bootstrap`
     (optional fine-tune, M0-gated behind an unmet depth-2 census anyway).
  2. Unbounded forest-rollout caches **OOM at scale** → **FIXED**: bounded-LRU
     `child_cache_max`/`ik_cache_max` (commit `4239d3a`, test 287).
  3. `top_k=200` forest rollout **crashes natively/silently** (~1.8 GB RSS, RAM 77% free → RDKit
     segfault, not OOM; cache fix didn't help) → **use `top_k<=40`** (stable regime).
- **Undertrained model under-produces** (near-empty forests → recall@15 0.0 at 2 epochs; reranker/
  beam 0.43). Needs ~25+ epochs; forest-eval is ~40 s/sample (union-stream retries when
  under-producing). A converged single-seed run is **~overnight on CPU**; multi-seed is
  compute-gated (Modal burned, GCP occupied — but this is CPU-bound, so GCP CPU-spot ≈ $1–2 total).
- **Tested runnable recipe:** `run_gflownet.py --no-bootstrap --top-k 40 --epochs ~25 --train-substrates 150 --eval-substrates 50 --n-samples 16`.
- **Converged + multi-seed result** (`results/gflownet_seed{0,1,2}_overnight.json`; 3 seeds, **VAL**,
  120 train / 40 eval / 12 samples / 25 epochs; ~3.6 h CPU/seed): the trained Set-GFlowNet **no longer
  under-produces** (recall@15 0.0→~0.29). But multi-seed **corrects the seed-0 over-read**:
  - **recall@15 mean±std:** gflownet **0.292±0.032** ≈ beam **0.311±0.039**, both < reranker
    **0.411±0.044** — the Set-GFlowNet does **not** beat the baselines on point recall (seed-0's
    "beats beam" was noise).
  - **diversity is real + stable:** pairwise-Tanimoto **0.214±0.003**, unique-scaffolds **38.7±1.9**,
    circles@0.4 **31.4±0.5**, modes 0.77±0.10.
  - **but no clean coverage *win*:** gflownet-union AUC 0.284±0.031 (tight) vs reranker-union AUC
    **0.266±0.237** (0.0–0.576 across seeds — corrupted by inconsistent under-production skips), so the
    seed-0 union edge does **not** survive as a defensible claim.
  - **Honest verdict:** the novel set-generation machinery works and yields diverse outputs, but on
    this data/scale shows **no robust recall or coverage advantage** over the simpler reranker. A clean
    comparative-diversity claim needs the EVAL-02 under-production guard fixed on the baseline arms +
    larger n / more seeds (GPU/compute). Stage-2 (method paper), separate from the diagnosis manuscript.

### D5 — Multi-step metabolism (depth ≥ 2) · low expected gain, targeted only
Chemically real (phase-I → phase-II), but depth-2 lifts the coverage ceiling only ~+0.012 (a long
tail). Pursue only for specific sequential cases, not as a general recall lever.

## 4. Manuscript & repo state
- Manuscript `docs/benchmark/manuscript.md`: §1–§14 drafted; §3 now carries concrete
  generator/filter architectures; the remaining paper-level punch-list (a few figures, MetaTrans on
  GLORYx, two paired CIs) is in its *Draft TODO / open items* section.
- Branch `claude/hungry-pasteur-25d746` is ahead of `main`; the redesign + full-1170 hybrid +
  architecture edits are committed locally, **not pushed**.

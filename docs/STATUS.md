# GRAIL — project status & directions

A single place to see **what is built** and **where the leverage is next**. Paper-level,
post-draft punch-list items (figures, remaining baselines, CI touch-ups) live in the manuscript's
own *Draft TODO / open items* section — this document is the project- and research-level roll-up
that sits above it. Last updated 2026-07-17.

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

**Upside = TAME's generality.** Reframe dead ⇒ TAME is the only measurement leg ⇒ all remaining ceiling
is whether **rank-flip reproduces in other domains** (retrosynthesis, molgen) — re-score existing
external predictions under the 5 match modes, no training. If it does, TAME becomes a measurement
phenomenon of molecular ML (main-track on different grounds). **This is the main next move.**

**Order:** (1) cross-domain rank-flip pilot on ONE external prediction set; (2) 1170 run overnight as a
*supporting* table (gates nothing); (3) optional same-pool ranker test (only if a cross-method P2 is
wanted); (4) abstract under D&B — TAME center, P2 as XMC section, GRAIL as self-measured ablator.
**Discipline adopted:** name the falsifying run *with* the thesis, not after.

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

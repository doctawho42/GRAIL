# GRAIL/TAME manuscript — first-draft design spec

> **Status:** design (2026-07-13). Feeds `writing-plans`. Assembles the **first coherent full draft** of
> the paper into `docs/benchmark/manuscript.md` from the already-verified no-compute material (~70% of the
> paper exists as internal notes). Compute-gated numbers are clearly-marked placeholders. Venue:
> **JCIM / J. Cheminformatics** (Q1 cheminformatics, rolling, GRAIL-primary framing).
> Context: `docs/benchmark/manuscript_readiness.md` (the section-by-section map this spec executes).

## Goal
Turn scattered, guardrail-voiced framing notes into one reviewable manuscript draft. This is **assembly +
de-guardrailing + prose**, not new science. Every number reconciles against `results/*.json`; where a
number is compute-gated or a citation unverified, insert an explicit placeholder rather than fabricate.

## Deliverable
- Create `docs/benchmark/manuscript.md` — the paper.
- Mark `docs/benchmark/paper_draft.md` and `docs/benchmark/DNB_FRAMING.md` **SUPERSEDED** (a one-line
  header note pointing to `manuscript.md`; do NOT delete — preserve history).

## Global constraints (binding; the final review checks each)
1. **All numbers come from `docs/GRAIL_FRAMING.md` / `results/*.json`, NEVER from `paper_draft.md`.** The
   stale draft's headline figures (ceiling 0.718, GRAIL 0.334/0.40, two-factor split, "MetaBench" name)
   are wrong-thesis — discard wholesale. Use ceiling **0.735** tautomer / 0.718 plain; GRAIL **0.330
   macro / 0.261 micro**; the three-factor decomposition; the name **TAME**.
2. **Honest anchor, verbatim in spirit:** GRAIL does NOT win on recall (0.330 macro / 0.261 micro < SyGMa
   0.572 < MetaPredictor 0.585). The paper's value = ceiling + diagnosis + protocol. State it early
   (abstract + intro). No claim, anywhere, that GRAIL wins.
3. **Micro/macro discipline** (as locked in the theory spine): the decomposition, Fig 2 waterfall, and the
   §Results conversion are **micro** (coverage 0.735 == ceiling; deployed 0.261; conversion 35.5%);
   cross-method recall (GRAIL 0.330, SyGMa 0.572, MetaPredictor 0.585) is **macro**; exactly ONE
   reconciliation sentence. No mixed-frame ratio.
4. **The factorization is a "decomposition"/identity — NEVER "theorem/verified/validated".** Preserve this
   honesty while converting to prose (collapse the repeated guardrail asides into one honest caveat).
5. **Reranker 0.500 (Prop 1) never near a headline; state it loses to SyGMa 0.558; a separate Stage-2
   artifact.** Prop 2's `e(r)∝π(r)` is an UNMEASURED assumption + `1/ê` an OPEN test. Prop 3
   single-step-conditional.
6. **Compute-gated numbers are placeholders, not fabrications.** Insert literal markers:
   - Headline deployed recall → `0.330 macro / 0.261 micro [PENDING: multi-seed mean±std over ≥3 seeds]`.
   - Any per-method comparison that mixes n → see constraint 7.
7. **The n=150-vs-1170 mismatch is scoped, not hidden.** The main cross-method table caption MUST state:
   "tier-2 comparators (BioTransformer, MetaPredictor, MetaTrans) are scored on the n=150 shared subset;
   GRAIL and SyGMa on the full n≈1170 clean test. A single-n rerun is future work." The honest-anchor
   comparison (GRAIL < SyGMa < MetaPredictor) must carry this caveat wherever it appears.
8. **Unverified citations are placeholders:** e.g. `[cite: MetaTrans — Litsa, Das, Kavraki 2020, Chem
   Sci; verify vol/DOI]`, `[cite: MetaPredictor — verify]`, `[cite: Dhaked 2019 / DataSAIL — verify DOI]`,
   `[resolve: "Gao 2026"]`. Verified ones (from `related_work_positioning.md`) go in as real cites: SyGMa
   Ridder & Wagener 2008 (doi:10.1002/cmdc.200700312); GLORYx de Bruyn Kops 2020
   (doi:10.1021/acs.chemrestox.0c00224); Boyce 2022 (doi:10.1016/j.comtox.2021.100208).
9. **NO Claude/AI/Co-Authored-By attribution** in any commit or the doc byline.
10. **Match mode `inchikey_tautomer`** is the paper's default; state it once in Methods.

## Manuscript structure & per-section spec
One `##` section per item. Sources are file paths / `results/*.json`; write PROSE (not note-voice).

### Title + Abstract
- Title: fresh, GRAIL-primary — conveys "rule-based metabolite-structure prediction, a coverage×selection×
  ranking diagnosis, and the TAME evaluation protocol." NOT "How You Match Decides Who Wins" (demoted).
- Abstract (~200 words): the ceiling 0.735; the coverage×selection×ranking decomposition (dominant
  selection loss); the honest anchor (GRAIL 0.330 < SyGMa 0.572 < MetaPredictor 0.585); TAME (tautomer-
  aware, leakage-audited, match-sensitivity). Source: `GRAIL_FRAMING.md` Thesis + Honest anchor (§7-31).

### 1. Introduction + numbered contributions
- Motivate around **coverage-conversion**, not "leaderboards aren't comparable." Honest anchor up front.
- Explicit numbered contributions (~4): (1) rule-bank coverage **ceiling** (0.735) as a diagnostic
  primitive; (2) an exact **coverage×selection×ranking recall decomposition** + three refutable
  propositions; (3) **TAME** — a standardized, tautomer-aware, leakage-audited matching protocol + a
  match-sensitivity ("rank-flip") analysis; (4) GRAIL as an interpretable, honestly-diagnosed instrument
  (one row, not a SOTA claim). Source: `GRAIL_FRAMING.md` Thesis.

### 2. Related Work
- Convert `docs/benchmark/related_work_positioning.md` to prose paragraphs: comparator methods (verified
  cites); prior multi-method comparisons (Scholz 2023, Boyce 2022) — **cite as corroboration** (their low
  precision, SyGMa overproduction, rule-vs-ML divergence, phase-2-harder all replicate ours); tautomer-
  matching prior art (Dhaked 2019, PubChem, QSAR-ready) justifying our protocol; leakage (DataSAIL);
  eval-robustness (Mishra 2021, Rodriguez 2021). Use the scoped-novelty statement (§6 of the memo).
  Unverified DOIs → placeholders per constraint 8.

### 3. Methods — GRAIL architecture
- Expand `GRAIL_FRAMING.md §1` (terse) using `docs/ARCHITECTURE.md` + `CLAUDE.md` invariants into prose:
  (i) generator — retrieval-scored multi-label rule selector over the 7,581-SMIRKS bank; (ii) RDKit rule
  application enumerating products; (iii) PU-trained, MCS-aware **pair filter** in the logit domain
  (PULoss/nnPU); deployment ranks by `filter_score × generator_score`. Note featurization dims + the
  positive-unlabeled setup. State `inchikey_tautomer` match here.

### 4. Methods — Formal framework (from §1.5)
- Source `GRAIL_FRAMING.md §1.5`. Write the generative latent-reaction mixture `P(m|s)=Σ P(r|s)·P(site|r,s)
  ·𝟙[apply=m]`, the stage→term mapping, and the **recall decomposition identity** `recall = coverage_bank
  · selection_retention · ranking_conversion`. De-guardrail: ONE "it is an accounting identity, not a
  theorem" caveat (not the repeated asides). Include the lever→factor map.

### 5. Methods — TAME evaluation protocol
- Source `GRAIL_FRAMING.md §Reproducibility` + `docs/benchmark/protocol.md` + the `paper_draft.md §4`
  worked example (D-alanine/acetone) — merge, rename MetaBench→TAME. Specify: the 5 match quotients
  (canonical / inchikey / inchi_no_stereo / tanimoto1 / **inchikey_tautomer** recommended), the
  molecule-disjoint leakage-audited split, the frozen multi-method re-scoring harness, recall@k +
  output-size co-reporting, select-on-val / touch-test-once.

### 6. Results — rule-bank coverage ceiling
- Source `GRAIL_FRAMING.md §2` + `results/benchmark_report.json`. Prose + a ceiling-vs-baselines table
  (ceiling **0.735** tautomer / 0.718 plain, 1910/2597; SyGMa **0.572** tautomer co-measured, n=1168).
  Move the tautomer-prefilter soundness (audit mismatch 0) to a footnote/supplement.

### 7. Results — external validity of the ceiling
- Source `GRAIL_FRAMING.md §2` external paragraph + `results/ceiling_external_validity.json`. Prose:
  internal 0.7355 [0.709,0.762] vs **external uncapped GLORYx-37 0.633 [0.531,0.733]** (n=37, wide CI);
  the composition covariate (OOS predicted external 0.738, recovers ~56%); the pool-capped 0.3715 is an
  artifact, never "the external ceiling." Optional small internal-vs-external scatter figure (placeholder).

### 8. Results — recall decomposition (the carrying result)
- Source `GRAIL_FRAMING.md §1.5 factor table` + `§3` + `results/recall_factorization.json`. **Fig 2 =
  `docs/benchmark/factorization_waterfall.svg`** with a manuscript caption. Micro: coverage 0.7355 ×
  selection 0.4885 × ranking 0.7256 = deployed 0.261; oracle 0.359; **conversion 35.5%**; the DOMINANT
  loss is selection. Block-bootstrap CIs. The ONE micro/macro reconciliation sentence lives here.

### 9. Results — honest-anchor certification
- Source `GRAIL_FRAMING.md §3` + `results/anchor_certification.json`. Prose: paired-bootstrap **Δ = −0.242
  [−0.271,−0.212]** (wholly < 0), McNemar **p ≈ 1.7e-44** (b=87/c=379), common-subset ceiling 0.736 ≈
  full (representative). Scope explicitly to **evaluation** variance (single deployed checkpoint) — this is
  where the `[PENDING: multi-seed]` caveat is stated. Optional paired-Δ/McNemar figure (placeholder).

### 10. Results — diagnosis levers + three Propositions
- Source `GRAIL_FRAMING.md §4` + `results/{prior_vs_learned,benchmark_report_depth2,benchmark_report_gap}.json`.
  A lever→factor table + a per-Proposition evidence box. **Proposition 1** (surrogate mismatch → ranking;
  confirmed listwise-InfoNCE +0.067, reranker 0.500 < SyGMa 0.558 — note its artifacts are Stage-2, cite
  `stage2_ranker_evidence.md`, currently 3-seed std not paired CI → soften to "confirmed on a held-out
  Stage-2 run" and add `[PENDING: paired CI]`). **Proposition 2** (propensity-PU → selection; prior beats
  learned Δ=−0.144; `e(r)∝π(r)` unmeasured; `1/ê` open test). **Proposition 3** (paradigm limit →
  coverage; depth-2 +0.012; external 0.633). Fold the withdrawn "learned beats prior" reversal into a
  one-line honesty note.

### 11. Results — match-sensitivity (TAME's headline eval result)
- Source `GRAIL_FRAMING.md §Supplement` + `results/{match_sensitivity_5method,rank_flip_ci}.json`. The
  **main cross-method table** (5 methods × 5 protocols) with `mean_output_size` (exposes SyGMa's large
  budget) — **caption carries the n=150-vs-1170 scoping (constraint 7)**. Strip DNB's dated UPDATE-1..5
  discovery voice. Keep the certified **differential-sensitivity interaction CI [+0.073,+0.171]** (the
  pre-declared primary endpoint, Holm) + the honest "per-pair flip not individually significant at n=150";
  the two independent rank-flips + MetaTrans non-monotonicity.

### 12. Limitations
- Write ONE section (none exists today). Consolidate from the two referee-risk registers + proposition
  guardrails: no recall win; precision is a pessimistic lower bound; single-checkpoint headline (multi-seed
  pending); n=150-vs-1170 comparability; external CI wide (n=37); Prop-2 unmeasured assumption; GRAIL loses
  to learned transformers.

### 13. Data & Code Availability
- (JCIM norm, replaces D&B Datasheet/Ethics.) The split (`fix_splits.py --molecule-disjoint` + leakage
  audit), the re-scoring harness, `regen_headline.sh`, the frozen per-substrate 5-method predictions, and
  the **DrugBank-derived data license constraint**. Note `[PENDING: commit frozen preds + leakage_fix_report.json]`.

### 14. Conclusion
- Fresh close from `GRAIL_FRAMING.md` "Net diagnosis": rule-based recall is coverage-conversion-limited (a
  dominant selection loss then a ranking loss); GRAIL as an interpretable instrument; TAME as the protocol.

### Fig 1 — pipeline schematic
- **Placeholder** in this pass: a captioned stub `[FIGURE 1: GRAIL 3-stage pipeline schematic — TO BUILD]`
  with the intended content described. (Real schematic is a post-draft cheap task.)

## Out of scope (track-list appended to `manuscript.md` as "## Draft TODO / open items")
- COMPUTE: multi-seed headline mean±std (`run_multiseed.py` ≥3 seeds); tier-2 tools on full 1170; optional
  compute-matched GFlowNet null.
- CHEAP (post-draft tasks): commit frozen tier-2 preds + `leakage_fix_report.json`; budget-matched
  leaderboard; MetaTrans↔SyGMa paired CI; MetaTrans on GLORYx-37; regenerate `rankflip.svg` +
  `scaling_curve.svg` on current numbers; verify all comparator DOIs + resolve "Gao 2026"; build Fig 1 +
  the anchor/external/long-tail figures; Prop-1 paired CI + commit its artifacts.

## Verification
- **Number-consistency gate:** every number in `manuscript.md` matches its `results/*.json` source or is a
  clearly-marked `[PENDING/cite/resolve]` placeholder — a grep for stale values (0.718 as GRAIL's ceiling
  headline, 0.334 as the deployed recall, "MetaBench") returns nothing except in the SUPERSEDED banners.
- **Guardrail gate:** `grep -niE "theorem|verified|validated" manuscript.md | grep -i decomposition` is
  empty; the reranker 0.500 never appears without "< SyGMa 0.558"; `e(r)∝π(r)` flagged unmeasured; the
  n=150 caveat is present in the cross-method table caption.
- **Adversarial referee pass** (multi-lens + refute) over the assembled draft at the end, as in the theory
  spine — a JCIM-referee lens, a number-accuracy lens, a guardrail lens.
- `make test` stays green (no code behavior change; this is docs).

## Notes for the plan
- One writing task per manuscript section (Title/Abstract, Intro+contribs, Related Work, each Methods
  subsection, each Results subsection, Limitations, Availability, Conclusion) — each independently
  reviewable; ordering: supersede-banners + skeleton first, then Methods, then Results (highest yield),
  then front-matter (abstract/contribs) once the body's numbers are placed, then Limitations/Conclusion.
- Prose voice: convert notes→manuscript, but preserve every honesty guardrail (constraints 2–8).

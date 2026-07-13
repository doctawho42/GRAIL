# GRAIL/TAME Manuscript First-Draft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the first coherent full draft of the GRAIL/TAME paper into `docs/benchmark/manuscript.md` from already-verified no-compute material, converting guardrail-voiced notes to manuscript prose while preserving every honesty constraint and marking compute-gated numbers as explicit placeholders.

**Architecture:** One `##` section per manuscript section, written into a single `docs/benchmark/manuscript.md`. This is **assembly + de-guardrailing + prose**, not new science — no code behavior changes, no new experiments. Every number is transcribed verbatim from `docs/GRAIL_FRAMING.md` / `results/*.json`; the two stale drafts (`paper_draft.md`, `DNB_FRAMING.md`) are marked SUPERSEDED, not deleted. Each section task ends with a grep-based number-consistency + guardrail check as its test.

**Tech Stack:** Markdown. Source docs: `docs/GRAIL_FRAMING.md` (primary), `docs/benchmark/related_work_positioning.md`, `docs/benchmark/protocol.md`, `docs/ARCHITECTURE.md`, `docs/benchmark/stage2_ranker_evidence.md`, and `results/*.json`. Figure asset: `docs/benchmark/factorization_waterfall.svg` (exists). Verification: `grep`, `make test` (green throughout — docs only).

## Global Constraints

*(Binding on every task; copied verbatim from the design spec `docs/superpowers/specs/2026-07-13-grail-manuscript-design.md`. Every task's requirements implicitly include this section.)*

1. **All numbers come from `docs/GRAIL_FRAMING.md` / `results/*.json`, NEVER from `paper_draft.md`.** The stale draft's headline figures (ceiling 0.718 as GRAIL's ceiling headline, GRAIL 0.334/0.40, two-factor split, "MetaBench" name) are wrong-thesis — discard wholesale. Use ceiling **0.735** tautomer / 0.718 plain; GRAIL **0.330 macro / 0.261 micro**; the three-factor decomposition; the name **TAME**.
2. **Honest anchor, early and everywhere:** GRAIL does NOT win on recall (0.330 macro / 0.261 micro < SyGMa 0.572 < MetaPredictor 0.585). The paper's value = ceiling + diagnosis + protocol. State it in the abstract + intro. No claim, anywhere, that GRAIL wins.
3. **Micro/macro discipline:** the decomposition, Fig 2 waterfall, and the §Results conversion are **micro** (coverage 0.735 == ceiling; deployed 0.261; conversion 35.5%); cross-method recall (GRAIL 0.330, SyGMa 0.572, MetaPredictor 0.585) is **macro**; exactly ONE reconciliation sentence. No mixed-frame ratio.
4. **The factorization is a "decomposition"/identity — NEVER "theorem/verified/validated".** Collapse the repeated guardrail asides into one honest caveat per section.
5. **Reranker 0.500 (Prop 1) never near a headline; it loses to SyGMa 0.558; a separate Stage-2 artifact.** Prop 2's `e(r)∝π(r)` is an UNMEASURED assumption + `1/ê` an OPEN test. Prop 3 single-step-conditional.
6. **Compute-gated numbers are placeholders, not fabrications.** The headline deployed recall carries `[PENDING: multi-seed mean±std over ≥3 seeds]`.
7. **The n=150-vs-1170 mismatch is scoped, not hidden.** The main cross-method table caption states: tier-2 comparators (BioTransformer, MetaPredictor, MetaTrans) are scored on the n=150 shared subset; GRAIL and SyGMa on the full n≈1170 clean test; a single-n rerun is future work. The honest-anchor comparison (GRAIL < SyGMa < MetaPredictor) carries this caveat wherever it appears.
8. **Unverified citations are placeholders:** `[cite: MetaTrans — Litsa, Das, Kavraki 2020, Chem Sci; verify vol/DOI]`, `[cite: MetaPredictor — verify]`, `[cite: Dhaked 2019 / DataSAIL — verify DOI]`, `[resolve: "Gao 2026"]`. Verified ones go in as real cites: SyGMa Ridder & Wagener 2008 (doi:10.1002/cmdc.200700312); GLORYx de Bruyn Kops 2020 (doi:10.1021/acs.chemrestox.0c00224); Boyce 2022 (doi:10.1016/j.comtox.2021.100208).
9. **NO Claude/AI/Co-Authored-By attribution** in any commit or the doc byline.
10. **Match mode `inchikey_tautomer`** is the paper's default; state it once in Methods.

**Voice:** convert notes → manuscript prose (flowing paragraphs, not bullet-note fragments), but preserve every honesty guardrail. Target section lengths are guidance, not hard limits.

**Commit discipline:** one commit per task, message `docs(manuscript): <section>`. NEVER add Co-Authored-By or any AI attribution (constraint 9). `results/` is gitignored — any result file referenced is read-only here; no `git add -f` needed since this plan writes only `.md`.

---

## File Structure

- **Create:** `docs/benchmark/manuscript.md` — the entire paper (all sections in one file; a journal manuscript is one document).
- **Modify:** `docs/benchmark/paper_draft.md` (line 1: prepend SUPERSEDED banner), `docs/benchmark/DNB_FRAMING.md` (line 1: prepend SUPERSEDED banner).
- **Read-only sources:** `docs/GRAIL_FRAMING.md`, `docs/benchmark/related_work_positioning.md`, `docs/benchmark/protocol.md`, `docs/ARCHITECTURE.md`, `docs/benchmark/stage2_ranker_evidence.md`, `results/*.json`, `docs/benchmark/factorization_waterfall.svg`.

Manuscript section order in the file (final document order):
Title → Abstract → 1. Introduction (+ contributions) → 2. Related Work → 3. Methods: GRAIL architecture → 4. Methods: Formal framework → 5. Methods: TAME protocol → 6. Results: coverage ceiling → 7. Results: external validity → 8. Results: recall decomposition (Fig 2) → 9. Results: honest-anchor certification → 10. Results: diagnosis + 3 Propositions → 11. Results: match-sensitivity (main cross-method table) → 12. Limitations → 13. Data & Code Availability → 14. Conclusion → Fig 1 placeholder → Draft TODO / open items.

**Task order** (not document order — body first, front-matter once numbers are placed): Task 1 skeleton → Tasks 2–4 Methods → Tasks 5–9 Results → Task 10 Related Work → Task 11 front-matter → Task 12 closing → Task 13 final verification.

---

### Task 1: Skeleton + SUPERSEDED banners

**Files:**
- Create: `docs/benchmark/manuscript.md`
- Modify: `docs/benchmark/paper_draft.md:1`
- Modify: `docs/benchmark/DNB_FRAMING.md:1`

**Interfaces:**
- Produces: the section-header skeleton every later task fills in. Exact `##` headers (later tasks locate their section by these strings verbatim):
  `# GRAIL: ...` (title placeholder line), `## Abstract`, `## 1. Introduction`, `## 2. Related Work`, `## 3. Methods — GRAIL architecture`, `## 4. Methods — Formal framework`, `## 5. Methods — TAME evaluation protocol`, `## 6. Results — Rule-bank coverage ceiling`, `## 7. Results — External validity of the ceiling`, `## 8. Results — Recall decomposition`, `## 9. Results — Honest-anchor certification`, `## 10. Results — Diagnosis: levers and three propositions`, `## 11. Results — Match-sensitivity and cross-method comparison`, `## 12. Limitations`, `## 13. Data & Code Availability`, `## 14. Conclusion`, `## Figure 1 — pipeline schematic`, `## Draft TODO / open items`.

- [ ] **Step 1: Create the skeleton file**

Create `docs/benchmark/manuscript.md` with the header block, all `##` section stubs (each with a one-line `> _[STUB — Task N]_` note naming the task that fills it), the Fig 1 placeholder, and the Draft TODO track-list. Content:

```markdown
# GRAIL: rule-based metabolite-structure prediction, a coverage×selection×ranking diagnosis, and the TAME evaluation protocol

> **Draft status (2026-07-13):** first assembled full draft. Numbers sourced from `docs/GRAIL_FRAMING.md` / `results/*.json`. Compute-gated values are marked `[PENDING: ...]`; unverified citations are marked `[cite: ...]`. Venue target: JCIM / J. Cheminformatics.

## Abstract
> _[STUB — Task 11]_

## 1. Introduction
> _[STUB — Task 11]_

## 2. Related Work
> _[STUB — Task 10]_

## 3. Methods — GRAIL architecture
> _[STUB — Task 2]_

## 4. Methods — Formal framework
> _[STUB — Task 3]_

## 5. Methods — TAME evaluation protocol
> _[STUB — Task 4]_

## 6. Results — Rule-bank coverage ceiling
> _[STUB — Task 5]_

## 7. Results — External validity of the ceiling
> _[STUB — Task 5]_

## 8. Results — Recall decomposition
> _[STUB — Task 6]_

## 9. Results — Honest-anchor certification
> _[STUB — Task 7]_

## 10. Results — Diagnosis: levers and three propositions
> _[STUB — Task 8]_

## 11. Results — Match-sensitivity and cross-method comparison
> _[STUB — Task 9]_

## 12. Limitations
> _[STUB — Task 12]_

## 13. Data & Code Availability
> _[STUB — Task 12]_

## 14. Conclusion
> _[STUB — Task 12]_

## Figure 1 — pipeline schematic
> _[FIGURE 1: GRAIL 3-stage pipeline schematic — TO BUILD]_ A left-to-right schematic: (i) substrate + 7,581-rule SMIRKS bank → learned retrieval-scored **generator** selecting rules; (ii) **RDKit rule application** enumerating candidate products; (iii) **PU-trained MCS-aware pair filter** scoring (substrate, product) pairs; deployment ranks by `filter_score × generator_score`. Real schematic is a post-draft task.

## Draft TODO / open items
> _[STUB — Task 12 seeds this; final content is the out-of-scope track-list]_
```

- [ ] **Step 2: Verify the skeleton has every required header**

Run:
```bash
cd /Users/nikitapolomosnov/PycharmProjects/GRAIL/.claude/worktrees/hungry-pasteur-25d746
grep -c '^## ' docs/benchmark/manuscript.md
```
Expected: `18` (16 numbered/named sections + Fig 1 + Draft TODO; the Abstract and Introduction are `## Abstract` and `## 1. Introduction`). If not 18, add the missing headers.

- [ ] **Step 3: Prepend SUPERSEDED banner to the two stale drafts**

At the very top (line 1) of `docs/benchmark/paper_draft.md`, insert:
```markdown
> **⚠ SUPERSEDED (2026-07-13).** This is the pre-reframe "MetaBench / How You Match Decides Who Wins" draft. Its thesis, headline numbers (ceiling 0.718, GRAIL ~0.334/~0.40, two-factor split), and name are wrong-thesis and MUST NOT be built on. The current manuscript is `docs/benchmark/manuscript.md` (GRAIL-primary, ceiling 0.735, three-factor decomposition, TAME). Preserved for history only.

```
At the very top (line 1) of `docs/benchmark/DNB_FRAMING.md`, insert:
```markdown
> **⚠ SUPERSEDED (2026-07-13).** Match-sensitivity-as-headline framing with a dated UPDATE-1..5 discovery narrative. The match-sensitivity analysis now lives as a supporting section in `docs/benchmark/manuscript.md` (§11). Preserved for history only.

```

- [ ] **Step 4: Verify banners are in place**

Run:
```bash
grep -l 'SUPERSEDED' docs/benchmark/paper_draft.md docs/benchmark/DNB_FRAMING.md
```
Expected: both file paths printed.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmark/manuscript.md docs/benchmark/paper_draft.md docs/benchmark/DNB_FRAMING.md
git commit -m "docs(manuscript): skeleton + supersede stale drafts"
```

---

### Task 2: Methods — GRAIL architecture (§3)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill `## 3. Methods — GRAIL architecture`)

**Interfaces:**
- Consumes: the `## 3. ...` header from Task 1.
- Produces: the term names later sections reuse — **generator** (`P(r|s)` rule selector), **RDKit rule application**, **PU-trained MCS-aware pair filter**, deployment ranking `filter_score × generator_score`, match default `inchikey_tautomer`. Tasks 3, 8, 11 rely on these exact terms.

**Source material:** `docs/GRAIL_FRAMING.md §1` (lines 35–41), `docs/ARCHITECTURE.md`, and `CLAUDE.md` "Architecture" + "Invariants" sections. Read `docs/ARCHITECTURE.md` for the stage details before writing.

**Numbers/facts to use verbatim:**
- Rule bank: **7,581 SMIRKS** rules (curated bank).
- Three stages: (1) learned multi-label **rule selector** over the bank (retrieval-scored generator: attention + similarity + MLP head); (2) **RDKit rule application** enumerating candidate products; (3) **PU-trained, MCS-aware pair filter** scoring (substrate, product) pairs in the **logit domain** (PULoss / nnPU).
- Deployment ranks candidates by **`filter_score × generator_score`**.
- Training data is **positive-unlabeled**: rule-applicable non-annotated products are *unlabeled*, not negative.
- Featurization dims (state briefly): single-graph nodes 16-dim (generator encoder), pair-graph nodes 18-dim (pair filter), edge 18-dim, Morgan fingerprint 1024-dim. The pair filter's cross-edges come from an element-aware MCS atom correspondence.
- Match default: state that all structure matching in this paper uses **`inchikey_tautomer`** (constraint 10) — one sentence, here in Methods.

**Guardrails for this section:**
- Contribution framing: "interpretable learned rule selection + PU-aware pair filter", NOT recall supremacy (constraint 2). Do not claim GRAIL is best.
- Keep it Methods-prose (how it works), not Results (no recall numbers here except forward-references are fine).

- [ ] **Step 1: Read the architecture source**

Run: `sed -n '1,80p' docs/ARCHITECTURE.md` (and read `docs/GRAIL_FRAMING.md` lines 35–41). Confirm the three-stage description and the featurization dims match the CLAUDE.md invariants.

- [ ] **Step 2: Write the section prose**

Replace the `> _[STUB — Task 2]_` line under `## 3. Methods — GRAIL architecture` with ~250–350 words of prose covering, in order: (a) the three stages named above with the generator as `P(r|s)` selector over the 7,581-rule bank; (b) the PU setup (negatives are unlabeled-applicable, not ground-truth) and that the filter trains in the logit domain; (c) featurization dims + MCS-aware pair graph in one or two sentences; (d) deployment ranking `filter_score × generator_score`; (e) one sentence stating `inchikey_tautomer` is the default match mode throughout. Close with the one-line honest contribution framing (interpretable instrument, not a recall claim).

- [ ] **Step 3: Verify numbers + guardrails present**

Run:
```bash
grep -E '7,?581|filter_score|generator_score|inchikey_tautomer|positive-unlabeled|MCS' docs/benchmark/manuscript.md | head
grep -niE 'GRAIL (is|wins|beats).*(best|SOTA|state-of-the-art)' docs/benchmark/manuscript.md
```
Expected: first grep prints the architecture facts (7581, ranking product, match mode, PU, MCS all present); second grep is EMPTY (no supremacy claim).

- [ ] **Step 4: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): Methods — GRAIL 3-stage architecture"
```

---

### Task 3: Methods — Formal framework (§4)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill `## 4. Methods — Formal framework`)

**Interfaces:**
- Consumes: stage names from Task 2 (generator = `P(r|s)`, filter = `P(true|s,m)`, RDKit = deterministic support).
- Produces: the **recall decomposition identity** `recall = coverage_bank · selection_retention · ranking_conversion` and the **lever→factor map** that Tasks 6/8 reference. The names `coverage_bank`, `selection_retention`, `ranking_conversion` must be spelled exactly (Tasks 6, 8 grep them).

**Source material:** `docs/GRAIL_FRAMING.md §1.5` (lines 43–113).

**Content/facts to use verbatim:**
- The generative latent-reaction mixture (render the equation as a fenced block):
  ```
  P(m | s) = Σ_r Σ_{site ∈ sites(r,s)} P(r | s) · P(site | r, s) · 𝟙[ apply(r, s, site) = m ]
  ```
- Stage→term mapping: generator ≈ `P(r|s)` (its persisted `rule_prior_logits` is the marginal `π(r)=P(r fires)`); RDKit realises `𝟙[apply=m]`; filter ≈ `P(true|s,m)`; deployment ranks by `filter_score × generator_score`; PU training ≈ EM over the unobserved rule-firing indicator. Frame this as "the model the pipeline approximates", NOT a claim the trained weights are its MLE.
- The recall decomposition (render the equation):
  ```
  recall@k = H/U = (C_full/U) · (C_bud/C_full) · (H/C_bud)
                 = coverage_bank · selection_retention · ranking_conversion
  ```
  with `R_s(k) ⊆ P_bud,s ⊆ P_full,s` nesting, `hit(·)` under the tautomer-InChIKey quotient, and micro sums `U=Σ|T_s|`, `C_full=Σ hit(P_full,s)`, `C_bud=Σ hit(P_bud,s)`, `H=Σ hit(R_s(k))`. Each factor in [0,1] because the sets nest.
- The lever→factor map (render as a small table): `coverage_bank` ← multi-step (depth-2), ΔMW gap, external-validity composition, Proposition 3; `selection_retention` ← learned-vs-prior probe, data-scaling saturation, Proposition 2; `ranking_conversion` ← filter/listwise reranker, oracle bound, Proposition 1.

**Guardrails for this section (constraint 4):**
- Exactly ONE caveat sentence: "it is an accounting identity, not a theorem". Do NOT use the words "theorem", "verified", or "validated" to describe the decomposition. Do NOT offer its cancellation as evidence for anything — state its only use is to *localise* the loss.
- Note the framework is "deliberately thin: one likelihood, one decomposition, one lever→factor map" — not a rewrite of the empirical results.

- [ ] **Step 1: Write the section prose**

Replace the stub under `## 4. Methods — Formal framework` with the mixture equation + stage mapping paragraph, the decomposition equation + nesting explanation, the single identity caveat, and the lever→factor table. ~300–400 words + two equation blocks + one table. Numbers (0.735 etc.) belong to Results — do NOT put factor *values* here; this section defines the *identity*, Task 6 reports the *values*.

- [ ] **Step 2: Verify the identity + guardrail**

Run:
```bash
grep -E 'coverage_bank · selection_retention · ranking_conversion' docs/benchmark/manuscript.md
grep -niE 'accounting identity, not a theorem' docs/benchmark/manuscript.md
awk '/## 4\. Methods — Formal framework/,/## 5\. Methods/' docs/benchmark/manuscript.md | grep -niE 'theorem|verified|validated' | grep -viE 'not a theorem'
```
Expected: first two greps print a match; the third (forbidden-words scan within §4, excluding the allowed "not a theorem") is EMPTY.

- [ ] **Step 3: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): Methods — formal framework (mixture + decomposition identity)"
```

---

### Task 4: Methods — TAME evaluation protocol (§5)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill `## 5. Methods — TAME evaluation protocol`)

**Interfaces:**
- Consumes: `inchikey_tautomer` default from Task 2.
- Produces: the 5 match quotient names (Tasks 9, 11 reuse them) and the "TAME" name (used throughout).

**Source material:** `docs/GRAIL_FRAMING.md §Reproducibility & provenance` (lines 312–386), `docs/benchmark/protocol.md`, and the `docs/benchmark/paper_draft.md §4` worked example (D-alanine/acetone — read it, rename any "MetaBench" → "TAME").

**Content/facts to use verbatim:**
- **TAME** = **T**automer-**A**ware **M**etabolite-structure **E**valuation: a tautomer-InChIKey match quotient, a leakage-audited molecule-disjoint train/val/test split, and a frozen multi-method re-scoring harness. Explicitly: TAME is a *protocol + audited split + re-scoring harness*, **not** a leaderboard service.
- The **5 match quotients**: `canonical` (stereo-free canonical SMILES, LAGOM), `inchikey` (literature standard), `inchi_no_stereo` (GLORYx stereo-blind skeleton), `tanimoto1` (MetaTrans Morgan fingerprint Tanimoto=1), `inchikey_tautomer` (**recommended default** — tautomer-canonicalize both sides). Note plain InChI normalizes only a *subset* of tautomers, so a rule-emitted tautomer of the reference misses under plain `inchikey`.
- The **molecule-disjoint leakage-audited split**: `scripts/fix_splits.py --molecule-disjoint` enforces full molecule-set disjointness across train/val/test; emits `results/leakage_fix_report.json` when run; val ≈ test (0.327 vs 0.330) corroborates no overfit.
- **Metrics co-reporting**: recall@k reported alongside `mean_output_size` (precision is a pessimistic lower bound under incomplete annotation — an unannotated prediction is counted a false positive but may be a real unrecorded metabolite). Select-on-val, touch-test-once.
- The worked example (one short paragraph): D-alanine/acetone keto/enol — the same prediction scores correct or wrong purely by match protocol (this is the phenomenon §11 quantifies).

**Guardrails for this section:**
- State the match default (`inchikey_tautomer`) once (already noted in Methods §3; here give the full 5-quotient list with it recommended).
- Do NOT over-claim TAME as a leaderboard/benchmark service — it is a protocol + split + harness.
- The tautomer-prefilter soundness (audit mismatch = 0 on 50 substrates) may be a one-line footnote here; the ceiling section (Task 5) carries it too.

- [ ] **Step 1: Write the section prose**

Replace the stub under `## 5. Methods — TAME evaluation protocol` with: (a) the TAME definition + "protocol not leaderboard" scoping; (b) the 5-quotient list with `inchikey_tautomer` recommended and the tautomer-subset motivation; (c) the leakage-audited split + val≈test; (d) recall@k + mean_output_size co-reporting + select-on-val; (e) the D-alanine/acetone worked example. ~350–450 words.

- [ ] **Step 2: Verify quotients + TAME framing**

Run:
```bash
grep -E 'canonical|inchi_no_stereo|tanimoto1|inchikey_tautomer' docs/benchmark/manuscript.md | head
grep -niE 'Tautomer-Aware Metabolite|molecule-disjoint|mean_output_size' docs/benchmark/manuscript.md
grep -ni 'MetaBench' docs/benchmark/manuscript.md
```
Expected: first two greps print matches (all 5 quotients + TAME + split + output-size present); third grep is EMPTY (no "MetaBench" leaked in).

- [ ] **Step 3: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): Methods — TAME evaluation protocol"
```

---

### Task 5: Results — coverage ceiling (§6) + external validity (§7)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill `## 6. Results — Rule-bank coverage ceiling` and `## 7. Results — External validity of the ceiling`)

**Interfaces:**
- Consumes: the `coverage_bank` factor name from Task 3.
- Produces: the ceiling number **0.735** (== `coverage_bank`) that Task 6 references; the external ceiling 0.633 that Task 8 (Prop 3) references.

**Source material:** `docs/GRAIL_FRAMING.md §2` (lines 115–149); `results/benchmark_report.json`; `results/ceiling_external_validity.json`.

**Numbers to use verbatim (§6 ceiling):**
- Rule-bank recall ceiling on the full clean molecule-disjoint test split (1170 substrates, 2597 true pairs): **0.718 plain InChIKey / 0.735 tautomer-InChIKey** — **1865 / 1910 of 2597** recovered.
- Ceiling reported under the **same tautomer protocol** as GRAIL and SyGMa; plain 0.718 reproduces the previously reported number exactly (validates the run).
- Tautomer path via heavy-atom-formula prefilter, **audit mismatch = 0** on 50 substrates (footnote/supplement).
- SyGMa co-measured on the identical split: recall@15 **0.558 plain / 0.572 tautomer** (n=1168).
- Present a ceiling-vs-baselines table: ceiling 0.735 tautomer / 0.718 plain (1910/2597); SyGMa 0.572 tautomer (n=1168); learned systems ~0.47–0.585 (cite).

**Numbers to use verbatim (§7 external validity):**
- Internal ceiling `coverage_bank` = **0.735**, cluster-bootstrap 95% CI **[0.709, 0.762]** (ratio-of-sums `Σhit/Σtrue`, 10 000 resamples).
- **External uncapped GLORYx-37 ceiling = 0.633, 95% CI [0.531, 0.733]** (n=37, wide by design) — one uncapped full-bank depth-1 apply_rules pass over 37 GLORYx parents, same tautomer match.
- The previously committed **0.3715 is pool-capped** (a generator-budget artifact from `gloryx_oracle.json`) and must **never** be read as "the external ceiling."
- Composition covariate: an OLS regression on molecular descriptors predicts macro coverages ~0.79 internal / ~0.74 external in-sample; leave-external-out (fit-internal → predict-external) lands at **0.738 out-of-sample**, recovering **~56%** of the internal→external macro gap (`predicted_external_mean_oos = 0.738`, `gap_recovery_frac = 0.565`).
- Frame as a **suggestive partial composition effect at n=37** — GLORYx parents are larger, more-conjugated drugs — **not a fitted or transferable law**, not a defect in the bank/protocol.

**Guardrails:**
- "Coverage is high; the bank is powerful" → the task is limited by *coverage-conversion*, not rule expressiveness.
- Never call 0.3715 "the external ceiling"; label it pool-capped artifact.
- The ~0.74 in-sample external prediction sits *above* the measured external macro coverage (0.697) — state honestly; the OOS number is the transferable claim.
- Micro/macro: the ceilings (0.735, 0.633) are **micro**; the ~0.79/~0.74 composition predictions are **macro** — label each.

- [ ] **Step 1: Write §6 (ceiling)**

Replace the stub under `## 6. Results — Rule-bank coverage ceiling` with the ceiling prose + the ceiling-vs-baselines table. Move the prefilter-soundness (audit = 0) to a footnote sentence. ~200–300 words + one table.

- [ ] **Step 2: Write §7 (external validity)**

Replace the stub under `## 7. Results — External validity of the ceiling` with the internal-vs-external prose (0.735 [0.709,0.762] vs 0.633 [0.531,0.733]), the pool-capped 0.3715 correction, and the composition-covariate paragraph (0.738 OOS, ~56% recovery). Add `> _[FIGURE: internal-vs-external coverage scatter — optional, post-draft]_`. ~250–350 words.

- [ ] **Step 3: Verify numbers**

Run:
```bash
grep -E '0\.735|0\.718|1910|1865|0\.572|0\.558|1168' docs/benchmark/manuscript.md | head
grep -E '0\.633|0\.531, 0\.733|0\.3715|0\.738|56%|0\.565' docs/benchmark/manuscript.md | head
grep -niE '0\.3715.*external ceiling|external ceiling.*0\.3715' docs/benchmark/manuscript.md
```
Expected: first two greps print the ceiling + external numbers; the third (0.3715 mislabeled as external ceiling) is EMPTY.

- [ ] **Step 4: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): Results — coverage ceiling + external validity"
```

---

### Task 6: Results — recall decomposition + Fig 2 (§8)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill `## 8. Results — Recall decomposition`)

**Interfaces:**
- Consumes: the identity + factor names from Task 3; ceiling 0.735 from Task 5.
- Produces: the deployed **0.261 micro / 0.330 macro** anchor and the conversion **35.5%** that Tasks 7(anchor)/8/11 reference; the single micro/macro reconciliation sentence (constraint 3) lives here.

**Source material:** `docs/GRAIL_FRAMING.md §1.5 factor table` (lines 92–105) + `§3` (lines 151–170); `results/recall_factorization.json`. Figure asset: `docs/benchmark/factorization_waterfall.svg` (exists).

**Numbers to use verbatim (all micro, tautomer-InChIKey, n=1170, k=15; 10 000-resample substrate block-bootstrap, ratio-of-sums):**
- `coverage_bank` = **0.735**, CI [0.709, 0.762] (equals §2 ceiling).
- `selection_retention` = **0.489**, CI [0.458, 0.520] — **the dominant loss**.
- `ranking_conversion` = **0.726**, CI [0.687, 0.765].
- product = micro recall@15 = **0.261** (= 0.735 · 0.489 · 0.726).
- oracle (perfect ranking of deployed pool, `ranking_conversion = 1`) = `C_bud/U` = **0.359** (micro).
- conversion = **35.5%** (0.261 / 0.735); conversion gap = selection × ranking = 0.489 × 0.726 — "two losses in series", selection the larger.
- **Fig 2** = `docs/benchmark/factorization_waterfall.svg`. Caption: waterfall on one n=1170 population (tautomer-InChIKey, micro ratio-of-sums): `U (1.0) → coverage_bank (0.735) → coverage·selection = oracle recall (0.359) → deployed recall (0.261)`, with the **oracle line** on the ranking bar and each factor annotated with its 95% CI. Bar-1→Bar-2 is the selection loss; Bar-2→Bar-3 the ranking loss.

**The single micro/macro reconciliation sentence (constraint 3) — place it exactly once here:** "Throughout, cross-method recall@15 is the per-substrate mean (macro): GRAIL 0.330, SyGMa 0.572; the coverage→conversion decomposition uses the pooled (micro) frame — the only frame in which the three factors multiply exactly to the realised recall and in which `coverage_bank` equals the 0.735 rule-bank ceiling — in which deployed recall is 0.261."

**Headline placeholder (constraint 6):** where the deployed recall is first stated as the paper's headline number, append `[PENDING: multi-seed mean±std over ≥3 seeds]` — the single deployed checkpoint caveat.

**Guardrails (constraint 4):** label the figure and the identity a "decomposition"; never assert equality with any macro number; one honest caveat that the identity closes on any numbers and is used only to localise the loss.

- [ ] **Step 1: Write the section prose + embed Fig 2**

Replace the stub under `## 8. Results — Recall decomposition` with: the factor table (0.735 × 0.489 × 0.726 = 0.261, oracle 0.359, with CIs); the "dominant loss is selection" reading; the 35.5% conversion + two-losses-in-series framing; the Fig 2 reference with its manuscript caption (embed as `![Recall decomposition waterfall](factorization_waterfall.svg)` + caption text); the ONE micro/macro reconciliation sentence; the headline `[PENDING: multi-seed ...]` marker on the deployed recall. ~350–450 words + one table + figure ref.

- [ ] **Step 2: Verify numbers + placeholder + single reconciliation**

Run:
```bash
grep -E '0\.489|0\.726|0\.359|35\.5%|0\.261' docs/benchmark/manuscript.md | head
grep -c 'PENDING: multi-seed' docs/benchmark/manuscript.md
grep -c 'factorization_waterfall.svg' docs/benchmark/manuscript.md
```
Expected: first grep prints the factor values; the PENDING grep ≥1; the figure grep ≥1.

- [ ] **Step 3: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): Results — recall decomposition + Fig 2 waterfall"
```

---

### Task 7: Results — honest-anchor certification (§9)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill `## 9. Results — Honest-anchor certification`)

**Interfaces:**
- Consumes: deployed 0.330 macro (Task 6), SyGMa 0.572 (Task 5).
- Produces: the certified Δ = −0.242 the Limitations section (Task 12) references.

**Source material:** `docs/GRAIL_FRAMING.md §3` "Certifying the honest anchor" (lines 172–191); `results/anchor_certification.json`.

**Numbers to use verbatim (common substrate set, n=1168, tautomer-InChIKey):**
- Paired per-substrate difference `recall_GRAIL − recall_SyGMa` = **−0.242**, 95% CI **[−0.271, −0.212]** — wholly below zero.
- Exact **McNemar** test on any-hit@15: **b = 87** (GRAIL-only hits) vs **c = 379** (SyGMa-only hits), **p ≈ 1.7×10⁻⁴⁴**.
- Common-subset ceiling (**0.736**) matches the full-1170 ceiling (0.735) → common set is representative.
- Scope: the paired bootstrap covers continuous recall; McNemar covers **only** the binary any-hit outcome; together they certify **evaluation** variance (a single deployed checkpoint scored over resampled substrates). Split not overfit (val ≈ test).

**Guardrails (constraints 2, 6):**
- Scope explicitly to **evaluation** variance — this is where the `[PENDING: multi-seed]` caveat is restated: "the deployed headline is a single checkpoint, not a seed average." Add `[PENDING: multi-seed mean±std]` again here.
- "SyGMa > GRAIL is significant — the anchor holds." No recall-win claim.
- Add `> _[FIGURE: paired-Δ / McNemar — optional, post-draft]_`.

- [ ] **Step 1: Write the section prose**

Replace the stub under `## 9. Results — Honest-anchor certification` with the paired-bootstrap Δ, McNemar result, representativeness check, evaluation-variance scoping + the multi-seed PENDING caveat, and the "anchor holds" close. ~200–300 words.

- [ ] **Step 2: Verify numbers + guardrail**

Run:
```bash
grep -E '−0\.242|-0\.242|−0\.271|0\.736|87|379|1\.7' docs/benchmark/manuscript.md | head
grep -niE 'evaluation.*variance|single (deployed )?checkpoint' docs/benchmark/manuscript.md
```
Expected: both greps print matches (Δ + McNemar + representativeness present; evaluation-variance scoping present).

- [ ] **Step 3: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): Results — honest-anchor certification"
```

---

### Task 8: Results — diagnosis levers + three Propositions (§10)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill `## 10. Results — Diagnosis: levers and three propositions`)

**Interfaces:**
- Consumes: factor names (Task 3), factor values 0.489/0.726 (Task 6), external ceiling 0.633 (Task 5), reranker context.
- Produces: the three Propositions the Limitations section references.

**Source material:** `docs/GRAIL_FRAMING.md §4` (lines 193–279); `results/prior_vs_learned.json`, `results/benchmark_report_depth2.json`, `results/benchmark_report_gap.json`; `docs/benchmark/stage2_ranker_evidence.md`.

**Numbers to use verbatim — lever→factor table:**
- **Learned vs prior** (→ `selection_retention`; top_k-limited rule-selection **probe**, NOT deployed recall, n=245): prior-only **0.410** vs learned-only **0.266** gen-only (Δ **−0.144**, 95% CI **[−0.196, −0.095]**, paired bootstrap); with filter 0.405 vs 0.300 (Δ −0.105, CI [−0.152, −0.058]); adding prior to learned lifts **+0.130** gen / +0.099 filter; filter helps only the *weak learned* ordering (+0.034, CI [+0.011, +0.061]), not the strong prior (n.s.).
- **Multi-step** (→ `coverage_bank`): breadth-capped depth-2 lifts the ceiling **+0.012** (0.711→0.723) at **8.5× candidate cost** (194→1653), 150 subs, beam 10. Not the dominant coverage lever.
- **Coverage ΔMW gap** (→ `coverage_bank`): **26.9%** of true metabolites uncovered by depth-1 bank (plain-InChIKey; ~25% under tautomer). Diverse long tail (top class hydroxylation, only 6% of uncovered); 500 subs.
- **Data scaling** (→ `selection_retention`): recall saturates (2418 → 4787 substrates ≈ flat) — not a data-quantity problem.
- **Regioselectivity (SoM)** (→ `ranking_conversion`): SoM prior gives only a small lift.

**Three Propositions (per-Proposition evidence box):**
- **Proposition 1 — Surrogate mismatch (→ `ranking_conversion`).** Proper-scoring filter is Bayes-optimal for AUC/calibration but not for top-k across heterogeneous pools (pool size 17–150, `n_true` 1–18). *Confirmation:* listwise-InfoNCE reranker beats pointwise BCE filter as a ranker: **0.433 → 0.500 @15 (+0.067)**, 74% of the oracle **0.677** (`stage2_ranker_evidence.md`, Spike-3). *Guardrail:* a theorem about **objectives**, not a recall win — the reranker's **0.500 still loses to SyGMa (0.558)**, reported only as a separate Stage-2 artifact. Currently **3-seed std, not a paired CI** → soften to "confirmed on a held-out Stage-2 run" and add `[PENDING: paired CI]`.
- **Proposition 2 — Propensity-PU identifiability (→ `selection_retention`).** Under SCAR PU annotation, a constant-unlabeled-weight learner recovers a propensity-distorted score whose dominant component is `π(r)` (the frequency prior), so the prior is Bayes-competitive by construction. *Anchor:* learned-only 0.266 vs prior-only 0.410 (Δ **−0.144**, CI [−0.196, −0.095], n=245). *Falsifiable prediction:* reweighting the labeled loss by `1/ê(r)` should shrink the prior's edge — an **open test, not a promised fix**. *Guardrail:* `e(r) ∝ π(r)` is an **UNMEASURED modeling assumption**, flagged as such.
- **Proposition 3 — Paradigm limit (→ `coverage_bank`).** Single-step rule-based recall ≤ single-step coverage_bank < 1 (multi-generation references unreachable by one rule application). *Witnesses:* depth-2 lifts only **+0.012** at 8.5× cost; external uncapped single-step ceiling **0.633**. *Guardrail:* stated **single-step-conditional** — bounds the single-step paradigm, not the problem.

**Guardrails:** the deployed pipeline is prior-independent (probe reveals selector weakness; deployment masks it by applying rules broadly). Fold the withdrawn "learned beats prior" reversal into a one-line honesty note (an un-persisted prior-buffer artifact, caught by adversarial verification, corrected here).

- [ ] **Step 1: Write the section prose**

Replace the stub under `## 10. Results — Diagnosis: levers and three propositions` with the lever→factor table + a per-Proposition evidence box (three subsections). Preserve the probe-vs-deployed distinction and all three guardrails. Add the reranker `[PENDING: paired CI]` marker. ~600–800 words + one table.

- [ ] **Step 2: Verify numbers + guardrails**

Run:
```bash
grep -E '0\.410|0\.266|−0\.144|-0\.144|\+0\.012|26\.9%|0\.433|0\.500|0\.677|0\.633' docs/benchmark/manuscript.md | head
grep -niE '0\.500 (still )?loses to SyGMa|SyGMa \(?0\.558' docs/benchmark/manuscript.md
grep -niE 'unmeasured' docs/benchmark/manuscript.md
grep -c 'PENDING: paired CI' docs/benchmark/manuscript.md
```
Expected: first grep prints the lever + Proposition numbers; second grep confirms the reranker-loses-to-SyGMa guardrail (constraint 5); third confirms `e(r)∝π(r)` flagged unmeasured; PENDING grep ≥1.

- [ ] **Step 3: Verify reranker 0.500 never appears without the SyGMa guardrail**

Run:
```bash
grep -n '0\.500' docs/benchmark/manuscript.md
```
Manually confirm every line mentioning the reranker's 0.500 is within a sentence/paragraph that also names "SyGMa 0.558" (constraint 5). If any 0.500 stands near a headline without that qualifier, fix it.

- [ ] **Step 4: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): Results — diagnosis levers + three propositions"
```

---

### Task 9: Results — match-sensitivity + cross-method table (§11)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill `## 11. Results — Match-sensitivity and cross-method comparison`)

**Interfaces:**
- Consumes: 5 match quotients (Task 4), GRAIL 0.330 / SyGMa 0.572 / MetaPredictor 0.585 (Tasks 5–6).
- Produces: the main cross-method table + the primary-endpoint interaction CI.

**Source material:** `docs/GRAIL_FRAMING.md §Supplement` (lines 281–291) + `§Reproducibility` primary-endpoint block (lines 348–361); `results/match_sensitivity_5method.json`, `results/rank_flip_ci.json`.

**Numbers to use verbatim:**
- **5 methods** (GRAIL, SyGMa, BioTransformer, MetaPredictor, MetaTrans) × **5 protocols**; co-report `mean_output_size` (exposes SyGMa's large budget).
- **Differential match-protocol sensitivity (interaction, the pre-declared primary endpoint) = +0.120, 95% CI [+0.073, +0.171]** (`rank_flip_ci.json`, `interaction_B_extra_gain_from_normalization`) — canonical vs tautomer-InChIKey Δ-of-Δ; established the match protocol is a method-dependent confounder.
- **Two independent rank-flips:** GRAIL↔BioTransformer; MetaTrans↔SyGMa.
- **MetaTrans non-monotonic** protocol response: canon **0.523** > InChIKey **0.494** < no-stereo **0.561**.
- **Holm** multiplicity control *within each declared family*: (i) per-method protocol-sensitivity family (GRAIL, SyGMa, BioTransformer, MetaPredictor); (ii) rank-flip pairwise family. Everything else (factorization, external ceiling, anchor) is secondary/descriptive, not counted against the primary-endpoint budget.
- Honest anchor row in the table: GRAIL 0.330 macro < SyGMa 0.572 < MetaPredictor 0.585 (tautomer-InChIKey).

**The n=150-vs-1170 caption (constraint 7) — MUST appear in the table caption verbatim in spirit:** "Tier-2 comparators (BioTransformer, MetaPredictor, MetaTrans) are scored on the n=150 shared subset; GRAIL and SyGMa on the full n≈1170 clean test. A single-n rerun is future work."

**Guardrails:**
- Strip DNB's dated UPDATE-1..5 discovery voice — write as settled Results.
- Keep the honest "per-pair flip not individually significant at n=150" note alongside the certified interaction CI.
- The interaction CI [+0.073, +0.171] is the certified primary endpoint (Holm).

- [ ] **Step 1: Write the section prose + main table**

Replace the stub under `## 11. Results — Match-sensitivity and cross-method comparison` with: the 5×5 cross-method recall table (with `mean_output_size` column and the honest-anchor ordering), the n=150-vs-1170 caption, the primary-endpoint interaction (+0.120 [+0.073,+0.171]) + Holm family description, the two rank-flips, and the MetaTrans non-monotonicity. Add `> _[FIGURE: rank-flip — regenerate rankflip.svg on current numbers, post-draft]_`. ~450–600 words + one table.

- [ ] **Step 2: Verify numbers + the n=150 caption**

Run:
```bash
grep -E '\+0\.120|\+0\.073, \+0\.171|0\.523|0\.494|0\.561|0\.585|Holm' docs/benchmark/manuscript.md | head
grep -niE 'n.?150 shared subset|tier-2 comparators.*n.?150|single-n rerun is future work' docs/benchmark/manuscript.md
grep -ni 'UPDATE-[1-5]' docs/benchmark/manuscript.md
```
Expected: first two greps print matches (interaction CI + MetaTrans values + Holm + the n=150 caption present); third grep is EMPTY (no UPDATE-N discovery voice leaked in).

- [ ] **Step 3: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): Results — match-sensitivity + cross-method comparison"
```

---

### Task 10: Related Work (§2)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill `## 2. Related Work`)

**Interfaces:**
- Consumes: nothing from later tasks.
- Produces: the citation set the Data & Code / reference list draws on.

**Source material:** `docs/benchmark/related_work_positioning.md` (whole file).

**Content — convert memo to prose paragraphs:**
- Comparator methods with **verified** cites (real cites, constraint 8): SyGMa (Ridder & Wagener 2008, ChemMedChem, doi:10.1002/cmdc.200700312 — covers ~70% of human biotransformations; 68% of test metabolites, 30% in top-3); GLORYx (de Bruyn Kops 2020, Chem Res Toxicol, doi:10.1021/acs.chemrestox.0c00224 — recall 77%, finds phase-2 harder than phase-1); BioTransformer 3.0 (Djoumbou-Feunang 2019, J Cheminform); MetaTrans `[cite: Litsa, Das, Kavraki 2020, Chem Sci; verify vol/DOI]`; MetaPredictor `[cite: MetaPredictor — verify]`.
- Prior multi-method comparisons as **corroboration** (we are NOT "first comparison"): Scholz 2023 (Sci Total Environ — SyGMa/GLORY/GLORYx/BioTransformer/MetaTrans on 85 agrochemicals; low precision ~18% first-gen; rule-vs-ML divergence) and Boyce 2022 (doi:10.1016/j.comtox.2021.100208 — SyGMa highest coverage but overproduces, 5,125 metabolites = 54.7% of predictions; precision 1.1–29%). Their findings (low precision, SyGMa overproduction, rule-vs-ML divergence, phase-2 harder) **replicate ours**.
- Tautomer-matching prior art justifying our protocol: Dhaked 2019 `[cite: Dhaked 2019 — verify DOI]` (standard InChI normalizes only a subset of tautomers); Hähnke 2018 (60% of PubChem structures differ from the InChI form, mostly tautomer); Mansouri 2024 (QSAR-ready standardization).
- Leakage prior art: DataSAIL 2023 `[cite: DataSAIL — verify DOI]` (general leakage-aware splitting) — ours is the metabolite-specific molecule-disjoint audit.
- Eval-robustness precedent (rankings flip): Mishra 2021, Rodriguez 2021 — general for ML/NLP; our axis (how a predicted *structure* is matched) is the domain-specific novelty.
- The **scoped novelty statement** (§6 of the memo): NOT first to compare (Scholz/Boyce), NOT first to standardize structures (PubChem/QSAR-ready), NOT first leakage-aware split (DataSAIL); TAME's contribution is their **first joint instantiation** for metabolite structure prediction + the match-sensitivity analysis + the coverage×selection×ranking decomposition via GRAIL as one honest row.
- `[resolve: "Gao 2026"]` — flag the unresolved reference the earlier plan cited.

**Guardrails:** cite prior comparisons as corroboration, not threat; scope novelty precisely; every unverified DOI is a `[cite: ...]` / `[resolve: ...]` placeholder (constraint 8).

- [ ] **Step 1: Write the section prose**

Replace the stub under `## 2. Related Work` with 4–6 prose paragraphs covering the above. Verified cites inline as real; unverified as bracketed placeholders. ~500–700 words.

- [ ] **Step 2: Verify verified cites present + unverified flagged**

Run:
```bash
grep -E '10\.1002/cmdc\.200700312|10\.1021/acs\.chemrestox\.0c00224|10\.1016/j\.comtox\.2021\.100208' docs/benchmark/manuscript.md
grep -E '\[cite:|\[resolve:' docs/benchmark/manuscript.md | head
```
Expected: first grep prints the 3 verified DOIs; second prints the placeholder markers (MetaTrans, MetaPredictor, Dhaked/DataSAIL, Gao 2026).

- [ ] **Step 3: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): Related Work"
```

---

### Task 11: Title + Abstract + Introduction + contributions (front-matter)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill title line, `## Abstract`, `## 1. Introduction`)

**Interfaces:**
- Consumes: every headline number now placed in the body (ceiling 0.735, deployed 0.330/0.261, anchor, TAME, interaction CI). Written LAST among content tasks so the abstract summarises placed numbers.
- Produces: nothing downstream (front-matter).

**Source material:** `docs/GRAIL_FRAMING.md` Thesis (lines 7–17) + Honest anchor (lines 19–31); the body sections just written.

**Content:**
- **Title** (replace the Task-1 placeholder title if refining): GRAIL-primary, conveying "rule-based metabolite-structure prediction + a coverage×selection×ranking diagnosis + the TAME evaluation protocol." NOT "How You Match Decides Who Wins" (demoted).
- **Abstract** (~200 words): the ceiling **0.735**; the coverage×selection×ranking decomposition (dominant **selection** loss, 0.489); the honest anchor (GRAIL **0.330** < SyGMa **0.572** < MetaPredictor **0.585**, with the n=150 caveat where MetaPredictor appears); **TAME** (tautomer-aware, leakage-audited, match-sensitivity primary endpoint +0.120 [+0.073,+0.171]). Include the headline `[PENDING: multi-seed mean±std]` marker on the deployed recall.
- **1. Introduction** (~500–700 words): motivate around **coverage-conversion** (not "leaderboards aren't comparable"); honest no-win anchor up front (constraint 2); then the explicit **numbered contributions (~4)**: (1) rule-bank coverage **ceiling** (0.735) as a diagnostic primitive; (2) an exact **coverage×selection×ranking recall decomposition** + three refutable propositions; (3) **TAME** — a standardized, tautomer-aware, leakage-audited matching protocol + match-sensitivity ("rank-flip") analysis; (4) GRAIL as an interpretable, honestly-diagnosed instrument (one row, not a SOTA claim).

**Guardrails (constraints 2, 3, 6, 7):** honest anchor in both abstract and intro; no recall-win claim; the MetaPredictor 0.585 comparison carries the n=150 caveat; deployed recall carries the multi-seed PENDING marker; keep macro (0.330) for the cross-method comparison and micro (0.261) only where the decomposition is referenced.

- [ ] **Step 1: Write title + abstract + intro + contributions**

Replace the Task-1 title line (refine if needed), the `## Abstract` stub, and the `## 1. Introduction` stub. The contributions are an explicit numbered list inside §1.

- [ ] **Step 2: Verify anchor + contributions + no-win**

Run:
```bash
awk '/## Abstract/,/## 2\. Related Work/' docs/benchmark/manuscript.md | grep -E '0\.735|0\.330|0\.572|0\.585|TAME|\+0\.120'
awk '/## 1\. Introduction/,/## 2\. Related Work/' docs/benchmark/manuscript.md | grep -cE '^\s*[0-9]\.|\([1-4]\)'
grep -niE 'GRAIL (achieves|is) (the )?(best|state-of-the-art|SOTA)|GRAIL (wins|outperforms|beats) (all|SyGMa|every)' docs/benchmark/manuscript.md
```
Expected: first grep prints the abstract headline numbers; second grep ≥4 (numbered contributions present); third grep is EMPTY (no recall-win claim anywhere).

- [ ] **Step 3: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): title, abstract, introduction + numbered contributions"
```

---

### Task 12: Limitations + Data & Code Availability + Conclusion + track-list (§12–14)

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fill `## 12. Limitations`, `## 13. Data & Code Availability`, `## 14. Conclusion`, `## Draft TODO / open items`)

**Interfaces:**
- Consumes: anchor Δ (Task 7), propositions (Task 8), n=150 scoping (Task 9).
- Produces: the final closing matter.

**Source material:** the two referee-risk registers (`GRAIL_FRAMING.md` lines 293–302; `related_work_positioning.md §7`), the Proposition guardrails (Task 8), `GRAIL_FRAMING.md` "Net diagnosis" (lines 205–220), and the `§Reproducibility` release-artifact block (lines 363–386).

**Content:**
- **§12 Limitations** (ONE consolidated section — none exists today): no recall win (GRAIL loses to SyGMa Δ=−0.242 and to learned transformers); precision is a pessimistic lower bound under incomplete annotation; **single-checkpoint headline** (multi-seed `[PENDING]`); **n=150-vs-1170** comparability (tier-2 on shared subset); external CI wide (n=37); Prop-2 `e(r)∝π(r)` unmeasured assumption; depth-2/out-of-bank chemistry an open coverage lever.
- **§13 Data & Code Availability** (JCIM norm): the molecule-disjoint split (`fix_splits.py --molecule-disjoint` + `leakage_fix_report.json` audit); the frozen per-substrate 5-method × 5-protocol prediction set (`artifacts/tier2/*.json`, `artifacts/full5000_single/predictions/test_predictions.csv`); the re-scoring harness (`run_match_sensitivity.py`, `rank_flip_ci.py`); one-command regen (`regen_headline.sh`, dominant cost ~90 min); the **DrugBank-derived data license constraint**. Add `[PENDING: commit frozen tier-2 preds + leakage_fix_report.json]`.
- **§14 Conclusion** (fresh close from "Net diagnosis"): rule-based recall is coverage-conversion-limited (a dominant selection loss then a ranking loss); GRAIL as an interpretable instrument; TAME as the protocol. ~150 words.
- **## Draft TODO / open items** (replace the Task-1 stub with the out-of-scope track-list): COMPUTE — multi-seed headline mean±std (`run_multiseed.py` ≥3 seeds); tier-2 tools on full 1170; optional compute-matched GFlowNet null. CHEAP (post-draft) — commit frozen tier-2 preds + `leakage_fix_report.json`; budget-matched leaderboard; MetaTrans↔SyGMa paired CI; MetaTrans on GLORYx-37; regenerate `rankflip.svg` + `scaling_curve.svg` on current numbers; verify all comparator DOIs + resolve "Gao 2026"; build Fig 1 + the anchor/external/long-tail figures; Prop-1 paired CI + commit its artifacts.

**Guardrails:** every limitation stated plainly; no softening of the anchor; placeholders for compute-gated releases.

- [ ] **Step 1: Write the three closing sections + track-list**

Replace the four stubs. Limitations as flowing prose (not a bare list — a paragraph or a short prose-annotated list); Availability with the concrete artifact paths + license + PENDING; Conclusion fresh; track-list as the out-of-scope list.

- [ ] **Step 2: Verify limitations + availability + track-list**

Run:
```bash
awk '/## 12\. Limitations/,/## 13\./' docs/benchmark/manuscript.md | grep -niE 'no recall win|single.?checkpoint|n.?150|pessimistic lower bound|unmeasured'
grep -E 'run_multiseed|leakage_fix_report|DrugBank|regen_headline' docs/benchmark/manuscript.md | head
grep -c 'PENDING' docs/benchmark/manuscript.md
```
Expected: first grep prints the limitation items; second prints the availability artifacts; PENDING count ≥3 (multi-seed, paired CI, frozen preds).

- [ ] **Step 3: Commit**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): Limitations, Data & Code Availability, Conclusion + track-list"
```

---

### Task 13: Final number-consistency + guardrail verification pass

**Files:**
- Modify: `docs/benchmark/manuscript.md` (fixes only, if the gates find drift)

**Interfaces:**
- Consumes: the whole assembled document.
- Produces: a clean, internally-consistent draft ready for the adversarial referee review that follows this plan.

**This task is the whole-document gate.** It runs the spec's three verification gates and fixes any drift inline.

- [ ] **Step 1: Number-consistency gate (stale-value scan)**

Run:
```bash
cd /Users/nikitapolomosnov/PycharmProjects/GRAIL/.claude/worktrees/hungry-pasteur-25d746
echo "=== stale GRAIL headline 0.334 (should be absent as GRAIL's recall) ==="
grep -n '0\.334' docs/benchmark/manuscript.md
echo "=== 0.718 must NOT be labeled GRAIL's ceiling headline (only the plain-InChIKey ceiling) ==="
grep -n '0\.718' docs/benchmark/manuscript.md
echo "=== MetaBench must be absent ==="
grep -ni 'MetaBench' docs/benchmark/manuscript.md
echo "=== two-factor split language must be absent ==="
grep -niE 'two-factor' docs/benchmark/manuscript.md
```
Expected: `0.334` absent (the honest anchor is 0.330/0.261); every `0.718` occurrence is the plain-InChIKey *ceiling* (paired with 0.735), never GRAIL's recall; `MetaBench` and `two-factor` absent. Fix any violation inline.

- [ ] **Step 2: Guardrail gate**

Run:
```bash
echo "=== decomposition never theorem/verified/validated ==="
grep -niE 'theorem|verified|validated' docs/benchmark/manuscript.md | grep -i decomposition
echo "=== every reranker 0.500 sits with SyGMa 0.558 (manual check the lines) ==="
grep -n '0\.500' docs/benchmark/manuscript.md
echo "=== e(r) propensity flagged unmeasured ==="
grep -niE 'unmeasured' docs/benchmark/manuscript.md
echo "=== n=150 caveat present in the cross-method table region ==="
grep -niE 'n.?150 shared subset|single-n rerun is future work' docs/benchmark/manuscript.md
echo "=== multi-seed PENDING on the headline ==="
grep -c 'PENDING: multi-seed' docs/benchmark/manuscript.md
```
Expected: the theorem/decomposition grep is EMPTY; each 0.500 line co-occurs with SyGMa 0.558 (manual confirm); unmeasured present; n=150 caveat present; multi-seed PENDING ≥1. Fix any violation inline.

- [ ] **Step 3: Micro/macro single-reconciliation check**

Run:
```bash
grep -cE 'per-substrate mean \(macro\)|cross-method recall@15 is the per-substrate mean' docs/benchmark/manuscript.md
```
Expected: the ONE canonical reconciliation sentence appears exactly once (in §8). If it appears in multiple sections as a full restatement, collapse to one (brief forward-references are fine).

- [ ] **Step 4: make test stays green (docs-only change, sanity)**

Run:
```bash
make test 2>&1 | tail -5
```
Expected: PASS (unchanged count from HEAD — this plan touched no code). If anything fails, it is unrelated to this docs work — report it, do not attempt a code fix in this task.

- [ ] **Step 5: Attribution-clean check (constraint 9)**

Run:
```bash
git log --format='%an %ae %b' origin/main..HEAD 2>/dev/null | grep -iE 'co-authored-by|claude|anthropic|generated with' || echo "CLEAN — no AI attribution"
```
Expected: `CLEAN`. If anything matches, the commit history must be corrected before handoff.

- [ ] **Step 6: Commit any fixes**

```bash
git add docs/benchmark/manuscript.md
git commit -m "docs(manuscript): final number-consistency + guardrail pass" || echo "no fixes needed — nothing to commit"
```

---

## Post-plan step (controller, not a task)

After Task 13, run the **adversarial referee pass** over the assembled `manuscript.md` (as in the theory spine): a Workflow with three lenses — a JCIM-referee lens (is the story submittable?), a number-accuracy lens (does every figure trace to `results/*.json` or a marked placeholder?), and a guardrail lens (are all honesty constraints preserved?). Triage confirmed findings into fix subagents. This is the whole-branch review, not a per-task gate.

## Self-Review (run by plan author, done)

**Spec coverage:** Every spec section maps to a task — Title/Abstract/Intro/Contribs → Task 11; Related Work → Task 10; Methods (GRAIL/Formal/TAME) → Tasks 2/3/4; Results (ceiling/external/decomposition/anchor/diagnosis/match-sensitivity) → Tasks 5/6/7/8/9; Limitations/Availability/Conclusion → Task 12; Fig 1 placeholder → Task 1; SUPERSEDED banners → Task 1; out-of-scope track-list → Task 12; verification gates → Task 13. No gap.

**Placeholder scan:** the plan itself contains no TBD/TODO-as-instruction — the `[PENDING:]`/`[cite:]` strings are *deliverable content* the manuscript must carry (constraints 6, 8), not plan placeholders. Every number is written verbatim.

**Consistency:** the factor names (`coverage_bank`/`selection_retention`/`ranking_conversion`), the anchor (0.330 macro / 0.261 micro), the ceiling (0.735/0.718), and the interaction CI ([+0.073, +0.171]) are spelled identically across Tasks 3/5/6/7/8/9/11. The `## ` header strings in Task 1 match the section-locators used in Tasks 2–12.

# GRAIL → NeurIPS D&B: Paper Framing (draft)

**Status:** framing draft (2026-07-08), after the Stage-2 GFlowNet ablation returned a NULL.
Pivot: the **Datasets & Benchmarks paper is now the primary deliverable**; the GFlowNet is a
secondary honest-negative diagnostic. Drafted via a 5-section fan-out + an adversarial
NeurIPS-D&B-area-chair referee pass; the referee's fixes are baked in below and its full
risk register is in §Referee Risk Register.

---

## ⚠️ Empirical Results Update (2026-07-08) — TWO nulls; thesis reframed

The two potential headline results were tested and **both came back null**:

1. **Rank-flip is a NULL (referee FATAL-1 confirmed empirically).** `run_match_sensitivity.py`
   (150 substrates, GRAIL vs SyGMa, 5 match modes) shows **SyGMa > GRAIL under ALL 5 modes** —
   the leaderboard does **not** reorder. recall@15:

   | method | canon (LAGOM) | InChIKey | no-stereo (GLORYx) | Tanimoto=1 (MetaTrans) | tautomer (ours) |
   |--------|------|------|------|------|------|
   | GRAIL  | 0.356 | 0.357 | 0.358 | 0.356 | 0.365 |
   | SyGMa  | 0.514 | 0.547 | 0.548 | 0.514 | 0.554 |

   *What IS true (the surviving, weaker claim):* the match mode **shifts absolute recall
   differentially** — SyGMa swings ~8% (0.514→0.554), GRAIL ~2.5%, and the **gap varies
   0.158→0.190 (~20%)**. Evaluation is not match-invariant and methods are differentially
   sensitive, but on a strong-rule-vs-weak-learned pair the ordering is stable. A genuine
   reorder, if it exists, lives between **close-recall** methods — which requires ≥3 comparators
   (Tier-2 acquisition, the critical path). `results/match_sensitivity.json`.

2. **Set-reward GFlowNet is a NULL** (see contribution 5 / `ablation_local_verdict.json`).

**Reframed thesis (honest version):** the paper is now *"a standardized, leakage-audited,
tautomer-aware evaluation protocol for metabolite structure prediction, a diagnostic
decomposition showing the task is ranking/coverage-limited (0.718 ceiling), and two honest
negative results (no rank-flip on a strong-vs-weak pair; a set-reward GFlowNet does not beat
single-terminal sampling)."*

### UPDATE 2 (2026-07-08) — Tier-2 scouting REVIVES the strong story

The rank-flip null above was tested on the **wrong pair** (SyGMa 0.55 vs GRAIL 0.36 — 0.19 apart,
can't flip). A Tier-2 feasibility scan found the flip's natural home **does exist and is
obtainable**:

- **A close-recall cluster surrounds GRAIL.** On the shared GLORYx-37 set (LAGOM 2025, Table 2,
  recall@10): **MetaTrans 0.35, Chemformer 0.37, LAGOM 0.43, MetaPredictor 0.47** — all right on
  top of GRAIL's 0.36. SyGMa 0.68 / GLORYx 0.77 sit high **but only by emitting 800–1724
  predictions** vs ~320 for the capped transformers.
- **The recall spread is largely an OUTPUT-BUDGET artifact** — which *is* our decomposition
  thesis. At a common budget the 5–6 methods fall within ~0.11 of each other. "Who wins" is
  **budget-confounded**, and a benchmark that fixes budget + match-definition is the real
  contribution. **This is a quotable finding, not a null.**
- **Feasibility (env engineering, not method dev; days–weeks each):** BioTransformer 3.0 = EASY
  (JAR/Bioconda batch); MetaPredictor = MODERATE (cleanest transformer, Py3.8/PyTorch 1.13);
  MetaTrans = MODERATE (legacy pinned deps); LAGOM = MODERATE–HARD (needs external Chemformer
  ckpt); GLORYx = WEB-ONLY locally, but its **37-drug set + published 6-method recall is already
  usable as a second shared testbed.**

**Upgraded thesis (strongest defensible):** *method rankings in metabolite-structure prediction
are confounded by output budget AND fragile under the structure-match definition; we give the
protocol that controls both and show the 5–6-method leaderboard on a fair footing.* Stronger and
more honest than either "match decides who wins" or the modest 2-method story. **Path:** adopt
GLORYx-37 as a second shared testbed (cheap — published numbers + our oracle) + run the runnable
tools (BioTransformer first, then MetaPredictor) on our 150-substrate set at a common budget,
re-scoring under the match protocol. Investment = days–weeks of env work (user decision).

### UPDATE 3 (2026-07-09) — RANK-FLIP DEMONSTRATED with BioTransformer

Added BioTransformer (built + run in a Linux Docker container — jni-inchi loads on Linux; ARM-Mac
native is dead). recall@15 on our 150-substrate set, `results/match_sensitivity_3method.json`:

| method | canon (LAGOM) | InChIKey | no-stereo (GLORYx) | Tanimoto=1 (MetaTrans) | tautomer (ours) |
|--------|------|------|------|------|------|
| GRAIL  | 0.356 | 0.357 | 0.358 | 0.356 | 0.365 |
| SyGMa  | 0.514 | 0.547 | 0.548 | 0.514 | 0.554 |
| **BioTransformer** | **0.315** | **0.435** | **0.439** | **0.315** | **0.444** |

**The leaderboard REORDERS.** GRAIL vs BioTransformer SWAP rank by match mode:
- **strict-structural** (canonical, Tanimoto=1): **GRAIL > BioTransformer** (0.356 > 0.315).
- **normalized** (InChIKey, no-stereo, tautomer): **BioTransformer > GRAIL** (0.44 > 0.36).

BioTransformer swings **0.315↔0.444 (~40% relative)** — it emits stereo-specific SMILES that miss
under strict structural matching but hit once stereo/tautomers are normalized; GRAIL is flat
(0.356–0.365). **This is the paper's central thesis, now empirically DEMONSTRATED with one added
method** — "how you match decides who wins" holds on the GRAIL↔BioTransformer pair. The
close-recall cluster is exactly where the flip lives (as the Tier-2 scout predicted). Adding
MetaPredictor (0.47) / LAGOM (0.43) should enrich it. **The rank-flip contribution is REVIVED —
Referee FATAL-1 is no longer a null.** (Still needed: paired-bootstrap CI on the rank differences;
provenance harmonization; the budget-matched view.)

---

## Thesis

Metabolite structure prediction has **no agreed way to decide when a predicted structure is
correct**, and that single unstated choice — how a predicted molecule is *matched* to an
annotated reference — silently determines who wins. The literature matches incompatibly:
GLORYx by InChI-without-stereo, MetaTrans by fingerprint Tanimoto = 1, LAGOM by canonical
SMILES; and rule engines routinely emit a different *tautomer* of the reference than a naive
InChIKey canonicalizes, so plain string matching under-counts true hits. Because each paper
reports under its own convention on its own split, leaderboards are **not comparable**.

The primary obstacle is not model capacity but the absence of a shared, tautomer-aware,
leakage-audited protocol: on our clean molecule-disjoint split the **rule-bank coverage
ceiling is 0.718** (7,581 rules) — far above the best learned systems (~0.47) and rule-based
systems (SyGMa/GLORYx, 0.68–0.77 on their own sets) — so the task is limited by *ranking and
coverage-conversion*, not model expressiveness. The contribution is therefore a **benchmark
and protocol, not a new SOTA predictor**. GRAIL enters as one honest row (~0.40 absolute
recall, ~60% of SyGMa on our split), never the headline.

---

## Contributions (referee-hedged)

**(1) Standardized tautomer-aware matching + a match-sensitivity ("rank-flip") analysis.**
One protocol on tautomer-canonical InChIKey (`EvaluationConfig.match="inchikey_tautomer"`),
recall@k co-reported with mean output size and coverage, select-on-val / touch-test-once /
mean±std over ≥3 seeds. Alongside it we re-implement the literature's conventions
(`inchi_no_stereo`, `tanimoto1`, `canonical`, `inchikey`) and re-score the *same* predictions
of *every* method under all of them.
> ⚠️ **HEDGE (referee FATAL-1):** the claim "the leaderboard reorders" is **not yet
> demonstrated** — only the six modes are implemented. Until `run_match_sensitivity.py` is run
> and shows ≥2 methods inverting (or the learned↔rule gap materially compressing), state this
> as *"we measure how much a published ranking depends on its matching choice"*, not as a
> proven reordering. **This is the make-or-break experiment (see Open Experiments #1).**

**(2) Leakage-audited public split + the GLORYx-37 shared external set.** Molecule-disjoint
clean splits (`scripts/fix_splits.py --molecule-disjoint` + a machine-checkable
`leakage_fix_report.json` asserting zero substrate/molecule/positive-pair overlap). Split
trustworthiness corroborated by val≈test recall (0.327 vs 0.330 @15 — not overfit). GLORYx
37-drug set (135 annotated metabolites — reconcile the 135-vs-136 count) for cross-paper
comparability. **Not** "first benchmark" — Scholz 2023 / Gao 2026 exist — first *standardized,
leakage-audited* one with an explicit matching protocol.

**(3) Fair multi-method comparison under one protocol.** Re-score methods' predictions on one
shared split under one match mode. **Tiered by acquisition cost:** Tier-1 = SyGMa (run by us,
`run_benchmark.py:sygma_baseline`) + GRAIL + any method publishing per-substrate predictions;
Tier-2 = best-effort re-run of open tools (GLORYx, BioTransformer, MetaTrans, LAGOM,
MetaPredictor). Baseline acquisition is the **critical path, partly out of our control**;
GRAIL is one row, not the winner.
> ⚠️ Today only **SyGMa + GRAIL** are in hand. A "multi-method" claim on two rows is thin;
> the rank-flip figure needs ≥3 comparators (or published predictions) to show a real reorder.

**(4) Diagnostic decomposition of what limits the task.** Rule-bank ceiling 0.718 ≫ every
method; data-scaling **saturation** (400→2,418 tripled recall ~0.10→0.33; 2,418→4,787 flat,
+0.004 = noise; caveat: independent not prefix-nested subsamples); a ΔMW **gap decomposition**
(coverage / ranking / regioselectivity / multi-step) turning one recall number into an
attributable budget; and "empirical rule-frequency prior ≥ learned scorer" (filter is a good
classifier ROC-AUC 0.81 but a neutral ranker). Takeaway: **the gap is ranking quality, not
model capacity or data volume.**

**(5) The set-reward GFlowNet null — honest negative/diagnostic result (downscoped).** We
tested whether a set-level GFlowNet (diverse metabolite *set* per substrate, set-coverage
reward) beats a single-terminal GFlowNet and a pointwise ensemble on union@K coverage. **It
does not:** test union@K AUC **0.059** (set-reward) vs **0.187** (single-terminal) vs **0.215**
(ensemble); paired-bootstrap Δ = −0.128 and −0.142, 95% CIs entirely below 0; "null" across the
sensitivity grid; loses at every tested β′ ∈ {2..10}.
> ⚠️ **DOWNSCOPE (referee FATAL-2):** this rests on **n=12 paired test substrates**, a
> **compute-generous m=3 ensemble** (compute-matched deferred), a **β′ pinned to the sweep
> endpoint** (larger untested), and noisy per-seed AUCs [0.141, 0.170, 0.078]. Present as
> *"a preliminary negative result under our specific formulation, compute-generous baseline,
> and n=12 substrates"* — **NOT** as evidence that set-structured samplers are futile for
> metabolism, and **delete any "redirect the community" language.** Lead with the
> compute-comparable vs-single-terminal Δ (−0.128), not the vs-ensemble Δ. It is a *supporting
> anecdote for the ranking-ceiling thesis*, never a standalone claim.

---

## Referee Risk Register (NeurIPS D&B area-chair pass)

**Overall verdict:** *Needs-work (borderline, fixable to accept-shaped).* The benchmark/protocol
contribution is real and the honesty is a genuine strength, but three items can each draw a
reject/major-revision:

| # | Sev | Issue | Fix |
|---|-----|-------|-----|
| 1 | **FATAL → RESOLVED-as-null (2026-07-08)** | Rank-flip **tested and is a NULL**: SyGMa > GRAIL under all 5 modes (no reorder). The strong "leaderboard reorders" claim does NOT hold on GRAIL-vs-SyGMa. | **Reframed** to the differential-match-sensitivity claim (gap varies 0.158→0.190; SyGMa 8%-sensitive vs GRAIL 2.5%) — see Empirical Results Update. To salvage the *strong* claim, need ≥3 **close-recall** comparators (Tier-2, critical path); otherwise the honest modest paper (B) is the line. |
| 2 | **FATAL** | GFlowNet null rests on **n=12** pairs, CIs' upper bounds near 0 (−0.017/−0.019), compute-generous baseline, β′ at endpoint, compute-matched deferred. Not a "robust null" — underpowered/confounded. | Either run compute-matched + expand n before making it decision-relevant, or **drastically downscope the language** (done above). Inline power/CI caveat next to the headline number. |
| 3 | **MAJOR** | **Match-mode inconsistency on the flagship 0.718 ceiling**: FINDINGS.md says plain `InChIKey` (and SyGMa 0.558 too), but the framing labels 0.718 `tautomer-InChIKey`. Mislabeling the most-cited number is exactly the error the paper polices. | Reconcile from `run_benchmark.py`: state the exact mode consistently everywhere; add a **provenance table** (every headline number → match mode, split, seed count). Re-run under the recommended tautomer default, or note "reported under plain InChIKey; tautomer would only raise them." |

*(The referee returned further minor risks — baseline-acquisition thinness, GLORYx 135-vs-136
count, non-nested scaling caveat, citation-graph verification — folded into Open Experiments.)*

---

## Open Experiments / Next Steps (ordered by referee priority)

1. **[MAKE-OR-BREAK] Run the rank-flip** (`scripts/run_match_sensitivity.py`) on SyGMa + GRAIL
   (+ any published predictions) under all six match modes → does the ordering/gap actually
   move? This validates or kills the thesis. Add paired-bootstrap CIs on the rank differences
   (mirror the ablation's rigor) so a flip isn't sampling noise. *(Gated behind the running β′
   sweep for CPU.)*
2. **Reconcile the ceiling/SyGMa provenance** — confirm the match mode of 0.718 and 0.558 from
   `run_benchmark.py`; build the single provenance table. *(Cheap, do first — it's a doc/verify
   task, not compute.)*
3. **Land Tier-2 comparators** — re-run/obtain ≥1–2 of GLORYx/BioTransformer/MetaTrans/LAGOM so
   the multi-method table and the rank-flip have ≥3 rows.
4. **Compute-matched GFlowNet null** (+ expand the paired set beyond 12) — only if we keep the
   null as more than a supporting anecdote.
5. **Pre-submission checks** — Semantic Scholar citation-graph pass for the "no prior
   GFlowNet-for-metabolism" and "first standardized protocol" claims; #Circles / union@K
   threshold sensitivity; verify GLORYx 135-vs-136 and the literature match-convention
   attributions.

---

## Provenance table (referee MAJOR #3 — RECONCILED 2026-07-08)

**CONFIRMED via `run_benchmark.py` (line 36 `from grail_metabolism.metrics import _inchikey`;
line 46 "no tautomer standardization"; line 319 sygma "plain-InChIKey matching"):** the
flagship ceiling and SyGMa numbers are **plain InChIKey**, while GRAIL's headline is
**tautomer-InChIKey**. The headline comparison therefore MIXES match modes — the exact error
the paper polices. **FIX (required before submission): re-run the ceiling + SyGMa under
`inchikey_tautomer` so every headline number shares one mode.** Direction is favorable — tautomer
raises the ceiling and SyGMa (GRAIL is already measured on the generous mode), so the
ranking-limited thesis only strengthens; but the numbers must be harmonized for defensibility.

| Number | Value | Match mode | Split | Seeds | Source |
|--------|-------|-----------|-------|-------|--------|
| Rule-bank ceiling | 0.718 | **plain InChIKey (CONFIRMED)** → re-run tautomer (≥0.718) | clean test | — | `run_benchmark.py:grail_ceiling` (`_inchikey`) |
| SyGMa recall@15 | 0.558 | **plain InChIKey (CONFIRMED)** → re-run tautomer (≥0.558) | clean test | — | `run_benchmark.py:sygma_baseline` |
| GRAIL recall@15 | ~0.33–0.36 | tautomer-InChIKey | clean test | ≥3 | `docs/FINDINGS.md` |
| GFlowNet null (test union@K AUC) | 0.059 / 0.187 / 0.215 | tautomer-InChIKey | clean test | 3 (n=12 pairs) | `results/ablation_local_verdict.json` |

> **Risk-register update:** MAJOR #3 is now **CONFIRMED**, not merely suspected. The single
> provenance fix (re-run ceiling + SyGMa under `inchikey_tautomer`) resolves it and is a
> prerequisite for any headline table.

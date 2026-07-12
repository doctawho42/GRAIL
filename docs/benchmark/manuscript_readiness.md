# Manuscript-readiness map (GRAIL + diagnosis + TAME)

> 2026-07-13, from a 5-lens readiness workflow that read the current docs + artifacts and independently
> re-confirmed the anchor (Δ=−0.242, CI[−0.271,−0.212], McNemar p≈1.7e-44, n=1168), the decomposition
> (0.735×0.488×0.726=0.261 with block-bootstrap CIs), and the pre-declared primary endpoint. **Verdict:
> ~70% of the manuscript already exists as high-quality internal notes; the remaining work is assembly +
> de-guardrailing + venue hygiene — NOT new science — gated by exactly ONE compute item (multi-seed
> headline) and ONE scoping decision (n=150 vs 1170).**

## The trap to kill first
A full manuscript draft **already exists but is STALE and wrong-thesis**: `docs/benchmark/paper_draft.md`
(2026-06-29, pre-reframe) is the demoted "MetaBench / How You Match Decides Who Wins" version with
contradicting numbers (ceiling **0.718** not 0.735; GRAIL **~0.334/~0.40** not 0.330 macro/0.261 micro;
a two-factor split; the name **MetaBench** not TAME; the withdrawn GFlowNet-null contribution). It is a
reusable prose **skeleton** only — its story, headline figures, and citations must be **discarded, not
built on**. `DNB_FRAMING.md` is likewise superseded (match-sensitivity-as-headline + dated UPDATE-1..5
discovery narrative). **Action: mark both SUPERSEDED so only the GRAIL_FRAMING thesis/numbers/TAME name
survive.**

## Manuscript outline — section-by-section status

Status legend: **write-from-existing** (content reconciles, just needs prose) · **rewrite-voice** (present
but in notes/guardrail voice) · **assemble** (numbers exist, table/figure not built) · **needs-new-work**
(write from scratch) · **missing**.

| # | Section | Status | Effort | Source material | To do |
|---|---|---|---|---|---|
| 1 | Title | needs-new-work | S | stale paper_draft:1 | fresh GRAIL-primary title; drop "How You Match…"/MetaBench |
| 2 | Abstract | needs-new-work | M | GRAIL_FRAMING §Thesis+anchor | write around 0.735 ceiling, coverage×selection×ranking, honest anchor, TAME |
| 3 | Intro + motivation | rewrite-voice | M | paper_draft:20-45 (wrong frame), GRAIL_FRAMING:7-31 | re-story around coverage-conversion limit; honest no-win anchor up front |
| 4 | **Contributions (numbered)** | needs-new-work | S | none for current thesis | 4-item list: ceiling; decomposition+3 props; TAME+match-sensitivity; honest GRAIL row |
| 5 | Related Work | rewrite-voice | M | `related_work_positioning.md` (current) | memo→prose; **verify all comparator DOIs**; resolve "Gao 2026" |
| 6 | Method — GRAIL architecture | write-from-existing | M | GRAIL_FRAMING §1 (terse), ARCHITECTURE.md, CLAUDE.md | expand to methods prose: generator/RDKit/PU-logit MCS pair filter/ranking |
| 7 | **Fig 1 — pipeline schematic** | missing | M | only raw `img/*.png` legacy | build 3-stage schematic |
| 8 | Method — Formal framework §1.5 | rewrite-voice | M | GRAIL_FRAMING:43-113 | de-guardrail (collapse "never a theorem" asides to one caveat); polish mixture+identity |
| 9 | Method — TAME protocol | rewrite-voice | M | GRAIL_FRAMING §Repro, protocol.md, paper_draft §4 example | one section; rename MetaBench→TAME; 5 quotients + split + frozen preds |
| 10 | Results — ceiling §2 | rewrite-voice | S | GRAIL_FRAMING:115-128, benchmark_report.json | prose + ceiling-vs-baselines table (0.735/0.718; SyGMa 0.572) |
| 11 | Results — external validity | rewrite-voice | M | GRAIL_FRAMING:130-149, ceiling_external_validity.json | prose + internal-vs-external scatter (0.633, OOS recovers ~56%) |
| 12 | **Main cross-method table** | assemble | M | match_sensitivity_5method.json | one table w/ mean_output_size + rank-flip CIs + honest-anchor row; **flag n=150** |
| 13 | Results — decomposition §1.5/§3 | write-from-existing | S | factor table + factorization_waterfall.svg | Fig 2 caption w/ CIs (0.735×0.488×0.726=0.261) |
| 14 | Results — anchor certification §3 | rewrite-voice | S | anchor_certification.json (verified) | prose + paired-Δ/McNemar figure; scope to EVALUATION variance |
| 15 | Results — diagnosis + 3 Propositions §4 | rewrite-voice | L | GRAIL_FRAMING:193-279 + JSONs | lever→factor table + per-Prop evidence box; **fix Prop-1 evidence (see gaps)** |
| 16 | Match-sensitivity supplement | rewrite-voice | M | GRAIL_FRAMING:281-291, DNB_FRAMING | strip UPDATE-1..5 voice; keep interaction +0.120 CI[+0.073,+0.171] |
| 17 | **Limitations** | needs-new-work | M | scattered in 2 referee registers + prop guardrails | ONE section (no "Limitations" heading exists today) |
| 18 | Reproducibility statement | write-from-existing | M | GRAIL_FRAMING §Repro, regen_headline.sh | prose + data-availability + license (DrugBank-derived) |
| 19 | **Datasheet / benchmark card** | missing | M | protocol.md, DATASETS.md (not datasheets) | write; resolve split-count conflicts (1170 vs 1246; GLORYx 135 vs 136) |
| 20 | **Ethics / broader impact** | missing | S | none | write (false-negative safety; responsible release of DrugBank-derived + 3rd-party preds) |
| 21 | **NeurIPS repro/author checklist** | missing | S | fillable from provenance table | complete (venue-conditional) |
| 22 | Conclusion | needs-new-work | S | GRAIL_FRAMING "Net diagnosis" (notes) | fresh close: coverage-conversion limit; GRAIL as instrument; TAME |

## Blocking gaps (block a first submittable draft)
1. **No assembled current-thesis manuscript** — only the stale MetaBench draft. (assembly, no compute)
2. **No numbered contributions list** for the current thesis. (write, no compute)
3. **Headline is a SINGLE deployed checkpoint — no multi-seed mean±std.** `anchor_certification.json`
   certifies only *evaluation* variance; CLAUDE.md itself mandates seed-averaged headlines. Every
   downstream object (anchor Δ, decomposition, all 3 Propositions) rests on this one point.
   `run_multiseed.py` exists but was never run on the deployed pipeline. **← the one true COMPUTE gate.**
4. **Leaderboard mixes n: tier2 preds (BioT/MetaPredictor/MetaTrans) are all n=150; GRAIL/SyGMa n≈1170.**
   So "MetaPredictor 0.585 vs SyGMa 0.572" (used to justify "GRAIL doesn't win") are **not on the same n**.
   → rerun tier2 on 1170 (compute) OR scope the n=150-vs-1170 split prominently in every table.
5. **No consolidated Limitations section** (grep: no such heading in the primary docs).
6. **NeurIPS D&B mandatory sections absent**: Ethics/broader-impact, Datasheet, author checklist,
   dedicated Conclusion. (venue-conditional blockers)
7. **Released "frozen predictions" + leakage audit NOT in the tree**: 0 files under `artifacts/` are
   git-tracked and `results/leakage_fix_report.json` is absent → the release + zero-overlap-audit claims
   are unbacked in-tree. (cheap: `git add -f` the frozen preds; run `fix_splits.py` to emit the report)
8. **Comparator citations self-flagged UNVERIFIED** (MetaPredictor, MetaTrans vol/DOI, Dhaked 2019,
   DataSAIL; "Gao 2026" unresolved). A reference list cannot ship. Do NOT trust paper_draft's asserted
   citations over the memo's flags. (cheap: lit lookup)
9. **Prop 1 evidence (the "already confirmed" reranker 0.433→0.500) rests on artifacts absent from the
   tree** (`reranker_gate_bi_test*.json`, `gflownet_m2_test*.json`) and has only a 3-seed std, not a
   paired CI. Currently prose-only in `stage2_ranker_evidence.md`. (cheap: commit artifacts + paired CI, OR downgrade Prop 1 to "suggestive")

## Missing experiments (referee-must-haves), classified
- **NEEDS COMPUTE** — (a) **multi-seed mean±std headline** (`run_multiseed.py` ≥3 seeds on deployed
  gen+filter) — #1 referee-must-have, unblocks gap 3; (b) **tier2 tools on the full 1170** for a single-n
  leaderboard (gap 4) — skip if scoping instead; (c) compute-matched GFlowNet null (n>12) — else keep
  the n=12 result a downscoped anecdote.
- **CHEAP (no training — frozen preds / re-scoring / lit lookup)** — budget-matched leaderboard
  (recall@k + precision + output-size across 5 methods) = the benchmark's strongest quotable claim;
  MetaTrans↔SyGMa paired rank-flip CI (Holm family names it, absent); paired per-substrate CI on the
  reranker +0.067 + commit its artifacts; commit frozen tier2 preds + `leakage_fix_report.json`;
  regenerate `rankflip.svg` + `scaling_curve.svg` on current tautomer/5-method numbers (both stale, Jun 29);
  verify comparator DOIs + resolve "Gao 2026"; MetaTrans on the GLORYx-37 external set; build Fig 1 +
  anchor Δ/McNemar figure + external scatter + ΔMW long-tail figure.
- **HAVE** — primary endpoint (interaction +0.120, CI[+0.073,+0.171]); anchor certification; external
  ceiling; recall decomposition (all reconcile against `results/*.json`).
- **OUT OF SCOPE** — GLORYx-the-tool (FAME3 weights unavailable) + LAGOM (no released checkpoint): cite
  published numbers only.

## Prioritized plan (framing-doc → first full draft)
**NOW (no compute):**
1. Freeze the thesis: mark `paper_draft.md` + `DNB_FRAMING.md` SUPERSEDED; write the numbered
   contributions list + fresh title + abstract.
2. Assemble the Results section from already-verified JSONs (highest yield): promote provenance table +
   §1.5 factor table to main tables; caption `factorization_waterfall.svg` as Fig 2; write ceiling /
   external-validity / decomposition / anchor / diagnosis subsections as de-guardrailed prose.
3. Write the Method: GRAIL 3-stage architecture (ARCHITECTURE.md + CLAUDE.md); de-guardrail §1.5;
   consolidate TAME (rename MetaBench→TAME); build Fig 1 pipeline schematic.
4. Write the missing prose: Limitations (from the two referee registers), Conclusion, Ethics, Datasheet,
   author checklist, data-availability+license.
5. Cheap evidence fills: commit frozen tier2 preds + `leakage_fix_report.json`; run re-scoring for the
   budget-matched leaderboard + MetaTrans↔SyGMa CI + MetaTrans-on-GLORYx; regenerate the two stale
   figures on current numbers; verify all DOIs + resolve "Gao 2026".
6. Decide the **n=150-vs-1170 posture**: scope prominently in every table caption now, OR queue the
   tier2-on-1170 rerun. Resolve split-count conflicts (1170 vs 1246; GLORYx 135 vs 136) in the datasheet.

**WAITS ON COMPUTE:**
7. `run_multiseed.py` ≥3 seeds on the deployed gen+filter → report the headline as mean±std (unblocks
   the single-checkpoint gap under the anchor, decomposition, and all propositions).
8. Rerun tier2 on the full 1170 (single-n leaderboard; skip if scoping); regenerate + commit the Prop-1
   reranker/GFlowNet artifacts + paired CI (or downgrade Prop 1 to "suggestive"); optional compute-matched
   GFlowNet null.

## Highest-leverage single move
Stand up ONE skeleton now: mark the stale drafts superseded, write the contributions list + abstract, and
drop the verified provenance table + §1.5 factor table + waterfall figure into a Results section. That one
act converts scattered notes into a reviewable draft and makes every remaining gap concrete.

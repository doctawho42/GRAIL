# GRAIL — primary framing: a rule-based metabolite-structure predictor and a diagnosis of what limits it

> **Status:** living framing doc for the GRAIL-primary paper (reframed 2026-07-10). The
> match-sensitivity benchmark (`docs/benchmark/DNB_FRAMING.md`) is now a **supporting**
> fair-evaluation section, not the headline. Angle chosen by the user: **"GRAIL + its diagnosis."**

## Thesis

Rule-based metabolite-structure prediction is limited **not by rule coverage but by the
conversion of that coverage into a ranked prediction set.** We present **GRAIL**, a rule-based
predictor that learns to *select* SMIRKS transformations over a large curated bank (7,581 rules)
and score the resulting (substrate, product) pairs, and we use it to **decompose where the
rule-based paradigm's headroom is lost** — coverage vs ranking vs regioselectivity vs multi-step.
The contribution is the architecture **plus** a rigorous, honest diagnosis; a standardized,
tautomer-aware, leakage-audited evaluation protocol (the benchmark) is the apparatus that makes
the diagnosis fair and comparable.

## Honest anchor (state this up front, everywhere)

**GRAIL does not win on recall.** On our clean molecule-disjoint split it reaches
**~0.334 recall@15** (tautomer-InChIKey, deployed pipeline; prior-independent — verified on both
checkpoints), below SyGMa (0.57 tautomer, full clean test) and MetaPredictor (0.585).
The paper's A*/Q1 value is therefore the **ceiling + the diagnosis + the protocol**, *not* a SOTA
claim. GRAIL enters its own comparison as one honest row. We never imply GRAIL is the best
predictor; we show it is an interpretable, rule-grounded instrument that exposes *why* the
paradigm plateaus.

## Section structure & evidence

### §1 — Method: GRAIL (the rule-based architecture)
Three stages: (1) a learned multi-label **rule selector** over the 7,581-SMIRKS bank
(retrieval-scored generator); (2) RDKit **rule application** enumerating candidate products;
(3) a **PU-trained, MCS-aware pair filter** scoring (substrate, product) pairs. Trained on
positive-unlabeled data (rule-applicable non-annotated products are unlabeled, not negative).
*Status: built, trained (full5000 single-mode). Contribution = interpretable learned rule
selection + PU-aware pair filter, not recall supremacy.*

### §2 — The rule-bank coverage ceiling (the hero number)
The bank's **recall ceiling** — the best achievable recall if the selector/filter were perfect —
on the full clean molecule-disjoint test split (1170 substrates, 2597 true pairs) is
**0.718 (plain InChIKey) / 0.735 (tautomer-InChIKey)** — 1865 / 1910 of 2597 recovered. Far above
the best learned systems (~0.47–0.585) and comparable to rule systems on their own sets.
**Coverage is high; the bank is powerful.** So the task is limited by *ranking and
coverage-conversion*, not rule expressiveness. The 0.735 is reported under the **same tautomer
protocol as GRAIL and SyGMa** (referee MAJOR-3 resolved — no more mixed match modes); the plain
0.718 reproduces the previously reported number exactly, validating the run. The tautomer path is
computed via a heavy-atom-formula prefilter **verified sound** against naive keying (audit
mismatch = 0 on 50 substrates). The same run co-measures the **SyGMa** baseline under both
protocols on the identical split: recall@15 **0.558 plain / 0.572 tautomer** (n=1168) — so the
ceiling, SyGMa, and GRAIL are now all reported under tautomer-InChIKey. *Source:
`results/benchmark_report.json` / `run_benchmark.py`.*

### §3 — The conversion gap (ceiling ≫ realized)
GRAIL converts only **~0.334 of the ~0.72 ceiling** into top-15 recall — a **~46% conversion**. The
gap is the paper's central diagnostic object. §4 decomposes it.

> **Note (checkpoint / prior — resolved by re-measurement).** The deployed GRAIL recall was
> re-checked with the prior-populated checkpoint `full5000_priors` (byte-identical learned weights,
> trained prior present): the exported test predictions give **0.335** vs `full5000_single`'s
> **0.334** — essentially identical. So the *deployed* pipeline is prior-independent (it applies
> rules broadly, so the prior's rule-reordering does not change coverage-limited recall), and the
> ~0.30–0.36 headline stands. The large prior effect (§4 row 1) appears only in a *top_k-limited
> rule-selection* probe, not in deployment — an earlier draft over-extrapolated that probe to
> claim a ~0.40 headline; that claim is withdrawn.

### §4 — Diagnosis: where the headroom goes (all rule-based, all measured)
| Lever | Finding | Evidence |
|---|---|---|
| **Ranking: learned vs prior** *(top_k-limited rule-selection probe — not the deployed recall)* | In a controlled probe (each mode picks its top-30 rules by its own score, same product loop + filter), a SyGMa-style **frequency prior significantly OUT-ranks the learned generator**: prior-only 0.410 vs learned-only 0.266 gen-only (Δ −0.144, 95% CI [−0.196, −0.095], paired bootstrap); with the filter 0.405 vs 0.300 (−0.105, CI [−0.152, −0.058]). Adding the prior to the learned scorer lifts **+0.130** gen / +0.099 filter (both significant) — the prior is the load-bearing rule-selection signal; the filter significantly helps only the *weak learned* ordering (+0.034, CI [+0.011,+0.061]), not the strong prior (n.s.). NB: these probe numbers isolate rule *ranking*; the **deployed** GRAIL (0.334) applies rules broadly and is prior-independent, so it neither inherits the +0.13 nor exposes the learned selector's weakness — the probe reveals it. | `scripts/prior_vs_learned.py`, `results/prior_vs_learned.json` (245 test subs, paired-bootstrap CI) |
| **Multi-step** | Breadth-capped depth-2 rule application lifts the ceiling by **only +0.012** (0.711→0.723) at **8.5× candidate cost** (194→1653). Multi-step is **not** the dominant coverage lever; most uncovered metabolites are genuinely out-of-bank. | `run_benchmark --depth 2`, `results/benchmark_report_depth2.json` (150 subs, beam 10) |
| **Coverage (ΔMW gap)** | **26.9%** of true metabolites are uncovered by the depth-1 bank (plain-InChIKey; ~1.7 pp lower, ~25%, under the tautomer protocol the ceiling uses). Misses are a **diverse long tail** (top class = hydroxylation, only 6% of uncovered) — one-off large conjugates (glucuronide, glutathione, sulfate, mercapturate) + unusual transforms. No single rule-family addition closes much. | `run_benchmark --gap-analysis`, `results/benchmark_report_gap.json` (500 subs) |
| **Data scaling** | Recall saturates (2418→4787 substrates ≈ flat) — the plateau is **not** a data-quantity problem. | `results/full{2500,5000}_single.log` |
| **Regioselectivity (SoM)** | Site-of-metabolism prior gives only a small lift — regioselective ranking is hard within the bank. | `results/train_som.log` |

**Net diagnosis:** the rule bank covers ~0.72–0.73; the deployed GRAIL surfaces only **~0.334** of
it in top-15 (a ~46% conversion). The gap is a **ranking** problem: from a candidate pool whose
coverage approaches the ceiling, GRAIL's filter+ranking promotes only ~0.334 into the top 15. A
controlled rule-selection probe sharpens *why the learned part does not help*: in a top_k-limited
setting the learned generator ranks rules **worse than a trivial frequency prior** (−0.14 gen-only,
significant; row 1). The deployed pipeline masks this by applying rules broadly (so the prior does
not change deployed recall — verified), but the probe shows the learned rule-selector is itself a
liability vs frequency. Neither multi-step nor any single rule-family addition moves coverage much
(misses are a diverse long tail). **The open problem is ranking — neither the learned generator nor
the filter surfaces the rich candidate pool well, and the learned rule-selector loses to a
frequency baseline.** (An earlier draft reported the opposite "learned beats prior"; that was an
artifact of a checkpoint whose prior buffer was un-persisted, caught by adversarial verification
and corrected here.)

### §Supplement — Fair evaluation: a standardized, tautomer-aware, leakage-audited protocol
The match-sensitivity ("rank-flip") analysis and the multi-method comparison move here. Their job
is to justify that the numbers above are apples-to-apples and that prior rule-vs-learned
comparisons were confounded (methods gain method-dependent, significant amounts from the match
protocol — differential sensitivity CI [+0.073, +0.171]; see `docs/benchmark/DNB_FRAMING.md`).

## Referee-risk register (GRAIL-primary)
- **"GRAIL loses — why publish it?"** → we do not claim a recall win; the contribution is the
  ceiling + diagnosis + protocol. Lead with that; keep GRAIL one honest row.
- **"Ceiling is trivially high because InChIKey nesting."** → resolved: ceiling reported under
  the SAME tautomer protocol as GRAIL (§2 provenance), prefilter verified sound.
- **"Depth-2 is a lower bound — maybe multi-step helps more."** → acknowledged (breadth-capped,
  beam 10); even the lower bound + the 8.5× cost make the point; state it as a bound.
- **"Prior-vs-learned n=245, single split."** → report CIs; it is a diagnostic, not a selection.
- **"Is the architecture novel enough?"** → position vs SyGMa (fixed rules) and transformers
  (no rule grounding): learned selection over a large SMIRKS bank + MCS-aware PU pair filter.

## Open items before submission
1. **Provenance** (in flight): full-set ceiling + SyGMa under tautomer → fill §2 hero number.
2. **prior-vs-learned CIs** (paired bootstrap on the learned−prior gap) for §4 rigor.
3. **Tier-2 for the supplement** (LAGOM/MetaTrans) to widen the match-sensitivity spread.
4. Fold `DNB_FRAMING.md` explicitly into the §Supplement narrative.

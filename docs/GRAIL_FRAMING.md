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
~0.30–0.36 recall@15 (tautomer-InChIKey), below SyGMa (0.55) and MetaPredictor (0.585). The
paper's A*/Q1 value is therefore the **ceiling + the diagnosis + the protocol**, *not* a SOTA
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
mismatch = 0 on 50 substrates). *Source: `results/benchmark_report.json` / `run_benchmark.py`.*

### §3 — The conversion gap (ceiling ≫ realized)
GRAIL converts only **~0.30 of the ~0.72 ceiling** into top-15 recall — a ~42% conversion. The
gap is the paper's central diagnostic object. §4 decomposes it.

### §4 — Diagnosis: where the headroom goes (all rule-based, all measured)
| Lever | Finding | Evidence |
|---|---|---|
| **Ranking: learned vs prior** | The learned rule-selection **beats a frequency prior** (+0.093 gen-only, recall@15). The prior is **redundant on top of** the learned scorer (blend == learned exactly). The **filter does real ranking work** (rescues the weak prior +0.074 vs strong learned +0.034). So the gap is *ranking quality*, **not** a dead learner beaten by a trivial baseline. | `scripts/prior_vs_learned.py`, `results/prior_vs_learned.json` (245 test subs) |
| **Multi-step** | Breadth-capped depth-2 rule application lifts the ceiling by **only +0.012** (0.711→0.723) at **8.5× candidate cost** (194→1653). Multi-step is **not** the dominant coverage lever; most uncovered metabolites are genuinely out-of-bank. | `run_benchmark --depth 2`, `results/benchmark_report_depth2.json` (150 subs, beam 10) |
| **Coverage (ΔMW gap)** | **26.9%** of true metabolites are uncovered by the depth-1 bank. Misses are a **diverse long tail** (top class = hydroxylation, only 6% of uncovered) — one-off large conjugates (glucuronide, glutathione, sulfate, mercapturate) + unusual transforms. No single rule-family addition closes much. | `run_benchmark --gap-analysis`, `results/benchmark_report_gap.json` (500 subs) |
| **Data scaling** | Recall saturates (2418→4787 substrates ≈ flat) — the plateau is **not** a data-quantity problem. | `results/full{2500,5000}_single.log` |
| **Regioselectivity (SoM)** | Site-of-metabolism prior gives only a small lift — regioselective ranking is hard within the bank. | `results/train_som.log` |

**Net diagnosis:** the rule bank covers ~0.72–0.73; GRAIL's learned generator + filter convert
only ~0.30 of it; the learned components genuinely help (beat prior, filter adds value) but leave
a large ranking gap; and neither multi-step nor any single rule-family addition moves coverage
much. **The open problem is ranking/conversion quality — honestly characterized, not hand-waved.**

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

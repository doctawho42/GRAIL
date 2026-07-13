> **⚠ SUPERSEDED (2026-07-13).** This is the pre-reframe "MetaBench / How You Match Decides Who Wins" draft. Its thesis, headline numbers (ceiling 0.718, GRAIL ~0.334/~0.40, two-factor split), and name are wrong-thesis and MUST NOT be built on. The current manuscript is `docs/benchmark/manuscript.md` (GRAIL-primary, ceiling 0.735, three-factor decomposition, TAME). Preserved for history only.

# How You Match Decides Who Wins: A Standardized, Leakage-Audited Benchmark for Metabolite Structure Prediction

*Working draft (NeurIPS Datasets & Benchmarks track). Sections marked [PENDING] await
deferred experiments (cross-method baselines; the empirical-prior-vs-learned study).*

## Abstract

Predicting the metabolite **structures** of a drug or xenobiotic is a core ADME-tox task,
yet the field has no shared benchmark and each method matches predicted to reference
structures by a different rule — GLORYx by stereo-free InChI, MetaTrans by Tanimoto=1, LAGOM
by canonical SMILES, others by InChIKey. We show these choices are not cosmetic: holding a
method's predictions fixed and only changing the match protocol moves recall substantially
and can reorder the leaderboard. We release **MetaBench**: a molecule-disjoint, leakage-
audited split; a single standardized, tautomer-aware matching protocol with an explicit
match-sensitivity analysis; and a fair multi-method comparison. We further decompose *what
limits the task*, separating coverage loss (unreachable by any rule) from ranking loss
(reachable but mis-ranked), and find ranking — not coverage or data — is the dominant gap for
learned rule-rankers. MetaBench gives the field a common, honest yardstick.

## 1. Introduction

Metabolite structure prediction underlies toxicity assessment, drug–drug interaction
analysis, and environmental fate modeling. Methods span expert rule systems (SyGMa, Meteor,
BioTransformer), rule+ML hybrids (GLORYx), and sequence/graph learners (MetaTrans,
MetaPredictor, LAGOM). The recurring published "leaderboard" (GLORYx 0.77, SyGMa 0.68,
MetaPredictor 0.47, LAGOM 0.43, MetaTrans 0.35) reads like a ranking, but tracing each number
to its source shows it is **not one measurement**: it mixes different *test sets*, *match
definitions*, **and prediction budgets** (GLORYx 0.77 is *uncapped* recall over 1,724
predictions; LAGOM 0.43 is *top-10* over ~328), and two of the five numbers are not even the
named method's own — they are downstream re-runs or mis-attributions (§5). The spread is
dominated by these protocol choices, not method quality.

Two problems compound. First, **matching is inconsistent**: a prediction that is "correct"
under one paper's protocol is "wrong" under another's (a stereoisomer matches under
stereo-free InChI and Tanimoto=1 but not strict InChIKey; a tautomer matches only under
tautomer canonicalization). Second, **splits are rarely leakage-audited**, and the
rule-vs-learned comparison is structurally unfair (rule systems have no held-out training
set). No prior benchmark addresses either.

**Contributions.** (1) **MetaBench**, a molecule-disjoint, leakage-audited
substrate→metabolite benchmark (train/val/test = 9,339/1,028/1,246 substrates). (2) A
**standardized, tautomer-aware matching protocol** plus a **match-sensitivity experiment**
that quantifies how much rankings depend on the match definition. (3) A **fair multi-method
comparison** under one protocol. (4) A **limit decomposition** separating coverage from
ranking loss, showing ranking is the dominant, data-saturating gap.

## 2. Related work

**Methods.** SyGMa [Ridder & Wagener, ChemMedChem 2008] applies expert rules with an
empirical reaction-likelihood score. GLORYx [de Bruyn Kops et al., Chem. Res. Toxicol. 2021]
adds FAME-based site-of-metabolism ranking. BioTransformer [Wishart et al., NAR 2022] is a
hybrid rule+ML system. MetaTrans [Litsa et al., Chem. Sci. 2020], MetaPredictor [Zhu et al.,
Brief. Bioinform. 2024], and LAGOM [Larsson et al., AI in Life Sci. 2025] are learned. Their
*own*-reported recalls are **not** the low ~0.35–0.47 figures often quoted for them: MetaTrans
reports 0.576@10 and MetaPredictor 0.739@15 (both under fingerprint Tanimoto=1, on their own
sets); the 0.35/0.47 are LAGOM's canonical-SMILES re-runs / a mis-attribution (§5). **No
shared leaderboard exists**; the recurring GLORYx external set (37 drugs/136 metabolites) is
the closest thing to a common hold-out, but even there reported numbers use different k and
matching.

**Evaluation.** Application studies [Scholz et al., Sci. Total Environ. 2023; Gao et al.,
JCIM 2026] run several tools but neither standardizes matching nor audits leakage. Tautomer
ambiguity in structure identity is known [Hähnke et al., 2018: ~60% of PubChem structures
differ from their InChI, mostly by tautomer] but unhandled in metabolite evaluation. We are
therefore not the first *comparison*, but the first *standardized, leakage-audited,
match-sensitivity* protocol. **Out of scope:** site-of-metabolism prediction (FAME3,
SMARTCyp — predict *where*, not the product) and spectral annotation (CASMI — rank database
candidates against MS/MS), which are distinct tasks.

## 3. The MetaBench benchmark

Task: from a parent SMILES, output a ranked set of metabolite SMILES. Data are
`substrate–metabolite` triples over an indexed SDF; negatives are rule-applicable but
unannotated products (positive-unlabeled). Splits are **molecule-disjoint** (no test/val
substrate in train; no test substrate in val), built and verified by an audit that checks
zero substrate and zero positive-pair overlap. We report the **rule-vs-learned fairness
caveat** explicitly: expert-rule systems have no held-out training set, so disjoint splits
constrain learned models only. (Full datasheet: `protocol.md`.)

## 4. Matching protocol and match-sensitivity

We expose every literature match rule as a set key: `exact` (canonical SMILES), `inchikey`,
`inchi_no_stereo` (InChIKey skeleton; GLORYx), `tanimoto1` (identical Morgan fingerprint;
MetaTrans), and `inchikey_tautomer` (tautomer-canonical InChIKey; **recommended**). A method's
fixed predictions are re-scored under all five.

**Illustration.** For predictions {D-alanine, acetone enol} against references {L-alanine,
acetone keto}, recall is **0.0 / 0.5 / 0.5 / 1.0** under strict-InChIKey / GLORYx / MetaTrans
/ tautomer — the same predictions, four different verdicts.

**Result — a top-of-leaderboard reorder, observed.** On the GLORYx-37 shared set, four methods
scored under one protocol (recall@15) give **a genuine #1 flip from the match rule alone**: with
SyGMa and MetaPredictor close, SyGMa is first under `canonical` (0.498 vs 0.477), strict
`inchikey` (0.492 vs 0.362), and `tanimoto1` (0.500 vs 0.478), while MetaPredictor is first under
`inchi_no_stereo` and the recommended `inchikey_tautomer` (0.504 vs 0.498) (Fig. rankflip). No
predictions change — only the definition of "match" — yet the winner does. The driver is stereochemistry:
MetaPredictor (like GRAIL) emits stereo-variant structures, so under the only stereo-*aware*
protocol (full InChIKey) it collapses 0.504 → 0.362 (1.4×) and SyGMa overtakes it by 0.13;
GRAIL swings 0.243 → 0.116 (2.1×); SyGMa preserves stereo and is protocol-robust (~0.49–0.50).
A second reorder comes from k: SyGMa leads at recall@5 (0.347 vs 0.244) but MetaPredictor
overtakes by @15. We recommend `inchikey_tautomer`: rules and learned models emit tautomers of
the reference that plain InChI does not normalize. Matching is only the first of **three**
protocol axes the literature conflates; the other two — **prediction budget** (uncapped vs
top-k) and **test set** — are quantified in §5, where holding method/set/match fixed and changing
only the budget moves a single tool's recall by ~0.18.

## 5. Baselines under one protocol

Each method contributes a ranked-prediction file, scored uniformly (canonical dedup → top-k →
match mode). On the GLORYx-37 shared set, recall@15 under the recommended `inchikey_tautomer`
protocol is **MetaPredictor 0.504, SyGMa 0.498, BioTransformer 0.373, GRAIL 0.243** (the top
pair reorders by protocol; §4). GRAIL is one fair entry — the contribution is the protocol and
the analysis, not a new SOTA. Notably MetaPredictor's standardized 0.504 *exceeds* its
"published 0.47", consistent with that 0.47 being a downstream re-run/mis-attribution rather
than its own number (below).

**The published "leaderboard" is not a leaderboard.** Before our standardized numbers can be
read against the literature, the literature numbers must be provenanced. The recurring
collation — GLORYx 0.77, SyGMa 0.68, MetaPredictor 0.47, LAGOM 0.43, MetaTrans 0.35 (LAGOM
2025, Table 2) — conflates three incomparable axes, and two entries are not the named method's
own number (`data/published_provenance.json`):

| quoted | what it actually is | k / budget | match | test set |
|---|---|---|---|---|
| GLORYx 0.77 | uncapped recall, 105/136 TP from **1,724** predictions (prec 0.061) | uncapped | InChI-no-stereo | GLORYx-37 |
| SyGMa 0.68 | uncapped recall, ~800 predictions (prec 0.12), GLORYx authors' re-eval | uncapped | InChI-no-stereo | GLORYx-37 |
| MetaPredictor 0.47 | **mis-attributed** (≈ SyGMa's top-5 in MetaPredictor's table / a re-run); its own recall is **0.739@15** | top-k | Tanimoto=1 | own 135-drug set |
| LAGOM 0.43 | top-10 recall, ~328 predictions | top-10 | canonical SMILES | GLORYx-136 |
| MetaTrans 0.35 | **LAGOM's re-run**; its own recall is **0.576@10** | top-10 | canonical SMILES | GLORYx-136 |

Within LAGOM's own Table 2, GLORYx/SyGMa are *quoted* (footnote a: uncapped, 1,724/800
predictions) while the learned models are *re-run* at top-10 over ~328 — so the headline
0.77-vs-0.43 spread is mostly the prediction budget, not the method.

**One axis, isolated.** Holding method, set, and matching fixed and changing only the budget:
SyGMa's published 0.68 is uncapped (~22 predictions/drug); capped at top-15 under our protocol
the *same tool* scores **0.498**, and SyGMa is protocol-robust across all our match modes
(~0.49–0.50). Almost the entire **0.68 → 0.50** gap is the prediction budget alone. A fixed
(k, match, set) is the minimum needed to compare any two of these methods.

## 6. What limits the task: a decomposition

A rule-based method loses a true metabolite to **coverage** (no rule reaches it) or
**ranking** (reached but below top-k). For our reference learned ranker (GRAIL, recall@15
≈ 0.334): the rule-bank single-step **ceiling is 0.718**, so coverage loss ≈ 0.282; the model
surfaces only ~47% of the *reachable* set in top-15, so **ranking loss ≈ 0.384 dominates**.
SyGMa, with a smaller rule bank, reaches 0.558 — it converts more of its reachable set,
i.e. its ranking is better. **The gap is ranking quality, not coverage.**

This gap is also **not a data ceiling**: recall@15 goes 0.10 → 0.330 (400 → 2,418 train
substrates) then plateaus at 0.334 (4,787) — 2× more data adds noise (Fig. scaling_curve).
And it is not the filter: a learned plausibility filter reaches ROC-AUC ~0.80 yet does not
beat the generator's own ranking at any scale. The unreachable tail is dominated by phase-I
transformations (phase-II ~6%), with ~50% a long mass-shift tail attributable to
regioselectivity (right reaction, wrong site) and genuinely multi-step metabolites.

**The learned scorer under-uses the empirical rule prior.** Our reference ranker blends a
learned rule score with an empirical per-rule prior (log-odds of yielding a true metabolite)
at weight `prior_strength`. Sweeping that weight at inference (no retraining) maps the
learned↔prior spectrum: pure-learned ranking gives recall@15 = 0.313, the as-trained blend
0.356, and up-weighting the prior (selected on val) **0.382 — a free +0.069 (+22%) over
pure-learned, closing the gap to SyGMa from 60% to 68%.** SyGMa's edge is precisely this
empirical reaction-frequency prior, which the learned model under-weights. Yet even the
prior-dominated limit (~0.40) stays below SyGMa (0.558): GRAIL's priors are sparse per-rule
statistics over 7,581 rules and are noisier than SyGMa's curated probabilities — *the
dominant ranking signal is the empirical prior; its estimation quality is the residual gap.*
This is a general lesson for learned rankers over large rule/template libraries.

## 7. Limitations

Annotations are incomplete (precision is a lower bound; we lead with recall + output size).
The rule-vs-learned fairness asymmetry is surfaced but not eliminated. Four methods are run
under the standardized protocol (SyGMa, MetaPredictor, BioTransformer, GRAIL); three more
(GLORYx-the-tool, MetaTrans, LAGOM) are cited from provenanced published numbers, as releasing
them is infeasible (missing weights / dependency rot). The limit decomposition mixes a
full-test ceiling with a test-subset recall; the exact same-set split is a pending run.

## 8. Conclusion

How you match decides who wins. MetaBench gives metabolite structure prediction a single,
leakage-audited, tautomer-aware yardstick, a match-sensitivity analysis that exposes the
protocol dependence, and a decomposition showing the field's gap is ranking, not coverage or
data — pointing the next methods at better generative ranking (e.g. set-level GFlowNet
sampling; follow-on work).

## References

(to be formatted) Ridder & Wagener 2008; de Bruyn Kops et al. 2021; Wishart et al. 2022;
Litsa et al. 2020; Zhu et al. 2024; Larsson et al. 2025; Scholz et al. 2023; Gao et al. 2026;
Hähnke et al. 2018.

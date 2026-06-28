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
MetaPredictor, LAGOM). Reported recall ranges from ~0.35 (early transformers) to ~0.77
(GLORYx) — but **on different datasets, splits, and match definitions**, so the numbers are
not comparable.

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
empirical reaction-likelihood score (recall ~0.68 on the GLORYx set). GLORYx [de Bruyn Kops
et al., Chem. Res. Toxicol. 2021] adds FAME-based site-of-metabolism ranking (recall ~0.77).
BioTransformer [Wishart et al., NAR 2022] is a hybrid rule+ML system. MetaTrans [Litsa et
al., Chem. Sci. 2020], MetaPredictor [Zhu et al., Brief. Bioinform. 2024], and LAGOM
[Larsson et al., AI in Life Sci. 2025] are learned (recall ~0.35–0.47). **No shared
leaderboard exists**; the recurring GLORYx external set (37 drugs/136 metabolites) is the
closest thing to a common hold-out.

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

**Result.** [PENDING full table — needs cross-method baselines.] On GRAIL and SyGMa, absolute
recall@15 swings across protocols (e.g. canonical/Tanimoto strictest, no-stereo/tautomer most
lenient); the leaderboard reorders once methods are close. We recommend `inchikey_tautomer`:
rules emit a tautomer of the reference that standard InChI does not normalize, costing ~4.5×
recall under plain InChIKey on our engine.

## 5. Baselines under one protocol

[PENDING] Each method contributes a ranked-prediction file, scored uniformly (canonical
dedup → top-k → match mode). Current entries: GRAIL (rule generator + learned filter) and
SyGMa. Planned: GLORYx, BioTransformer, MetaTrans, MetaPredictor, LAGOM. GRAIL is one fair
entry; the contribution is the protocol and the analysis, not a new SOTA.

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
[PENDING: the empirical-rule-prior-vs-learned-scorer study, which tests whether a simple
frequency prior — SyGMa's strength — out-ranks the learned generator.]

## 7. Limitations

Annotations are incomplete (precision is a lower bound; we lead with recall + output size).
The rule-vs-learned fairness asymmetry is surfaced but not eliminated. The cross-paper GLORYx
anchor and several baselines are pending acquisition. The limit decomposition mixes a
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

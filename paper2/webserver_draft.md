# GRAIL: rule-grounded prediction of xenobiotic metabolite structures with a fully declared evaluation

⟨AUTHORS⟩
⟨AFFILIATIONS⟩

**Note on naming.** `GRAIL` is used throughout as a placeholder. The name is publicly
attached, in a companion manuscript, to a configuration that loses to every comparator,
and NAR treats a same-named successor to a published tool as an *update*, which carries a
two-year embargo and a "significant changes" bar. A new name removes both problems at once.
See §Open decisions.

---

## Abstract

Predicting the metabolite structures a xenobiotic will be converted into supports toxicology,
drug design and environmental risk assessment. Existing tools report a recall figure computed
under evaluation choices — how a predicted structure is matched to a reference, how many
candidates a method is allowed to emit — that their papers do not state, so their numbers are
not comparable to each other. We present GRAIL, a rule-grounded predictor that names the
transformation rule behind every prediction, ranks its entire output, and reports its
performance under five matching criteria and eleven output budgets against every comparator
with predictions on the same substrates, with all per-substrate predictions released.

On 291 test substrates the comparison divides by budget rather than resolving into an ordering.
At the tightest budgets SyGMa leads, at $k = 1$ by 0.1534 against 0.1053 with the paired interval
excluding zero; in the middle MetaPredictor leads without separating; from $k = 15$ GRAIL leads,
and at $k = 30$ and $k = 50$ its interval excludes zero against every comparator. The advantage
is at depth: the comparators saturate on lists of 10.7 and 40.5 candidates while GRAIL continues
to have candidates to rank. Against MetaTox alone, the incumbent web service, GRAIL leads at all
nine budgets. At an output budget of 15, where MetaTox emits about what GRAIL's interactive mode
emits, GRAIL recovers 0.5672 of annotated metabolites in macro recall against 0.5406. We additionally report the ceiling of the approach:
18.3% of reference metabolites lie outside what the rule bank reaches in one step, and we
show this bound is invariant to how a transformation type is defined, making it a property of
the annotation corpus rather than of our implementation. GRAIL is freely available at
⟨SERVICE_URL⟩; code, split, frozen predictions and evaluation harness at
`github.com/doctawho42/GRAIL`.

**Keywords:** xenobiotic metabolism; metabolite prediction; reaction rules; evaluation protocol; web server

---

## Introduction

When a drug or an environmental chemical enters an organism, enzymes convert it into other
compounds. Knowing which compounds — their structures, not merely their number — matters for
toxicology, for drug design, and for environmental fate assessment, because a metabolite can
be more toxic, more persistent or more active than the parent.

Two families of computational predictors dominate. Rule-based systems — SyGMa
[Ridder & Wagener, 2008], BioTransformer [Djoumbou-Feunang et al., 2019], GLORY and GLORYx
[de Bruyn Kops et al., 2019, 2020], MetaTox — encode biotransformations as reaction templates
and enumerate products by applying them. Sequence-to-sequence systems — MetaTrans
[Litsa et al., 2020], MetaPredictor [Zhu et al., 2024] — translate the substrate's string
representation into metabolite strings. A further wave has appeared recently: a chemical
language model [Larsson et al., 2025], a mechanistically informed graph framework
[Zhou et al., 2025], a web server predicting pathway, site and product together
[Zhou et al., 2026], and a preprint predicting all three jointly [Chavan, 2026].

Comparing any two of these is harder than it appears, and the difficulty is not a matter of
effort. A recall figure depends on at least two settings that papers in this area report apart
from their rankings or do not report at all: the rule deciding when a predicted structure
counts as the reference — canonical string equality, a hash, a stereochemistry-blind hash, a
tautomer-aware hash — and the number of candidates a system is allowed to emit. A system that
emits seventy candidates and one that emits eight are not measured on the same axis, and
neither convention is stated in a form a reader can reproduce.

GRAIL is built on the opposite premise: **every evaluation choice is declared, and the system
is reported under all of them rather than under one.** The claims below are therefore made as
a sweep, not as a single number, and the artifacts sufficient to recompute any cell are
released.

The design contribution is narrower than the evaluation contribution and we state it as such.
GRAIL's architecture is a rule bank, a learned rule selector and a learned pair filter — a
familiar shape. What changed the outcome was neither a new network nor new chemistry, but the
form in which two existing scores are combined: replacing their product with a rank fusion is
worth +0.0733 in recall on held-out data, larger than any architectural change we tested.

---

## Materials and Methods

### Overview

GRAIL predicts in three stages, each separately inspectable:

1. a **generator** scores every rule in a bank of 7,581 SMIRKS templates against the substrate;
2. selected rules are **applied** with RDKit and the products standardised and deduplicated;
3. a **filter** scores each (substrate, candidate) pair, and candidates are ordered by a
   **rank fusion** of the filter and generator orderings.

Every emitted candidate carries the identity of the rule that produced it and the site at
which it fired.

### Rule bank

The deployed bank holds 7,581 templates, of which 7,580 parse under RDKit 2022.09.5. It is
the deduplicated union of four earlier curated collections (474, 656, 500 and 1,051 rules)
together with newly mined templates: 1,724 curated and 5,856 mined.

Mining reads only the molecule-disjoint training split. For each annotated substrate–product
pair a ring-aware maximum common substructure is located; atoms whose element, formal charge,
hydrogen count, aromaticity or degree changed, atom pairs whose bond order changed, and
leaving or entering atoms are flagged as the reaction centre; the centre is expanded by one
bond and emitted as a mapped SMIRKS template. A template survives only if applying it to its
own source substrate regenerates the annotated product after canonicalisation, and a
selectivity filter rejects templates producing more than 200 distinct products on average
across a 50-substrate sample.

Support across mined templates is heavily skewed: 4,274 of 5,856 (73%) rest on a single
training pair and only 457 on five or more. Re-running the miner on the same split re-derives
5,856 of 5,856 templates and none new; mining is saturated on this corpus.

### Generator

The generator estimates $P(r \mid s)$, the probability that rule $r$ applies to substrate $s$,
for the whole bank in one forward pass. Substrate graphs carry 16-dimensional node and
18-dimensional edge features; a three-layer GATv2 encoder ($16 \to 192 \to 256 \to 128$, batch
normalisation, ReLU, dropout 0.1, mean-pooled readout) embeds the substrate, and a parallel
multilayer perceptron embeds a 1024-bit Morgan fingerprint, the two combined by a gated fusion.
Each rule is embedded independently by a rule-graph encoder of the same family
($128 \to 192 \to 128$) alongside a six-dimensional branch of global rule features.

The logit sums four learned terms, each with its own learned scalar: a cross-attention term in
which the rule embedding queries the substrate's atom states through single-head attention; a
global term, the cosine similarity between substrate and rule embeddings; a local term, the
cosine between the attention context and the rule; and a term reflecting whether the rule's
SMARTS matches. To this are added a per-rule bias and a frequency-prior term at weight 0.4 —
the log-odds that the rule yields a true metabolite, estimated from training statistics — and
a penalty of 7.5 is subtracted for rules whose SMARTS does not match, which masks inapplicable
rules before the sigmoid. Where a rule fires at several sites the per-firing probabilities are
combined by noisy-OR.

Training is multi-label rule classification with a margin-ranking auxiliary at weight 0.25 and
margin 0.45. Rules that match the substrate but carry no annotation are down-weighted by half
rather than treated as negatives, which makes the objective positive-unlabelled: an annotated
metabolite is a positive, and a rule-applicable product without an annotation is unlabelled,
because a missing annotation does not establish that a transformation does not occur.
Optimisation uses Adam at $10^{-4}$ with early stopping.

### Application and standardisation

Selected rules are applied with RDKit and every product standardised, with tautomer
canonicalisation, before deduplication under the matching key. Applying the whole bank leaves 669.9
distinct candidates per substrate on the validation split after deduplication under the matching
key, and 588.7 on the comparison set.

Standardisation dominates runtime. Called on every product of every rule, it accounted for
94–99% of cold generator time; on one 29-atom substrate a cold pass took 63.44 s of which
62.63 s was normalisation. Two registered changes address this (§Runtime).

### Filter

The filter scores a (substrate, candidate) pair. Substrate and candidate are embedded
independently, each through its own three-layer GATv2 encoder of the same dimensions with a
mean-pooled readout; the two 128-dimensional embeddings are concatenated with both Morgan
fingerprints and the resulting 2304-dimensional vector mapped through a three-layer perceptron
($\to 128 \to 64 \to 1$) to a single logit. The estimator is selectable — a non-negative
positive-unlabelled risk on logits, or binary cross-entropy on probabilities; the released
configuration uses the second.

A variant encoding one merged substrate–product graph, with cross-edges placed by an
element-aware maximum-common-substructure correspondence, was implemented and measured. It
loses on both accuracy (recall@15 0.357 against 0.366, paired difference $-0.0090$,
95% CI $[-0.017, -0.001]$, $n = 1{,}170$) and cost ($34.0$ ms against $9.1$ ms per pair) and is
not deployed.

### Rank fusion

Both stages produce an opinion about each candidate: the filter's pair score and the
generator's probability for the rule that produced it. The deployed system previously ordered
candidates by their product. A product is scale-sensitive: if one scorer occupies a narrow
range and the other a wide one, the wide one determines the order and the narrow one
contributes almost nothing.

The deployed combination is instead reciprocal rank fusion,

$$\mathrm{RRF}(c) \;=\; \sum_{i} \frac{1}{K + r_i(c)}, \qquad K = 60,$$

where $r_i(c)$ is the rank of candidate $c$ under scorer $i$. Only positions enter, so the two
opinions contribute equally regardless of their scales.

This was registered as hypothesis **H7** with a threshold of $+0.05$ and evaluated on the
validation split, where it had not been selected: $+0.0733$, 95% CI $[+0.0477, +0.1020]$. On
the reporting set, where the variant had been chosen among four, it gave $+0.109$; the
shrinkage to two thirds is the measured cost of that selection.

### Pool cap

Before fusion the candidate pool is truncated to the 100 highest-scoring by the generator.
Registered as **H9** with a threshold of $+0.015$; measured $+0.0260$, 95% CI
$[+0.0033, +0.0469]$, mean pool $669.9 \to 98.0$. The cap does not cost recall, it supplies it:
it removes a small number of candidates that were a substrate's only correct answer
($-0.0077$ at $k = 1$ and $5$) and a much larger number that were occupying the middle of the
list ($+0.0748$ at $k = 50$).

The value 100 was fixed by a rule rather than by search — the smallest round cap at twice the
largest reported budget, so that the widest column is not a tautology. A cap of 50 scored
$0.0015$ higher, inside noise.

### Group rank

Candidates group naturally by molecular formula. An oracle that orders these groups by whether
they contain a reference is worth $+0.1729$ over the ranking it is measured against, which is the
product ordering with each formula group emitted as a block on the uncapped pool of 588.7
candidates. That is not the deployed ranking, and the figure is therefore an upper bound on a
configuration the server does not run; the deployed comparison is the H12 row below. Two mechanisms for exploiting this were registered and
both are reported here because their failure is informative.

**H8** trained a listwise group scorer whose ordering was applied to whole groups, emitted as
blocks. Registered threshold $+0.05$; measured on the 291 comparison substrates, $-0.1203$,
95% CI $[-0.1590, -0.0828]$. The
deficit decomposes into $-0.0647$ attributable to the design — block emission is itself worse
than interleaved emission, $0.4376$ against $0.5023$ — and $-0.0556$ to the model. The oracle
had been measured in blocked form, so more than a third of the headroom it promised was the
return of what the design spends.

**H12** used the same model with nothing retrained, entering its group score as a third
ranking in the same fusion:

$$\mathrm{score}(c) \;=\; \frac{1}{60 + r_{\text{filter}}(c)} + \frac{1}{60 + r_{\text{gen}}(c)} + \frac{1}{60 + r_{\text{group}}(c)}.$$

Registered threshold $+0.02$; measured on the 291, $+0.0286$, 95% CI $[+0.0085, +0.0502]$. The
effect is not selection: all twenty configurations beat the base on the 293-substrate validation
split, within a band of $0.0114$, and 70% of that validation margin survived the move to the
291.

**H14** tested the converse composition — the group signal as a gate applied before fusion,
with the H9 budget and no new parameter. Registered threshold $+0.02$; measured on the 291, $-0.0286$,
separated from zero against the hypothesis. The mechanism is closed by an oracle rather than by
model quality: **perfect group ranking through the gate gives $-0.0361$**, worse than the
generator cap it replaces. The median substrate has one group containing a reference, while
filling a hundred candidates admits at least seven groups; after the single informative group
the gate spends its remaining budget on groups its signal cannot separate, whereas the
generator cap orders every candidate. The group signal is close to binary and the generator
score is graded, so any mechanism that lets the former decide discards the latter's resolution.

The two results are symmetric: $+0.0286$ as a nudge on top of the generator's ordering,
$-0.0286$ as a replacement for it.

Entering a *binary* group ordering — one that knows only which groups hold a reference — as the
third term reaches the same $0.5308$ the trained scorer reaches. That is not an upper bound and
we do not use it as one: a binary ordering leaves every reference-free group tied, and on
validation the trained scorer exceeds it, $+0.0412$ against $+0.0260$. What the pair does show is
that the $+0.1729$ measured against the blocked product ordering is not available to either
registered composition, and the reason differs between them: H8's design spends more than it
draws, and H14's is bounded below the cap it replaces even under an oracle. ⟨ORACLE_BY_TYPE: whether the same headroom exists under a transformation-type
grouping rather than a formula grouping — running at time of writing⟩

### Emission

Under the product combination, the deployed system emitted candidates scoring within
$\alpha = 0.5$ of the leader. This rule does not transfer to a rank fusion and the reason is
structural: the worst candidate scores $0.88$ of the best under fusion against $0.016$ under
the product, because a reciprocal-rank sum is a rank statistic without scale. At
$\alpha = 0.5$ the rule admits $16.22$ candidates from a pool of $16.3$ — it truncates nothing.

The deployed emission is therefore the whole pool of the operating budget, with no threshold
and no parameter; the size of an answer is set by the rule budget and by the substrate's
chemistry. Two figures for MetaTox's output size appear in this paper and mean different things:
$30.9$ where its list is truncated at the widest budget reported, and $36.34$ where it is read
untruncated in the emission grid. The comparison table uses the first, because a budget of fifty
is the widest column it has. Registered as **H11** with a criterion of zero lost cells over a grid of five
matching criteria by eleven budgets: 0 of 55.

### Two operating modes

| | interactive | exhaustive |
|---|---|---|
| rules applied | 30 | 7,581 |
| mean candidates emitted | 15.6 | 98.1 |
| median time per substrate | 0.39 s | 5.28 s |
| mean time per substrate | 2.36 s | — |
| 90th percentile | 2.59 s | — |
| slowest substrate | 231.0 s | did not finish |

Both medians are over the 294 validation substrates and measure the same boundary, everything
before the filter. The interactive mode's mean is six times its median because of one substrate:
a 291-heavy-atom PEGylated peptide taking 231 s, which the exhaustive mode has never finished at
any budget. A service must publish the tail as well as the median, and the tail here is a
molecule a user can plausibly submit.

The rule budget of 30 is recorded in the deployed checkpoint; every pool in the project had
been built with the whole bank, and what that setting costs had never been measured.
Registered as **H10** with a ceiling of $+0.05$ for what the whole bank buys: on validation
$+0.0092$, 95% CI $[-0.0231, +0.0399]$, mean pool $669.9 \to 17.0$, all 294 validation
substrates scored in 1,510 s.

On the reporting set the same quantity is $+0.0556$, 95% CI $[+0.0235, +0.0876]$, just past the
ceiling. The discrepancy is reported rather than resolved, and it has a reading: at $k \le 10$
the whole bank buys nothing on either population, and from $k = 15$ it buys more the wider the
budget, because the cheap arm runs out of candidates. **The threshold turned out to be a
property of the budget requested, not of the system.**

### Web server

⟨SERVER: implementation stack, input formats accepted, output formats, batch limits,
sample-data mechanism, help pages and tutorial, licence — NAR requires free non-commercial
use under a standard licence such as CC BY-SA and forbids mandatory login. These are hard
eligibility criteria and must be confirmed against the hosting platform.⟩

---

## Evaluation protocol

### Which population carries which number

Three populations appear in this paper and no figure is stated without one. The **test split** is
1,170 substrates carrying 2,597 references and is where the coverage ceiling and the type census
are measured. The **validation split** is 293 substrates carrying 655 references and is where
every registered ranking and timing threshold is checked, because a threshold checked where its
rule was chosen is not a check. The **comparison set** is the 291 substrates on which every
method predicts, and is where the comparison and the group-ranking results are reported. The
tables below name the population in every row.

### Data and split

Splits are substrate-disjoint. A read-only audit canonicalises every SMILES and verifies zero
substrate overlap and zero positive-pair overlap across each split pair; all three pairs return
zero. Molecular overlap is non-zero and expected — 2,408 molecules occur in both train and
test, and 243 test substrates appear as molecules in train — and is recorded so that it is not
later discovered as leakage.

The evaluated test set is 1,170 substrates carrying 2,597 reference metabolites. Of the 1,170,
845 are parent drugs and 325 are themselves annotated products of another substrate in the
split; methods do not treat the two alike, and the mixture is reported rather than averaged
silently.

Comparison with MetaTox is on the 291 test substrates for which both systems supply
predictions. **Populations are never mixed**: every figure below names the set it is computed
on.

### Matching criteria

What counts as a hit is a declared axis with five settings: canonical SMILES equality; the
full InChIKey; its stereochemistry-blind first block; Tanimoto similarity of one over Morgan
fingerprints; and a tautomer-aware hash, which canonicalises tautomers on both sides and
generates a canonical SMILES without stereochemistry. The tautomer-aware key is the default,
because the references neither carry stereochemistry nor distinguish tautomers, and standard
InChI normalises only a subset of them. All five are reported for every comparison.

### Budgets and aggregation

Recall at cut-off $k$ for substrate $s$ is

$$\text{recall@}k \;=\; \frac{\left| Y_s \cap \hat{Y}_s^{(k)} \right|}{|Y_s|},$$

with $Y_s$ the references and $\hat{Y}_s^{(k)}$ the first $k$ predictions. Two aggregations are
distinguished throughout and never mixed:

$$\text{macro} = \frac{1}{|S|}\sum_{s} \frac{|Y_s \cap \hat{Y}_s|}{|Y_s|}, \qquad
\text{micro} = \frac{\sum_{s} |Y_s \cap \hat{Y}_s|}{\sum_{s} |Y_s|}.$$

Intervals are paired bootstrap over substrates, 10,000 resamples, seed 0.

Precision is reported but not used to order systems. Under incomplete annotation an
unannotated but real metabolite counts as a false positive, so every precision figure is
pessimistic by a bounded amount, and a system emitting fewer candidates is rewarded for
reasons unrelated to its chemistry.

### The form of the claim

No single budget is privileged. A claim of the form "our system leads" is made for a cell only
where that cell's interval excludes zero, and the whole sweep is reported with each budget
carrying its own interval. Where an ordering is not established, the table says so. Working to
move one cell would be working for a number, and is not done.

### Comparators and their availability

SyGMa 1.1.0 is taken from the installed module. BioTransformer is pinned by digest rather than
by version string, because "3.0" names several builds: jar `c70cad91…`, 115,415,197 bytes,
upstream commit `6432cf88` of 2023-05-25, its 994 templates loaded at freeze time rather than
taken from a constant.

Four systems published in 2025–26 could not be run as released. We report this as a result,
because it bears directly on whether any current comparison in this area is possible:

| system | released | missing | what would be required |
|---|---|---|---|
| LAGOM | training code, Apache-2.0 | any fine-tuned checkpoint | training it ourselves, i.e. comparing against our own reimplementation |
| DeepMetab | weights (81 LFS objects, 307.8 MB, all verified) and site prediction | `SOM/Reaction.py`, withdrawn by the author for copyright reasons; only 3.7 bytecode remains | the module, or a reimplementation of that stage |
| DeepCYP | web form | models or an API | submitting 1,170 substrates to a third-party server |
| Metabolite-Gen | preprint | no repository found | code |

The list is closed by a rule: a comparator that could not be obtained is reported as
unavailable with its reason, is not dropped silently, and the list is not extended after the
first run.

---

## Results

### Comparison with published predictors

On the 291 shared test substrates carrying 665 annotated metabolites, micro recall by budget. A prediction equal to the substrate is dropped before the budget for every method alike, the convention `results/four_method_291.json` uses.

| $k$ | GRAIL exhaustive | GRAIL interactive | MetaTox | SyGMa | MetaPredictor |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0902 | 0.1053 | 0.0406 | **0.1534** | 0.1398 |
| 3 | 0.2271 | 0.2406 | 0.1323 | **0.2887** | 0.2767 |
| 5 | 0.3098 | 0.3203 | 0.2180 | **0.3744** | 0.3549 |
| 8 | 0.4150 | 0.4015 | 0.3113 | 0.4271 | **0.4346** |
| 10 | 0.4526 | 0.4331 | 0.3880 | 0.4556 | **0.4707** |
| 15 | **0.5353** | 0.4797 | 0.5143 | 0.4842 | 0.4797 |
| 20 | **0.5789** | 0.4872 | 0.5519 | 0.4992 | 0.4797 |
| 30 | **0.6391** | 0.4902 | 0.6015 | 0.5068 | 0.4797 |
| 50 | **0.7038** | 0.4902 | 0.6271 | 0.5113 | 0.4797 |

Mean list length: GRAIL exhaustive 98.1, GRAIL interactive 15.6, MetaTox 30.9, SyGMa 40.5, MetaPredictor 10.7.

Read against the strongest comparator at each budget, the picture divides in three.

**GRAIL trails, interval excluding zero:** $k=1$ (SyGMa 0.1534 against GRAIL interactive 0.1053), $k=3$ (SyGMa 0.2887 against GRAIL interactive 0.2406), $k=5$ (SyGMa 0.3744 against GRAIL interactive 0.3203).

**Neither separates:** $k=8$ (MetaPredictor 0.4346 against GRAIL exhaustive 0.4150), $k=10$ (MetaPredictor 0.4707 against GRAIL exhaustive 0.4526), $k=15$ (MetaTox 0.5143 against GRAIL exhaustive 0.5353), $k=20$ (MetaTox 0.5519 against GRAIL exhaustive 0.5789).

**GRAIL leads, interval excluding zero:** $k=30$ (MetaTox 0.6015 against GRAIL exhaustive 0.6391), $k=50$ (MetaTox 0.6271 against GRAIL exhaustive 0.7038).

The advantage is at depth and not at the head of the list. SyGMa leads at the tightest budgets and MetaPredictor in the middle; both saturate, MetaPredictor at 0.4797 from $k=15$ on a mean list of 10.7, SyGMa at 0.5113 on 40.5. GRAIL's exhaustive mode keeps rising because it keeps having candidates.

Against MetaTox alone, the incumbent web service and the system this work set out to replace, the exhaustive mode leads at every budget: $k=1$ +0.0496*, $k=3$ +0.0947*, $k=5$ +0.0917*, $k=8$ +0.1038*, $k=10$ +0.0647*, $k=15$ +0.0211, $k=20$ +0.0271, $k=30$ +0.0376*, $k=50$ +0.0767*, where an asterisk marks an interval excluding zero.

Where a method runs out of candidates the budget stops measuring ranking, and the counts are reported for every arm: at $k=15$, 181 of 291 interactive lists are shorter than the budget, 273 of MetaPredictor's and 34 of MetaTox's; at $k=50$ the counts are 282, 291 and 249.

### Where the design came from

Every deployed choice is a registered hypothesis with a threshold and a falsification
condition, tested on a population where it was not selected.

| | fixed | threshold | measured | tested on |
|---|---|---:|---|---|
| H7 | rank fusion instead of product | $+0.05$ | $+0.0733$ $[+0.0477, +0.1020]$ | validation |
| H9 | pool cap 100 by generator | $+0.015$ | $+0.0260$ $[+0.0033, +0.0469]$ | validation |
| H10 | rule budget 30 from checkpoint | $\le +0.05$ | $+0.0092$ $[-0.0231, +0.0399]$ | validation |
| H11 | emission = whole pool | 0 cells lost | 0 of 55 | grid $5 \times 11$ |
| H12 | group score as third rank | $+0.02$ | $+0.0286$ $[+0.0085, +0.0502]$ | 291 |
| H8 | group score, blocked | $+0.05$ | $-0.1203$ $[-0.1590, -0.0828]$ | 291 |
| H14 | group score as a gate | $+0.02$ | $-0.0286$, separated against | 291 |
| H13 | standardise survivors only | $\ge 10\times$ | $2.95\times$ | validation, 293 paired |
| H15 | survivors + tautomer budget 200 | median $< 10$ s | 5.28 s | validation, 293 paired |

Of fifteen registered hypotheses, six are confirmed and three failed. `paper2/preregistration.md`

### Runtime

Standardisation was the binding constraint and its diagnosis was wrong twice before it was
right. Three hypotheses were tested and rejected by measurement: the pairwise filter's
maximum-common-substructure step (the generator outruns the filter a hundred to one);
rule application (all 7,580 templates on a 109-atom molecule in 5.2 s, 20,655 products, at most
0.58 s per rule); and substrate size as an axis (two 43-atom substrates, 64.7 s and no
completion). The cause was product standardisation, at 94–99% of cold generator time. A cache
had concealed it: a repeated substrate runs 16–47 times faster, so all early timings on the
branch were partly warm.

Moving standardisation to surviving candidates (**H13**) freed recall ($-0.0015$ against a cap
of $0.01$) but failed on time: $2.95\times$ against a registered $10\times$. The mechanism was
right and the arithmetic was not — enumeration collapsed by $13.9\times$, and standardising the
hundred survivors became 72% of the new arm. Removing 94–99% of the work does not remove
94–99% of the time; the remainder becomes the new majority.

Adding a tautomer enumeration budget of 200 (**H15**) passes both halves: median time
$49.85 \to 5.28$ s against a registered target below 10 s, recall loss $0.0015$ against a cap
of $0.01$. The target was set in seconds rather than as a multiple, precisely because H13 had
shown a multiple to measure change where the question is fitness for use. Part of the apparent speed-up is machine load. The enumeration stage is identical in the two
survivors arms, so the ratio of its medians, $1.73$, measures load and nothing else. Dividing
that out of the standardisation step's ratio leaves $2.52\times$ attributable to the tautomer
budget itself, against $3.93$ predicted by the invariance curve; and applying it to the whole arm
puts the median substrate at $9.12$ s under the load the comparison arm ran at, still inside the
registered target of ten seconds. The headline $5.28$ s is measured on a machine running only
this job.

The budget lives in a module-level singleton shared with the tautomer matching key, so lowering
it would silently have moved what counts as a hit. A private enumerator was introduced; keys on
96 molecules are identical and the global setting is untouched.

### The ceiling, and why it is a property of the corpus

Applying the whole bank to every test substrate recovers 0.817 of references in one step
(micro; 0.860 macro). The remaining 18.3% — 475 of 2,597 — divide by the reaction-centre change
they require: 337 need a transformation type the bank does not hold at all, 98 need a type it
holds whose rule did not fire on that substrate, and 40 cannot be typed.

Relaxing the convention-dependent primitives in the templates — hydrogen counts and
degree/connectivity constraints, the constructs whose interpretation varies between rule
libraries — recovers **3 of the 98**, moving the ceiling $0.8171 \to 0.8183$, at 240 additional
candidates per recovered reference. The a-priori bound had been computed in advance: 385 of
4,417 types (8.7%) hold a carrier the relaxation reaches, predicting $\approx 8.5$. Carrying a
relaxable primitive is necessary and sufficient only about a third of the time.

The 337 misses of absent type collapse into **312 distinct types**, of which 294 occur once and
carry 87.2% of the mass; half the mass requires 144 types. The obvious objection — that this is
an artifact of how finely a type is defined — is answered by varying the definition:

| definition of type | types | singletons | mass in singletons | names a transformation |
|---|---:|---:|---:|---|
| exact multiset with counts | 312 | 294 | 87.2% | yes |
| set of bond classes | 284 | 247 | 73.3% | yes |
| participating element pairs | 69 | 30 | 8.9% | **no** |
| number of changed bonds | 44 | 9 | 2.7% | **no** |

The tail holds at every granularity at which a type still determines a product, and collapses
only where it stops doing so. What this table supports is not that 87.2% of the gap is
singletons under our definition; it is that **the bound is invariant to the definition of a
type across the whole range in which the definition is meaningful**, which makes it a fact
about the annotation corpus rather than about our choice.

Nor is the missing chemistry available in the largest reaction library we hold. Typing the
42,554 templates of the standard retrosynthesis library in the metabolic direction gives 9,885
types, of which **14** are among the 312 missing, carrying 18 of the 337 misses — 5.3% of the
gap. The control that makes this readable: bank and library share 308 types, 7.0% of the bank,
so the intersection measures real absence rather than two incomparable formalisms. Synthetic
organic chemistry and metabolism do overlap; the missing chemistry lies outside both.

---

## Discussion

### Where the advantage is, and where it is not

The comparison does not produce a winner and we do not present one. SyGMa recovers more at the
head of the list, and at $k \le 5$ the interval separates against us. What GRAIL does is keep
improving where the others stop: MetaPredictor is flat from $k = 15$ because 273 of its 291 lists
are shorter than that budget, and SyGMa gains 0.0271 between $k = 15$ and $k = 50$ against
GRAIL's 0.1685. A user reading three candidates is better served by SyGMa; a user reading fifty
is better served here, and the paper says both.

That division has a consequence for how such tools are compared. A single recall figure at a
single budget would have named a different winner depending on which budget its author chose,
and every one of those choices would have been defensible. This is the argument for reporting
the sweep rather than a cell, made on our own numbers rather than about someone else's.

### What the system offers that its predecessor does not

Three things, none of which is a recall figure.

Every prediction names the transformation rule that produced it and the site at which it fired,
so a user can accept or reject a candidate on chemical grounds rather than on a score.

The whole output is ranked. MetaTox ranks 5,177 of its 10,601 predictions by a
metabolite-likeness score and leaves the remainder in file order; more than half of what it
returns carries no ordering at all.

Every evaluation choice is declared and every cell of the resulting grid is reported. To our
knowledge no other metabolite predictor publishes its matching criterion, its output budget,
its coverage ceiling and its per-substrate predictions together, which is what makes the
comparison above reproducible by a third party.

### What the numbers do not support

GRAIL does not lead at the tight budgets: at $k = 1$, $3$ and $5$ SyGMa leads with the paired
interval excluding zero, and at $k = 8$ and $10$ MetaPredictor leads without separating. At
$k = 15$ and $20$ no arm separates from the strongest comparator. The lead that is established
is at $k = 30$ and $50$.

It is not compared against the 2025–26 wave, because that wave cannot be run as released. Its
precision figures are bounded by annotation completeness and are not used to order systems. And
the advantage at wide budgets is partly a statement about output length on all sides rather than
about ranking quality: where a method's list is shorter than the budget the budget has stopped
measuring ranking, and the counts are given for every arm.

### The ceiling bounds this formulation and not the task

The 18.3% shortfall is a property of the corpus the templates are mined from, established
above by invariance rather than asserted. Mining is saturated: a re-run derives no new template.
Extending coverage therefore requires new reaction corpora, and the census narrows the search —
synthetic corpora are excluded by measurement, leaving enzymatic and regulatory sources.
Metabolite identifications from radiolabelled human ADME studies, reported in regulatory
submissions, are human *in vivo* data on exactly the population predicted here and are, to our
knowledge, not systematically mined.

We note that this bound applies to any predictor trained on a comparable corpus, including
sequence-to-sequence systems, which do not report it.

### Availability of comparators as a finding

Of four systems published in this area in 2025–26, none could be run on our split as released.
This is not a complaint about individual authors; it is the reason a reader cannot presently
verify any ranking in this literature, and it is the concrete case for the recommendation that
leaderboards publish per-item predictions rather than aggregate scores.

---

## Limitations

Predictions are single-step; applying the bank at depth 2 lifts the ceiling by ≈0.012 at 8.5
times the candidate cost and is not deployed. Precision is bounded by annotation completeness.
The rule-budget threshold in H10 proved to be a property of the requested budget rather than of
the system, and is reported as such. The group-ranking headroom of $+0.1729$ is real and
unreached by both registered compositions for structural reasons. The interactive mode runs out
of candidates above $k = 10$, and so do the comparators: at $k = 50$, 291 of MetaPredictor's 291
lists and 249 of MetaTox's are shorter than the budget, so the widest column measures list length
on every side and not only on ours. The comparison population is the 291 substrates on which all
methods predict, which is a quarter of the evaluated test split and is not a random quarter of it.
⟨ANY_FURTHER⟩

---

## Availability

⟨SERVICE_URL⟩ — free, ⟨LICENCE⟩, no registration required ⟨CONFIRM⟩.
Source, split manifest, frozen predictions of every comparator, evaluation harness, provenance
sweep and pre-registration: `github.com/doctawho42/GRAIL`.

A split manifest pins by digest the three triple files, the substrate sets, the evaluated test
set, the stratum files, the rule bank and all third-party artifacts; `--verify` recomputes and
names any leaf that has moved. Every committed numeric artifact records the commit that wrote
it, and readers verify that record; the sweep is green on 38 pinned artifacts.

---

## Author contributions / Funding / Conflicts

⟨TBD⟩

---

# Placeholders to fill

Everything computable from the released artifacts has been filled and the entries below
are the ones no run can supply. Each was checked against the artifacts before being
removed from this list; the closed ones are recorded in the paragraph beneath the table.

| marker | what is needed |
|---|---|
| `⟨AFFILIATIONS⟩` | affiliations |
| `⟨ANY_FURTHER⟩` | limitations the authors know of and this draft does not |
| `⟨AUTHORS⟩` | author list |
| `⟨CONFIRM⟩` | confirmation that the platform requires no login; NAR forbids mandatory registration |
| `⟨LICENCE⟩` | the licence; NAR requires free non-commercial use under a standard licence |
| `⟨ORACLE_BY_TYPE⟩` | the oracle recomputed under a transformation-type grouping, running |
| `⟨SERVER⟩` | implementation stack, input and output formats, batch limits, sample data, help pages and tutorial, all of them NAR eligibility criteria |
| `⟨SERVICE_URL⟩` | the deployed URL |
| `⟨TBD⟩` | author contributions, funding, conflicts |

Closed since the first draft:

- `⟨VERSION⟩` — RDKit 2022.09.5, the version the bank parses under
- `⟨N_SHARED⟩` — 291, from results/deployment_table.json
- `⟨N_DISTINCT⟩` — the sentence was wrong in the other direction: 669.9 is the count AFTER deduplication on validation, 588.7 on the comparison set, and no raw figure was needed
- `⟨T_INTERACTIVE⟩` — 0.39 s median over 294 validation substrates, from results/mode_timings.json. The 1.85 s this was to be filled with is a mean over a forty-substrate draw from a different population
- `⟨POP_H13⟩`, `⟨POP_H15⟩` — validation, 293 paired substrates, from the two verdict artifacts
- `⟨REGISTRY_URL⟩` — paper2/preregistration.md

**Numbers deliberately not used**, because they belong to the superseded architecture
(product fusion, $\alpha$ emission) and would be stale here: the F1 table with GRAIL at 0.192,
recall 0.219, precision 0.222, output 2.09, and its comparators. If a precision or F1 table is
wanted, it must be recomputed under the deployed fusion.

**Reference list** is by name and year only above; ⟨BIBTEX⟩ to be pulled from the companion
manuscript, whose bibliography already holds all of them.

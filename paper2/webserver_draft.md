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
performance under five matching criteria and ten output budgets against every comparator
with predictions on the same substrates, with all per-substrate predictions released.

On 291 test substrates the comparison divides by budget rather than resolving into an ordering.
At the tightest budgets SyGMa leads, at $k = 1$ by 0.1534 against 0.1053 with the paired interval
excluding zero; in the middle MetaPredictor leads without separating; from $k = 15$ GRAIL leads,
and at $k = 30$ and $k = 50$ its interval excludes zero against every comparator. The advantage
is at depth: the comparators saturate on lists of 10.7 and 40.5 candidates while GRAIL continues
to have candidates to rank. Against MetaTox alone, the incumbent web service, GRAIL is ahead at all nine
budgets and the paired interval excludes zero at seven of them, by margins from $+0.0376$ to
$+0.1038$; at $k = 15$ and $k = 20$ the interval includes zero and those two budgets are
reported as unresolved. We additionally report the ceiling of the approach:
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
[de Bruyn Kops et al., 2019, 2020], MetaTox [Rudik et al., 2017] — encode biotransformations as reaction templates
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

### A worked example

Gemcitabine is annotated in the corpus with four metabolites: the deamination to dFdU that
inactivates it, and the three sequential 5′ phosphorylations that activate it. Both operating
modes were run on it and every field the pipeline computes was kept.

| annotated metabolite | interactive: rank / rule | exhaustive: rank / rule / site |
|---|---|---|
| dFdU, deamination | 4 / 4913 | 13 / 4913 / atoms 0,1,2,4,15,16,17 |
| dFdCMP, 5′-monophosphate | not produced | 8 / 2198 / atoms 8,9 |
| dFdCTP, 5′-triphosphate | not produced | 16 / 1745 / atoms 8,9 |
| dFdCDP, 5′-diphosphate | not produced | 21 / 2139 / atoms 8,9 |

Figure 4 shows where the rules fire and where the two arms put the four answers.

The interactive mode applies its 30-rule budget in 3.34 s and returns 18 candidates, ranking the
deamination fourth. The three phosphorylations are not among the rules that budget selects, so no
ranking could have recovered them. The exhaustive mode applies the whole bank in 14.18 s, returns
100 candidates and produces all four, at micro recall 0.25 at $k = 10$, 0.50 at $k = 15$ and
1.00 at $k = 30$. Both timings are one substrate on one run, not the medians of the mode table.
The division the sweep shows across 291 substrates is visible on this single molecule: the cheap
arm is right about what it emits and runs out, the wide arm keeps having candidates.

The attribution is the part a score cannot supply. All three phosphorylations fire at the same
substrate atoms, the 5′ hydroxymethyl, under three different rules, and the deamination fires on
the pyrimidine ring. A user reading the output sees which transformation is claimed and where,
and can reject a candidate on chemical grounds rather than on a number. All four rules here are
mined rather than curated, so the half of the bank with no expert provenance is the half carrying
this molecule.

The pool artifacts do not carry these two fields. `build_val_pools` calls the generator with
`compute_sites=False` and keeps two elements of a four-tuple, so the rule identity and the firing
atoms are computed and discarded on the way to the file. `scripts/typed_edit/case_study.py`
keeps all four, and `results/case_study.json` holds the whole ranked pool of both arms.

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
generator's probability for the rule that produced it. The obvious way to combine two such
scores is their product, and it is scale-sensitive. If one scorer occupies a narrow range and
the other a wide one, the wide one determines the order and the narrow one contributes almost
nothing.

The deployed combination is instead reciprocal rank fusion
[Cormack, Clarke & Buettcher, 2009],

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
they contain a reference is worth $+0.3143$ of micro recall@15 over the ranking it is measured
against, and $+0.3814$ in macro. That ranking
is the product ordering with each formula group emitted as a block, on the uncapped pool of 588.7
candidates, and it is not what the server runs. The figure is therefore an upper bound on a
configuration nobody deploys, and the comparison that bears on the deployed system is the H12 row
below. Two mechanisms for spending this headroom were registered before either was built, and
both are reported here because they fail in different and informative ways.

The registration of H8 was written against an earlier value of this oracle, $+0.1729$, which
`paper2/preregistration.md` records and which a later recomputation of
`wide_pool_analysis_implicit.json` superseded. The registration is left as it was written, since
a preregistration that is edited to match its outcome registers nothing; the figure above is the
current artifact's.

**H8** trained a listwise group scorer whose ordering was applied to whole groups, emitted as
blocks. Registered threshold $+0.05$; measured on the 291 comparison substrates, $-0.1203$,
95% CI $[-0.1590, -0.0828]$. The
deficit decomposes into $-0.0647$ attributable to the design and $-0.0556$ to the model: block
emission is by itself worse than interleaved emission, $0.4376$ against $0.5023$. Because the
oracle is measured in blocked form, more than a third of the headroom it promises is the return
of what the design spends.

**H12** used the same model with nothing retrained, entering its group score as a third
ranking in the same fusion:

$$\mathrm{score}(c) \;=\; \frac{1}{60 + r_{\text{filter}}(c)} + \frac{1}{60 + r_{\text{gen}}(c)} + \frac{1}{60 + r_{\text{group}}(c)}.$$

Registered threshold $+0.02$; measured on the 291, $+0.0286$, 95% CI $[+0.0085, +0.0502]$. The
effect is not selection: all twenty configurations beat the base on the 293-substrate validation
split, within a band of $0.0114$, and 70% of that validation margin survived the move to the
291.

**H14** tested the converse composition: the group signal as a gate applied before fusion, with
the H9 budget and no new parameter. Registered threshold $+0.02$; measured on the 291, $-0.0286$,
separated from zero against the hypothesis.

What closes this mechanism is an oracle rather than model quality. **Perfect group ranking through
the gate gives $-0.0361$**, worse than the generator cap it replaces. The median substrate has one
group containing a reference, while filling a hundred candidates admits at least seven groups. After
the single informative group, the gate spends its remaining budget on groups its signal cannot
separate; the generator cap, by contrast, orders every candidate. The group signal is close to
binary and the generator score is graded, so any mechanism that lets the former decide discards
the latter's resolution.

The two results are symmetric: $+0.0286$ as a nudge on top of the generator's ordering,
$-0.0286$ as a replacement for it.

Entering a *binary* group ordering — one that knows only which groups hold a reference — as the
third term reaches the same $0.5308$ the trained scorer reaches. That is not an upper bound and
we do not use it as one: a binary ordering leaves every reference-free group tied, and on
validation the trained scorer exceeds it, $+0.0412$ against $+0.0260$. What the pair does show is
that the headroom measured against the blocked product ordering is not available to either
registered composition, and the reason differs between them: H8's design spends more than it
draws, and H14's is bounded below the cap it replaces even under an oracle.

**The grouping itself was the untested assumption.** A molecular formula may be standing in for
the transformation rather than carrying the signal, so the same oracle was recomputed on the
deployed pool — the H9 cap, ordered by the deployed fusion — under four partitions of one set of
candidates. A finer partition wins at an oracle for nothing: in the limit of one candidate per
group the oracle becomes a perfect ranker. The fourth partition is therefore a control, a random
partition whose group-size multiset is exactly the type partition's on each substrate, so it
matches the granularity and carries no chemistry.

| partition | recall@15 | over fusion | groups per substrate |
|---|---:|---|---:|
| none (fusion, the deployed ranking) | 0.5353 | — | — |
| molecular formula | 0.6602 | $+0.1248$ $[+0.0853, +0.1654]$ | 41.5 |
| random, matched to type's group sizes | 0.6632 | $+0.1278$ $[+0.0809, +0.1757]$ | 51.4 |
| transformation type | 0.7128 | $+0.1774$ $[+0.1404, +0.2175]$ | 51.4 |
| both | 0.7158 | $+0.1805$ $[+0.1454, +0.2180]$ | 53.7 |

Read against the control rather than against fusion, the four rows separate differently.
Formula does not beat a random partition of its granularity: $-0.0030$,
95% CI $[-0.0396, +0.0332]$. **Everything the formula oracle promises is delivered by any
partition of that shape**, and the chemistry in it is not measurable here. Transformation type
does beat the control, $+0.0496$, 95% CI $[+0.0248, +0.0768]$, and beats formula directly,
$+0.0526$, 95% CI $[+0.0269, +0.0824]$. Their conjunction adds nothing to type alone,
$+0.0030$, 95% CI $[-0.0044, +0.0131]$: type absorbs formula rather than complementing it.

This relocates the headroom without enlarging it. It remains an oracle, and H8 is the standing
demonstration that a blocked design returns part of what its oracle promises; what changes is
which label a mechanism would have to learn. That label is the transformation type, which is the
vocabulary the rule bank's coverage is already defined on and the label space registered as H1,
so the question is one already on the register rather than a fourth mechanism.

The typing step runs a maximum-common-substructure search under a wall-clock timeout and treats
a cancelled search as untypeable, so this artifact is not byte-reproducible. Two runs on the same
machine differed in 2 of approximately 29,100 typings; every figure in the table above was
identical to four decimals, and the only cell that moved was the control at $k = 1$, by $0.0015$.

### Emission

The natural emission rule under a product combination is a score ratio: emit every candidate
scoring within $\alpha = 0.5$ of the leader. It does not transfer to a rank fusion, for a
structural reason. A reciprocal-rank sum is a rank statistic without scale, so the worst
candidate scores $0.88$ of the best under fusion against $0.016$ under the product; at
$\alpha = 0.5$ the rule admits $16.22$ candidates from a pool of $16.3$, truncating nothing.

The deployed emission is therefore the whole pool of the operating budget, with no threshold
and no parameter; the size of an answer is set by the rule budget and by the substrate's
chemistry. Two figures for MetaTox's output size appear in this paper and mean different things:
$30.9$ where its list is truncated at the widest budget reported, and $36.34$ where it is read
untruncated in the emission grid. The comparison table uses the first, because a budget of fifty
is the widest column it has. Registered as **H11** with a criterion of zero lost cells over a grid of five matching
criteria by ten budgets, 50 cells, together with five further cells that read MetaTox at its own
untruncated emission rather than at a shared budget: 0 of 55 lost.

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

Three evaluation populations appear in this paper, each a stated subset of a split rather
than the split itself, and no figure is given without naming one.

The **evaluated test set** is the 1,170 substrates of the 1,198-substrate clean test split that
carry at least one reference, and it holds all 2,597 references of that split; the 28 excluded
substrates carry none. It is where the coverage ceiling and the type census are measured.

The **validation draw** is a declared sample of 300 substrates at seed 0 from the 1,020-substrate
clean validation split. Of these, 294 produced a candidate pool and carry 657 references, and the
293 held by both arms of a paired comparison carry 655. It is where every registered ranking and
timing threshold is checked, because a threshold checked on the substrates its rule was chosen on
is not a check. Sampling is without replacement, so a different cap yields a different set of the
same size rather than a subset, and the cap and seed are recorded with every figure drawn from it.

The **comparison set** is the 291 substrates on which every method predicts, carrying 665
references. It is where the comparison and the group-ranking results are reported.

The tables below name the population in every row.

### Data and split

⟨CORPUS: the provenance of the annotated substrate–metabolite corpus — its source, its release
or access date, and the licence under which it is used. Nothing in the draft or the repository
records this, and it is the first question a reader will ask of every number below.⟩

The corpus is divided into three substrate-disjoint splits: 9,011 training substrates carrying
17,454 annotated pairs, 1,020 validation substrates carrying 2,085, and 1,198 test substrates
carrying 2,597. A read-only audit canonicalises every SMILES and verifies zero substrate overlap
and zero positive-pair overlap across each split pair; all three pairs return zero. Molecular
overlap is non-zero and expected, because a molecule that is a substrate in one split may appear
as a product in another: 2,408 molecules occur in both train and test, and 243 test substrates
appear as molecules in train. Both counts are recorded so that neither is later discovered and
mistaken for leakage.

Of the 1,170 evaluated test substrates, 845 are parent drugs and 325 are themselves annotated
products of another substrate in the split. Methods do not treat the two alike, so the mixture is
reported rather than averaged silently.

The comparison set carries no training contamination: 0 of its 291 substrates appear in train or
validation under a canonical key. The same audit is the reason no external evaluation set is
reported here — of the 37 GLORYx parent substrates, 24 (64.9%) are inside our train or validation
splits, so a figure on that set would measure memorisation as much as prediction.

**Populations are never mixed**: every figure below names the set it is computed on.

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
Figure 1 plots the same sweep, shaded by where the paired interval separates.

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

Of the fifteen registered hypotheses, nine are checked here: six confirmed and three failed.
The remaining six (H1–H6) are registered against work this paper does not report — a typed label
space, informed node features, learned abstention, a ranked prediction of intervention sizes, an
out-of-sample null and a negative control — and are stated with their thresholds so that a later
result on any of them cannot be a threshold chosen after the fact. The register, with every
prediction as it was written and every outcome as it was measured, is
`paper2/preregistration.md`.

### Runtime

Standardisation is the binding constraint. Three other candidates were tested against it and
rejected by measurement: the pairwise filter's maximum-common-substructure step (the generator
outruns the filter a hundred to one); rule application (all 7,580 templates on a 109-atom
molecule in 5.2 s, 20,655 products, at most 0.58 s per rule); and substrate size as an axis (two
43-atom substrates, 64.7 s and no completion). What remains is product standardisation, at
94–99% of cold generator time (Figure 2). A cache conceals it in ordinary use: a repeated
substrate runs 16–47 times faster, so any timing not taken cold measures a mixture of the two
regimes, and every figure in this section is cold.

Moving standardisation to surviving candidates (**H13**) freed recall ($-0.0015$ against a cap
of $0.01$) but failed on time: $2.95\times$ against a registered $10\times$. The mechanism holds
and the arithmetic does not. Enumeration collapses by $13.9\times$, and standardising the hundred
survivors becomes 72% of the new arm: removing 94–99% of the work does not remove 94–99% of the
time, because the remainder becomes the new majority.

Adding a tautomer enumeration budget of 200 (**H15**) passes both halves: median time
$49.85 \to 5.28$ s against a registered target below 10 s, recall loss $0.0015$ against a cap
of $0.01$. The target is registered in seconds rather than as a multiple, because a multiple measures
change where the question is fitness for use. Part of the apparent speed-up is machine load. The enumeration stage is identical in the two
survivors arms, so the ratio of its medians, $1.73$, measures load and nothing else. Dividing
that out of the standardisation step's ratio leaves $2.52\times$ attributable to the tautomer
budget itself, against $3.93$ predicted by the invariance curve; and applying it to the whole arm
puts the median substrate at $9.12$ s under the load the comparison arm ran at, still inside the
registered target of ten seconds. The headline $5.28$ s is measured on a machine running only
this job.

The budget lives in a module-level singleton shared with the tautomer matching key, so setting
it globally would silently move what counts as a hit. The budget is therefore applied through a
private enumerator: the global setting is untouched, and matching keys on 96 molecules are
identical with and without it.

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
only where it stops doing so (Figure 3). What this table supports is not that 87.2% of the gap is
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
so a user can accept or reject a candidate on chemical grounds rather than on a score
(§A worked example).

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
the system, and is reported as such. The group-ranking headroom is unreached by both registered
compositions for structural reasons, and under a granularity-matched control the part of it
attributable to a molecular formula does not separate from zero; what remains is an oracle over
transformation types, which bounds a mechanism nobody has built. The interactive mode runs out
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
names any leaf that has moved. Every committed numeric artifact records the commit that wrote it,
and readers verify that record; the sweep is green on 40 pinned artifacts.

Two of the forty are too large to commit: the train and validation candidate pools, 47.2 MB and
48.6 MB, holding 201,103 and 196,282 candidates with both component scores for each. They are
deposited at ⟨ZENODO_DOI⟩ under CC BY 4.0, compressed to 9.5 MB and 9.6 MB. `paper2/zenodo_manifest.json`
pins the SHA-256 of both the raw file and the uploaded archive together with the record counts,
and `scripts/zenodo_deposit.py --verify --dir <download>` checks a download against it. The
archive is gzipped at a fixed level with a zero timestamp, so its digest is a function of the
content alone and can be pinned at all.

That guarantee covers the pinned set and not the directory: of 258 files under `results/`, 117
carry no stamp and 40 were written by a producer that has since changed. Those files support no
number in this paper.

---

## Author contributions / Funding / Conflicts

⟨TBD⟩

---

## References

Cited above by name and year. Each is present in `paper2/refs.bib` under the key given, so the
conversion to LaTeX is a substitution and not a search.

| cited as | key in `refs.bib` |
|---|---|
| Chavan, 2026 | `Chavan_2026` |
| Cormack, Clarke & Buettcher, 2009 | `cormack2009` |
| de Bruyn Kops et al., 2019 (GLORY) | `de_Bruyn_Kops_2019` |
| de Bruyn Kops et al., 2020 (GLORYx) | `de_Bruyn_Kops_2020` |
| Djoumbou-Feunang et al., 2019 (BioTransformer) | `Djoumbou_Feunang_2019` |
| Larsson et al., 2025 | `Larsson_2025` |
| Litsa et al., 2020 (MetaTrans) | `Litsa_2020` |
| Ridder & Wagener, 2008 (SyGMa) | `Ridder_2008` |
| Rudik et al., 2017 (MetaTox) | `Rudik_2017` |
| Zhou et al., 2025 | `Zhou_2025` |
| Zhou et al., 2026 | `Zhou_2026` |
| Zhu et al., 2024 (MetaPredictor) | `Zhu_2024` |

`refs.bib` holds 53 entries; the 12 above are what this text cites. The remainder support the
evaluation-methodology argument and are cited in the companion manuscript.

---

# Placeholders to fill

Everything computable from the released artifacts has been filled and the entries below
are the ones no run can supply. Each was checked against the artifacts before being
removed from this list; the closed ones are recorded in the paragraph beneath the table.

| marker | what is needed | blocks submission |
|---|---|---|
| `⟨CORPUS⟩` | the provenance of the annotated substrate–metabolite corpus: source, access date, licence | yes |
| `⟨SERVER⟩` | implementation stack, input and output formats, batch limits, sample data, help pages and tutorial, all of them NAR eligibility criteria | yes |
| `⟨SERVICE_URL⟩` | the deployed URL | yes |
| `⟨LICENCE⟩` | the licence; NAR requires free non-commercial use under a standard licence | yes |
| `⟨CONFIRM⟩` | confirmation that the platform requires no login; NAR forbids mandatory registration | yes |
| `⟨ZENODO_DOI⟩` | the DOI, minted on publishing the deposit; the bundle and manifest are built and verified, the upload needs a token and is the author's action | yes |
| `⟨AUTHORS⟩` | author list | yes |
| `⟨AFFILIATIONS⟩` | affiliations | yes |
| `⟨TBD⟩` | author contributions, funding, conflicts | yes |
| `⟨ANY_FURTHER⟩` | limitations the authors know of and this draft does not | no |

`⟨CASE_STUDY⟩` is listed as blocking because the claim it would demonstrate — that every
prediction names its rule and site — is the paper's main non-numerical claim, and no released
artifact currently carries a rule field per candidate.

Resolved from the artifacts:

- `⟨VERSION⟩` — RDKit 2022.09.5, the version the bank parses under
- `⟨N_SHARED⟩` — 291, from results/deployment_table.json
- `⟨N_DISTINCT⟩` — the sentence was wrong in the other direction: 669.9 is the count AFTER deduplication on validation, 588.7 on the comparison set, and no raw figure was needed
- `⟨T_INTERACTIVE⟩` — 0.39 s median over 294 validation substrates, from results/mode_timings.json. The 1.85 s this was to be filled with is a mean over a forty-substrate draw from a different population
- `⟨POP_H13⟩`, `⟨POP_H15⟩` — validation, 293 paired substrates, from the two verdict artifacts
- `⟨REGISTRY_URL⟩` — paper2/preregistration.md
- `⟨ORACLE_BY_TYPE⟩` — computed; see the group-rank section
- `⟨CASE_STUDY⟩` — gemcitabine, both modes, from `results/case_study.json`. Producing it found
  that `_firing_atoms` returned an empty tuple for every candidate: the localisation is handed
  the raw product of `RunReactants`, whose implicit valence is not computed, the MCS inside
  raises, and the except clause swallowed it. Fixed, with a guard in `test_audit_fixes.py`
- `⟨CITE_METATOX⟩` — Rudik et al., 2017, JCIM 57(4):638–642, added to `refs.bib` as `Rudik_2017`
- `⟨BIBTEX⟩` — the reference table above; `cormack2009`, `Rudik_2017` and `de_Bruyn_Kops_2019`
  were missing from `refs.bib` and have been added, the first of them cited by the LaTeX and
  therefore rendering as a broken reference until now

**Numbers deliberately not used**, because they belong to the superseded architecture
(product fusion, $\alpha$ emission) and would be stale here: the F1 table with GRAIL at 0.192,
recall 0.219, precision 0.222, output 2.09, and its comparators. If a precision or F1 table is
wanted, it must be recomputed under the deployed fusion.

**Citations in the LaTeX manuscript.** `paper2/body.tex` currently cites one work in its whole
body. Every comparator it names — SyGMa, MetaTox, BioTransformer, GLORYx, MetaTrans,
MetaPredictor — appears without a citation, though all six are in `refs.bib`. The reference
table above gives the key for each so the wiring is a substitution.

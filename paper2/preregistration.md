# Preregistration: typed local edit for metabolite prediction

**Status:** draft. This file is committed and given an external timestamp **before the first
run**, together with the frozen split and the stratum membership files.

The principle is the one the leaderboard paper states in its own appendix: *the test was
specified before it could be run, and specified so that it could fail.*

The unit of analysis is the substrate throughout. Intervals are paired bootstraps over
substrates, 10,000 resamples. Multiplicity is corrected with Holm inside a declared family, and
each hypothesis names its own.

---

## 0. What is frozen before registration

| artifact | why |
|---|---|
| the train/valid/test split, substrate-disjoint, with the zero-overlap audit | nothing else matters without it |
| `strata/sparse_at_rule_dense_at_type.txt` and its complement | without them H1 is post hoc |
| `strata/trivial_automorphism.txt` and `strata/nontrivial_automorphism.txt` | the two arms of H6 |
| the type vocabulary, its signature definition and its radius, plus the curve over both | the choice of signature is itself an undeclared choice |
| `analyse.py` in full | thresholds cannot move after the numbers are seen |
| the comparator set, including LAGOM, DeepMetab, DeepCYP and Metabolite-Gen | adding a comparator after a run is choosing a cell |

### 0.1 Measurements already taken, which therefore cannot be hypotheses

These were measured on the clean test split before this file was written. They are inputs. A
prediction whose answer is already in an artifact is not a preregistration, so anything below
is stated as a fixed quantity and never as a hypothesis.

Two type vocabularies run through this work and they are close in size, so each is named
wherever it is counted. The **bond-delta** vocabulary keys a type by the multiset of changed
bonds between mapped atoms (`reaction_types.canonical_type`, 4,417 types); it is what E.1 means
by "the bank holds this type". The **signature** vocabulary keys a type by the step-0
reaction-centre signature (4,456 types at radius 0 with hydrogen dropped); it is what the H1
stratum is built on. `check_prereg.py` fails any sentence that quotes a count from either
without naming which.

| quantity | value | artifact |
|---|---|---|
| coverage ceiling as deployed | 0.8171 | `results/coverage_gap_types.json` |
| uncovered references, split by type | 337 novel / 98 known / 40 untypeable | same |
| ceiling after relaxing the hydrogen-count primitives | 0.8175 (+0.0004) | `results/typed_edit_known_type_recovery.json` |
| ceiling after relaxing hydrogen count and connectivity | 0.8183 (+0.0012) | same |
| of the 98 known-type misses, recovered by that relaxation | 3 | same |
| the ceiling on that recovery, from the carriers alone | 8.5 of 98 (8.7%) | `results/typed_edit_type_carriers.json` |
| realised over ceiling | 0.35 | derived from the two rows above |
| bond-delta types with at least one carrier the relaxation touches | 383 of 4,417 (8.7%) | `results/typed_edit_type_carriers.json` |
| training pairs in labels with at least five examples, rules as labels | 64.4% | `results/typed_edit_type_curve.json` |
| the same, signature types as labels | 77.9% | same |
| test substrates in the H1 stratum | 277 of 1,170 (23.7%) | `results/h1_stratum.json` |
| test references in the H1 stratum | 376 of 2,536 (14.8%) | same |
| pool coverage, whole bank, no selector, k = 82 | 0.7326 on 1,128 substrates | `results/bank_without_selection_full.json` |
| the same on the 245-substrate subsample | 0.7602 | `results/bank_without_selection.json` |
| SyGMa recall@15 on the same 1,128 | 0.5707 | `results/bank_without_selection_full.json` |
| test substrates with a trivial automorphism group | 445 of 1,170 (38.0%) | `results/h6_stratum.json` |
| of the 725 with a symmetry, those whose largest orbit is a pair | 615 (84.8%) | same |

The first version of this file predicted that soft admissibility would return about half of the
98 and lift the ceiling to 0.836. It returns three and lifts it to 0.8183. H1 and H5 are
rewritten below around what the measurement leaves standing, and the mechanism they used to
name is recorded here as a closed question rather than deleted.

### 0.1c The deployment population, and what is already measured on it

The 291 substrates of `results/four_method_291.json` are where the MetaTox comparison is
decided. They are listed here rather than under 0.1 because they are not the test split, and
because a set that carries a release claim needs its own membership audit.

| quantity | value | artifact |
|---|---|---|
| of the 291, substrates seen in training or validation | **0** | `results/external_overlap_audit.json` |
| the same audit on the GLORYx external set | 24 of 37 (64.9%) | same |
| the same on the shared 150-substrate subset | 0 of 150 | same |
| pool coverage, whole bank, no selector, uncapped | 0.8105 micro / 0.8458 macro | `results/wide_pool_analysis_implicit.json` |
| references no budget can reach | 126 of 665 (18.9%) | derived from the pool artifact |
| substrates whose pool holds none of their references | 24 of 291 | same |
| micro recall@15, filter x generator | 0.3549 | `results/wide_pool_analysis_implicit.json` |
| micro recall@15, rank fusion | 0.5023 | `results/rrf_vs_metatox.json` |
| micro recall@15, perfect choice of formula group | 0.6692 | `results/wide_pool_analysis_implicit.json` |
| micro recall@15, perfect choice of isomer within group | 0.3654 | same |
| MetaTox micro recall@15 | 0.5143 | `results/four_method_291.json` |
| capping candidates per group at m = 5, 3, 2, 1 | 0.4902 / 0.4707 / 0.4526 / 0.3850 | `results/group_decode.json` |

The last row is a closed question, not an input to one. Every cap tested falls below the
uncapped ranking and the family is monotone in how tight the cap is, so spreading the budget
across formula groups is not an approximation to choosing among them. H8 is registered below on
what that leaves.

### 0.1a Three of our own

Three numbers produced while building these instruments did not describe what they appeared to
describe: a rule bank of 7,581 that was silently 844, a ceiling literal that survived the
correction of the code that wrote it, and this registration's own H1 stratum, 227 of whose 376
pairs were in it by notation. Each is set out in `paper2/three_instances.md` with its mechanism,
what caught it, and its fix. They are recorded because they are the only honest explanation of
why the checks below sit where they sit, and because a methodological claim that has only ever
been tested on other people's work has not been tested.

### 0.1b The pool, measured on the split rather than on a fifth of it

The project's target function was derived from a 245-substrate subsample: a pool coverage of
0.7602, a recall@15 of 0.5277 without a selector, and therefore a top-15 retention of 0.694.
Re-measured on all 1,170 substrates the ladder sits lower and one of its conclusions does not
survive. (A first re-measurement ran on 1,128: `--max-substrates 1198` was passed in the belief
that 1,198 was the whole split, but the cap applies to the 1,246 substrate indices in the
triples, so it sampled. The full run reproduces it to within 0.0002 on every cell, which is
worth stating because the correction was necessary and changed nothing.)

| k | 245: GRAIL | SyGMa | gap | 1,128: GRAIL | SyGMa | gap | interval excludes zero |
|---|---|---|---|---|---|---|---|
| 8 | 0.4151 | 0.5208 | −0.106 | 0.3723 | 0.5259 | −0.154 | yes |
| 15 | 0.5277 | 0.5755 | −0.048 | 0.4715 | 0.5708 | **−0.099** | **yes** |
| 32 | 0.6519 | 0.5889 | +0.063 | 0.5994 | 0.5885 | **+0.011** | **no** |
| 64 | 0.7345 | 0.5930 | +0.141 | 0.6971 | 0.5911 | +0.106 | yes |
| 82 | 0.7602 | 0.5930 | +0.167 | 0.7327 | 0.5923 | +0.140 | yes |

**The inequality this measures, which is about a class and not about us.** Breadth buys +0.140
of coverage at k = 82 and is charged 0.2612 on the way down to k = 15, against SyGMa's 0.0215.
Breadth is bought for 0.140 and paid for with 0.240, a ratio of **1.71 against**. A bank that
reaches further pays more at truncation than it reaches, and that cannot be seen without a
ceiling, which nobody else in this field publishes.

Two readings change. On the subsample the bank without a selector was **statistically
indistinguishable from SyGMa at the budget the field reports**, which was the strongest form of
the claim that the bank is not the bottleneck; on the split the gap at k = 15 is −0.099 and its
interval excludes zero. And the crossover at k = 32 was separated from zero on the subsample and
is not on the split (+0.014, interval covering zero). The subsample was favourable, not harsh --
the opposite of what an earlier note in this project asserted, and for a reason that has nothing
to do with the ceiling literal that note relied on.

**The target arithmetic, restated.** Retention is recall@15 over pool coverage: 0.4715 / 0.7327
= **0.6435**, against 0.694 on the subsample. The operational number is not the stretch but the
**break-even**: matching SyGMa at k = 15 needs 0.5708 / 0.7327 = **0.7790**, so a ranker must
close 13.5 points of retention to reach zero and everything above that is margin. At 0.85 the
margin over SyGMa is 0.052, not the 0.070 the subsample implied.

**What may and may not be frozen from this.** The pool is clean: it is the bank applied without
a selector, and the generator does not enter it. **Retention is not**: it is ranking, ranking is
the filter times the generator, and the generator is trained on the label matrix whose
convention disagrees with the firing convention on a Jaccard of 0.456. The pool coverage of
0.7326 is therefore a fixed quantity in the sense of §0.1; the retention of 0.6435 is recorded
as the value under the current supervision and is re-measured after that defect is fixed, before
any target is set against it.

### 0.2 The split, frozen

`paper2/split_manifest.json` fingerprints what is frozen: the content digest of each clean
triples file, an order-independent digest of each split's substrate set, the digest of the
evaluated test set and of every stratum file, the digest of the rule bank, since a bank
that moves moves every ceiling, and the digests of the third-party comparators that can be
pinned at all. `scripts/typed_edit/freeze_split.py --verify` recomputes all of
it and names any leaf that moved. The dataset is external and cannot live in the repository; a
fingerprint can, and without one "the split is frozen" is a claim a reader cannot check.

| | substrates | positive pairs |
|---|---|---|
| train | 9,011 | 17,454 |
| validation | 1,020 | 2,085 |
| test | 1,198 | 2,597 |
| test, as evaluated | 1,170 | 2,597 |

The evaluated test set is smaller than the split because 28 substrates carry no parseable
reference; that set, not the triples file, is what every number in this registration is
computed on, and it has its own digest.

Cross-split overlap, from `results/leakage_fix_report.json`, recomputed rather than asserted:
substrate overlap and positive-pair overlap are zero for all three split pairs. Molecule
overlap is not zero and is not meant to be: 2,408 molecules appear in both train and test,
because a product annotated in one split can be a substrate in another. 243 test substrates
appear as molecules in train and 115 as annotated metabolites of a train substrate. That is a
property of a metabolic corpus, not a leak of the label, and it is recorded so that nobody has
to rediscover it as one.

### 0.2a What the emission rule is and is not entitled to claim

The pool-relative emission rule leads the five-method table on macro F1 at the published cell:
0.192 against MetaTrans's 0.177, emitting 2.09 candidates against the deployed 8.65 and SyGMa's
74.15. Two facts bound what may be said about that, and both are measured.

**The interval.** Of the five paired contrasts, two separate from zero: **+0.049
[+0.013, +0.086]** against our own deployed output policy and **+0.141 [+0.090, +0.196]**
against SyGMa's volume strategy. Against the three learned methods the differences are +0.015
to +0.030 with intervals covering zero. The lead over them is not established and is not
claimed.

**The grid.** Run in every cell of five criteria crossed with ten budgets the comparators are
read at, the rule beats every comparator under every criterion at budgets of 15 and above --
25 cells of 25 -- and loses or ties at 8 and below. The boundary is where the comparators'
output falls to the rule's own two candidates: below it a comparator is read at a comparable
size and its ranking decides, above it the rule's output policy decides.

Together these say what the contribution is. It is a declared emission policy that beats an
undeclared budget and a volume strategy, on the axis and at the budgets the field reports.
It is not a better ranking: at matched small output the comparators win, which is the same
thing recall@15 says when it falls from 0.365 to 0.219. A paper claiming an F1 leadership from
the published cell alone would be selecting a cell, which is the error this series is about.

---

### 0.3 The comparator set, closed

Adding a comparator after a run is choosing a cell, so the list is closed here. A comparator
that cannot be obtained by the freeze is reported in the paper as unavailable, with the reason;
it is not silently dropped, and no comparator is added later.

| comparator | what is pinned | status |
|---|---|---|
| SyGMa \citep{Ridder_2008} | `sygma == 1.1.0`, declared in `pyproject.toml`, run in process | pinned |
| BioTransformer \citep{Djoumbou_Feunang_2019} | `BioTransformer3.0_20230525.jar`, sha256 `c70cad91…`, from bitbucket.org/wishartlab/biotransformer3.0jar at commit `6432cf88` (2023-05-25); the 994 templates the reach figure reads are three JSON files in this repository, digested in the manifest and loaded to a count of 994 | pinned |
| MetaTrans \citep{Litsa_2020} | frozen predictions only | its pipeline no longer reproduces here, which is why the five-method table rests on 150 shared substrates |
| MetaPredictor \citep{Zhu_2024} | third-party weights, cited rather than shipped | predictions frozen |
| GLORYx \citep{de_Bruyn_Kops_2020} | the external 37-substrate set | 24 of the 37 overlap the training substrates; the rank-flip section reports the 13 that do not |
| LAGOM \citep{Larsson_2025} | code at github.com/tsofiac/LAGOM, Apache-2.0 | **no fine-tuned checkpoint is published**; running it means training it ourselves, on data including DrugBank |
| DeepMetab \citep{Zhou_2025} | weights obtained: 81 LFS objects, 307.8\,MB, each verified against its declared sha256 | **the generation module is withheld**: `SOM/Reaction.py` is absent from the tree and the author states it was removed (issue \#1, 2025-12-29) |
| DeepCYP \citep{Zhou_2026} | a web service, deepcyp.scbdd.com | **web only**, and named by the DeepMetab author as the route to run DeepMetab |
| Metabolite-Gen \citep{Chavan_2026} | preprint only, not peer reviewed | **no code repository found**, and the record could not be retrieved to read its availability statement |
| MetaTox | named in the plan | **no run in this repository** |

Six of the ten are pinned or frozen and four are not, which is the honest state of the list on
the day it is closed. The four were pursued rather than assumed unavailable, and the outcome is
recorded per comparator in `results/comparator_acquisition.json` with the URL each fact was
read from: **none of the 2025-26 wave can be run end to end as released.** One of the four
publishes usable weights and withholds the module that turns a predicted site into a metabolite;
one publishes code and no checkpoint; one is a web service; one publishes neither. This is a
fact about that literature and not a difficulty of ours, and it is reported as such rather than
as a set of comparators we chose to leave out. The BioTransformer row is pinned by the digest of the jar itself and by
the upstream commit, not by a version string: `3.0` names several builds, and the 994 templates
the appendix reports are counted from files whose digests are in the manifest beside it. The four systems of the 2025-26 wave are the ones that set the operating
point this work has to beat, so a claim of a state of the art without them is not available
whatever the outcome; if they cannot be run, what the paper may claim is dominance over the
five it did run, named as such.

---

**The four that do not run, as a table.** Each was pursued to the point where the obstacle was
identified, and each fact carries the URL and the date it was read from in
`results/comparator_acquisition.json`; `scripts/check_comparators.py` re-derives the local half,
so the report cannot drift from the machine it describes.

| system | what is released | what is missing | what running it would take |
|---|---|---|---|
| LAGOM \citep{Larsson_2025} | training code, Apache-2.0 | any fine-tuned checkpoint | training it ourselves, which yields our reimplementation and not LAGOM |
| DeepMetab \citep{Zhou_2025} | weights (81 LFS objects, 307.8 MB, all verified) and site/substrate prediction | `SOM/Reaction.py`, removed by the author for stated copyright reasons; only its 3.7 bytecode remains | the module, or a reimplementation of the step this work compares on |
| DeepCYP \citep{Zhou_2026} | a web form | any model or documented API | submitting 1,170 substrates to a third party's server, at a scale it does not advertise |
| Metabolite-Gen \citep{Chavan_2026} | a preprint, not peer reviewed | any code repository found by search or on the author's accounts | the code, or the availability statement the record does not serve |

None of the four is unavailable through neglect: three publish something and one publishes a
paper. What none publishes is a path from a released artifact to a prediction on our substrates.
That is the concrete case for the recommendation the first paper makes -- publish either a score
per cell of the declared grid or the predictions themselves -- and it is worth more coming from
a failed attempt than from an assertion.

## H1 — the label space, not the admissibility gate

**Prediction.** Scoring a type-indexed candidate pool beats scoring a rule-indexed one on macro
recall@15, **and the gain concentrates in the substrates whose reference transformation needs a
label that is sparse as a rule and dense as a type**, by a factor of at least

    K = 2.5

That is: the stratum's share of the gain is at least 2.5 times its share of the references.
The prediction is registered as an enrichment factor and not as a share, because the stratum's
size depends on how a mined SMIRKS is matched to the catalog, and a prediction stated as a
share would move every time that join moved.

**The three definitions, and which one is primary.** The rule-level lookup can be done by exact
SMIRKS string, or by a signature that does not depend on how a template was written. All three
are declared and the prediction must hold under **each** of them: the conjunction is the test,
and the three ratios with their intervals are the robustness report. The **primary** definition,
named here in advance, is membership confirmed by every key, which is the most conservative of
the three and the only one no join choice can inflate.

| key | pairs | substrates | share of the 2,536 typeable references |
|---|---|---|---|
| exact string | 376 | 277 | 14.8% |
| signature, radius 2 | 149 | 124 | 5.9% |
| signature, radius 1 | 143 | 118 | 5.6% |
| **primary: every key** | **143** | **118** | **5.6%** |

`strata/sparse_at_rule_dense_at_type_intersection.txt` holds the primary membership and its
complement sits beside it; the per-key files and the sensitivity are in
`results/h1_join_sensitivity.json`. At the primary definition K = 2.5 asks for 14% of the gain;
at the widest it asks for 37%. Both exclude a diffuse effect, and neither is reachable by
accident.

**Why 2.5 and not the number the stratum suggests.** The factor implied before any of this was
measured lies in [2.4, 3.4]: the first version of H1 (98 of 475 references, half the gain) is
2.43, and the string-join stratum with the same half is 3.38. 2.5 sits at the conservative end
of that interval, so it cannot be a fit to the strata that were later built, and it does not
inherit the inflation the string join turned out to carry.

**The feasibility check, registered with the prediction.** The gain inside the stratum cannot
exceed the stratum, so with N typeable references, S in the stratum and a gain of G references,
the requirement is satisfiable only while

    K <= N / G

At N = 2,536 that is a gain under 1,014 references for K = 2.5, under 746 for 3.4, and under
285 for 8.9. A factor of 8.9 would therefore be **falsified by its own success**: let the
intervention recover more than 285 references and the required share runs into an arithmetic
ceiling. When G is known, K·p is compared against S/G before the verdict is read. If the
requirement exceeds the ceiling the test was incapable, and that is reported as incapacity and
not as a failure of the hypothesis, exactly as an MDE is reported for H6.

**Evaluated.** Δ = recall(typed pool) − recall(rule pool) on the full clean test split; the
share is the same contrast computed over the stratum's references, divided by Δ.

**Failure:** Δ ≤ 0, **or** the enrichment falls below 2.5 under any of the three keys while the
feasibility check passes. The second is a substantive failure at a numerical success: the model
won, and not by the mechanism claimed. The mechanism would then be reformulated and
re-registered rather than written up after the fact.

**Family:** H1 alone, m = 1. It shared a family with H5 while both were about soft
admissibility; they now test different mechanisms and are corrected apart.

**What this replaces.** The first version predicted that half the gain would fall in a stratum
of references whose type the bank holds and whose rule did not fire. The stratum behind it was
built on an exact string join, which put 227 of its 376 pairs there by notation rather than by
sparsity. The failed construction is recorded rather than removed.

---

## H7 — the fusion rule, declared before it is checked

**Why this exists.** Ranking the selector-free pool by reciprocal rank fusion of the two
component scores beat ranking by their product by 0.109 of micro recall@15 on the 291 MetaTox
substrates. That number is not usable: four combination rules were computed on those substrates
and the best was taken, which is an argmax over the set it is reported on. The rule is therefore
fixed here, in advance of any further measurement, and checked where it has not been looked at.

**The rule, fixed.** For a substrate's pool, rank the candidates twice — once by the filter
score, once by the generator score — and order them by

    score(c) = 1/(60 + rank_filter(c)) + 1/(60 + rank_generator(c))

The constant 60 is the one Cormack, Clarke and Buettcher published with the method in 2009. It
is not tuned here and no other value is computed. No weighting between the two rankings is
introduced: an asymmetric fusion would be a second free parameter, and the point of this
registration is that there is none.

**Prediction.** On the validation split, which no measurement in this project has been read on
for this purpose, the fusion rule beats the product on micro recall@15 over the same
selector-free pool, and the margin is **at least +0.05**, half of what it showed on the 291.

**Failure:** the margin is below +0.05, or negative. Either says the 0.109 was a property of
the substrates it was measured on rather than of the combination, and the deployed product is
kept.

**What is not claimed.** That rank fusion is the right combination. It is a combination that is
scale-free, has no fitted parameter, and can be checked; a trained listwise ranker is the thing
it stands in for, and this registration exists so that the stand-in cannot quietly become the
result.

**Family:** H7 alone, m = 1.

**Outcome, checked 2026-08-25.** Supported. On the validation split, micro recall@15 is
0.3756 under the product and 0.4489 under rank fusion, a difference of **+0.0733** against the
registered +0.05, 95% paired-bootstrap CI [+0.0477, +0.1020] over 10,000 resamples of
substrates. The margin is roughly two thirds of the 0.109 the rule showed on the substrates it
was chosen on, which is the direction an argmax predicts. Recorded in
`results/h7_verdict.json`; the pool it reads is `results/val_pools.json`.

**Deviation from the declared population.** 293 of the 294 substrates were measured. Index 83
is a 291-heavy-atom PEGylated peptide, four times the size of the next largest substrate in the
population; the pair filter's MCS alignment did not terminate on it in over three hours and it
holds 2 of the population's 657 references. Rather than drop it, the difference is recomputed
with both of its references credited first to the product and then to the fusion: the margin
lies in [+0.0700, +0.0761] whichever way it would have fallen, so the absence cannot have
produced the verdict. The absent index and its references are recorded in the pool artifact.

**Bias that remains.** The checkpoints were selected on validation, so validation is not clean
for them in the strict sense. The selection was made on the product-shaped objective the fusion
rule is being compared against, so the residual bias favours the product and the margin
reported here is a lower bound rather than an inflated one.

---

## H8 — the group scorer, registered before it is built

**Why this exists.** The oracle decomposition on the deployment population says the ordering
loss is almost entirely a question of which molecular formula to spend the budget on. Handing
the ranker a perfect choice of formula group lifts micro recall@15 from 0.3549 to 0.6692; a
perfect choice of isomer inside a group lifts it to 0.3654. Rank fusion, registered as H7 and
confirmed on validation, reaches 0.5023, so 0.167 separates it from the between-group oracle
(which orders within a group by the deployed product, the ordering worth 0.011). The cheap route
to that gap is closed: capping how many candidates one group may take in the early slots hurts
at every strength tested and monotonically in the cap, because the oracle's advantage is
selective rather than diversifying. What is left is the model H7 already named rank fusion a
stand-in for, and this registration fixes it before it is written.

**What a group is.** The molecular formula of the candidate, from
`rdMolDescriptors.CalcMolFormula` on the standardised product. It is the partition the oracle
decomposition and the decoding measurement were computed on. No second partition is computed,
so nothing here is evidence about grouping by reaction type.

**The model, fixed.** A scorer over groups rather than candidates. For a substrate and one group
of its pool it consumes only quantities already available without further chemistry:

- the substrate encoding the generator already produces;
- the group's size, and the maximum and mean of its members' filter and generator scores;
- the elemental difference between the group's formula and the substrate's, as a fixed-length
  count vector over the elements the bank can add or remove.

It is trained with a listwise objective over the groups of one substrate, softmax cross-entropy
against the indicator of which groups contain an annotated reference. Its output sets the order
of the groups; inside a group, members keep the order rank fusion gives them, because that
ordering is worth 0.011 and is not what is being changed.

**Split discipline.** Trained on the training split, every hyperparameter chosen on validation,
reported on the 291. The audit in section 0.1c shows none of the 291 appears in training or
validation, so the population is held out from both. One thing was chosen by looking at the 291:
the direction, because the oracle decomposition that motivates a group scorer was computed
there. No parameter is, and the threshold below is fixed before the model exists.

**Prediction.** On the 291, micro recall@15 under the group scorer exceeds micro recall@15
under rank fusion by at least **+0.05**, with a paired-bootstrap CI of the difference that
excludes zero.

**Operational corollary, registered with it.** At the same budget the gap to MetaTox turns
positive with a CI excluding zero. MetaTox stands at 0.5143 and rank fusion at 0.5023, so the
corollary needs 0.012 of the registered 0.05. A primary that holds while the corollary does not
would mean the scorer helped without reaching the budget where the comparison is decided. Both
are reported whichever way each falls.

**Failure:** the margin is below +0.05, or negative. That would say the between-group headroom
is not reachable by a scorer over these features, leaving it a property of the reference set
rather than of anything the pool carries, and rank fusion stays what ships.

**What is not claimed.** That the oracle's 0.6692 is attainable. It is an upper bound computed
with knowledge of the answer, and the registered margin is under a third of the distance to it.

**Family:** H8 alone, m = 1.

---

## H9 — the pool cap, fixed before it is checked

**Why this exists.** The filter builds one pair graph per candidate, so the work it does is
proportional to the pool: 588.7 candidates on average over the deployment population, 4,614 at
the worst. Capping the pool bounds that work, and the cap can be applied on the generator score,
which is available before any pair graph is built.

**What this cap does not bound.** The envelope sweep in `results/cost_envelope.json` shows the
filter is not where a large substrate spends its time. At 40 heavy atoms the generator takes
369.6 seconds and the filter 3.4, a ratio of a hundred to one that widens with size, and the
substrate that did not finish in three hours was not finishing inside the generator. A pool
cannot be capped before it has been enumerated, so this cap saves nothing on the cost that
actually grows. The input envelope is a separate question about the generator's rule budget and
is not addressed by H9; what H9 bounds is the filter's share, and what it is checked on below is
ranking.

Measured on the 291, the cap does not cost recall; it gains. Eight caps were computed there and
the table is in `results/pool_cap_cost.json`, which is an argmax over the set it is reported on
and is recorded as an upper bound rather than a result. This registration fixes the cap and the
margin before the check.

**The cap, fixed.** Keep the 100 highest-scoring candidates by generator score, then order the
survivors by the fusion rule H7 registers. 100 is not the best of the eight: 50 scored 0.0015
higher at k=15, which is inside the noise of either. It is chosen by a rule stated before the
comparison is quoted -- the smallest round cap that is at least twice the largest budget any
comparison in this project reports, so that no reported budget is constrained by the cap itself.
A cap of 50 would make the k=50 column a tautology.

**Prediction.** On the validation split, micro recall@15 under the cap exceeds micro recall@15
without it by at least **+0.015**, roughly half the +0.0331 it showed on the 291, with a
paired-bootstrap CI of the difference that excludes zero.

**Failure:** the margin is below +0.015, or negative.

**What the failure does not undo.** The cap is adopted as a cost guard whichever way the margin
falls, because bounding the pool at 100 bounds the filter's work by six times on the mean and
forty-six on the worst substrate, and the registration's own prediction is that this costs
nothing. Only the claim that it *improves* ranking is at stake here. A margin between zero and
+0.015 means the guard is free and the gain was a property of the 291; a margin below zero means
the guard has a price, and the price is then reported with it.

**What is not claimed.** That 100 is optimal, or that the mechanism is understood. The plausible
mechanism is that fusion weights the filter's ranking equally with the generator's, so
candidates the filter likes and the generator does not float up from deep in the pool; gating on
the generator first removes them. That is a hypothesis about why, and nothing here tests it.

**Family:** H9 alone, m = 1.

**Outcome, checked 2026-08-26.** Supported. On the validation split micro recall@15 is 0.4489
uncapped and 0.4748 capped, a difference of **+0.0260** against the registered +0.015, 95%
paired-bootstrap CI [+0.0033, +0.0469] over 10,000 resamples of substrates. The margin is about
four fifths of the +0.0331 the cap showed on the 291 it was read off. The mean pool falls from
669.9 to 98.0, so the guard it was registered as holds independently of the gain. Recorded in
`results/h9_verdict.json`.

The same absent substrate as H7 is bounded rather than excluded: the difference lies in
[+0.0228, +0.0289] whichever arm its two references are credited to, so the absence cannot have
produced the verdict.

**The shape of the gain.** The cap costs a little at the tightest budgets and pays increasingly
with depth: -0.0077 at k=1 and k=5, +0.0061 at k=10, then +0.0259, +0.0397, +0.0641 and +0.0748
at k=15, 20, 30 and 50. Gating on the generator removes candidates that were occasionally the
single best answer and removes far more that were crowding the middle of the list.

**On the deployment population**, which the audit in section 0.1c shows is held out from
training and validation both, the cap turns every budget's gap to MetaTox positive: +0.0917,
+0.0617, +0.0211, +0.0271, +0.0391 and +0.0797 at k = 5, 10, 15, 20, 30 and 50, with the
bootstrap excluding zero at 5, 10, 30 and 50. Before the cap the comparison lost at 15, 20 and
30. These are the deployment report and not the evidence for the cap, which is the validation
check above.

---

## H2 — informed node features

**Prediction.** Adding reactivity (hydrogen abstraction energy, SMARTCyp-style) and
accessibility (2DSASA) to the node features improves the **ordering of sites within a type**
more than it improves overall recall, in standardised units:
Δ(site-top1)/σ(site-top1) > Δ(recall@15)/σ(recall@15).

**site-top1** is the share of cases where the annotated site is scored top among the admissible
sites of ONE type; counted only on substrates where that type has at least two admissible sites
and at least one annotated.

**Failure:** the inequality does not hold. The features would then be extra capacity rather
than chemistry, and calling them informed would not be supportable.

**Why this shape.** The diagnosis appendix reports that a site-of-metabolism prior used as a
reweighting was worth at most +0.011 and says outright that it did not isolate why. This
measures the effect where it has to be if the claim is true, rather than in the aggregate.

**Counter-evidence addressed in the text whatever the outcome:** Li, Green et al., *JACS* 2024,
where the gain from quantum descriptors falls as the corpus grows; GraphCySoM, *JCIM* 2024,
which argues topology suffices.

**Family:** H2 alone, m = 1.

---

## H3 — learned abstention

**Prediction.** A stop logit reaches a macro F1 no worse than the tuned rule of the emission
appendix (α = 0.5 gives 0.209 against 0.195 for the best global constant), **with no α**, on the
same test set.

**Failure:** the learned stop is worse than the tuned constant. That would itself be a result:
set size is not learnable from this supervision, which would sharpen the appendix finding that
the count is not predictable from substrate descriptors, to not predictable from the pool
either.

**Family:** H3 alone, m = 1.

---

## H4 — a ranked prediction of intervention sizes

**Prediction.** Four interventions, ordered by their gain in macro recall@15 on the shared test
set, come out in this order:

1. removing the selector as a hard gate — **measured floor +0.176** (n = 245);
2. fixing the convention in the supervision (label matrix and firing convention agree on a
   Jaccard of 0.456);
3. typing the label space;
4. informed node features.

**Failure:** any pair inverted with an interval excluding zero.

This is the most exposed hypothesis in the set and is registered for that reason: it tests
whether the world model the system is built from is right, not whether something works. The
relaxation of convention-dependent primitives is not on this list and never was; what the
measurement above closed is H5's mechanism, not this ordering.

**On the floor this list starts from.** The +0.176 is measured on a 245-substrate subsample of
the test split, drawn at random (`rng.choice`, without replacement, seed 44) and not sliced, so
it is a sample and not a selection. Its full-split value is being computed and will be recorded
in §0.1 when it lands. **H4 predicts an ordering, not a magnitude**, so a floor that moves
without inverting a pair leaves the prediction untouched; a floor that inverts one inverts it
before the freeze and the registration says so rather than re-deriving the list around the new
number.

An earlier note in this project asserted that the 245-substrate subsample was systematically
harder than the split, on the strength of a `ceiling_on_this_subset` of 0.7284. That literal
predates a correction that took the same quantity to 0.8007 on the same substrates, against
0.8171 on the full split -- a difference consistent with sampling. The assertion was wrong and
is retracted here; see `paper2/three_instances.md`.

**Family:** the six pairwise comparisons, m = 6.

---

## H5 — the same null, out of sample

**Prediction.** Relaxing the convention-dependent primitives raises the coverage ceiling by
**at most 0.005** on the validation split and on the external GLORYx substrates, neither of
which this measurement has touched.

On the test split the same relaxation is worth +0.0012, and that number is an input in §0.1,
not a prediction. What is predicted is that the near-null transfers: the reason it is small is
structural rather than particular to one split. 82% of the bank's bond-delta types are carried
only by mined rules with no relaxable carrier at all, and the mined majority carries such
primitives on 2.8% of its rules against 51.1% of the curated fifth, so a split whose references
need mined chemistry cannot be helped much by this lever whichever split it is.

One quantity from that measurement transfers to any other attempt at relaxation and is recorded
for it. The carriers put a ceiling of about 8.5 of the 98 on what this lever could reach; three
were reached. **Carrying a relaxable primitive is necessary for recovery and sufficient in about
a third of cases**, because the rule must also fire on that substrate and build that reference.
A ceiling computed from carriers alone should be discounted by roughly that factor before it is
used to justify the work.

**Failure:** the ceiling rises by more than 0.005 on either set. That would mean the near-null
is a property of the test split rather than of the bank, and the structural account above is
wrong.

**Family:** H5 over two sets, m = 2.

**What this replaces.** The first version predicted a rise to at least 0.836, being half of the
98 known-type misses. Three of the 98 are recovered. The prediction failed before the freeze,
which is what a preregistration is for; it is rewritten rather than removed, and the failed
version is recorded here.

---

## H6 — a negative control

**Prediction.** Pooling probability over automorphism classes

- **does not** change macro recall@15 on substrates whose graph has no symmetry: the interval
  on the paired difference covers zero;
- **does** improve it on substrates whose largest orbit is three or more.

**Why the treatment arm is the orbit-three arm and not every symmetric substrate.** Of the 725
substrates carrying a symmetry, 615 (84.8%) have a largest orbit of two: one pair of equivalent
atoms and nothing else. Pooling over a pair can at most double one candidate's mass, so a
prediction made over all 725 is a prediction made mostly where the mechanism can barely act.
The treatment arm is therefore the 110 substrates with a largest orbit of three or more, named
here in advance. The 725-substrate contrast is reported beside it as a secondary, because its
arm is the larger one and its interval the tighter.

| arm | file | n | mean recall@15 | sd |
|---|---|---|---|---|
| control, trivial group | `strata/trivial_automorphism.txt` | 445 | 0.3046 | 0.4112 |
| treatment, largest orbit ≥ 3 | `strata/orbit_ge3.txt` | 110 | 0.3455 | 0.4401 |
| secondary, any symmetry | `strata/nontrivial_automorphism.txt` | 725 | 0.3609 | 0.4379 |

**The minimum detectable effect, registered with the prediction.** A negative control that
passes because it could not have failed is worse than no control, so the MDE is reported with
the verdict whatever the verdict is. At 80% power and α = 0.05 two-sided, the paired MDE is
2.8·σ_d/√n. σ_d, the spread of the per-substrate paired difference, is not knowable before the
run; substituting the spread of recall itself gives a conservative upper bound, since a paired
intervention that changes few substrates has σ_d well below σ:

    control arm    n = 445   MDE ≤ 0.055
    treatment arm  n = 110   MDE ≤ 0.118
    secondary      n = 725   MDE ≤ 0.046

The realised σ_d and the MDE it implies are computed from the run and reported in the same
table as the contrasts. **A null on the control arm is read as evidence only if its realised MDE
is below the effect the treatment arm shows**; a control too blunt to have seen the effect it is
controlling for is reported as blunt.

**Failure:** it improves recall on both arms, or on neither while the treatment arm's MDE is
below the improvement the secondary arm shows. The first says the mechanism is not the one
claimed and something that changed alongside it is doing the work; the second says the same by
a different route.

**Family:** H6 alone, m = 1, all three arms inside it.

**The definition.** Symmetry classes come from RDKit's canonical ranking with ties left unbroken
and chirality excluded, the latter because the pipeline strips stereochemistry before the rules
fire. Every heavy atom in its own class means the automorphism group is trivial, and that
direction is exact, which is the direction a control arm needs. The orbit histogram is 445 at
one, 615 at two, 75 at three, 31 at four, 3 at six and 1 at eight.

---


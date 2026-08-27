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

### 0.2b How the comparison with MetaTox may be stated

The comparison is the sweep, not a cell. Every budget the comparators are read at is reported
together, with the interval on each, and no budget is privileged by being quoted alone.

That rule is not a preference. Paper one is about the error of reading a leaderboard at the cell
that flatters the reader, and a project that argued the point and then fought for k=15 -- the one
budget where its own interval covers zero -- would be applying to itself a standard it calls
substitution in others. So the claim is stated as it falls:

> The whole bank leads MetaTox at k = 1, 3, 5, 8, 10, 15, 20, 30 and 50, with the paired-bootstrap interval excluding zero at k = 1, 3, 5, 8, 10, 30 and 50. At k = 15 and 20 the interval does not separate.
> The trained budget leads MetaTox at k = 1, 3, 5, 8 and 10, with the paired-bootstrap interval excluding zero at k = 1, 3, 5, 8 and 10. It trails with the interval excluding zero at k = 20, 30 and 50. At k = 15 the interval does not separate.
> Mean list length: whole bank 98.9, trained budget 15.7, MetaTox 31.0, on 291 substrates.

That block is generated from `results/deployment_table.json` by `scripts/sweep_claim.py` and a
test holds this file to containing exactly what it produces. Written by hand it was wrong within
the hour: it said the point estimate leads at k=15 and k=20, which is true of the whole bank and
false of the trained budget, where the gaps are -0.0346 and -0.0647 and the second separates
against us. A sentence that cannot drift from its artifact is worth more than one that happens to
be right today.

The artifact carries every budget, both arms, the intervals, and the count of substrates whose
list is shorter than the budget on each side, so a reader sees the whole sweep rather than the
row an author chose.

**The consequence for what is open.** "The lead at k=15 is not established" is not an open
question that further work should close; it is a line in that table. Work aimed at moving that
one cell would be work aimed at a number rather than at the service, and this file records that
so the temptation is on paper rather than in someone's head.

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

**Scope, fixed before the check.** The oracle numbers this registration is built on are the
whole-bank pool's. On the pool the trained rule budget produces, which H10 fixed after H8 was
written, a perfect choice of formula group is worth **+0.0120** of micro recall@15 -- 8.5 groups
per substrate against the whole bank's 160.4 -- so the registered margin of +0.05 is not
reachable there and H8 is not testable on the shipped configuration. It is checked on the
whole-bank pool, where the same oracle is worth +0.1729. The result is therefore about a
configuration that costs forty times what ships, and any margin it produces is read against
+0.1729 rather than against zero.

**A restriction on the training population, recorded rather than discovered later.** The pipeline
does not terminate on every substrate: `results/cost_envelope.json` times 106 and the smallest
that failed to finish in 600 seconds has 42 heavy atoms, while everything below 40 completed.
Four substrates of the training draw, at 45, 51, 58 and 60 heavy atoms, blocked their shards
indefinitely. The training population is therefore restricted to substrates of at most **40
heavy atoms**, the largest round threshold strictly below any observed failure, which removes 37
of 386 and leaves 349.

The restriction is on training only; the 291 are reported whole. That makes a distribution shift
between the two, and its size is measured rather than assumed: 17 of the 291 exceed 40 heavy
atoms, 5.8 per cent of the substrates carrying 6.3 per cent of the references, and the largest
is 109. The scorer will be asked about molecules larger than any it was trained on, and the
verdict reports the margin on those seventeen separately so the reader can see whether the shift
costs anything.

**Outcome, checked 2026-08-27. Failed.** On the 291 the scorer reaches 0.3820 of micro recall@15
against rank fusion's 0.5023, a margin of **-0.1203** with a CI of [-0.1590, -0.0828] against a
registered +0.05. The corollary fails with it: against MetaTox the gap is -0.1323. Recorded in
`results/h8_verdict.json`. Twenty configurations were trained, the grid extended once when its
winner sat at the widest layer it contained, and every one of them selected on validation.

**The failure has two halves and the registration owns one of them.** H8 fixes that the scorer
orders the groups and leaves the order inside a group alone. That forces every member of a
formula group to be emitted adjacently, and blocking is not free: the *same* fusion ranking, with
group members kept together, scores 0.4376 against 0.5023 interleaved. The design costs
**-0.0647** before the model does anything, and the model then costs a further **-0.0556**
relative to that blocked baseline. The +0.1729 of oracle headroom this hypothesis was built on
was measured in the blocked form, so its first 0.0647 was never headroom at all -- it was the
design buying back what the design had spent, and nobody had measured that before registering.

**Two explanations that the artifact rules out.** The distribution shift is not it: above 40
heavy atoms the scorer reaches 0.1667 against fusion's 0.2619, and at or below it reaches 0.3965
against 0.5185, so it is no worse in proportion on the molecules larger than any it saw. Capacity
is not it either: validation recall across widths of 32, 64, 128, 256 and 512 runs 0.4117,
0.4100, 0.4149, 0.4100 and 0.4198, which is flat.

**What the failure is, then.** On validation the scorer beat the blocked baseline by +0.0243; on
the 291 the same contrast is -0.0556. A margin of that size, picked as the best of twenty
configurations on one split, does not survive the move to another, which is what selecting on a
margin near the noise buys. Under it sits the feature set the registration fixed: the maximum and
mean of the members' raw filter and generator scores and no rank, while the baseline it must beat
is built from ranks. The scorer was asked to rebuild a strong ordering in a representation that
ordering does not use, and it did not. That reading was written down before the check, not after.

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

## H10 — the rule budget, registered while the answer is still running

**Why this exists.** The generator's cost is not the filter's and not RDKit's. Applying all
7,580 templates to a 109-heavy-atom substrate takes 5.2 seconds; standardising the products they
produce is 94 to 99 per cent of the generator's cold time, and the number of products is set by
how many templates are applied. The envelope sweep finds substrates of 42, 43, 56, 57, 61, 67
and 68 heavy atoms that do not finish in 600 seconds, so this is an operating limit rather than
an efficiency question.

The budget is already a parameter, `top_k`, and the shipped checkpoint records **30** for it.
Every pool this project has built passed 7,581 instead, to get a bank-wide pool with no
selector. Nothing has measured what that costs or buys.

**The operating point, and the rule that picks it.** 30, because it is the value the checkpoint
records: the configuration the generator was trained and selected under. It is not read off the
curve this registration is written against, and the curve is deliberately incomplete at the time
of writing -- two of its six arms have reported, the arm at the whole bank has not started, and
the prediction below is made without it.

**Prediction.** On the validation split, the whole bank buys **at most +0.05** of micro recall@15
over the trained budget of 30, both measured with the H9 cap applied and both ranked by the H7
fusion rule.

**Failure:** the whole bank buys more than +0.05. The rule budget would then not be a free guard,
and what it costs must be stated as a price rather than adopted as a saving.

**What is measured beside it, and is not a prediction.** The wall clock. At the trained budget a
substrate costs 1.85 seconds against 56.67 at a budget of 100, and the arms above 100 are slower
still; whichever way the recall falls, the cost side is a measurement and is reported as one.
Costs are wall clock on one machine with the standardisation caches cleared between arms, which
is the state a service meets a molecule in and not the state most of this project's earlier
timings were taken in.

**What is not claimed.** That 30 is the best budget. The curve exists to show the shape, and its
draw of 40 substrates carries 92 references, so a difference of 0.03 in recall is three
references and the arms cannot be separated on it. The shape chooses the operating point; the
validation split, with 293 substrates and 655 references, decides the margin.

**Family:** H10 alone, m = 1.

**Outcome, checked 2026-08-26.** Supported. On validation the whole bank buys **+0.0092** of
micro recall@15 over the trained budget of 30, against a registered ceiling of +0.05; the 95%
paired-bootstrap CI is [-0.0231, +0.0399], which contains zero, so at this budget the whole
bank's advantage is not distinguishable from none. The mean pool falls from 669.9 candidates to
17.0. Recorded in `results/h10_verdict.json`.

The substrate the whole bank never finished is in the trained arm: all 294 validation substrates
completed in 1,510 seconds, the 291-heavy-atom peptide among them. The absent substrate is the
one the whole-bank pool lacks, and bounding its two references both ways puts what the bank buys
in [+0.0061, +0.0122], so the absence cannot have produced the verdict.

**Where the trained budget cannot answer.** Thirty templates yield seventeen candidates, so above
a budget of fifteen the arm is not ranking worse, it is running out of things to return. At
k=15, 167 of 293 substrates already hold fewer candidates than the budget; at k=50, 287 of 293
do. The whole bank's +0.1863 at k=50 is therefore almost entirely the trained arm's empty list
and not a ranking difference, and it is not evidence about ordering. What the check establishes
is bounded to budgets the arm can fill: at fifteen and below the trained budget matches the whole
bank at a thirty-ninth of the pool, and at fifty it does not compete.

**A provenance gap, recorded rather than assumed away.** The whole-bank pool artifact predates
the flag that records the rule budget, so it carries none. It was built by `build_val_pools.py`
when that script passed 7,581 unconditionally, which git records and the artifact does not.

**The deployment population does not agree, and the disagreement is the finding.** Repeating the
same contrast on the 291, where H10 was not registered and which is therefore a report rather
than a second test, the whole bank buys **+0.0556** at k=15 with a CI of [+0.0235, +0.0876] --
just past the ceiling that validation cleared at +0.0092. The two populations give the same
answer at tighter budgets and diverge as the budget widens:

| budget | what the whole bank buys, on the 291 | 95% CI |
|---|---|---|
| 5 | -0.0090 | [-0.0394, +0.0224] |
| 10 | +0.0165 | [-0.0145, +0.0463] |
| 15 | +0.0556 | [+0.0235, +0.0876] |
| 20 | +0.0917 | [+0.0574, +0.1262] |
| 50 | +0.2165 | [+0.1738, +0.2601] |

So the registered threshold is not a property of the system but of the budget it is asked at.
At ten and below the whole bank buys nothing on either population, both intervals containing
zero. At fifteen and above it buys, and the amount rises with the budget because the trained
arm's list runs out: 178 of the 291 hold fewer than fifteen candidates. What H10 establishes is
that the trained budget is free at the budgets it can fill, and what it does not establish is
anything about the budgets it cannot.

---

## H11 — the emission rule, and why it cannot be a threshold

**Why the old rule does not carry over.** Section 0.2a registers a pool-relative rule: emit every
candidate scoring at least alpha times the best, with alpha = 0.5. That was defined on the
product of the filter and generator scores. The ranking is now the rank fusion H7 registers, and
the rule does not survive the change. On the deployment pools at the trained rule budget, alpha
= 0.5 emits 2.31 candidates by the product and **16.22 of a pool of 16.3** by fusion. The reason
is structural rather than incidental: fusion is a rank statistic whose values carry no scale. In
one pool of seven the fusion scores run from 0.03252 to 0.03055, a ratio of 0.939, where the
product spans 2.55e-04. A relative threshold on such a quantity is a rank cutoff wearing a
threshold's clothes, and its knob is violent -- alpha would have to reach 0.98 before the output
fell to two candidates. Choosing an alpha for fusion would be fitting an output size and calling
it a policy.

**The rule, fixed.** The service emits, for a substrate, every candidate the trained rule budget
produces, ordered by the H7 fusion rule, with nothing truncated. There is no emission parameter.
What decides the size of the answer is the rule budget H10 fixed at the value the checkpoint
records, and the chemistry of the substrate; on the 291 this emits 15.7 candidates on average
against MetaTox's 31.0. The H9 cap of 100 remains in the pipeline and does not bind at this
budget, so it is part of the configuration and not part of the emission.

**What is already known and is therefore not predicted.** The direction of the F1 comparison is
implied by numbers already in `results/deployment_table.json`: the rule emits half of MetaTox's
volume and recovers 0.4902 of the references against MetaTox's 0.6271, so it leads on precision
and trails on recall. Restating that ordering as a prediction would be describing an artifact.

**Prediction.** The lead survives the grid rather than living in a cell. Run over the five
matching criteria of section 0.2a crossed with the budgets MetaTox is read at, the rule beats
MetaTox on macro F1 in **every cell**, and the count of cells lost is zero.

**Failure:** any cell is lost. The rule would then be a policy that wins where it is quoted, which
is the error this series exists to catch, and what may be claimed shrinks to the cells that hold.

**The bias that runs in our favour, declared.** Precision is computed against annotated positives
only, so an unannotated true metabolite counts as a false positive. That penalises volume, and
this rule emits half of what it is compared against. The F1 comparison is therefore biased toward
the rule and the artifact reports precision, recall and mean output beside it so the direction of
the bias is visible in the same table. No claim about ranking quality is made from F1; the
ranking claims are H7, H9 and H10, measured at matched budgets.

**What is not claimed.** That emitting the whole pool is the best policy for a user. A service
that must return thirty candidates cannot use this rule, because at the trained budget 282 of the
291 pools hold fewer than fifty and 178 hold fewer than fifteen; for that service the
configuration is the whole bank, and the comparison at matched volume is the one in
`results/deployment_table.json` rather than this one.

**Family:** H11 alone, m = 1.

**Outcome, checked 2026-08-26.** Supported. **Zero cells lost** of the fifty declared, and zero
of the fifty-five once MetaTox's own emission is counted as a column. The rule's macro F1 is
0.1175 to 0.1531 depending on the matching criterion; MetaTox's best cell under every criterion
is at a budget of ten and reaches 0.0905 to 0.1343. Recorded in `results/h11_grid.json`.

| criterion | the rule | MetaTox's best cell |
|---|---|---|
| canonical | 0.1175 | 0.0905 at k=10 |
| inchikey | 0.1530 | 0.1215 at k=10 |
| inchi_no_stereo | 0.1531 | 0.1249 at k=10 |
| tanimoto1 | 0.1180 | 0.0906 at k=10 |
| inchikey_tautomer | 0.1530 | 0.1343 at k=10 |

**The part that does not rest on the declared bias.** F1 favours the smaller emitter here and the
rule emits 16.31 against MetaTox's 36.34, so the F1 lead is not the interesting number. The
components are, and they say something F1 cannot be blamed for: read at a budget of fifteen,
where MetaTox emits about what the rule emits, its macro recall is 0.5406 against the rule's
**0.5672**. At comparable output the rule recovers more, and no accounting of unannotated
positives changes that comparison, because both sides emit the same amount. At its own emission
MetaTox recovers more, 0.6868, on 2.2 times the output.

**The gate failed first, on this file's own error.** It was written to compare MetaTox's macro
recall at its own emission against the 0.6708 recorded in `results/vs_metatox.json`. That number
is MetaTox capped at fifty, which that artifact does and this grid does not; the k=50 column
reproduces it exactly. The cell the gate names was corrected to the one the number was computed
in, and the grid was re-run rather than the discrepancy explained away.

---

## H12 — the group scorer again, without the constraint that sank it

**Why this exists.** H8 failed by -0.1203 and the artifact splits the failure: -0.0647 of it is
the design and -0.0556 the model. H8 required the scorer to order groups and leave the order
inside one alone, which forces every member of a formula group to be emitted adjacently, and the
same fusion ranking blocked that way scores 0.4376 against 0.5023 interleaved. This registers the
same scorer under a composition that cannot block.

**What is reused and what is not.** The model is the one H8 registered: the same features, the
same listwise objective over groups, the same training population. Nothing is refitted on
anything new. What changes is only how its output enters the ranking, and the hyperparameters
are re-selected on validation under the new composition, because selecting them under the old one
would be selecting for a property that no longer decides anything.

**The composition, fixed.** The group score enters as a third ranking of the fusion rule H7
registers, with the same constant and no weight:

    score(c) = 1/(60 + rank_filter(c)) + 1/(60 + rank_generator(c)) + 1/(60 + rank_group(c))

where `rank_group(c)` is the competition rank of c's formula group under the scorer. This
introduces no free parameter: 60 is H7's published constant and the three rankings are unweighted
for the reason H7 gives, that an asymmetric fusion would be a second free parameter. Candidates
of different groups interleave exactly as they do under the two-way rule, so the design cannot
spend the 0.0647 that H8's did.

**Prediction.** On the 291, the three-way fusion beats the two-way by at least **+0.02** of micro
recall@15, with a paired-bootstrap CI of the difference excluding zero.

**Where +0.02 comes from.** It is the margin that decides the comparison this project exists to
settle, not a number read off a curve. At k=15 the two-way fusion reaches 0.5023 against
MetaTox's 0.5143, so +0.012 ties and +0.02 leads. A gain smaller than that would be real and
would not change what may be claimed.

**Failure:** the margin is below +0.02, or negative. That would say the scorer's ordering of
groups carries nothing the filter and generator rankings do not already carry, and the
between-group headroom stays a property of an oracle rather than of anything learnable from these
features. Rank fusion as H7 registers it remains what ships.

**What is not claimed, and what H8 already ruled out.** That a group scorer cannot work. The
features are the ones H8 fixed -- the maximum and mean of the members' raw filter and generator
scores, with no rank -- and a scorer given rank information is a different hypothesis that is not
registered here. H8's artifact already excludes two other explanations: capacity, flat across
widths from 32 to 512, and the training population's 40-heavy-atom restriction, since the scorer
was no worse in proportion on the substrates above it.

**Family:** H12 alone, m = 1.

**Outcome, checked 2026-08-27.** Supported. On the 291 the three-way fusion reaches 0.5308 of
micro recall@15 against the two-way's 0.5023, a margin of **+0.0286** with a CI of [+0.0085,
+0.0502] against the registered +0.02. Both arms are uncapped. Recorded in
`results/h12_verdict.json`. The effect is not a selection: all twenty configurations beat the
two-way baseline on validation and they cluster between 0.5089 and 0.5203, where H8's twenty
straddled its baseline and its winner did not survive the move to another split.

**The ceiling of this composition is the same +0.0286.** Entering a perfect group ranking as the
third component -- knowing exactly which formula groups hold a reference -- also reaches 0.5308.
The model is not merely close to that, it is level with it at this budget. Which says less about
the model than about the composition: a reciprocal-rank term is bounded, so however good the
group ranking is it can only nudge, and the +0.1729 the oracle showed under H8's blocked form is
not reachable this way either. H8's design spent that headroom; H12's design cannot draw on it.

**Where `oracle_third` stops being a ceiling.** It ranks groups binarily, so every group without
a reference is tied and it orders nothing among them. Past k=15 the model's graded ranking does
better -- 0.5865 against 0.5729 at twenty, 0.6421 against 0.6195 at thirty -- and calling the
binary arm an upper bound at those budgets would be wrong. It bounds only the budget at which
knowing which groups hold a reference is the whole question.

**The threshold met its arithmetic and not its purpose.** +0.02 was chosen because at k=15 the
two-way fusion sits at 0.5023 against MetaTox's 0.5143, so that margin would turn a tie into a
lead. The point estimate does lead, 0.5308 against 0.5143, but the paired-bootstrap interval on
that contrast is [-0.0238, +0.0586] and covers zero. The margin is established against the
two-way rule, which is what H12 predicted; the lead over MetaTox it was sized to produce is not.

**The same check on validation, which is not clean for this model.** The hyperparameters were
selected there, so the number below is the selection's own and is optimistic by construction. It
is run because the difference between it and the 291 measures what the selection bought.
Recorded in `results/h12_verdict_validation.json`.

| population | three-way | two-way | margin | 95% CI |
|---|---|---|---|---|
| validation, selected on | 0.4901 | 0.4489 | **+0.0412** | [+0.0192, +0.0621] |
| the 291, held out from train and validation both | 0.5308 | 0.5023 | **+0.0286** | [+0.0085, +0.0502] |

About seventy per cent of the validation margin survives the move, and the +0.0126 that does not
is what choosing among twenty configurations on that split is worth. H8's margin under its own
composition was +0.0243 on validation and -0.0556 on the 291; this one keeps its sign and most
of its size, which is the difference between an effect and a selection.

**`oracle_third` is exceeded on validation, which settles what it is.** The model reaches 0.4901
against the binary oracle's 0.4748. An arm that can be beaten is not an upper bound: ranking
groups binarily orders nothing among the groups holding no reference, so a graded ranking has
room the oracle does not use. The artifact records that beside the ratio so a share above one is
not read as a model beating a ceiling.

**A comparator absent is not a comparator beaten.** The first run of this check on validation
reported the three-way arm ahead of MetaTox by +0.4901 with the interval excluding zero. MetaTox
was run on the 291 and nowhere else, so every one of its lists on validation is empty and that
figure was the arm's own recall wearing a gap's clothes. The check now counts how many substrates
the comparator covers and refuses the contrast at zero rather than computing it.

---

## H13 — standardisation off the enumeration loop

**Why this exists.** Standardising products is **93.7 to 98.7 per cent** of the generator's cold
time across the seven substrates measured in `results/generator_cost_split.json`. The enumeration
calls it on every product of every rule: 366 standardisations on a four-heavy-atom substrate,
4,976 on a twenty-five-atom one, median 3,571. After the H9 cap at most a hundred survive, so the
ratio paid to kept runs from 3.7 to 49.8 and is above twenty for every substrate above thirteen
heavy atoms. Applying all 7,580 templates the checkpoint initialises to a 109-heavy-atom
substrate takes 5.2 seconds; the minutes are spent canonicalising tautomers of products nothing
will rank.

This is registered before the group work because its effect is larger. If it holds, the whole-bank
arm becomes interactive and the two-configuration split may collapse to one; closing the +0.1729
between groups would not do that.

**The change, fixed.** During enumeration, deduplicate products on the cheap canonical SMILES --
`MolToSmiles(mol, isomericSmiles=False)`, the `canonical` branch the code already has -- and run
the full `standardize_mol` only on the candidates that survive the H9 cap of 100. Nothing else
changes: the same templates, the same budget, the same fusion, the same cap.

**Why this is a hypothesis and not a refactor.** The normalised SMILES is the deduplication key,
and the candidate score is a noisy-or over every rule that produces the same key. Two tautomers
of one product collapse under standardisation and do not under canonicalisation, so the change
splits their evidence and moves the surviving candidates' scores. It also feeds the filter a raw
tautomer where it was trained on a standardised one. Both are real effects on the numbers and
neither can be waved through.

**Prediction.** On validation, micro recall@15 falls by at most **0.01**, and the median
per-substrate generator time falls by at least **ten times**.

**Where 0.01 comes from.** It is below every margin this file's results turn on: H12's +0.02,
H9's +0.015, and the +0.012 that separates the two-way fusion from MetaTox at k=15. A change
costing less than that cannot move any claim recorded here, which is the definition of free that
matters.

**Failure:** recall falls by more than 0.01, or the time does not fall tenfold. A cost above the
threshold makes this a trade rather than a saving, and the trade is then reported with both sides
and decided separately.

**What is not claimed.** That the filter is indifferent to being fed a raw tautomer. It was
trained on standardised inputs and this feeds it something else, which is one of the two
mechanisms that could produce a failure; the verdict reports recall with the cap applied and
without it, so a loss concentrated in the filter's scores can be told from one in the
deduplication.

**Family:** H13 alone, m = 1.

**Outcome, checked 2026-08-28. Failed, on the half that was not about recall.** Recorded in
`results/h13_verdict.json`, over the 293 substrates both arms hold.

| | registered | measured | |
|---|---|---|---|
| recall@15 loss | at most 0.01 | **0.0015**, CI [-0.0124, +0.0081] | holds |
| median time falls by | at least 10x | **2.95x**, 49.85 s to 16.92 s | fails |

**The mechanism was right and the arithmetic was not.** Enumeration collapses exactly as the
rationale said it would: against the old median of 49.85 seconds it now takes 3.59, a factor of
13.9. What the rationale did not carry is that standardising the hundred survivors costs 11.88
seconds, which is **72 per cent of the new arm**. Removing 94 to 99 per cent of the work does not
remove 94 to 99 per cent of the time when what remains becomes the new majority.

**Recall is free, and that part stands whatever happens to the timing.** The change costs 0.0015
at k=15 with the interval covering zero, and at k=10 it is worth +0.0107 with the interval
excluding zero. Deduplicating on the cheap canonical form during enumeration, which splits
tautomers that used to merge and moves the noisy-or scores, does not cost accuracy on this split.
The mechanism the registration warned might break it -- feeding the filter a raw tautomer -- does
not arise, because the survivors are standardised before the filter sees them.

**What the survivors arm can do that the other cannot.** It finished all 294 validation
substrates including the 291-heavy-atom peptide at index 83, which the whole-product arm has
never finished in any run at any budget. That is not what H13 predicted and is not claimed as its
result, but it is in the artifact.

**Where the remaining time is, for whoever registers the next one.** 11.88 seconds for a hundred
molecules is 119 milliseconds each, and `results/tautomer_budget.json` already measures what a
smaller tautomer budget buys: 200 gives 3.9 times the speed for three points of invariance, 500
gives 1.8 for half a point. Combining this change with a budget would plausibly clear tenfold,
and would be a different hypothesis with a different cost to declare.

---

## H14 — the group signal as a gate rather than a third ranking

**Why this exists.** H12 works and is bounded. The scorer's group ranking entered as a third
reciprocal-rank term is worth +0.0286, and a *perfect* group ranking entered the same way is
worth the same +0.0286: the term is bounded, so however good the ranking is it can only nudge.
The +0.1729 the oracle shows is real and neither registered composition reaches it -- H8's spent
more than it drew and H12's barely draws.

The way out is not a multiplicative composition, which would restore the scale sensitivity H7
just found worse. It is H9's own result: capping the pool did not cost recall, it gained it, by
removing candidates that were crowding the middle of the list. So the group signal belongs
*before* the fusion, as what decides which candidates the cap keeps, rather than after it as a
term the fusion bounds.

**The gate, fixed.** H9 keeps the 100 highest-scoring candidates by generator score. This keeps
100 candidates too, but chooses them by group: take the formula groups in the order the scorer
ranks them, admitting each group's members whole, until 100 candidates are admitted; rank the
admitted set by the H7 fusion. The budget is H9's registered 100 and no new parameter is
introduced. The last group admitted may take the total past 100, and it is admitted whole rather
than cut, because cutting it would reintroduce a within-group decision this hypothesis does not
make.

**Prediction.** On the 291, micro recall@15 exceeds the H9 configuration -- top 100 by generator,
then fusion -- by at least **+0.02**, with the paired-bootstrap interval excluding zero.

**Where +0.02 comes from.** The same place as H12's: at k=15 the deployed configuration reaches
0.5353 against MetaTox's 0.5143, and the margin is set at the size that would make the lead
survive its interval rather than merely lead in point estimate. It is not read off the group
oracle, which would be reading the answer.

**Failure:** the margin is below +0.02, or negative. Taken with H12 that would say the
between-group signal is worth about +0.03 however it is composed, and the oracle's +0.1729 is a
property of knowing the answer rather than of anything these features carry.

**What is not claimed.** That 100 is the right budget for a gate as opposed to a cap. It is
H9's number, reused so that this hypothesis introduces nothing of its own, and a sweep over it
would be a different registration.

**Family:** H14 alone, m = 1.

**Outcome, checked 2026-08-28. Failed.** On the 291 the gate reaches 0.5068 of micro recall@15
against the H9 cap's 0.5353, a margin of **-0.0286** with a CI of [-0.0558, -0.0044] against a
registered +0.02. Recorded in `results/h14_verdict.json`.

**The design is bounded below the thing it must beat.** A *perfect* group ranking used as this
gate reaches 0.4992, which is **-0.0361** against the cap with the interval excluding zero. No
scorer can clear the threshold through this gate, because knowing exactly which groups hold a
reference does not clear it either. That is not a statement about the model.

**Why, measured.** The median substrate has **one** group holding a reference, and the gate must
admit at least seven groups to reach a hundred candidates -- more in the scorer's own order,
since it does not take the largest first. So after the single informative group the gate spends
the rest of the budget on groups its signal cannot order at all, while the cap spends the same
budget by generator score, which orders every one of them. The gate throws away a graded signal
to use a nearly binary one.

**Read against H12, which it was meant to improve on.** H12 enters the same group score as a
third reciprocal-rank term and gains +0.0286; H14 uses it to select and loses 0.0286. The same
magnitude with the opposite sign, and `gate - three_way` is **-0.0241** measured directly. The
group signal is worth something as a nudge on top of the generator's ordering and is worth less
than nothing as a replacement for it over most of the budget.

**The validation grid said so before the 291 did.** One of twenty configurations beat the H9
baseline there, by +0.0179, and the other nineteen fell below it, spread 0.4781 to 0.5203. H12's
twenty all beat their baseline in a band of 0.0114. A single winner out of twenty is the shape
H8 had, and it did not transfer then either.

**What this closes.** Both remaining ways of spending the between-group signal have now been
registered and checked: as a term inside the fusion, which is bounded to +0.0286 by the fusion's
own arithmetic, and as a gate on the budget, which is bounded below zero by the signal's
coarseness. The +0.1729 the blocked oracle showed is not reachable by either, and a third attempt
would need a mechanism neither of these has rather than another composition of the same score.

---

## H15 — the survivors arm with a bounded tautomer budget

**Why this exists.** H13 failed on time and not on recall. Standardising only the candidates that
survive the cap costs 0.0015 of micro recall@15 with the interval covering zero, and moves the
median from 49.85 seconds to 16.92, a factor of 2.95 against a registered ten. The reason is in
its artifact: enumeration collapses by 13.9 times and standardising the hundred survivors costs
11.88 of the remaining 16.92 seconds, 72 per cent of the new arm. What is left to attack is the
cost of standardising one molecule, and `results/tautomer_budget.json` already measures it
against the enumerator's budget.

**The budget, fixed at 200.** The rule is stated before the consequence is computed: the largest
reduction whose invariance on enumerated tautomers stays within three points of the shipped
setting. The measured curve is 0.9652 at the shipped 1,000, 0.9602 at 500, **0.9353 at 200** and
0.9005 at 100, so 200 is the last value inside three points and 100 is the first outside it. It
is not the value that reaches the target below; that value is not consulted in choosing it, and
if 200 falls short the hypothesis fails rather than the budget moving.

**Why the target is a number of seconds and not a factor.** H13 asked for a tenfold fall and got
2.95 while removing 94 to 99 per cent of the work, because the remainder became the majority.
A factor measures the change and the question is whether the arm is usable: a synchronous request
that returns in ten seconds is slow and possible, one that takes fifty is neither. So the target
is absolute.

**Prediction.** On validation, with the survivors arm of H13 and the tautomer budget at 200:

1. the median per-substrate time before the filter falls **below 10 seconds**, from 49.85 under
   the shipped configuration; and
2. micro recall@15 falls by at most **0.01** against the whole-product baseline, the same ceiling
   and the same reason as H13 -- it is below every margin this file's results turn on.

Both must hold. A single failure is a failure.

**Failure:** the median stays at or above 10 seconds, or micro recall@15 falls by more than 0.01.

Each half means something different. If the time holds and the recall does not, the budget has
started breaking the tautomer-invariant matching the whole comparison rests on, and the trade is
reported with both sides rather than taken. If the recall holds and the time does not, the floor
is the enumeration itself -- 3.59 seconds in H13's artifact, which no tautomer budget touches --
and the next lever is not standardisation at all.

**The diagnostic that is reported and is not a gate.** `tautomer_near_miss` found zero of 655
validation references lost to the canonicaliser at the shipped budget, against an upper bound of
ten from a screen that was too loose. It is re-run at 200 and its count reported beside the
verdict. It is not a threshold, because a reference that stops matching shows up in the recall
number already and two gates on one quantity would double-count it.

**What is not claimed.** That 200 is the right budget for the paper's *matching*, which is a
different question from the budget the generator uses to deduplicate and rank. If this holds, the
scoring-time criterion stays at the shipped setting unless something separately registered moves
it.

**Family:** H15 alone, m = 1.

**Outcome, checked 2026-08-28. Supported.** Both halves hold, on the 293 substrates the arm and
the whole-product baseline share. Recorded in `results/h15_verdict.json`.

| | registered | measured | |
|---|---|---|---|
| median time before the filter | below 10 s | **5.28 s**, from 49.85 | holds |
| recall@15 loss | at most 0.01 | **0.0015**, CI [-0.0124, +0.0081] | holds |

At k=10 the arm is worth +0.0092 with the interval excluding zero, and at every other budget the
change is inside the noise. The whole-bank configuration is interactive: a synchronous request
at the median now returns in five seconds where it took fifty.

**The factor is not all ours, and the artifact divides it.** The two survivors arms did not run
under the same machine load -- H13's ran beside the whole-product arm, this one with only its own
shards -- and the enumeration is *identical* between them, canonical deduplication with no
tautomer budget in it at all. Its time therefore measures the load and nothing else: 3.59 seconds
against 2.09, a factor of 1.73. Dividing that out of the standardisation's 4.35 leaves **2.52
attributable to the budget**, against the 3.93 the budget curve predicted for 1,000 to 200. The
registered target is absolute and is unaffected by any of this; the factor is not, and would have
been overstated by a third had nobody looked.

**What the budget actually changed.** Of 27,595 candidates in the bounded arm, **263** -- 0.95
per cent -- standardise to a different molecule than at the shipped budget. Across all 293
substrates that moves the pools' key sets by 21: twelve keys appear only at 200 and nine only at
1,000. The matching key is computed at the shipped 1,000 in both arms and is untouched; a private
enumerator with a private cache does the survivors, and the global pair is left exactly as it
ships, which was checked before the run on 96 molecules.

**A diagnostic this file first reported as a tautology.** The first version asked how many
candidates matched *by SMILES* between the two arms had a different key, and answered zero of
27,332. That count can only ever be zero: a candidate the lower budget standardised differently
carries a different SMILES and never enters the comparison. It was replaced by the two numbers
above, which can be other than zero and are.

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


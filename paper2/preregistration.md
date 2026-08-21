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
| bond-delta types with at least one carrier the relaxation touches | 383 of 4,417 (8.7%) | `results/typed_edit_type_carriers.json` |
| training pairs in labels with at least five examples, rules as labels | 64.4% | `results/typed_edit_type_curve.json` |
| the same, signature types as labels | 77.9% | same |
| test substrates in the H1 stratum | 277 of 1,170 (23.7%) | `results/h1_stratum.json` |
| test references in the H1 stratum | 376 of 2,536 (14.8%) | same |
| test substrates with a trivial automorphism group | 445 of 1,170 (38.0%) | `results/h6_stratum.json` |
| of the 725 with a symmetry, those whose largest orbit is a pair | 615 (84.8%) | same |

The first version of this file predicted that soft admissibility would return about half of the
98 and lift the ceiling to 0.836. It returns three and lifts it to 0.8183. H1 and H5 are
rewritten below around what the measurement leaves standing, and the mechanism they used to
name is recorded here as a closed question rather than deleted.

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

**Family:** the six pairwise comparisons, m = 6.

---

## H5 — the same null, out of sample

**Prediction.** Relaxing the convention-dependent primitives raises the coverage ceiling by
**at most 0.005** on the validation split and on the external GLORYx substrates, neither of
which this measurement has touched.

On the test split the same relaxation is worth +0.0012, and that number is an input in §0.1,
not a prediction. What is predicted is that the near-null transfers: the reason it is small is
structural rather than particular to one split. 82% of the bank's types are carried only by
mined rules with no relaxable carrier at all, and the mined majority carries such primitives on
2.8% of its rules against 51.1% of the curated fifth, so a split whose references need mined
chemistry cannot be helped much by this lever whichever split it is.

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


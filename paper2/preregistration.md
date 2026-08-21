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
| `strata/trivial_automorphism.txt` | the control arm of H6 |
| the type vocabulary, its signature definition and its radius, plus the curve over both | the choice of signature is itself an undeclared choice |
| `analyse.py` in full | thresholds cannot move after the numbers are seen |
| the comparator set, including LAGOM, DeepMetab, DeepCYP and Metabolite-Gen | adding a comparator after a run is choosing a cell |

### 0.1 Measurements already taken, which therefore cannot be hypotheses

These were measured on the clean test split before this file was written. They are inputs. A
prediction whose answer is already in an artifact is not a preregistration, so anything below
is stated as a fixed quantity and never as a hypothesis.

| quantity | value | artifact |
|---|---|---|
| coverage ceiling as deployed | 0.8171 | `results/coverage_gap_types.json` |
| uncovered references, split by type | 337 novel / 98 known / 40 untypeable | same |
| ceiling after relaxing the hydrogen-count primitives | 0.8175 (+0.0004) | `results/typed_edit_known_type_recovery.json` |
| ceiling after relaxing hydrogen count and connectivity | 0.8183 (+0.0012) | same |
| of the 98 known-type misses, recovered by that relaxation | 3 | same |
| bank types with at least one carrier the relaxation touches | 383 of 4,417 (8.7%) | `results/typed_edit_type_carriers.json` |
| training pairs in labels with at least five examples, rules as labels | 64.4% | `results/typed_edit_type_curve.json` |
| the same, types as labels | 77.9% | same |
| test substrates in the H1 stratum | 277 of 1,170 (23.7%) | `results/h1_stratum.json` |
| test references in the H1 stratum | 376 of 2,536 (14.8%) | same |

The first version of this file predicted that soft admissibility would return about half of the
98 and lift the ceiling to 0.836. It returns three and lifts it to 0.8183. H1 and H5 are
rewritten below around what the measurement leaves standing, and the mechanism they used to
name is recorded here as a closed question rather than deleted.

---

## H1 — the label space, not the admissibility gate

**Prediction.** Scoring a type-indexed candidate pool beats scoring a rule-indexed one on macro
recall@15, **and at least 50% of the gain falls in the stratum
`sparse_at_rule_dense_at_type`**: substrates at least one of whose reference transformations
needs a label that carries fewer than five training pairs when rules are the labels and at
least five when types are, pooled.

That stratum is where typing is supposed to act. Rules as labels leave 35.6% of the training
pairs in labels with fewer than five examples; types as labels leave 22.1%. The prediction is
that the 13.5 points which move are where the recall moves too.

**Evaluated.** Δ = recall(type-indexed) − recall(rule-indexed) on the full test split;
Δ_stratum, the same contrast over the references in the stratum only; share =
Δ_stratum · |stratum| / (Δ · |all|).

**Failure:** Δ ≤ 0, **or** share < 0.5. The second is a substantive failure at a numerical
success: the model won, and not by the mechanism claimed. The mechanism would then be
reformulated and re-registered rather than written up after the fact.

**Family:** H1 alone, m = 1. It shared a family with H5 while both were about soft
admissibility; they now test different mechanisms and are corrected apart.

**The stratum, as built.** `strata/sparse_at_rule_dense_at_type.txt` holds **277 of the 1,170
test substrates (23.7%)**, carrying **376 of the 2,536 typeable reference transformations
(14.8%)**. Its complement is committed beside it. Of the 2,536, the exact mined rule is in the
catalog for 1,564 (61.7%), 1,259 (49.6%) are sparse at rule level and 1,653 (65.2%) are dense
at type level; the stratum is the intersection of the last two. Sixty-one references do not
type through the mining route and are in neither file.

**Why the threshold is 0.5.** A gain spread uniformly over the split would put 14.8% of itself
here, that being the stratum's share of the references. Asking for half is asking for 3.4 times
the uniform expectation, so a typed model that simply scores better everywhere fails this
hypothesis while passing its own contrast. The threshold was fixed before the stratum was
built; the enrichment factor it implies is recorded now that the sizes are known, and the
threshold is not moved to suit them.

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

**Prediction.** Pooling over automorphism classes
- does **not** change recall@15 on substrates with a trivial automorphism group (the interval
  on the difference covers zero);
- **does** improve it on substrates with a non-trivial one.

**Failure:** it helps on both strata. The mechanism would then not be the one claimed, and
whatever changed alongside it would have to be found separately.

**Family:** H6 alone, m = 1, both strata inside it.

---

## What is not predicted, and why

- **Multi-step.** Depth-2 application lifts the ceiling by about 0.012 at 8.5 times the cost.
  The architecture supports the recursion; no claim rests on it.
- **Precision as a headline.** The closure appendix measured the one checkable component of
  annotation incompleteness at 0.0016–0.0035, two orders below what would be needed. Precision
  here is bounded by the corpus, not by the model.
- **Dominance over the grid.** That is a form for stating a result, not a hypothesis: it either
  holds or it does not, and there is nothing to predict.

---

## Compliance check

`scripts/check_prereg.py` reads this file and the final manuscript. A hypothesis is registered
only if it declares a prediction, a failure condition and a family with a size. Every sentence
in the manuscript that claims an effect must name the hypothesis entitling it, and every
registered hypothesis must appear carrying a claim or be reported as having failed. It exits
non-zero otherwise, and it runs in the default test suite.

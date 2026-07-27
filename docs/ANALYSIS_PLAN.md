# Analysis plan: the match-sensitivity endpoint

This document specifies the endpoint, the estimator and the analysis population for the
match-sensitivity results, so that the specification can be read independently of the prose that
reports them.

**Status.** Partly pre-specified, and the split matters. The *design* was fixed in advance:
a draft dated 2026-06-28 names all five matching criteria and specifies that each method's frozen
predictions be re-scored under all of them, twelve days before the first match-sensitivity
artifact (2026-07-10). The *estimator* was not: the differential framing and the paired bootstrap
enter the history on 2026-07-10, seven hours after the first result, in a commit whose own message
describes it as reframing that result. No correction for multiplicity was planned or applied.

This document was written later still. Read it as a specification of what was done, with the
design half pre-dated and the analysis half not.

## Endpoint

The endpoint is the **protocol × method interaction**: how much more one method gains than another
when the matching criterion is relaxed. For methods $A$, $B$ and criteria $q_1 \to q_2$,

    interaction(A, B) = [recall_{q2}(A) - recall_{q1}(A)] - [recall_{q2}(B) - recall_{q1}(B)]

The endpoint is deliberately differential rather than a single method's gain, because every method
gains something under a more tolerant criterion. Only the differential can reorder a leaderboard,
and reordering is the claim.

Individual reversals are **not** the endpoint. A reversal is a consequence of an interaction large
enough relative to the gap between two methods, and is reported as an illustration.

## Estimator

Paired bootstrap over substrates, which are the independent unit:

- 10,000 resamples, seed 0.
- Substrates are resampled with replacement; all methods and both criteria are recomputed on the
  same resampled index, so the pairing is preserved.
- Intervals are the 2.5th and 97.5th percentiles of the resampled statistic.
- Recall vectors are per-substrate; a substrate contributing no reference metabolite is excluded
  from the denominator, not counted as zero.
- Interactions and per-method changes are computed from unrounded values, and may differ by 0.001
  from the difference of the rounded figures printed in the tables.

## Population

- Internal: the shared subset of 150 substrates for which frozen third-party predictions exist for
  all five methods, drawn from the clean, substrate-disjoint test split.
- External: the 37 GLORYx parent substrates that carry at least one reference metabolite.

Predictions are frozen. No method is re-run between criteria; only the matching map changes.

## Multiplicity

Six method pairs are tested on each set. **No correction for multiple comparisons is applied**, and
the paper says so where the results are reported. Under a Bonferroni correction at six tests the
threshold would be 0.0083; the two interactions the paper highlights have intervals that exclude
zero comfortably, but the individual reversals do not survive as independently significant and are
not claimed to.

## Criterion ladder

Where a criterion relaxes more than one thing at once, the intermediate criterion is reported so
the components are separable. On the external set this is `inchikey → inchi_no_stereo →
inchikey_tautomer`, which separates the stereochemistry relaxation from the tautomer relaxation.

## Artifacts

| quantity | artifact |
|---|---|
| internal five-method, five-criterion table | `results/match_sensitivity_5method.json` |
| external four-method interaction and CIs | `results/gloryx_rank_flip_ci.json` |
| external criterion ladder | `results/gloryx_criterion_ladder.json` |
| recall decomposition | `results/recall_factorization.json` |

The toolkit version that reproduces these numbers is recorded in the artifacts that postdate this
document; earlier artifacts were produced under the same environment, RDKit 2022.09.5.

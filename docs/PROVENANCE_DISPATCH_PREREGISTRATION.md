# Where the dispatch gain lives: the prediction, and what would kill the framing that makes it

Written and committed **before** the measurement runs, and before the subset arms exist in any
script. The framing this tests was written a day earlier, on evidence that does not decide it, and
the whole point of registering the prediction now is that the framing can lose.

## What is already known, so the information state is on the record

Measured, on the 245-substrate subsample, `results/dispatch_paired_ci.json`:

- dispatching the hydrogen convention per template across the whole 7,581-rule bank reaches
  **0.8148** against the better of the two global settings, **0.7989**;
- the paired residual is **+0.0159 [0.0056, 0.0284]** micro, **+0.0205 [0.0055, 0.0388]** macro,
  both excluding zero.

Measured, `results/ceiling_by_provenance.json` and `results/provenance_knob_attribution.json`:

- the bank partitions exactly into 1,715 curated and 5,866 mined templates;
- the pre-registered classifier sends **675** templates to the expanding convention, every one of
  them curated and **none** of the 5,866 mined;
- expanding hydrogens carries −0.647 of the mined-minus-curated reversal, the validity floor +0.002.

**Not measured, and the subject of this document:** how that +0.0159 distributes across the two
collections. No arm of any script has yet been run on a subset of the bank.

## The framing under test

The appendix now says the bank is *a union of two collections written under different hydrogen
conventions, and any single global setting is the wrong one for one of them*. That sentence claims
the loss is created by **mixing** — by having to choose one setting for an object that is two
objects. A weaker and duller reading fits the same evidence: curated templates simply want
expansion, mined ones do not, and the dispatch gain is just the curated templates being given what
they want, with the mined majority contributing nothing to the loss and nothing to its repair.

These two readings differ in one measurable way, stated below. If the dull reading is right, the
appendix sentence is an overclaim and gets restated.

## Definitions, fixed here

For a rule subset `R` and a hydrogen convention `c ∈ {expanded, implicit}`:

    reach(R, c)  = micro coverage of references by applying R to each substrate under convention c,
                   canonical normalisation, no validity floor, tautomer-InChIKey matching
    reach_disp(R)= the same, applying each rule under the convention the pre-registered classifier
                   assigns it (docs/DISPATCH_PREREGISTRATION.md, unchanged, not re-derived)
    residual(R)  = reach_disp(R) - max(reach(R, expanded), reach(R, implicit))

Subsets: `full` = all 7,581; `curated` = the 1,715 absent from `mined_only.txt`; `mined` = the 5,866
in it. The partition is exact and already gated.

**Population:** the full clean test split, n = 1,170, as widened. If the widened family has not
landed when this runs, it runs on the 245 subsample and says so; the predictions are qualitative and
do not depend on which.

**Estimator:** paired bootstrap over substrates, 10,000 draws, seed 0, micro (ratio of sums), the
estimator already in `scripts/dispatch_paired_ci.py`. Macro reported beside it as a different
estimand, never instead of it.

## Predictions, in advance

**P1 — `residual(mined) = 0`, exactly.** Zero of the 5,866 mined templates carry a hydrogen atom
primitive, so the classifier sends none of them anywhere and dispatch must reduce to the identity
map over that subset. This is the null that checks the instrument, not the claim: it is the same
role SyGMa played in the earlier registration. **If it returns anything other than 0 to four
decimals, the subsetting is wrong and every other number in the run is uninterpretable.**

**P2 — `residual(curated) > 0`, interval excluding zero.** 675 of 1,715 want the expansion that
curated's better global setting may deny them. If this is 0 there is nothing to distribute and P3
is untestable; that outcome is reported as an untestable prediction rather than quietly dropped.

**P3 — the residual is superadditive: `residual(full) > residual(curated) + residual(mined)`.**
Given P1 this is `residual(full) > residual(curated)`. This is the prediction that carries the
framing. Mixing two differently-conventioned collections into one bank should create loss that
neither collection carries alone, because the global setting that is best for the union is worse for
each part than that part's own best.

## What kills the framing

**If `residual(curated) ≥ residual(full)`**, the mixing account is refuted. The loss is then wholly
inside the curated collection, the mined majority neither creates nor repairs any of it, and the
appendix sentence must be restated as *"675 hand-written templates require the expansion that the
bank's global setting denies them"* — a real result, a smaller one, and not the one currently
written. It is reported as a refutation, in those words.

**If P3 holds but the margin's interval covers zero**, the direction is reported as a point estimate
and explicitly not claimed as certified, in the same terms `negative.tex` used for +0.016 before the
paired run existed.

## What the run may not do

- The classifier is imported from `hydrogen_dispatch.py`; it is not re-implemented, re-tuned, or
  given a subset-specific variant. A classifier chosen per subset would be an oracle.
- The `$(...)`-unclassifiable policy stays frozen at the **whole bank's** majority convention, not
  at each subset's majority. Recomputing the majority per subset would let the policy respond to the
  subset, which is exactly the degree of freedom the earlier registration closed.
- No subset other than the three named here is scored. There is no reporting of a best-performing
  partition.

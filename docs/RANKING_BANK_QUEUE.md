# Queue: three retraining experiments (points 5, 6, 7)

Each needs training, so none is a config flag like the support-gate. This records what each changes,
the prediction it would be registered against, what retrains, and the order to run them in once the
rule-gate result is back. The rule-gate holds the GPU now; these follow it, and the first-stage
measurement noted under each is free (no model) and settles whether the training is worth spending.

## 7. Merge duplicate templates by type instead of compensating with a weight

**What.** The bank is 7,581 templates over 4,417 canonical types. noisy-or treats them as
independent witnesses, the bank violates that, and the manuscript compensates with a tuned weight
in the hybrid rule. Merging the templates of one type into a single template with a generalised
environment makes the independence hold by construction rather than by calibration, and mechanically
raises per-template support: 4,655 of 5,856 mined templates rest on one training pair, and after a
type-merge the median support rises because a type's pairs pool.

**Free first stage (no model).** Group the bank by canonical_type, and measure: how many types have
>1 template, what the support distribution becomes when a type's pairs pool, and -- the decisive
one -- the exhaustive coverage of the merged bank against the deployed bank. If coverage falls, the
generalised environment is too loose and the merge is closed before any training. scripts/merge_by_type.py
below does the counting half now.

**First stage RESULT (results/merge_by_type.json).** Run. The support half of the argument does
NOT hold: 7,581 rules collapse to 4,417 types, but the median per-unit support is 0 per template
and 0 per type, and the share resting on <=1 training pair RISES from 0.76 to 0.79 rather than
falling, because 3,610 types carry a single template and their share of the type space exceeds
their share of the rule space. Merging pools support only in the 807 types with more than one
template (largest 143). So the support motivation is closed before training; what remains is
independence -- those 807 types stop being counted as multiple witnesses by noisy-or -- and whether
that alone beats the tuned hybrid weight is what a run would test. The generalised-environment
coverage measurement is still the gate before spending it.

**Retrains.** The generator, because the label space changes (fewer, pooled rules). The filter can
be reused or retrained.

**Prediction (to register before training).** recall@15 under the merged bank >= the deployed bank
at equal output budget, AND the noisy-or / hybrid gap closes (the tuned weight stops being load-
bearing). Falsification: if the merged bank needs the same hybrid weight, the independence argument
did not buy anything the calibration was not already buying.

**Caveat.** Generalising a type's environment is the hard part and is exactly point 3's hierarchy:
a type merged to its radius-0 center fires everywhere (the bank is already at precision 0.032), so
the merge must keep the most specific shared environment, not collapse to the bare type. Merge and
hierarchy are the same construction at different levels and should be built together.

## 6. Harder negatives for the filter

**What.** The filter trains on every rule-applicable unannotated product as a negative, undiscerning.
The informative negatives are products of the same transformation type on a different site -- the
discrimination S22 audits (92.3% vs a 35.2% null) and which the filter never sees in its training
signal.

**The problem this must not ignore.** The filter trains under nnPU: its negatives are *unlabeled*,
not negatives, because the annotation is positive-unlabelled -- the whole premise of the paper. A
product of the same type on a different site is exactly the kind of thing that could be a real,
unannotated metabolite. Weighting it up as a hard negative amplifies the PU noise the nnPU loss
exists to absorb. So this is not a plain hard-negative-mining change; it is a change to what the PU
prior is estimated against, and it can make ranking worse by teaching the filter to reject true
metabolites. Any run must be read against that, and the guardrail below is the point.

**Free first stage (no model).** Type the negatives (pair_to_type) and measure how many share a
positive's type on the same substrate. If that set is small, there is little signal to mine and the
change is not worth the PU risk; if it is large, so is the risk, and the run has to be careful.

**Retrains.** Only the filter.

**Prediction.** recall@15 rises AND precision at k=50 rises (the filter rejects same-type-wrong-site
products it used to rank). Guardrail, from the PU concern: recall on a held-out set of *known*
metabolites does not fall -- if teaching the filter these negatives costs it true positives, the
intervention is rejected however much precision improves.

## 5. Exact atom map for the filter, not MCS

**What.** The merged substrate-product graph (filter mode pair) lost once: recall@15 0.357 vs 0.366,
-0.0090 [-0.017,-0.001], at 34.0 ms vs 9.1. But its cross-edges were placed by MCS -- an inferred
correspondence. The true correspondence is known: the template that produced a candidate carries the
atom map. This re-runs the same architecture with the exact map, so the earlier loss may have been
the correspondence failing rather than the idea.

**Free first stage (no model).** On the substrates where the pair filter and the single filter
disagree, measure how often the MCS correspondence differs from the template's atom map. If they
rarely differ, the exact map cannot rescue the architecture and the idea is closed; if they often
differ, the earlier loss is plausibly the correspondence and the run is worth it.

**Retrains.** The filter, in pair mode, and the plumbing: the generator's rule_id per candidate has
to reach the filter so the template's map can be used, which the factorized path already threads.

**Prediction.** the pair filter with the exact map >= the single filter (0.366), reversing the
earlier -0.0090. Falsification: if it still trails, the merged-graph architecture loses on its own
merits and not on the correspondence.

## Order

7's first stage first (it is the cheapest measurement and feeds both 3 and 5's reasoning about
specificity), then 5's disagreement measurement, then 6's -- all three free. Whichever survives its
free stage is trained first; 6 trains only the filter and is cheapest to train, 7 trains the
generator and is dearest. None depends on the rule-gate result, so they can be measured now and
trained whenever the GPU is free.

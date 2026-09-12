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

**First stage RESULT (results/mcs_vs_template_map.json).** Run on 55 substrates x 60 rules. Of 858 applied pairs, 238 (27.7%) have an MCS correspondence that sends at least one substrate atom to a different product atom than the template's exact map does; 3.9% of all aligned atoms are misplaced. The MCS is right most of the time atom-for-atom but wrong on more than a quarter of pairs, often at the reaction center where symmetry defeats it. Point 5 is NOT closed by its free stage: the exact map is a genuinely different alignment on a quarter of pairs, so the pair filter's earlier -0.0090 may well have been the correspondence, and the exact-map re-run is worth training.

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

## Change panel: what survived, and the invariance that kills half of it

A design panel proposed changes across six lenses (ranking, calibration, filter, generator, pool,
ensemble); each was checked against what is already measured. Ten of twenty-three were renames of
levers already rejected (score shrinkage, the k-adaptive blend, coverage expansion, hard-negative
reweighting). Of the survivors, the one rated most promising was measured directly and failed, and
a structural fact disposes of a whole family of the rest.

**Cap by the fused rank instead of the generator score: MEASURED, worse (results/cap_by_fused_rank.json).**
The generator cap drops 39 in-pool references the filter favours (pool_cap_cost.json). Keeping the
same budget of 100 but selecting it by the fused rank was expected to recover them. It does the
reverse: -0.0702 [-0.0936, -0.0478] at k=15 on validation, -0.0647 on the comparison set, both
significant, with the deployed arm reproduced exactly (0.4748 / 0.5353). The weaker filter pulls
noise into the top-100 faster than it recovers the dropped refs, so the generator cap's noise
suppression is load-bearing. Closed.

**The rank-invariance that closes the calibration family.** The deployed fusion (scripts/typed_edit/_rrf.py)
orders candidates by reciprocal fusion of their two competition RANKS, and the cap keeps the top-100
by raw generator score. Both are invariant to any monotone, per-axis rescaling of the scores. So
temperature/Platt/isotonic calibration, MC-dropout averaging over an eval-mode scorer with no
variance, and SWA/EMA weight averaging cannot change the order they are applied to -- the same
reason score shrinkage's optimum was lambda=0. Any lever that only rescales scores is dead on
arrival; only a lever that changes which candidate OUTRANKS which can move recall@k. That leaves
two routes: change the pool composition (the cap experiment above, worse), or retrain so a model's
score ORDER differs. Everything below is the second route, and needs compute.

### Compute-gated survivors (retrain to change the rank order; GPU-bound)

**A. Difference-aware readout in the pair filter.** filter.py Filter.forward (pair mode) reads the
merged sub+prod graph with a single global_mean_pool, averaging the ~5% minimal-delta signal of a
dehydrogenation (the class GRAIL trails furthest, and where the diagnostic puts 12/23 refs in the
pool but ranked past the budget) into near-invariance. Replace with a structured readout:
encoder.forward_nodes (exists) then split substrate/product pools by the flag column at node index
16 and concatenate a delta vector. PU-safe (label-free), changes the filter's discrimination order
on minimal-delta pairs. Prediction: recall@15 rises, concentrated on dehydrogenation; falsify if
the class-conditioned recall does not move.

**B. train_on_candidates as the release filter config.** The deployed filter trains on MolFrame.negs
(derived from gen_map), a different distribution from the wide generator-top-k pool it ranks at
inference; the machinery to train on the generator's own top-k exists (workflows/ensemble.py
generate_filter_training_data, candidate_generation_top_k=200). Aligning train and deploy
distributions changes the filter's rank order on the pool it must rank. nnPU-correct (unannotated
candidates stay unlabeled, not true-negative), val-selected on recall@15. Falsify if aligned
negatives do not beat the deployed filter on validation.

**C. Type-shared rule-id embedding (refinement to the support-gated id already training).** The
id-zeroing ablation gains at the head (+0.037@1, +0.027@5) but loses at depth (-0.021@30): the
per-template id overfits a rare template's single positive at the head yet carries signal at depth.
Form id_r = g(n_r)*template_id_r + type_id_{type(r)} so a rare template borrows its reaction type's
pooled id (canonical_type over rule_keys; type stats pooled as _update_rule_statistics already does).
Sequence AFTER the current rulegate run reports, since it builds on that id_gate. Falsify if the
type term does not recover the depth loss without giving back the head gain.

**D. Eval-time match_scale override.** generator._forward_generation_logits adds match_scale*log_counts
to every rule logit, a systematic bonus for rules matching the substrate at many sites (promiscuous
aromatic hydroxylation over few-site desaturation). match_scale is a learned parameter with no
inference knob. NOT frozen-measurable: the bonus is baked into the generator score in the frozen
pools, so testing an override requires a generation pass with the knob exposed, not a re-rank.
Add the inference override, regenerate the pool at a swept match_scale, measure on validation.
Falsify if curbing the bonus does not lift few-site classes without sinking the many-site ones.

### Order

D is the cheapest (one generation pass, no training) and directly tests the promiscuity tilt that
the dehydrogenation diagnostic implicates, so it runs first when compute frees. Then A (filter-only
retrain, cheap) and B (filter-only retrain), then C after the rulegate run reports. All are
validation-selected with the deployed arm reproduced as the gate, on the same terms as the cap
experiment above.

## Point 3 (radius hierarchy): MEASURED, closed (results/radius_hierarchy_coverage.json)

The caveat above predicted it and the measurement confirms it. Mining the same 400 train pairs at
radii 0/1/2 and applying each to 40 test substrates:

  radius 0 (bare center): coverage 0.5976, mean pool 88.3
  radius 1 (deployed):    coverage 0.5122, mean pool 53.6
  radius 2 (specific):    coverage 0.4268, mean pool 18.6
  most-specific-that-fires hierarchy: 0.4268 / 18.6 -- identical to radius 2

Two findings. Coverage rises only by going MORE general (r0 is +0.085 over the deployed r1) at 1.65x
the pool -- the precision cost this bank cannot afford at 0.032, and the wrong end of the pipeline
when the binding constraint is ranking a noisy pool, not coverage. And the naive hierarchy
degenerates to radius 2: the specific rules fire SOMETHING on all 40 substrates, so
"most-specific-that-fires" never falls back and lands on the worst coverage. A hierarchy that helped
would need a per-rule confidence/support signal to decide fallback -- which is point 7's merge/support
work, not a pure radius mechanism. Radius alone is not a lever; closed.

## Point B (train_on_candidates): RESULT -- not selected at subsample scale (results/rulegate_candfilter_result.json)

Trained locally on CPU, the cpu_baseline recipe with filter.train_on_candidates=true, seed 42,
compared only to rulegate_cpu_baseline. The filter's negatives came from the generator's own top-200
(176,544 candidates, 427 missed positives added back) instead of MolFrame.negs.

Validation (the selection metric) is a wash: recall@5 -0.0011, recall@15 -0.0037, every budget within
the seed sd 0.0107. On test recall@5 rose +0.0303 -- the largest head signal of any survivor -- but
recall@15 fell -0.0129, and the filter's global discrimination dropped (mcc -0.081, auc -0.075). The
covariate-shift alignment is mechanically real: the filter trades global binary accuracy for accuracy
on the head of the pool it actually ranks. But on one seed at subsample scale it does not move the
metric selection is made on, so B is not adopted on its own terms. The only survivor with a head
signal worth a second look: a cheap second seed would say whether the +0.030 test head separates from
noise. Same helps-head-loses-depth shape as the id-gate; representation and filter-objective levers
both nudge the head and cost depth without breaking the ranking ceiling.

## Point A (difference_readout) + the survivor frontier: RESULT (results/rulegate_survivors_summary.json)

A note on scope: A was written for the pair filter's single mean-pool, but the deployed filter is
mode=single (two independent encoders concatenated), which has no merged-pool averaging problem. A
was adapted to the deployed mode: append the signed delta (prod_emb - sub_emb) to the [sub, prod]
readout so a minimal-delta transformation is represented directly. Trained on the cpu_baseline
recipe, seed 42, compared to the matched baseline.

A behaves like the others. Validation (selection): recall@1 +0.0148, recall@3 +0.0120 -- the clearest
head gain on the selection metric of any arm -- but recall@15 -0.0119. Filter discrimination is
unchanged (mcc +0.002, auc +0.001), so the gain is a cleaner head readout, not a better binary
filter.

**The frontier.** With A done, all four matched arms are in one table (rulegate_survivors_summary.json):

  recall@k delta vs baseline, validation:   id_gate   candfilter   diffreadout
    r@1                                       -0.0093    +0.0075      +0.0148
    r@3                                       +0.0079    +0.0102      +0.0120
    r@15                                      -0.0050    -0.0037      -0.0119

Three unrelated mechanisms -- representation (id-gate), filter objective (train-on-candidates),
filter features (difference-readout) -- and one shape: each lifts the head (k=1,3) and loses the
budget (k=15). None lifts the whole curve, and most deltas are within the seed sd 0.0107, so no arm
is an individually significant win at k=15. The consistent head-for-depth trade across three places
in the pipeline is the result: at this scale the ranking ceiling is a frontier the generator's rule
scores set, and reshaping how the FILTER reads a candidate, or how a rule id is represented, slides
along it rather than raising it. A headline gain needs either a multi-seed confirmation of a head
lever at the k the product is read at, or a change to what the GENERATOR ranks -- which is point D
(match_scale), the one remaining survivor and the only one that touches the generator's scores.

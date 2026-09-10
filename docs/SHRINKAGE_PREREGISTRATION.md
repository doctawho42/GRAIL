# Preregistration: reinforcement shrinkage of the rule scores

Registered before the experiment was run. The commit that adds this file is the timestamp; the
producer and its result come in later commits, and this document is not edited to match them.

## The mechanism under test

The exhaustive arm applies the whole bank of 7,581 templates. 4,271 of them are never a positive
label in training and a further 1,520 are positive for exactly one substrate, so three quarters of
the bank carries a score estimated from zero or one example. Those scores are noise, and a noisy
high score displaces a correct candidate produced by a well-supported template past the output
budget. The interactive arm applies only the 30 highest-scoring templates, which are less exposed
to that noise, and the manuscript's own per-class table shows it beating the exhaustive arm on
exactly the classes where such displacement would bite.

Shrinkage pulls each per-rule score toward the frequency prior in proportion to how little the
rule was seen:

    score'(rule) = n/(n + lambda) * score(rule) + lambda/(n + lambda) * prior(rule)

where n is the rule's count of training positives (results/rule_train_positives from the label
cache 5a3f22a1e5962bbb, which reproduces the published never/eq1/ge2 counts), score is the deployed
generator score, and prior is sigmoid(rule_prior_logits) from the checkpoint. At n -> infinity the
score is unchanged; at n = 0 it becomes the prior. Shrunk scores are then aggregated by the
deployed noisy-or and ranked by the deployed fusion, so nothing else changes.

lambda is swept on the validation draw and the value maximising validation recall@15 is spent once
on the comparison set. Selection is on validation, never on the population the result is reported
on.

## The prediction

Group G is named before the run, from the deployed noisy-or per-class table
(results/error_by_chemistry.json), as the classes where the interactive arm already beats the
exhaustive arm at k=15:

    G = { sulfation, hydrolysis, deamination, isomerisation (no formula change), demethylation }

**Primary, directional (family m = 1).** The mean gain in exhaustive-arm recall@15 that shrinkage
delivers on G exceeds the mean gain on the complement:

    mean_{c in G} [ recall@15_shrunk(c) - recall@15_deployed(c) ]
        >  mean_{c not in G} [ recall@15_shrunk(c) - recall@15_deployed(c) ]

The sign is fixed here: shrinkage is predicted to help G more than the rest. This is content-ful
and not a tautology. G is identified by a different quantity (interactive vs exhaustive) than the
one shrinkage moves (a score correction by training support); the mechanism could raise the classes
where the prior happens to be strong, or raise everything uniformly, and either would refute this.

**Falsification.** If mean_gain(G) <= mean_gain(not G), the mechanism is rejected: the correction
does not land where the displacement account says it should.

**Guardrail, separate from the mechanism test.** Shrinkage at the validation-selected lambda must
not lower the exhaustive arm's headline recall@15 on the comparison set below the deployed 0.5353.
A correction that buys the per-class pattern at the cost of the headline is reported as a failure
of the intervention even if the directional test passes.

**Reported regardless.** The full per-class delta table, the lambda sweep on both populations, the
headline at every budget, and the transfer gap between what validation promised and what the
comparison set delivered. A prediction reported only when it succeeds is not a preregistration.

## Result (added after the run; the prediction above is unchanged)

**Rejected. The mechanism is dead at every strength.**

lambda was swept on the validation draw and the best value was lambda = 0, i.e. no shrinkage:
validation recall@15 is 0.4748 at lambda = 0 and falls to 0.41-0.47 for every lambda > 0. The
discipline therefore spent lambda = 0 on the comparison set, and every per-class delta is zero.

The prediction about group G could not be tested content-fully, because the intervention collapsed
to a no-op rather than because it raised the wrong classes. So a comparison-set sweep was run for
diagnosis only -- never for selection -- to separate "the mechanism is dead" from "the discipline
would not take it". It is dead: on the comparison set too, lambda = 0 is the best value
(recall@15 0.5353), and shrinkage only lowers it (0.4737 at lambda = 1, recovering to 0.5128 at
lambda = 64 as the score approaches the prior but never reaching lambda = 0). The comparison-best
lambda is 0, so there was nothing for the discipline to fail to take.

**Reading.** Pulling a rule's score toward the frequency prior in proportion to its training
support does not help, at any strength, on either population. The displacement account it was built
on -- that low-support mined rules carry noisy high scores that push correct candidates past the
budget -- is not refuted as a description of the bank, but it is refuted as something this
correction fixes: the deployed cap at 100 and the reciprocal-rank fusion already absorb whatever
those rules do, and shifting their scores toward the prior only moves good candidates with the bad.

The guardrail passed trivially, because lambda = 0 is a no-op. That is not the intervention
succeeding; it is the intervention declining to fire.

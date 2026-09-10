# Preregistration: support-gated rule identity

Registered before training. The commit that adds this file is the timestamp. The implementation
(f3a3eeb) is in place and off by default; the intervention is a positive id_gate_lambda, which
needs retraining, and this document fixes what that training must show before it is run.

## The mechanism under test

The rule representation is graph_encoded + id_embedding + meta. A variance decomposition put 82%
of it in the per-rule id. A rule with one training positive fits its id to that one example, and
zeroing the id at inference gains +0.037 at a budget of one and +0.027 at five while losing 0.021
at thirty (results/ablate_id_embedding.json): the id is a frequency lookup that helps frequent
rules at depth and buries rare rules' correct candidates at the head.

The gate makes the id contribute n/(n+lambda), n the rule's training positive count. A rare rule
then leans on the template's chemistry (graph + meta); a frequent one keeps its lookup. The
frequency signal is not lost when the id is gated, because it also lives, separately, in
rule_prior_logits.

This is not the score-level shrinkage this repository rejected (results/shrinkage_rerank.json).
That corrected a trained model's scores at inference and did nothing at any strength. This changes
what the graph encoder is *trained* to carry: with the id gated off for rare rules, the chemistry
must learn to score them, which inference-time correction cannot induce.

## The training design

Two runs under identical conditions -- same seed, same data, same schedule, same early-stopping
rule -- differing only in id_gate_lambda:

  lambda = 0    the retrained baseline (gate off)
  lambda = L*   the intervention

The baseline is retrained rather than read from the deployed checkpoint, so the comparison is the
gate alone and not the gate confounded with a different training run. If more than one L is
trained, L* is chosen on the validation draw (recall@5) and spent once on the comparison set; if
compute allows only one positive L, it is fixed here at L = 8 (the value at which a rule needs
about eight training positives to recover half its id) and reported whether or not it wins.

## The prediction

**Primary, directional.** On the comparison set, the exhaustive arm's recall@5 under the gate
exceeds its recall@5 at lambda = 0, both retrained identically:

    recall@5(lambda=L*) > recall@5(lambda=0)

The head of the list is where the ablation said the id hurts, so the head is where the gate must
help. The sign is fixed here.

**Falsification.** If recall@5(L*) <= recall@5(lambda=0), the gate does not help the head and the
mechanism is rejected, exactly as the score-level version was.

**Mechanism check (secondary).** The share of the rule representation's variance carried by the
graph encoding rises from the 17% it holds at lambda = 0. If recall@5 rose but the graph share did
not, the gain is not coming from the chemistry learning to score rare rules, and the reading is
wrong even if the number is right.

**Guardrail, separate from the mechanism.** recall@15 under the gate does not fall below the
lambda = 0 baseline's by an interval excluding zero. The gate is meant to move the head without
surrendering the depth; a gate that buys the head by losing the budget the headline is read at is
reported as a failure of the intervention even if the primary test passes.

**Reported regardless.** Both arms at every budget under both lambdas, the per-class deltas, the
variance decomposition before and after, and -- if a lambda sweep is run -- the transfer gap
between the validation choice and the comparison result. A prediction reported only on success is
not a preregistration.

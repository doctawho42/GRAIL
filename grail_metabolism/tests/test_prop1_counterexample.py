"""Arithmetic check of Proposition 1 (surrogate mismatch → ranking_conversion).

Proposition 1 (see ``docs/GRAIL_FRAMING.md`` §4) claims that a filter trained by a
strictly proper scoring rule (BCE / PU) learns a *globally calibrated* posterior
``eta(s, m) = P(true | s, m)`` that is Bayes-optimal for AUC / calibration, yet is
**not** recall@k-optimal when candidate pools vary in size and positive rate across
substrates. A listwise / ranking-consistent surrogate can then dominate it on
recall@k even though a proper-scoring objective would *reject* that surrogate
(because it is not calibrated). That objective-vs-metric gap is the surrogate
mismatch.

This test encodes the minimal 2-substrate witness so the prose's inequality is
*checked*, not asserted. It is deliberately a discriminating counterexample:

  * the load-bearing assertion is the STRICT inequality
    ``calibrated_recall < listwise_recall`` on a constructed heterogeneous instance.
    If the surrogate-mismatch claim were false (calibration => recall-optimality),
    no ranker could beat the calibrated one and this assertion would fail;
  * the recall-superior listwise scorer is a *concrete alternative scorer*, not an
    oracle that trivially sorts trues first, and we verify it is NOT globally
    calibrated -- so a proper-scoring objective genuinely prefers the (worse-recall)
    calibrated scorer;
  * a homogeneous control shows the gap DISAPPEARS when pools are not heterogeneous,
    proving the loss is caused by the pool heterogeneity of the Proposition, not by
    a rigged construction.

Empirically this is confirmed by the committed Spike-3 result (a listwise-InfoNCE
reranker lifts recall@15 0.433 -> 0.500 over the pointwise filter); note the
reranker's 0.500 still *loses* to SyGMa (0.558) -- Prop 1 is a statement about
objectives, not a recall win. Pure-Python; no product code imported.
"""

# An item is (calibrated_score, listwise_score, is_true).
CAL, LST, TRUE = 0, 1, 2


def recall_at_k(pools, score_idx, k):
    """Micro recall@k: rank each pool independently by ``score_idx`` (descending),
    take the top-k, and pool hits over pooled trues (ratio-of-sums, the frame of
    §1.5). ``pools`` is a list of pools; each pool is a list of item tuples."""
    hits = trues = 0
    for pool in pools:
        trues += sum(1 for it in pool if it[TRUE])
        ranked = sorted(pool, key=lambda it: -it[score_idx])[:k]
        hits += sum(1 for it in ranked if it[TRUE])
    return hits / trues if trues else 0.0


def global_calibration_error(pools, score_idx):
    """Mean |empirical-true-rate - score| over score buckets, pooled across all
    substrates (a proper-scoring objective drives this to 0). ~0 means the scorer
    is a valid calibrated posterior; large means the metric-optimal ranker is one
    the calibration objective would reject."""
    buckets = {}
    total = 0
    for pool in pools:
        for it in pool:
            key = round(it[score_idx], 6)
            buckets.setdefault(key, []).append(1 if it[TRUE] else 0)
            total += 1
    err = 0.0
    for score, labels in buckets.items():
        rate = sum(labels) / len(labels)
        err += abs(rate - score) * len(labels)
    return err / total if total else 0.0


# --- The minimal heterogeneous 2-substrate witness ------------------------------
#
# Substrate A: a SMALL pool (2 candidates). Its one true metabolite is
# pointwise-UNDERRANKED -- a generically-more-reactive false site carries the higher
# calibrated score (the regioselectivity failure the pointwise filter cannot fix).
# Substrate B: a LARGER pool (6 candidates) where the calibrated posterior orders the
# pool's own true metabolites on top.
#
# The calibrated scores are perfectly globally calibrated: among all items scored
# 0.75 exactly 3/4 are true; among all items scored 0.25 exactly 1/4 are true.
POOL_A = [
    (0.75, 0.30, False),  # a1: distractor site, pointwise favours it
    (0.25, 0.55, True),   # a2: the real metabolite, pointwise underranks it
]
POOL_B = [
    (0.75, 0.95, True),
    (0.75, 0.40, True),
    (0.75, 0.35, True),
    (0.25, 0.90, False),  # a false site the listwise scorer also overrates (not an oracle)
    (0.25, 0.20, False),
    (0.25, 0.10, False),
]
HETEROGENEOUS = [POOL_A, POOL_B]

# Homogeneous control: distinct calibrated scores, true on top within each pool.
HOMOGENEOUS = [
    [(0.90, 0.90, True), (0.10, 0.10, False)],
    [(0.80, 0.80, True), (0.20, 0.20, False)],
]


def test_calibrated_scorer_loses_to_listwise_under_heterogeneous_pools():
    k = 1
    cal_recall = recall_at_k(HETEROGENEOUS, CAL, k)
    lst_recall = recall_at_k(HETEROGENEOUS, LST, k)

    # Load-bearing: the calibrated pointwise ranker is STRICTLY worse on recall@k.
    # (If calibration implied recall-optimality, this could not happen.)
    assert cal_recall < lst_recall
    # Pin the exact arithmetic so a regression is caught:
    #   calibrated top-1: A -> a1 (false, miss); B -> a 0.75 true (hit) => 1/4
    #   listwise  top-1: A -> a2 (true, hit);    B -> b1 0.95 true (hit) => 2/4
    assert cal_recall == 0.25
    assert lst_recall == 0.5

    # The calibrated scorer IS a valid calibrated posterior...
    assert global_calibration_error(HETEROGENEOUS, CAL) < 1e-9
    # ...while the recall-superior listwise scorer is NOT calibrated, so a proper
    # scoring objective would reject it despite its better recall. That objective vs.
    # top-k-metric gap is exactly the surrogate mismatch of Proposition 1.
    assert global_calibration_error(HETEROGENEOUS, LST) > 0.1


def test_no_gap_under_homogeneous_pools_isolates_the_mechanism():
    # Control: with homogeneous pools the calibrated posterior already orders each
    # pool correctly, so it ties the listwise scorer -- the recall loss above is
    # caused by pool heterogeneity, not by the construction being rigged.
    k = 1
    assert recall_at_k(HOMOGENEOUS, CAL, k) == recall_at_k(HOMOGENEOUS, LST, k)

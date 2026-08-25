"""The one implementation of the fusion rule H7 registers.

    score(c) = 1/(60 + rank_filter(c)) + 1/(60 + rank_generator(c))

Ranks are 1-based competition ranks: tied scores share the lower rank. The alternative reading,
a candidate's position in a sorted list, makes the score depend on how the sort broke ties, so
two runs over the same pool can disagree. They did: the same 291 substrates gave 0.4992 under
positions and 0.5023 under ranks. A registered formula must have one implementation, and this is
it -- import from here rather than writing it again.
"""
from __future__ import annotations

RRF_K = 60  # Cormack, Clarke and Buettcher 2009; not tuned


def competition_ranks(items, score):
    """1-based competition ranks, descending by `score`; tied items share the lower rank."""
    order = sorted(range(len(items)), key=lambda i: -score(items[i]))
    out, prev, cur = [0] * len(items), None, 1
    for pos, i in enumerate(order, 1):
        v = score(items[i])
        if v != prev:
            prev, cur = v, pos
        out[i] = cur
    return out


def rrf_order(cands, k=RRF_K, filter_key="filter", generator_key="generator"):
    """Order a pool by reciprocal rank fusion of its two component scores."""
    rf = competition_ranks(cands, lambda c: c[filter_key])
    rg = competition_ranks(cands, lambda c: c[generator_key])
    idx = sorted(range(len(cands)), key=lambda i: -(1.0 / (k + rf[i]) + 1.0 / (k + rg[i])))
    return [cands[i] for i in idx]

"""The one implementation of the release ranking (H7 fusion, H9 cap, P2 dedup order).

Order of operations, which is not interchangeable: dedup by tautomer key in descending order of the
product of the two component scores, then cap by generator score, then fuse by reciprocal rank,
then drop the parent. Promoted here from the analysis scripts so there is a single implementation
the deploy and the measurements share.
"""
from __future__ import annotations

from typing import Callable, Sequence

RRF_K = 60   # Cormack, Clarke and Buettcher 2009; not tuned
CAP = 100    # H9


def competition_ranks(items: Sequence, score: Callable) -> list:
    """1-based competition ranks, descending by `score`; tied items share the lower rank."""
    order = sorted(range(len(items)), key=lambda i: -score(items[i]))
    out, prev, cur = [0] * len(items), None, 1
    for pos, i in enumerate(order, 1):
        v = score(items[i])
        if v != prev:
            prev, cur = v, pos
        out[i] = cur
    return out


def reciprocal_rank_fusion(cands, k=RRF_K, filter_key="filter", generator_key="generator"):
    """Order candidate dicts by reciprocal rank fusion of their two component scores."""
    rf = competition_ranks(cands, lambda c: c[filter_key])
    rg = competition_ranks(cands, lambda c: c[generator_key])
    idx = sorted(range(len(cands)), key=lambda i: -(1.0 / (k + rf[i]) + 1.0 / (k + rg[i])))
    return [cands[i] for i in idx]


def release_order(pool, self_key, cap=CAP, rrf_k=RRF_K):
    """The deployed order, returning the ordered candidate dicts (parent dropped)."""
    cands = sorted(pool, key=lambda c: -(c["filter"] * c["generator"]))
    seen, dedup = set(), []
    for c in cands:
        if not c["key"] or c["key"] in seen:
            continue
        seen.add(c["key"])
        dedup.append(c)
    keep = sorted(dedup, key=lambda c: -c["generator"])[:cap]
    return [c for c in reciprocal_rank_fusion(keep, k=rrf_k) if c["key"] != self_key]

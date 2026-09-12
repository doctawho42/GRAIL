"""Re-exports the one fusion implementation, now in the library. Import from here or from
grail_metabolism.model.ranking; they are the same functions."""
from grail_metabolism.model.ranking import RRF_K, competition_ranks, reciprocal_rank_fusion


def rrf_order(cands, k=RRF_K, filter_key="filter", generator_key="generator"):
    return reciprocal_rank_fusion(cands, k=k, filter_key=filter_key, generator_key=generator_key)

from __future__ import annotations

from typing import Dict

from ..config import FilterConfig, GeneratorConfig
from ..model.filter import Filter, GATv2Filter, GCNFilter, GINFilter, MolPathFilter, MorganOnlyFilter
from ..model.generator import Generator
FILTER_TYPES = {
    "Filter": Filter,
    "GATv2Filter": GATv2Filter,
    "GCNFilter": GCNFilter,
    "GINFilter": GINFilter,
    "MolPathFilter": MolPathFilter,
    "MorganOnlyFilter": MorganOnlyFilter,
}


def build_rule_dict(rules: list[str]) -> Dict[str, object]:
    return {rule: Generator._cached_rule_graph(rule) for rule in rules}


def build_generator(config: GeneratorConfig, rules: list[str]) -> Generator:
    rule_dict = build_rule_dict(rules)
    return Generator(
        rule_dict=rule_dict,
        in_channels=config.in_channels,
        edge_dim=config.edge_dim,
        arg_vec=config.hidden_dims,
        rp_arg_vec=config.rule_hidden_dims,
        projection_dim=config.projection_dim,
        use_maccs_pretraining=config.use_maccs_pretraining,
        scoring=config.scoring,
        conv_kind=config.conv_kind,
        top_k=config.top_k,
        use_fingerprint=config.use_fingerprint,
        rank_weight=config.rank_weight,
        ranking_margin=config.ranking_margin,
        unlabeled_weight=config.unlabeled_weight,
        prior_strength=config.prior_strength,
        use_applicability_mask=config.use_applicability_mask,
        applicability_penalty=config.applicability_penalty,
        candidate_aggregation=config.candidate_aggregation,
    )


def build_filter(config: FilterConfig):
    model_cls = FILTER_TYPES[config.model_type]
    kwargs = {}
    if model_cls in {Filter, GATv2Filter, GCNFilter, GINFilter}:
        kwargs = {
            "use_graph": config.use_graph,
            "use_fingerprint": config.use_fingerprint,
            "dropout": config.dropout,
        }
        if model_cls is Filter:
            kwargs["conv_kind"] = config.conv_kind
    elif model_cls is MolPathFilter:
        kwargs = {
            "molpath_cutoff": config.molpath_cutoff,
            "molpath_y": config.molpath_y,
            "molpath_hidden": config.molpath_hidden,
        }
    return model_cls(
        in_channels=config.in_channels,
        edge_dim=config.edge_dim,
        arg_vec=config.hidden_dims,
        mode=config.mode,
        **kwargs,
    )

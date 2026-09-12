from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict

from ..config import FilterConfig, GeneratorConfig, SoMConfig
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


RULE_GRAPH_CACHE_ENV = "GRAIL_RULE_GRAPH_CACHE"
RULE_GRAPH_CACHE_DIR_ENV = "GRAIL_RULE_GRAPH_CACHE_DIR"
RULE_GRAPH_CACHE_FORMAT = 1  # bump when from_rule's output layout changes


def _rule_graph_cache_path(rules: list[str]) -> Path:
    """Where the built graphs for exactly this rule list live.

    Keyed by the digest of the rule strings in order, so a different bank, a reordered bank or an
    edited template cannot hit a stale entry: the key IS the input. The directory can be moved
    with GRAIL_RULE_GRAPH_CACHE_DIR, because an installed tree is not always writable and a cache
    nobody can write is a cache that never helps.
    """
    digest = hashlib.sha256("\n".join(rules).encode("utf-8")).hexdigest()[:32]
    override = os.environ.get(RULE_GRAPH_CACHE_DIR_ENV)
    root = (Path(override) if override
            else Path(__file__).resolve().parents[2] / "artifacts" / "rulecache")
    return root / f"v{RULE_GRAPH_CACHE_FORMAT}_{len(rules)}_{digest}.pt"


def build_rule_dict(rules: list[str]) -> Dict[str, object]:
    """Rule string -> rule graph, optionally read from a disk cache.

    Building the graphs is most of the cost of constructing a generator over a large bank (about a
    minute for the released 6,970 templates), and it is pure: the graph depends on the template
    string and nothing else. For a deployment that starts a process per request that minute is the
    dominant cost, so the graphs can be cached to disk and read back in under a second.

    The cache is OFF unless GRAIL_RULE_GRAPH_CACHE is set, because every measurement in this
    repository constructs models through this function and none of them should start reading a
    file that an earlier run wrote. Cache problems of any kind fall back to building, so the
    worst case is the current cost and never a wrong graph.
    """
    if not os.environ.get(RULE_GRAPH_CACHE_ENV):
        return {rule: Generator._cached_rule_graph(rule) for rule in rules}

    import torch

    path = _rule_graph_cache_path(rules)
    try:
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            # The digest is in the filename, but a truncated or half-written file can still
            # exist, so what was loaded is checked against what was asked for.
            if (payload.get("format") == RULE_GRAPH_CACHE_FORMAT
                    and payload.get("rules") == rules
                    and len(payload.get("graphs", ())) == len(rules)):
                # Seed the in-process cache with the loaded graphs, then read back through the
                # same accessor the uncached path uses, so both paths hand out clones of one
                # cached graph rather than the loaded object in one case and a clone in the other.
                for rule, graph in zip(rules, payload["graphs"]):
                    Generator._cached_rule_graph(rule, existing=graph)
                return {rule: Generator._cached_rule_graph(rule) for rule in rules}
    except Exception:
        pass  # any cache trouble is a cache miss, never an error

    built = {rule: Generator._cached_rule_graph(rule) for rule in rules}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a private temporary name and rename, so a concurrent reader sees either the
        # previous file or the complete new one and never a partial write.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        torch.save({"format": RULE_GRAPH_CACHE_FORMAT, "rules": rules,
                    "graphs": [built[rule] for rule in rules]}, tmp)
        os.replace(tmp, path)
    except Exception:
        pass  # a bank we cannot cache is still a bank we can use

    return built


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
        propensity_weighting=config.propensity_weighting,
        propensity_a=config.propensity_a,
        propensity_b=config.propensity_b,
        prior_strength=config.prior_strength,
        use_applicability_mask=config.use_applicability_mask,
        applicability_penalty=config.applicability_penalty,
        candidate_aggregation=config.candidate_aggregation,
        id_gate_lambda=getattr(config, "id_gate_lambda", 0.0),
    )


def build_filter(config: FilterConfig):
    model_cls = FILTER_TYPES[config.model_type]
    kwargs = {}
    if model_cls in {Filter, GATv2Filter, GCNFilter, GINFilter}:
        kwargs = {
            "use_graph": config.use_graph,
            "use_fingerprint": config.use_fingerprint,
            "dropout": config.dropout,
            "difference_readout": getattr(config, "difference_readout", False),
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


def build_som(config: SoMConfig):
    """Single construction path for the SoM regioselectivity prior (model.som.SoMPredictor)."""
    from ..model.som import SoMPredictor
    from ..utils.transform import EDGE_DIM, SINGLE_NODE_DIM

    return SoMPredictor(
        in_channels=SINGLE_NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dims=config.hidden_dims,
        out_dim=config.out_dim,
        conv_kind=config.conv_kind,
        dropout=config.dropout,
    )

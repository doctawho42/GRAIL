from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdChemReactions
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from ._graph import GraphEncoder
from .wrapper import GGenerator
from ..utils.preparation import MolFrame, _normalize_smiles_cached, generate_vectors, safe_run_reactants
from ..utils.transform import EDGE_DIM, FINGERPRINT_DIM, SINGLE_NODE_DIM, from_rdmol, from_rule


def _pad_dims(values: Sequence[int], size: int, default: int) -> List[int]:
    padded = list(values[:size])
    while len(padded) < size:
        padded.append(default)
    return padded


def _strip_grouping(smarts: str) -> str:
    text = str(smarts).strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1]
    return text


def _split_rule(rule: str) -> Tuple[str, str]:
    left, right = rule.split(">>", 1)
    return _strip_grouping(left), _strip_grouping(right)


def _safe_logit(probabilities: torch.Tensor) -> torch.Tensor:
    probs = probabilities.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(probs / (1.0 - probs))


def _rule_metadata(rule: str) -> List[float]:
    sub_smarts, prod_smarts = _split_rule(rule)
    sub = Chem.MolFromSmarts(sub_smarts)
    prod = Chem.MolFromSmarts(prod_smarts)
    if sub is None or prod is None:
        return [0.0] * 6
    sub_atoms = float(sub.GetNumAtoms())
    prod_atoms = float(prod.GetNumAtoms())
    mapped_sub = float(sum(atom.HasProp("molAtomMapNumber") for atom in sub.GetAtoms()))
    mapped_prod = float(sum(atom.HasProp("molAtomMapNumber") for atom in prod.GetAtoms()))
    aromatic_sub = float(sum(atom.GetIsAromatic() for atom in sub.GetAtoms()))
    aromatic_prod = float(sum(atom.GetIsAromatic() for atom in prod.GetAtoms()))
    return [
        sub_atoms / 32.0,
        prod_atoms / 32.0,
        mapped_sub / max(sub_atoms, 1.0),
        mapped_prod / max(prod_atoms, 1.0),
        abs(sub_atoms - prod_atoms) / 16.0,
        (aromatic_sub + aromatic_prod) / max(sub_atoms + prod_atoms, 1.0),
    ]


def _timeout_buffer(last_epoch_seconds: float) -> float:
    return max(30.0, min(300.0, 0.2 * max(last_epoch_seconds, 0.0)))


class RuleParse(nn.Module):
    def __init__(
        self,
        rule_dict: Dict[str, Data],
        arg_vec: Sequence[int],
        embedding_dim: int = 128,
        use_molpath: bool = False,
        molpath_hidden: Optional[int] = None,
        molpath_cutoff: Optional[int] = None,
        molpath_y: Optional[float] = None,
    ) -> None:
        super().__init__()
        del use_molpath, molpath_hidden, molpath_cutoff, molpath_y
        hidden = _pad_dims(arg_vec, 3, embedding_dim)
        self.rule_keys = list(rule_dict.keys())
        self.rule_graphs = [rule_dict[key] for key in self.rule_keys]
        self.rule_batch = Batch.from_data_list(self.rule_graphs) if self.rule_graphs else None
        self.encoder = GraphEncoder(
            in_channels=SINGLE_NODE_DIM,
            edge_dim=EDGE_DIM,
            hidden_dims=hidden[:2],
            out_dim=embedding_dim,
            conv_kind="gatv2",
            dropout=0.1,
        )
        meta = [_rule_metadata(rule) for rule in self.rule_keys]
        meta_tensor = torch.tensor(meta, dtype=torch.float32) if meta else torch.empty((0, 6), dtype=torch.float32)
        self.register_buffer("rule_meta", meta_tensor, persistent=False)
        self.id_embedding = nn.Embedding(max(len(self.rule_keys), 1), embedding_dim)
        self.meta_encoder = nn.Sequential(
            nn.Linear(6, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.rule_batch is None:
            return torch.empty((0, self.id_embedding.embedding_dim), device=device)
        rule_batch = self.rule_batch.to(device)
        encoded = self.encoder(rule_batch)
        ids = self.id_embedding.weight[: encoded.size(0)]
        meta = self.meta_encoder(self.rule_meta.to(device))
        return self.norm(encoded + ids + meta)


class MoleculeAugmentor:
    @staticmethod
    def atom_masking(graph: Data, mask_ratio: float = 0.15) -> Data:
        masked = graph.clone()
        num_nodes = masked.x.size(0)
        if num_nodes <= 1:
            return masked
        num_mask = max(1, min(num_nodes - 1, int(num_nodes * mask_ratio)))
        mask_indices = torch.randperm(num_nodes)[:num_mask]
        masked.x[mask_indices] = 0.0
        return masked

    @staticmethod
    def edge_masking(graph: Data, mask_ratio: float = 0.15) -> Data:
        masked = graph.clone()
        num_edges = masked.edge_index.size(1)
        if num_edges == 0:
            return masked
        num_keep = max(1, int(num_edges * (1.0 - mask_ratio)))
        keep = torch.randperm(num_edges)[:num_keep]
        masked.edge_index = masked.edge_index[:, keep]
        masked.edge_attr = masked.edge_attr[keep]
        return masked

    @classmethod
    def augment_batch(cls, batch: Batch) -> Batch:
        graphs = batch.to_data_list()
        augmented = []
        for graph in graphs:
            transformed = cls.atom_masking(graph) if np.random.rand() < 0.5 else cls.edge_masking(graph)
            augmented.append(transformed)
        return Batch.from_data_list(augmented)


def get_maccs_smarts() -> List[str]:
    smarts = []
    for index in range(1, 167):
        try:
            pattern = MACCSkeys.smartsPatts[index]
            smarts.append(pattern[0] if pattern else "")
        except Exception:
            smarts.append("")
    return smarts


class GeneratorObjective(nn.Module):
    def __init__(self, rank_weight: float = 0.25, ranking_margin: float = 0.45, unlabeled_weight: float = 1.0) -> None:
        super().__init__()
        self.rank_weight = float(rank_weight)
        self.ranking_margin = float(ranking_margin)
        # Weight applied to applicable-but-unobserved (target=0) rules. Metabolite
        # annotations are incomplete, so these are positive-unlabeled, not true
        # negatives. unlabeled_weight < 1 down-weights them so the generator is not
        # punished at full strength for proposing plausible-but-unannotated products,
        # which otherwise suppresses recall.
        self.unlabeled_weight = float(unlabeled_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        pos_weight: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise RuntimeError(f"Shape mismatch: {logits.shape} != {targets.shape}")
        weights = pos_weight.view(1, -1).expand_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=weights)
        probabilities = torch.sigmoid(logits)
        focal = torch.where(targets > 0.5, (1.0 - probabilities).pow(2.0), probabilities.pow(1.0))
        # Down-weight unobserved-applicable negatives (PU-aware).
        label_weight = torch.where(
            targets > 0.5,
            torch.ones_like(targets),
            torch.full_like(targets, self.unlabeled_weight),
        )
        effective = mask * label_weight
        masked = (bce * (1.0 + focal) * effective).sum() / effective.sum().clamp_min(1.0)

        positive_mask = (targets > 0.5) & (mask > 0.0)
        negative_mask = (targets <= 0.5) & (mask > 0.0)
        if self.rank_weight <= 0.0 or not positive_mask.any() or not negative_mask.any():
            return masked

        pos_logits = logits.masked_fill(~positive_mask, float("inf"))
        neg_logits = logits.masked_fill(~negative_mask, float("-inf"))
        hardest_positive = pos_logits.min(dim=1).values
        hardest_negative = neg_logits.max(dim=1).values
        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if valid.any():
            ranking = torch.relu(self.ranking_margin - hardest_positive[valid] + hardest_negative[valid]).mean()
            return masked + self.rank_weight * ranking
        return masked


class Generator(GGenerator):
    _rule_graph_cache: Dict[str, Data] = {}

    def __init__(
        self,
        rule_dict: Dict[str, Data],
        in_channels: int,
        edge_dim: int,
        arg_vec: Optional[Sequence[int]] = None,
        rp_arg_vec: Optional[Sequence[int]] = None,
        projection_dim: int = 128,
        use_maccs_pretraining: bool = False,
        scoring: Literal["bilinear", "dot", "mlp", "retrieval"] = "retrieval",
        conv_kind: Literal["gatv2", "gcn", "gin"] = "gatv2",
        top_k: int = 15,
        use_molpath: bool = False,
        molpath_hidden: Optional[int] = None,
        molpath_cutoff: Optional[int] = None,
        molpath_y: Optional[float] = None,
        use_fingerprint: bool = True,
        rank_weight: float = 0.25,
        ranking_margin: float = 0.45,
        unlabeled_weight: float = 1.0,
        prior_strength: float = 0.4,
        use_applicability_mask: bool = True,
        applicability_penalty: float = 7.5,
        candidate_aggregation: Literal["max", "mean", "noisy_or", "hybrid"] = "noisy_or",
    ) -> None:
        super().__init__()
        self._prime_rule_graph_cache(rule_dict)
        self.rules = {rule: graph for rule, graph in zip(rule_dict.keys(), self._get_rule_graphs(list(rule_dict.keys())))}
        self.rule_names = list(self.rules.keys())
        self.num_rules = len(self.rule_names)
        hidden = _pad_dims(arg_vec or [128, 256], 2, 128)
        rule_hidden = _pad_dims(rp_arg_vec or [128, 128, 128], 3, 128)
        self.scoring = scoring
        self.embed_dim = projection_dim
        self.use_fingerprint = use_fingerprint
        self.prior_strength = float(prior_strength)
        self.use_applicability_mask = use_applicability_mask
        self.applicability_penalty = float(applicability_penalty)
        self.candidate_aggregation = candidate_aggregation
        self.default_top_k = max(1, int(top_k))

        self.parser = RuleParse(
            rule_dict=self.rules,
            arg_vec=rule_hidden,
            embedding_dim=projection_dim,
            use_molpath=use_molpath,
            molpath_hidden=molpath_hidden,
            molpath_cutoff=molpath_cutoff,
            molpath_y=molpath_y,
        )
        # Inference cache for the encoded rule bank (see _rule_embeddings). The rule graphs
        # and encoder weights are fixed during inference, so the ~7.5k-rule encode is done
        # once and reused across substrates instead of re-run every forward (the dominant
        # per-substrate cost). Invalidated on any grad-enabled (training) forward.
        self._rule_embedding_cache: Optional[torch.Tensor] = None
        self.substrate_encoder = GraphEncoder(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_dims=hidden,
            out_dim=projection_dim,
            conv_kind=conv_kind,
            dropout=0.1,
        )
        self.graph_norm = nn.LayerNorm(projection_dim)
        if use_fingerprint:
            self.fp_encoder = nn.Sequential(
                nn.Linear(FINGERPRINT_DIM, projection_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(projection_dim, projection_dim),
            )
            self.substrate_fusion = nn.Sequential(
                nn.Linear(2 * projection_dim, projection_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(projection_dim, projection_dim),
            )
        else:
            self.fp_encoder = None
            self.substrate_fusion = None
        self.substrate_norm = nn.LayerNorm(projection_dim)
        self.node_key = nn.Linear(projection_dim, projection_dim, bias=False)
        self.node_value = nn.Linear(projection_dim, projection_dim, bias=False)
        self.rule_query = nn.Linear(projection_dim, projection_dim, bias=False)

        pair_dim = (6 * projection_dim) + 2
        self.rule_mlp = nn.Sequential(
            nn.Linear(pair_dim, 2 * projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * projection_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, 1),
        )
        self.bilinear = nn.Parameter(torch.empty(projection_dim, projection_dim))
        self.bias = nn.Parameter(torch.zeros(self.num_rules))
        self.global_scale = nn.Parameter(torch.tensor(1.0))
        self.local_scale = nn.Parameter(torch.tensor(0.7))
        self.match_scale = nn.Parameter(torch.tensor(0.25))
        self.dropout = nn.Dropout(0.1)
        self.projection_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )
        self.atom_predictor = nn.Linear(projection_dim, SINGLE_NODE_DIM)
        self.maccs_head = nn.Linear(projection_dim, 166)
        # Normalization for products enumerated at generation time. "canonical" is ~5x
        # faster than tautomer "standardize" and must match the dataset normalization the
        # model was trained on (set to "canonical" when trained with standardize=False).
        self.gen_normalization: Literal["standardize", "canonical"] = "standardize"
        # GFlowNet STOP-action head: P(stop | molecule) competes with the rule actions.
        self.stop_head = nn.Linear(projection_dim, 1)
        self.use_maccs_pretraining = use_maccs_pretraining
        self.pretrained = False
        self.objective = GeneratorObjective(rank_weight=rank_weight, ranking_margin=ranking_margin, unlabeled_weight=unlabeled_weight)
        self.rule_patterns = [self._compile_reactant_pattern(rule) for rule in self.rule_names]
        self.rule_reactions = [self._compile_reaction(rule) for rule in self.rule_names]
        self._applicability_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.calibrated_threshold: Optional[float] = None
        self.register_buffer("rule_prior_logits", torch.zeros(self.num_rules), persistent=False)
        self.register_buffer("pos_weight", torch.ones(self.num_rules), persistent=False)
        nn.init.xavier_uniform_(self.bilinear)

    @classmethod
    def _placeholder_rule_graph(cls) -> Data:
        return Data(
            x=torch.zeros((1, SINGLE_NODE_DIM), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, EDGE_DIM), dtype=torch.float32),
        )

    @classmethod
    def _cached_rule_graph(cls, rule: str, existing: Optional[Data] = None) -> Data:
        if rule not in cls._rule_graph_cache:
            graph = existing
            if graph is None:
                try:
                    graph = from_rule(rule)
                except Exception:
                    graph = None
            if graph is None:
                graph = cls._placeholder_rule_graph()
            cls._rule_graph_cache[rule] = graph
        return cls._rule_graph_cache[rule].clone()

    def _prime_rule_graph_cache(self, rule_dict: Dict[str, Data]) -> None:
        for rule, graph in rule_dict.items():
            self._cached_rule_graph(rule, existing=graph)

    def _get_rule_graphs(self, rules: List[str]) -> List[Data]:
        return [self._cached_rule_graph(rule) for rule in rules]

    @staticmethod
    def _compile_reactant_pattern(rule: str) -> Optional[Chem.Mol]:
        try:
            reactant_smarts, _ = _split_rule(rule)
            return Chem.MolFromSmarts(reactant_smarts)
        except Exception:
            return None

    @staticmethod
    def _compile_reaction(rule: str):
        try:
            return rdChemReactions.ReactionFromSmarts(rule)
        except Exception:
            return None

    def _rule_embeddings(self, device: torch.device) -> torch.Tensor:
        self.parser.to(device)
        if torch.is_grad_enabled():
            # Training / differentiable (GFlowNet) path: encoder weights change each step and
            # gradients must flow, so recompute. Also drop any inference cache so a later
            # no-grad pass cannot reuse embeddings from before a weight update.
            self._rule_embedding_cache = None
            return self.parser().to(device)
        # Inference (no_grad): the rule bank and encoder weights are fixed across substrates,
        # so encode the rule graphs ONCE and reuse. This is the dominant per-substrate cost.
        # score_rules runs this in eval mode, so dropout is off and parser() is deterministic
        # -- the single cached pass equals re-encoding every substrate.
        cache = self._rule_embedding_cache
        if cache is None or cache.device != device:
            cache = self.parser().to(device)
            self._rule_embedding_cache = cache
        return cache

    def _batch_index(self, data: Data, device: torch.device) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            return torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        return batch.to(device)

    def _reshape_rule_tensor(self, value: Optional[torch.Tensor], batch_size: int, fill: float, device: torch.device) -> torch.Tensor:
        if value is None or self.num_rules == 0:
            return torch.full((batch_size, self.num_rules), fill_value=fill, dtype=torch.float32, device=device)
        return value.float().view(batch_size, self.num_rules).to(device)

    def _compose_substrate_embedding(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = data.x.device
        node_states = self.substrate_encoder.forward_nodes(data)
        batch_index = self._batch_index(data, device)
        graph_embedding = self.graph_norm(global_mean_pool(node_states, batch_index))
        if not self.use_fingerprint:
            return graph_embedding, graph_embedding, node_states, batch_index

        batch_size = graph_embedding.size(0)
        fp = getattr(data, "fp", None)
        if fp is None:
            fp = torch.zeros((batch_size, FINGERPRINT_DIM), dtype=torch.float32, device=device)
        else:
            fp = fp.float().view(batch_size, -1).to(device)
        assert self.fp_encoder is not None
        assert self.substrate_fusion is not None
        fp_embedding = self.fp_encoder(fp)
        fused = self.substrate_fusion(torch.cat((graph_embedding, fp_embedding), dim=1))
        return graph_embedding, self.substrate_norm(fused), node_states, batch_index

    def _pairwise_inputs(
        self,
        substrate_embedding: torch.Tensor,
        node_states: torch.Tensor,
        batch_index: torch.Tensor,
        rules: torch.Tensor,
        rule_mask: torch.Tensor,
        rule_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dense_nodes, node_mask = to_dense_batch(node_states, batch_index)
        keys = self.node_key(dense_nodes)
        values = self.node_value(dense_nodes)
        queries = self.rule_query(rules)
        attn = torch.einsum("bnd,rd->brn", keys, queries) / math.sqrt(max(self.embed_dim, 1))
        attn = attn.masked_fill(~node_mask.unsqueeze(1), -1e4)
        local_weights = torch.softmax(attn, dim=-1)
        local_context = torch.einsum("brn,bnd->brd", local_weights, values)

        substrate = substrate_embedding.unsqueeze(1).expand(-1, self.num_rules, -1)
        rule_bank = rules.unsqueeze(0).expand(substrate_embedding.size(0), -1, -1)
        log_counts = torch.log1p(rule_counts.clamp_min(0.0)) / math.log(5.0)
        match_features = torch.stack((rule_mask, log_counts), dim=-1)

        pair_features = torch.cat(
            (
                substrate,
                rule_bank,
                local_context,
                substrate * rule_bank,
                torch.abs(substrate - rule_bank),
                local_context * rule_bank,
                match_features,
            ),
            dim=-1,
        )
        return pair_features, local_context, match_features

    def _forward_generation_logits(self, data: Data) -> torch.Tensor:
        _, substrate_embedding, node_states, batch_index = self._compose_substrate_embedding(data)
        batch_size = substrate_embedding.size(0)
        rules = self._rule_embeddings(substrate_embedding.device)
        if rules.numel() == 0:
            return torch.empty((batch_size, 0), device=substrate_embedding.device)

        rule_mask = self._reshape_rule_tensor(getattr(data, "rule_mask", None), batch_size, 1.0, substrate_embedding.device)
        rule_counts = self._reshape_rule_tensor(getattr(data, "rule_counts", None), batch_size, 0.0, substrate_embedding.device)

        if self.scoring == "bilinear":
            logits = torch.einsum("bi,ij,rj->br", substrate_embedding, self.bilinear, rules)
        elif self.scoring == "dot":
            logits = F.normalize(substrate_embedding, dim=1) @ F.normalize(rules, dim=1).T
        else:
            pair_features, local_context, match_features = self._pairwise_inputs(
                substrate_embedding,
                node_states,
                batch_index,
                rules,
                rule_mask,
                rule_counts,
            )
            interaction_logits = self.rule_mlp(pair_features).squeeze(-1)
            global_similarity = F.normalize(substrate_embedding, dim=1) @ F.normalize(rules, dim=1).T
            local_similarity = F.cosine_similarity(local_context, rules.unsqueeze(0), dim=-1)
            if self.scoring == "mlp":
                logits = interaction_logits + (self.match_scale * match_features[..., 1])
            else:
                logits = (
                    interaction_logits
                    + (self.global_scale * global_similarity)
                    + (self.local_scale * local_similarity)
                    + (self.match_scale * match_features[..., 1])
                )

        logits = logits + self.bias.view(1, -1) + (self.prior_strength * self.rule_prior_logits.view(1, -1))
        if self.use_applicability_mask and self.num_rules:
            logits = logits - ((1.0 - rule_mask) * self.applicability_penalty)
        return logits

    def forward(self, data: Data, mode: str = "generation", return_logits: bool = False) -> torch.Tensor:
        # Generation is the dominant path: compute the substrate embedding ONCE inside
        # _forward_generation_logits. The previous code also called
        # _compose_substrate_embedding here and discarded it, doubling encoder compute
        # on every train/eval/inference forward.
        if mode == "generation":
            logits = self._forward_generation_logits(data)
            if return_logits:
                return logits
            return torch.sigmoid(logits)

        _, embeddings, _, _ = self._compose_substrate_embedding(data)
        if mode == "contrastive":
            return F.normalize(self.projection_head(embeddings), dim=1)
        if mode == "masked_modeling":
            return embeddings
        if mode == "maccs":
            return torch.sigmoid(self.maccs_head(embeddings))
        raise ValueError(f"Unsupported mode: {mode}")

    def _rule_applicability(self, substrate: str, mol: Optional[Chem.Mol] = None) -> Tuple[np.ndarray, np.ndarray]:
        if substrate in self._applicability_cache:
            return self._applicability_cache[substrate]
        if mol is None:
            mol = Chem.MolFromSmiles(substrate)
        mask = np.zeros(self.num_rules, dtype=np.float32)
        counts = np.zeros(self.num_rules, dtype=np.float32)
        if mol is None:
            self._applicability_cache[substrate] = (mask, counts)
            return mask, counts
        for index, pattern in enumerate(self.rule_patterns):
            if pattern is None:
                continue
            try:
                matches = mol.GetSubstructMatches(pattern, uniquify=True, maxMatches=4)
            except Exception:
                matches = ()
            if matches:
                mask[index] = 1.0
                counts[index] = float(min(len(matches), 4))
        self._applicability_cache[substrate] = (mask, counts)
        return mask, counts

    def _get_applicability(self, substrate: str, mol: Optional[Chem.Mol] = None) -> Tuple[np.ndarray, np.ndarray]:
        return self._rule_applicability(substrate, mol)

    def _single_records(self, data: MolFrame) -> List[Data]:
        if not data.single:
            data.singlegraphs()
        records = []
        for substrate, graph in data.single.items():
            if substrate not in data.map or substrate not in data.reaction_labels:
                continue
            datum = graph.clone()
            mask, counts = self._get_applicability(substrate, data.mol_structs.get(substrate))
            datum.y = torch.tensor(data.reaction_labels[substrate], dtype=torch.float32)
            datum.rule_mask = torch.tensor(mask, dtype=torch.float32)
            datum.rule_counts = torch.tensor(counts, dtype=torch.float32)
            records.append(datum)
        return records

    def _single_loader(self, records: List[Data], batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(records, batch_size=batch_size, shuffle=shuffle)

    def _compute_epoch_loss(self, records: Sequence[Data], batch_size: int, device: torch.device) -> float:
        if not records:
            return float("inf")
        loader = self._single_loader(list(records), batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        num_batches = 0
        self.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = self(batch, return_logits=True)
                targets = batch.y.view(logits.size(0), -1).float()
                rule_mask = self._reshape_rule_tensor(getattr(batch, "rule_mask", None), logits.size(0), 1.0, logits.device)
                loss = self.objective(logits, targets, rule_mask, self.pos_weight.to(logits.device))
                total_loss += float(loss.item())
                num_batches += 1
        return total_loss / max(num_batches, 1)

    def _update_rule_statistics(self, records: Sequence[Data]) -> None:
        if not records or self.num_rules == 0:
            return
        targets = torch.stack([record.y.float() for record in records], dim=0).view(len(records), self.num_rules)
        mask = torch.stack([record.rule_mask.float() for record in records], dim=0).view(len(records), self.num_rules)
        positives = targets.sum(dim=0)
        exposures = mask.sum(dim=0)
        negatives = ((1.0 - targets) * mask).sum(dim=0)
        pos_weight = ((negatives + 1.0) / (positives + 1.0)).clamp(1.0, 25.0)
        prior_prob = torch.where(
            exposures > 0.0,
            (positives + 1.0) / (exposures + 2.0),
            torch.full_like(exposures, 0.01),
        )
        with torch.no_grad():
            self.pos_weight.copy_(pos_weight.to(self.pos_weight.device))
            self.rule_prior_logits.copy_(_safe_logit(prior_prob).to(self.rule_prior_logits.device))

    def _create_pretraining_graphs(self, smiles_list: Sequence[str]) -> List[Data]:
        graphs = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            graph = from_rdmol(mol)
            if graph is not None:
                graphs.append(graph)
        return graphs

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        similarity = torch.mm(z1, z2.T) / max(temperature, 1e-8)
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(similarity, labels)

    def _masked_loss(self, pooled: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prediction = self.atom_predictor(pooled)
        return F.mse_loss(prediction, targets)

    def _masked_node_modeling(self, batch: Batch, mask_ratio: float = 0.15) -> torch.Tensor:
        # True masked-atom modeling: zero out a subset of nodes on the INPUT graph,
        # encode the corrupted graph, and reconstruct the ORIGINAL features of exactly
        # the masked nodes. This forces the encoder to infer missing structure from
        # context, unlike the previous task which regressed the mean of its own
        # unmasked input (a near-trivial identity).
        x = batch.x.float()
        num_nodes = x.size(0)
        if num_nodes == 0:
            return x.sum() * 0.0
        num_mask = max(1, int(num_nodes * mask_ratio))
        mask_indices = torch.randperm(num_nodes, device=x.device)[:num_mask]
        original = x[mask_indices].clone()
        masked = batch.clone()
        masked.x = x.clone()
        masked.x[mask_indices] = 0.0
        node_states = self.substrate_encoder.forward_nodes(masked)
        prediction = self.atom_predictor(node_states[mask_indices])
        return F.mse_loss(prediction, original)

    def pretrain(
        self,
        smiles_list: Sequence[str],
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 1e-4,
        contrastive_ratio: float = 0.5,
        save_path: Optional[str] = None,
    ) -> None:
        graphs = self._create_pretraining_graphs(smiles_list)
        if not graphs:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-6)
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        contrastive_epochs = max(1, int(epochs * contrastive_ratio))

        for epoch in range(epochs):
            self.train()
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                if epoch < contrastive_epochs:
                    aug1 = MoleculeAugmentor.augment_batch(batch).to(device)
                    aug2 = MoleculeAugmentor.augment_batch(batch).to(device)
                    loss = self._contrastive_loss(self(aug1, mode="contrastive"), self(aug2, mode="contrastive"))
                else:
                    loss = self._masked_node_modeling(batch)
                loss.backward()
                optimizer.step()
        self.pretrained = True
        if save_path:
            self.save_pretrained_weights(save_path)

    def pretrain_maccs(
        self,
        smiles_list: Sequence[str],
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> None:
        if not self.use_maccs_pretraining:
            return
        graphs = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            graph = from_rdmol(mol)
            if graph is None:
                continue
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
            # RDKit MACCS keys span 167 bits where bit 0 is an always-zero placeholder
            # and bits 1..166 are the real keys. Use range(1, 167) so the 166-wide head
            # learns real keys (previously it wasted a slot on bit 0 and dropped bit 166).
            graph.maccs = torch.tensor([fingerprint.GetBit(index) for index in range(1, 167)], dtype=torch.float32)
            graphs.append(graph)
        if not graphs:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(
            list(self.substrate_encoder.parameters()) + list(self.maccs_head.parameters()),
            lr=lr,
            weight_decay=1e-6,
        )
        criterion = nn.BCELoss()
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            self.train()
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = criterion(self(batch, mode="maccs"), batch.maccs.view(batch.num_graphs, -1))
                loss.backward()
                optimizer.step()

    def comprehensive_pretrain(
        self,
        smiles_list: Sequence[str],
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-4,
        contrastive_ratio: float = 0.4,
        maccs_ratio: float = 0.3,
        masked_ratio: float = 0.3,
        save_path: Optional[str] = None,
    ) -> None:
        total = contrastive_ratio + maccs_ratio + masked_ratio
        if total <= 0:
            return
        contrastive_ratio /= total
        maccs_ratio /= total
        masked_ratio /= total
        self.pretrain(
            smiles_list=smiles_list,
            epochs=max(1, int(epochs * (contrastive_ratio + masked_ratio))),
            batch_size=batch_size,
            lr=lr,
            contrastive_ratio=contrastive_ratio / max(contrastive_ratio + masked_ratio, 1e-8),
        )
        self.pretrain_maccs(
            smiles_list=smiles_list,
            epochs=max(1, int(epochs * maccs_ratio)),
            batch_size=batch_size,
            lr=lr,
        )
        if save_path:
            self.save_pretrained_weights(save_path)

    def save_pretrained_weights(self, path: str) -> None:
        torch.save(
            {
                "substrate_encoder": self.substrate_encoder.state_dict(),
                "parser": self.parser.state_dict(),
                "projection_head": self.projection_head.state_dict(),
                "atom_predictor": self.atom_predictor.state_dict(),
                "maccs_head": self.maccs_head.state_dict(),
                "bilinear": self.bilinear.data,
                "bias": self.bias.data,
                "rule_prior_logits": self.rule_prior_logits.data,
                "pos_weight": self.pos_weight.data,
            },
            path,
        )

    def load_pretrained_weights(self, path: str) -> None:
        weights = torch.load(path, map_location="cpu")
        self.substrate_encoder.load_state_dict(weights["substrate_encoder"])
        self.parser.load_state_dict(weights["parser"])
        self.projection_head.load_state_dict(weights["projection_head"])
        self.atom_predictor.load_state_dict(weights["atom_predictor"])
        self.maccs_head.load_state_dict(weights["maccs_head"])
        self.bilinear.data.copy_(weights["bilinear"])
        self.bias.data.copy_(weights["bias"])
        if "rule_prior_logits" in weights and self.rule_prior_logits.numel():
            self.rule_prior_logits.data.copy_(weights["rule_prior_logits"])
        if "pos_weight" in weights and self.pos_weight.numel():
            self.pos_weight.data.copy_(weights["pos_weight"])
        self.pretrained = True

    def fit(
        self,
        data: MolFrame,
        lr: float = 1e-4,
        verbose: bool = True,
        eps: int = 20,
        gamma: int = 2,
        freeze_pretrained: bool = False,
        batch_size: int = 64,
        weight_decay: float = 1e-6,
        val_data: Optional[MolFrame] = None,
        patience: int = 7,
        min_delta: float = 1e-4,
        timeout_seconds: Optional[float] = None,
    ) -> "Generator":
        del gamma
        if not data.single:
            data.singlegraphs()
        if not data.reaction_labels:
            data.label_reactions(self.rule_names)

        records = self._single_records(data)
        self._update_rule_statistics(records)
        loader = self._single_loader(records, batch_size=batch_size)

        val_records: List[Data] = []
        if val_data is not None:
            if not val_data.single:
                val_data.singlegraphs()
            if not val_data.reaction_labels:
                val_data.label_reactions(self.rule_names)
            val_records = self._single_records(val_data)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        if freeze_pretrained and self.pretrained:
            for parameter in self.substrate_encoder.parameters():
                parameter.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda parameter: parameter.requires_grad, self.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        best_loss = float("inf")
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        loss_history: List[float] = []
        val_loss_history: List[float] = []
        early_stopped_epoch: Optional[int] = None
        timed_out = False
        stop_reason = "completed"
        last_epoch_seconds: Optional[float] = None
        started = time.perf_counter()

        def _sync_training_state() -> None:
            self.best_state_ = best_state
            self.best_loss_ = best_loss if loss_history else None
            self.best_val_loss_ = best_val_loss if val_records and val_loss_history else None
            self.loss_history_ = list(loss_history)
            self.val_loss_history_ = list(val_loss_history)
            self.epochs_trained_ = len(loss_history)
            self.early_stopped_epoch_ = early_stopped_epoch
            self.timed_out_ = timed_out
            self.timeout_seconds_ = timeout_seconds
            self.stop_reason_ = stop_reason
            self.last_epoch_seconds_ = last_epoch_seconds

        _sync_training_state()
        for epoch in range(eps):
            if timeout_seconds is not None and last_epoch_seconds is not None:
                elapsed = time.perf_counter() - started
                remaining = timeout_seconds - elapsed
                required_seconds = last_epoch_seconds + _timeout_buffer(last_epoch_seconds)
                if remaining <= required_seconds:
                    timed_out = True
                    stop_reason = "timeout"
                    if verbose:
                        print(
                            f"generator stopping before epoch={epoch + 1} "
                            f"remaining={remaining:.1f}s estimated_epoch={last_epoch_seconds:.1f}s"
                        )
                    break
            epoch_started = time.perf_counter()
            self.train()
            epoch_loss = 0.0
            num_batches = 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = self(batch, return_logits=True)
                targets = batch.y.view(logits.size(0), -1).float()
                rule_mask = self._reshape_rule_tensor(getattr(batch, "rule_mask", None), logits.size(0), 1.0, logits.device)
                loss = self.objective(logits, targets, rule_mask, self.pos_weight.to(logits.device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            if num_batches == 0:
                stop_reason = "no_batches"
                break
            avg_loss = epoch_loss / num_batches
            loss_history.append(float(avg_loss))
            best_loss = min(best_loss, avg_loss)
            if val_records:
                val_loss = self._compute_epoch_loss(val_records, batch_size=batch_size, device=device)
                val_loss_history.append(float(val_loss))
                if verbose:
                    print(f"generator epoch={epoch + 1} train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_state = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}
                    self.best_state_ = best_state
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    early_stopped_epoch = epoch + 1
                    stop_reason = "early_stopping"
                    if verbose:
                        print(f"generator early stopping at epoch={epoch + 1} patience={patience}")
                    break
            elif verbose:
                print(f"generator epoch={epoch + 1} loss={avg_loss:.4f}")
            last_epoch_seconds = time.perf_counter() - epoch_started
            _sync_training_state()
            if timeout_seconds is not None and (time.perf_counter() - started) >= timeout_seconds:
                timed_out = True
                stop_reason = "timeout"
                _sync_training_state()
                if verbose:
                    print(f"generator stopping at epoch={epoch + 1} timeout_seconds={timeout_seconds}")
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        _sync_training_state()
        return self

    def _graph_for_substrate(self, sub: str) -> Tuple[Optional[Chem.Mol], Optional[Data]]:
        mol = Chem.MolFromSmiles(sub)
        if mol is None:
            return None, None
        graph = from_rdmol(mol)
        if graph is None:
            return mol, None
        mask, counts = self._get_applicability(sub, mol)
        graph.rule_mask = torch.tensor(mask, dtype=torch.float32)
        graph.rule_counts = torch.tensor(counts, dtype=torch.float32)
        return mol, graph

    @torch.no_grad()
    def score_rules(self, sub: str, return_mask: bool = False):
        mol, graph = self._graph_for_substrate(sub)
        if mol is None or graph is None:
            empty = np.array([], dtype=np.float32)
            if return_mask:
                return empty, empty
            return empty
        first_param = next(iter(self.parameters()), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        batch = Batch.from_data_list([graph]).to(device)
        # Score in eval mode so inference is deterministic (encoder dropout off) -- this also
        # lets _rule_embeddings cache the rule-bank encoding across substrates. Restore the
        # prior mode afterward so a caller mid-training is unaffected.
        was_training = self.training
        if was_training:
            self.eval()
        try:
            scores = self(batch).view(-1).detach().cpu().numpy()
        finally:
            if was_training:
                self.train()
        if return_mask:
            rule_mask = graph.rule_mask.detach().cpu().numpy()
            return scores, rule_mask
        return scores

    def action_distribution(self, sub: str):
        """Differentiable GFlowNet forward-policy logits over {child products} ∪ {STOP}.

        Returns (children: List[str], child_logits: Tensor[len(children)], stop_logit:
        Tensor[1]). The RDKit enumeration (which children exist) is the fixed environment,
        but the logit assigned to each child (logsumexp over the rules that produce it) and
        the STOP logit are differentiable w.r.t. the generator — this is the policy the
        Trajectory-Balance objective trains. NOT wrapped in no_grad on purpose.
        """
        first_param = next(iter(self.parameters()), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        mol, graph = self._graph_for_substrate(sub)
        if mol is None or graph is None:
            return [], torch.empty(0, device=device), torch.zeros(1, device=device)
        batch = Batch.from_data_list([graph]).to(device)
        _, embedding, _, _ = self._compose_substrate_embedding(batch)
        rule_logits = self._forward_generation_logits(batch).view(-1)  # [num_rules], differentiable
        stop_logit = self.stop_head(embedding).view(-1)  # [1]
        rule_mask = (
            graph.rule_mask.detach().cpu().numpy()
            if getattr(graph, "rule_mask", None) is not None
            else np.ones(self.num_rules, dtype=np.float32)
        )
        child_to_rules: Dict[str, List[int]] = {}
        for index in np.where(rule_mask > 0.0)[0]:
            index = int(index)
            if index >= len(self.rule_reactions):
                continue
            reaction = self.rule_reactions[index]
            if reaction is None:
                continue
            for product_tuple in safe_run_reactants(reaction, mol):
                for product in product_tuple:
                    try:
                        smiles = Chem.MolToSmiles(product)
                    except Exception:
                        continue
                    for fragment in smiles.split("."):
                        fragment = fragment.strip()
                        if not fragment:
                            continue
                        try:
                            normalized = _normalize_smiles_cached(fragment, self.gen_normalization)
                        except Exception:
                            continue
                        if normalized:
                            child_to_rules.setdefault(normalized, []).append(index)
        children = list(child_to_rules.keys())
        if not children:
            return [], torch.empty(0, device=device), stop_logit
        child_logits = torch.stack(
            [
                torch.logsumexp(rule_logits[torch.tensor(child_to_rules[child], dtype=torch.long, device=device)], dim=0)
                for child in children
            ]
        )
        return children, child_logits, stop_logit

    @torch.no_grad()
    def calibrate_threshold(
        self,
        val_data: MolFrame,
        rules: Optional[Sequence[str]] = None,
        target: Literal["recall_at_precision", "f1"] = "f1",
        min_precision: float = 0.1,
        verbose: bool = True,
    ) -> Tuple[float, float]:
        selected_rules = list(rules) if rules is not None else self.rule_names
        if not val_data.reaction_labels:
            val_data.label_reactions(selected_rules)

        all_scores: List[float] = []
        all_labels: List[float] = []

        self.eval()
        for substrate in val_data.map:
            try:
                scores, mask = self.score_rules(substrate, return_mask=True)
            except Exception:
                continue
            if scores.size == 0:
                continue
            labels = val_data.reaction_labels.get(substrate)
            if labels is None:
                continue
            labels_np = np.asarray(labels, dtype=np.float32)
            if labels_np.shape[0] != scores.shape[0]:
                continue
            valid = mask > 0.0
            if not valid.any():
                continue
            all_scores.extend(scores[valid].tolist())
            all_labels.extend(labels_np[valid].tolist())

        if not all_scores:
            self.calibrated_threshold = 0.5
            return 0.5, 0.0

        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        labels_t = torch.tensor(all_labels, dtype=torch.float32)
        best_threshold = 0.5
        best_metric = -1.0

        for threshold_step in range(1, 100):
            threshold = threshold_step / 100.0
            predictions = (scores_t >= threshold).float()
            tp = float((predictions * labels_t).sum().item())
            fp = float((predictions * (1.0 - labels_t)).sum().item())
            fn = float((((1.0 - predictions) * labels_t)).sum().item())
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
            if target == "recall_at_precision":
                if precision >= min_precision and recall > best_metric:
                    best_metric = recall
                    best_threshold = threshold
            elif f1 > best_metric:
                best_metric = f1
                best_threshold = threshold

        self.calibrated_threshold = best_threshold
        if verbose:
            print(
                f"generator threshold calibrated={best_threshold:.3f} "
                f"target={target} metric={best_metric:.4f}"
            )
        return best_threshold, best_metric

    def _aggregate_candidate_scores(self, scores: Sequence[float]) -> float:
        if not scores:
            return 0.0
        if self.candidate_aggregation == "max":
            return float(max(scores))
        if self.candidate_aggregation == "mean":
            return float(sum(scores) / len(scores))
        clipped = np.clip(np.asarray(scores, dtype=np.float32), 1e-6, 1.0 - 1e-6)
        noisy_or = float(1.0 - np.prod(1.0 - clipped))
        if self.candidate_aggregation == "noisy_or":
            return noisy_or
        return float((0.65 * float(clipped.max())) + (0.35 * noisy_or))

    @torch.no_grad()
    def generate_scored(
        self,
        sub: str,
        pca: bool = False,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[tuple[str, float]]:
        del pca
        mol, _ = self._graph_for_substrate(sub)
        if mol is None:
            return []
        scores, rule_mask = self.score_rules(sub, return_mask=True)
        if scores.size == 0:
            return []

        active_pool = np.where(rule_mask > 0.0)[0] if self.use_applicability_mask else np.arange(scores.shape[0])
        if active_pool.size == 0:
            active_pool = np.arange(scores.shape[0])

        if threshold is not None:
            active = active_pool[scores[active_pool] >= threshold]
            if active.size == 0 and top_k is not None:
                ranked_pool = active_pool[np.argsort(scores[active_pool])[::-1]]
                active = ranked_pool[:top_k]
            elif top_k is not None and active.size > top_k:
                # Apply top_k AFTER thresholding so a low calibrated threshold cannot
                # emit nearly every applicable rule; keep only the top_k best-scoring.
                active = active[np.argsort(scores[active])[::-1][:top_k]]
        else:
            if top_k is None:
                top_k = min(self.default_top_k, max(1, active_pool.size))
            ranked_pool = active_pool[np.argsort(scores[active_pool])[::-1]]
            active = ranked_pool[:top_k]

        candidate_scores: Dict[str, List[float]] = {}
        ranked_indices = sorted((int(index) for index in active), key=lambda index: float(scores[index]), reverse=True)
        for index in ranked_indices:
            if index >= len(self.rule_reactions):
                continue
            reaction = self.rule_reactions[index]
            if reaction is None:
                continue
            rule_score = float(scores[index])
            outcomes = safe_run_reactants(reaction, mol)
            seen_products = set()
            for product_tuple in outcomes:
                for product in product_tuple:
                    try:
                        smiles = Chem.MolToSmiles(product)
                    except Exception:
                        continue
                    for fragment in smiles.split("."):
                        fragment = fragment.strip()
                        if not fragment:
                            continue
                        try:
                            normalized = _normalize_smiles_cached(fragment, self.gen_normalization)
                        except Exception:
                            continue
                        if normalized in seen_products:
                            continue
                        seen_products.add(normalized)
                        candidate_scores.setdefault(normalized, []).append(rule_score)

        ranked = [(candidate, self._aggregate_candidate_scores(values)) for candidate, values in candidate_scores.items()]
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    @torch.no_grad()
    def generate(
        self,
        sub: str,
        pca: bool = False,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[str]:
        return [candidate for candidate, _ in self.generate_scored(sub, pca=pca, top_k=top_k, threshold=threshold)]

    @torch.no_grad()
    def jaccard(self, test_frame: MolFrame) -> List[float]:
        scores = []
        for substrate, products in test_frame.map.items():
            predicted = set(self.generate(substrate))
            real = {_normalize_smiles_cached(product, self.gen_normalization) for product in products}
            union = predicted | real
            scores.append(len(predicted & real) / len(union) if union else 0.0)
        return scores

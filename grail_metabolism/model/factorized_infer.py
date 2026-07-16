"""Factorized inference: type -> site -> filter ranking.

`FactorizedGenerator` (Task 3/4) supplies two per-substrate signals: `type_logits`
(graph-level P(type|substrate) over the coarse reaction-type vocabulary, Task 1) and
`site_logits` (per-atom "how plausible is this the reacting atom", the SoM-style
localization signal, `model/som.py`). Neither alone enumerates candidate metabolite
structures -- that still requires applying RDKit reaction rules. `generate` wires the
three together: select the most likely reaction types, apply their rules broadly, then
rank the resulting candidate pool by `P(type) * site_plausibility * filter_score` and
dedup by tautomer-InChIKey (the recall-correct match convention this codebase uses
throughout, see `metrics._tautomer_inchikey`).

See docs/superpowers/plans/2026-07-14-grail-factorized-generator.md (Task 6) and
.superpowers/sdd/task-6-brief.md.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from rdkit import Chem

from ..metrics import _tautomer_inchikey
from ..utils.preparation import (
    _clean_product_smiles,
    _compile_rule_pattern,
    _compile_rule_reaction,
    safe_run_reactants,
)
from ..utils.transform import from_rdmol
from .som import product_som_score

__all__ = ["build_rule_by_type", "generate"]


def build_rule_by_type(rule_to_type: Dict[str, int]) -> Dict[int, List[str]]:
    """Invert Task 1's `{smirks: type_id}` map into `{type_id: [smirks, ...]}`.

    The `-1` ("other"/rare, below-`min_pairs`) bucket is dropped: the dense type head has
    no output slot for it, so it can never be a "selected top type" at inference.
    """
    out: Dict[int, List[str]] = {}
    for smirks, type_id in rule_to_type.items():
        if type_id < 0:
            continue
        out.setdefault(type_id, []).append(smirks)
    return out


def _apply_rule(sub_h: Chem.Mol, smirks: str, timeout: float, max_products: int) -> List[str]:
    """Apply one SMIRKS rule to an `AddHs`-prepared substrate -> candidate product SMILES.

    Mirrors `scripts/train_factorized.py`'s per-rule timeout+cap loop (compiled
    pattern/reaction + `safe_run_reactants` + `_clean_product_smiles`), NOT
    `utils.preparation.apply_rules_to_molecule`'s hardcoded 5s/500-product default --
    a short, explicit per-rule budget is what keeps a pathological substrate (a large or
    highly symmetric ring match) from stalling the whole eval loop.
    """
    pattern = _compile_rule_pattern(smirks)
    rxn = _compile_rule_reaction(smirks)
    if pattern is None or rxn is None:
        return []
    try:
        if not sub_h.HasSubstructMatch(pattern):
            return []
    except Exception:
        return []
    outcomes = safe_run_reactants(rxn, sub_h, timeout=timeout, max_products=max_products)
    products: List[str] = []
    for product_tuple in outcomes:
        for prod in product_tuple:
            try:
                smi = Chem.MolToSmiles(prod)
            except Exception:
                continue
            products.extend(_clean_product_smiles(smi))
    return products


def generate(
    model,
    filter_model,
    rule_by_type: Dict[int, Sequence[str]],
    sub: str,
    top_types: int = 10,
    max_output: int = 15,
    rule_timeout: float = 1.0,
    rule_max_products: int = 20,
) -> List[str]:
    """Factorized type -> site -> filter ranking for one substrate.

    1. Score every reaction type by `P(type|s)` (`model.type_logits`) and every atom by
       site plausibility (`model.site_logits`).
    2. Take the `top_types` highest-scoring types and apply every rule mapped to each
       (`rule_by_type`, from `build_rule_by_type`) to the substrate, enumerating candidate
       products via RDKit (per-rule timeout + product cap).
    3. Score each candidate as `P(type) * site_plausibility(reacting atoms) *
       filter_model.score_batch(...)` -- the type prior, WHERE-on-the-molecule prior, and
       the learned (substrate, product) plausibility score, respectively.
    4. Dedup by tautomer-InChIKey (keeping the best-scoring occurrence), sort by score
       descending, return up to `max_output` SMILES.

    Returns an empty list if the substrate doesn't parse, has no atoms, or no rule fires.
    """
    sub_mol = Chem.MolFromSmiles(sub)
    if sub_mol is None:
        return []
    data = from_rdmol(sub_mol)
    if data is None or data.x.size(0) == 0:
        return []

    model.eval()
    with torch.no_grad():
        type_probs = torch.sigmoid(model.type_logits(data)).view(-1).cpu().numpy()
        site_probs = torch.sigmoid(model.site_logits(data)).view(-1).cpu().numpy()

    n_types = type_probs.shape[0]
    k = min(max(int(top_types), 0), n_types)
    if k == 0:
        return []
    top_type_ids = [int(i) for i in (-type_probs).argsort()[:k]]

    sub_h = Chem.AddHs(Chem.Mol(sub_mol))

    # tautomer-InChIKey -> (prior_score, smiles); keeps only the best-scoring occurrence of
    # a metabolite reached via more than one rule/type, and preserves first-seen order as a
    # stable tie-break for the final sort.
    best: Dict[str, tuple] = {}
    order: List[str] = []

    for type_id in top_type_ids:
        type_score = float(type_probs[type_id])
        for smirks in rule_by_type.get(type_id, ()):
            for prod_smi in _apply_rule(sub_h, smirks, rule_timeout, rule_max_products):
                site_score = product_som_score(site_probs, sub_mol, prod_smi)
                prior_score = type_score * site_score
                try:
                    key = _tautomer_inchikey(prod_smi)
                except Exception:
                    continue
                prev = best.get(key)
                if prev is None:
                    order.append(key)
                    best[key] = (prior_score, prod_smi)
                elif prior_score > prev[0]:
                    best[key] = (prior_score, prod_smi)

    if not best:
        return []

    smiles_list = [best[key][1] for key in order]
    prior_scores = [best[key][0] for key in order]
    filter_scores = filter_model.score_batch(sub, smiles_list)

    ranked = sorted(
        zip(smiles_list, prior_scores, filter_scores),
        key=lambda row: row[1] * row[2],
        reverse=True,
    )
    return [row[0] for row in ranked[:max_output]]


class FactorizedReranker:
    """Re-rank a candidate pool by the factorized ``P(type|s)·P(site|type,s)`` signal.

    This is the deployable form of §10's hybrid re-rank: on the identical broad pool,
    ``filter×gen×type×site`` beats ``filter×gen`` by a paired **+0.0165 (95% CI [0.006, 0.027],
    n=1170)** (``results/hybrid_rerank_full1170.json``). It multiplies a per-candidate
    ``type × site`` factor into the ranking and NEVER gates a candidate out (rank-only, honoring
    the same lesson as the SoM prior), so with no reranker the deployment ranking is unchanged.
    """

    def __init__(self, model, rule_to_type: Dict[str, int], rule_names: Sequence[str], aggregation: str = "max") -> None:
        self.model = model
        self.model.eval()
        self.rule_to_type = dict(rule_to_type)
        self.rule_names = list(rule_names)
        self.aggregation = aggregation

    @classmethod
    def load(cls, checkpoint, vocab_path, rule_names: Sequence[str], aggregation: str = "max") -> "FactorizedReranker":
        import json
        from pathlib import Path

        from .factorized import FactorizedGenerator

        model = FactorizedGenerator.load(checkpoint)
        rule_to_type = json.loads(Path(vocab_path).read_text())["rule_to_type"]
        return cls(model, rule_to_type, rule_names, aggregation=aggregation)

    def multipliers(self, sub_mol, detailed) -> List[float]:
        """Per-candidate ``type × site`` multiplier, index-aligned to ``detailed``.

        ``detailed`` is the ``generate_scored_with_details`` output
        ``(smiles, gen_score, rule_id, firing_atoms)``. A candidate whose rule maps to no known
        type falls back to the mean type score (a neutral prior), and an un-localizable site
        yields the neutral 1.0 from ``product_som_score`` -- so the reranker only reshapes rank.
        """
        from .som import product_som_score

        data = from_rdmol(sub_mol)
        with torch.no_grad():
            type_scores = torch.sigmoid(self.model.type_logits(data))[0].cpu().numpy()
            site_scores = torch.sigmoid(self.model.site_logits(data)).cpu().numpy()
        type_floor = float(type_scores.mean())
        out: List[float] = []
        for row in detailed:
            smi, _gscore, rid = row[0], row[1], row[2]
            smirks = self.rule_names[rid] if 0 <= rid < len(self.rule_names) else None
            tid = self.rule_to_type.get(smirks, -1) if smirks is not None else -1
            tfac = float(type_scores[tid]) if 0 <= tid < len(type_scores) else type_floor
            sfac = product_som_score(site_scores, sub_mol, smi, self.aggregation)
            out.append(tfac * sfac)
        return out

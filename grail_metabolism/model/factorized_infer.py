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

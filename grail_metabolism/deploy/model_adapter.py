"""Load the released (bank, generator, filter) pair and rank metabolites with the release order."""
from __future__ import annotations

from pathlib import Path

import torch

from grail_metabolism.config import FilterConfig, GeneratorConfig
from grail_metabolism.model.ranking import release_order
from grail_metabolism.workflows.factory import build_filter, build_generator

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "grail_metabolism/resources/extended_smirks_released.txt"
CKPT = ROOT / "artifacts/full5000_released/checkpoints"


def _load(path, build_fn):
    p = torch.load(path, map_location="cpu", weights_only=False)
    m = build_fn(p["arch"], p.get("rules"))
    m.load_state_dict(p["state_dict"], strict=False)
    m.calibrated_threshold = p.get("calibrated_threshold")
    m.eval()
    return m


class ReleasedModel:
    def __init__(self, generator, filter_model, timeout_seconds):
        self.generator = generator
        self.filter = filter_model
        self.timeout_seconds = timeout_seconds

    def rank(self, smiles, top_k):
        from rdkit import Chem

        from grail_metabolism.metrics import _tautomer_inchikey
        # Reject an unparseable substrate here, explicitly: generate_scored() swallows a bad
        # SMILES internally (it just returns []), which the CLI would otherwise conflate with a
        # substrate that parses fine but has no rule-bank candidates ("no_metabolites"). Raising
        # lets predict_cli.predict_rows classify this row as "no_parse", distinct from that case.
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"cannot parse substrate SMILES: {smiles!r}")
        # apply the WHOLE released bank (not the generator's default top_k): the release ranking is
        # defined on whole-bank pools, so top_k here is the rule count, and the CLI's --top-k caps
        # the RANKED metabolites afterwards, below.
        scored = self.generator.generate_scored(smiles, top_k=self.generator.num_rules)
        if not scored:
            return []
        products = [s for s, _ in scored]
        filter_scores = self.filter.score_batch(smiles, products)     # aligned to products
        cands, self_key = [], _tautomer_inchikey(smiles)
        for (prod, gen), filt in zip(scored, filter_scores):
            try:
                key = _tautomer_inchikey(prod)
            except Exception:
                continue
            cands.append({"smiles": prod, "generator": float(gen), "filter": float(filt), "key": key})
        ordered = release_order(cands, self_key=self_key)[:top_k]
        return [(c["smiles"], c["filter"] * c["generator"]) for c in ordered]


def load_released_model(timeout_seconds=120):
    rules = [l.strip() for l in open(BANK) if l.strip()]
    gen = _load(CKPT / "generator.pt", lambda a, r: build_generator(GeneratorConfig(**a), r or rules))
    filt = _load(CKPT / "filter.pt", lambda a, r: build_filter(FilterConfig(**a)))
    return ReleasedModel(gen, filt, timeout_seconds)

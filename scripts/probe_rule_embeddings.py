#!/usr/bin/env python3
"""Cheap selection-test (a): does the rule GNN already absorb permutation/relabel variants?

from_rule uses atom-map numbers ONLY to add substrate<->product correspondence cross-edges (topology),
not as node features -> permutation/relabel variants are isomorphic graphs -> a permutation-invariant
GNN should map them to ~identical embeddings. If so, merging those ~43% permutation-variant duplicates
gives ~0 selection benefit (the encoder is already invariant to them); the only payoff is deploy hygiene.

Test: for every structural (canonical-SMIRKS) cluster of size>=2, measure the mean pairwise cosine
DISTANCE among member embeddings from the DEPLOYED encoder, vs a random-pair baseline. Near-zero
within-cluster spread relative to baseline => the GNN absorbs the redundancy.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from rdkit.Chem import AllChem
from rdkit import RDLogger

from grail_metabolism.config import GeneratorConfig
from grail_metabolism.utils.preparation import load_default_rules
from grail_metabolism.workflows.factory import build_generator

RDLogger.DisableLog("rdApp.*")
DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
_MAP = re.compile(r":(\d+)]")


def canon_key(smirks):
    try:
        rxn = AllChem.ReactionFromSmarts(smirks)
        if rxn is None:
            return None
        cs = AllChem.ReactionToSmiles(rxn, canonical=True)
    except Exception:
        return None
    remap = {}
    for m in _MAP.finditer(cs):
        o = m.group(1)
        if o not in remap:
            remap[o] = str(len(remap) + 1)
    smi = _MAP.sub(lambda mm: ":" + remap[mm.group(1)] + "]", cs)
    qs, bs = [], []
    for side in (rxn.GetReactants(), rxn.GetProducts()):
        for mol in side:
            for a in mol.GetAtoms():
                qs.append(re.sub(r":\d+", "", a.GetSmarts()))
            for b in mol.GetBonds():
                bs.append(b.GetSmarts())
    return smi + "||" + "|".join(sorted(qs)) + "##" + "|".join(sorted(bs))


def main() -> int:
    torch.manual_seed(0)
    s = torch.load(DEPLOYED_GEN, map_location="cpu", weights_only=False)
    gen = build_generator(GeneratorConfig(**s["arch"]), s.get("rules"))
    gen.load_state_dict(s["state_dict"], strict=False)
    gen.eval()
    rules = list(getattr(gen, "rules", s.get("rules")) or load_default_rules())
    print(f"generator rules: {len(rules)}", flush=True)

    with torch.no_grad():
        emb = gen._rule_embeddings(torch.device("cpu"))  # (num_rules, D)
    emb = F.normalize(emb.float(), dim=1)
    print(f"rule embeddings: {tuple(emb.shape)}", flush=True)

    # structural clusters
    groups = defaultdict(list)
    for i, r in enumerate(rules):
        k = canon_key(r)
        groups[k if k else f"__raw__{i}"].append(i)
    clusters = [v for v in groups.values() if len(v) >= 2]
    print(f"structural clusters (size>=2): {len(clusters)}", flush=True)

    def mean_pair_cos_dist(idxs):
        if len(idxs) < 2:
            return None
        vs = emb[idxs]
        sims = []
        for a, b in combinations(range(len(idxs)), 2):
            sims.append(float(torch.dot(vs[a], vs[b])))
        return 1.0 - sum(sims) / len(sims)

    within = [d for c in clusters if (d := mean_pair_cos_dist(c)) is not None]
    within_mean = sum(within) / len(within)

    # random-pair baseline
    rng = torch.Generator().manual_seed(0)
    n = emb.size(0)
    base = []
    for _ in range(2000):
        i, j = torch.randint(0, n, (2,), generator=rng).tolist()
        if i != j:
            base.append(1.0 - float(torch.dot(emb[i], emb[j])))
    base_mean = sum(base) / len(base)

    # worst (max within-cluster spread) to check the tail
    worst = max(within)
    frac_tiny = sum(1 for d in within if d < 0.01) / len(within)

    print("\n=== TEST (a): does the GNN absorb permutation/relabel variants? ===", flush=True)
    print(f"mean within-structural-cluster cosine distance : {within_mean:.5f}", flush=True)
    print(f"random-pair baseline cosine distance           : {base_mean:.5f}", flush=True)
    print(f"ratio within/baseline                          : {within_mean/base_mean:.4f}", flush=True)
    print(f"worst within-cluster distance                  : {worst:.5f}", flush=True)
    print(f"fraction of clusters with dist < 0.01          : {frac_tiny:.3f}", flush=True)
    verdict = ("GNN ABSORBS them (merging permutation variants ~0 selection benefit; deploy hygiene only)"
               if within_mean / base_mean < 0.1 else
               "GNN does NOT fully absorb them (merging could still concentrate signal)")
    print(f"VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

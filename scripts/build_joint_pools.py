#!/usr/bin/env python3
"""Precompute frozen gen+filter candidate pools for D3 joint training.

The joint-training arm fine-tunes the factorized generator's type/site heads to *rank* true
metabolites given the FIXED deployed generator and filter, instead of the bolt-on re-ranker's
independently-trained heads. Everything the ranking loss needs except the trainable head outputs
is fixed per candidate, so we precompute it once here:

  per substrate -> list of candidates, each with:
    tid       : reaction-type id (rule -> coarse type vocab), or -1 (falls back to mean type prob)
    reacting  : substrate atom indices at the reacting site (product_som_score's `_reacting_atoms`;
                the SAME localization the deployed reranker aggregates the site head over)
    lf, lg    : log filter score, log generator score (fixed context in the combined log-score)
    y         : 1 if the candidate's tautomer-InChIKey matches an annotated metabolite, else 0

The trainer (`scripts/train_joint_factorized.py`) rebuilds each substrate graph from `sub` and
forwards the trainable heads; only tid/reacting/lf/lg/y are needed as fixed data. Shardable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from rdkit import Chem

from grail_metabolism.config import FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.model.som import _reacting_atoms
from grail_metabolism.utils.preparation import _normalize_smiles_cached
from grail_metabolism.workflows.factory import build_filter, build_generator
from scripts.abstention_frontier import load_split_map

DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
DEPLOYED_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
VOCAB = ROOT / "grail_metabolism" / "resources" / "coarse_type_vocab.json"
_EPS = 1e-6


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--sample", type=int, default=0, help="cap #substrates (0 = all in the slice)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--filter-cap", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    generator = _load(DEPLOYED_GEN, lambda a, r: build_generator(GeneratorConfig(**a), r))
    assert float(generator.rule_prior_logits.std()) > 0.1, "degenerate prior; use full5000_priors"
    generator.gen_normalization = "canonical"
    filt = _load(DEPLOYED_FILTER, lambda a, r: build_filter(FilterConfig(**a)))
    gen_threshold = getattr(generator, "calibrated_threshold", None)
    rule_to_type = json.loads(VOCAB.read_text())["rule_to_type"]
    rule_names = generator.rule_names

    smap = load_split_map(args.split)
    items = sorted(smap.items())
    items = items[args.start:(args.end or None)] if (args.start or args.end) else items
    if args.sample:
        items = items[: args.sample]
    print(f"{args.split}: {len(items)} substrates [{args.start}:{args.end or 'end'}]  top_k={args.top_k}", flush=True)

    out_rows = []
    t0 = time.time()
    n_pos = n_cand = 0
    for i, (sub, prods) in enumerate(items, 1):
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)} ({time.time()-t0:.0f}s) cands={n_cand} pos={n_pos}", flush=True)
        sub_mol = Chem.MolFromSmiles(sub)
        if sub_mol is None:
            continue
        true_keys = set()
        for p in prods:
            try:
                true_keys.add(_tautomer_inchikey(p))
            except Exception:
                continue
        detailed = generator.generate_scored_with_details(sub, top_k=args.top_k, threshold=gen_threshold, compute_sites=False)[: args.filter_cap]
        if not detailed:
            continue
        smis = [_normalize_smiles_cached(d[0], "canonical") for d in detailed]
        fscores = filt.score_batch(sub, smis)
        cands = []
        has_pos = False
        for (raw, gscore, rid, _sites), smi, fs in zip(detailed, smis, fscores):
            prod_mol = Chem.MolFromSmiles(smi)
            if prod_mol is None:
                continue
            smirks = rule_names[rid] if 0 <= rid < len(rule_names) else None
            tid = rule_to_type.get(smirks, -1) if smirks is not None else -1
            reacting = sorted(int(a) for a in _reacting_atoms(sub_mol, prod_mol))
            try:
                y = 1 if _tautomer_inchikey(smi) in true_keys else 0
            except Exception:
                y = 0
            cands.append({"tid": int(tid), "reacting": reacting,
                          "lf": math.log(max(float(fs), _EPS)), "lg": math.log(max(float(gscore), _EPS)), "y": y})
            n_cand += 1
            if y:
                has_pos = True
                n_pos += 1
        if cands and has_pos:  # only substrates with >=1 recoverable positive teach the ranking loss
            out_rows.append({"sub": sub, "n_atoms": sub_mol.GetNumAtoms(), "cands": cands})

    Path(args.out).write_text(json.dumps({"split": args.split, "n": len(out_rows), "rows": out_rows}))
    print(f"Wrote {args.out}: {len(out_rows)} substrates with a recoverable positive, {n_cand} candidates", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

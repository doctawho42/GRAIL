#!/usr/bin/env python3
"""Hybrid eval: keep the FULL-bank broad pool (coverage 0.735), re-rank with the factorized signal.

The factorized redesign (scripts/eval_factorized.py) type-GATES application to the top-N types'
rules, capping coverage at ~47% -> recall 0.256. This tests the design's actual intent ("apply
broadly, do NOT gate"): take the SAME broad candidate pool that the selection-breadth ablation
ranks to 0.413 (deployed generator, top_k=300 rules, ranked by filter x gen), and swap ONLY the
ranking signal to the factorized P(type|s) x P(site|type,s). Three rankings on the identical pool:

  (a) filter x gen               -- the broad+filter baseline (should reproduce ~0.413)
  (b) filter x type x site       -- the factorized signal alone
  (c) filter x gen x type x site -- combined

If (b) or (c) > (a) on the same substrates, the learned factorized signal improves ranking of the
broad pool while keeping coverage -- the hybrid beats broad+filter. Deployed generator prior is
restored from full5000_priors (the shipped full5000_single generator.pt has it zeroed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rdkit import Chem

from grail_metabolism.config import FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.model.factorized import FactorizedGenerator
from grail_metabolism.model.som import product_som_score
from grail_metabolism.utils.transform import from_rdmol
from grail_metabolism.workflows.factory import build_filter, build_generator
from scripts.run_benchmark import load_test_map

DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
DEPLOYED_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
FACTORIZED = ROOT / "artifacts" / "factorized_v1" / "checkpoints" / "factorized.pt"
VOCAB = ROOT / "grail_metabolism" / "resources" / "coarse_type_vocab.json"
KS = [5, 10, 15]


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    return model


def _dedup_top(cands_scored, mo):
    """cands_scored: list[(smiles, score)] -> tautomer-deduped top-mo smiles (score desc)."""
    out, seen = [], set()
    for smi, _ in sorted(cands_scored, key=lambda x: -x[1]):
        try:
            k = _tautomer_inchikey(smi)
        except Exception:
            k = smi
        if k in seen:
            continue
        seen.add(k)
        out.append(smi)
        if len(out) >= mo:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-substrates", type=int, default=250)
    ap.add_argument("--top-k", type=int, default=300, help="rules applied for the broad pool (matches the 0.413 ablation)")
    ap.add_argument("--max-output", type=int, default=15)
    ap.add_argument("--filter-cap", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "hybrid_rerank.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    generator = _load(DEPLOYED_GEN, lambda a, r: build_generator(GeneratorConfig(**a), r))
    assert float(generator.rule_prior_logits.std()) > 0.1, "degenerate prior; use full5000_priors"
    generator.gen_normalization = "canonical"
    filt = _load(DEPLOYED_FILTER, lambda a, r: build_filter(FilterConfig(**a)))
    fact = FactorizedGenerator.load(FACTORIZED)
    fact.eval()
    rule_to_type = json.loads(VOCAB.read_text())["rule_to_type"]
    rule_names = generator.rule_names
    gen_threshold = getattr(generator, "calibrated_threshold", None)

    test_map = load_test_map(None, 42)
    items = list(test_map.items())[: args.max_substrates] if args.max_substrates else list(test_map.items())
    print(f"substrates: {len(items)}  top_k={args.top_k}", flush=True)

    rows = {"a_filter_gen": [], "b_filter_type_site": [], "c_all": []}
    t0 = time.time()
    for i, (sub, prods) in enumerate(items, 1):
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        sub_mol = Chem.MolFromSmiles(sub)
        data = from_rdmol(sub_mol) if sub_mol is not None else None
        real = sorted(prods)
        if data is None or data.x.size(0) == 0:
            for key in rows:
                rows[key].append({"predicted": [], "real": real})
            continue
        # broad pool with provenance (rule_id -> type)
        detailed = generator.generate_scored_with_details(sub, top_k=args.top_k, threshold=gen_threshold, compute_sites=False)[: args.filter_cap]
        if not detailed:
            for key in rows:
                rows[key].append({"predicted": [], "real": real})
            continue
        with torch.no_grad():
            type_scores = torch.sigmoid(fact.type_logits(data))[0].cpu().numpy()
            site_scores = torch.sigmoid(fact.site_logits(data)).cpu().numpy()
        type_floor = float(type_scores.mean())
        smis = [d[0] for d in detailed]
        fscores = filt.score_batch(sub, smis) if smis else []
        a, b, c = [], [], []
        for (smi, gscore, rid, _sites), fs in zip(detailed, fscores):
            fs = float(fs)
            smirks = rule_names[rid] if 0 <= rid < len(rule_names) else None
            tid = rule_to_type.get(smirks, -1) if smirks is not None else -1
            tfac = float(type_scores[tid]) if 0 <= tid < len(type_scores) else type_floor
            sfac = product_som_score(site_scores, sub_mol, smi, "max")
            a.append((smi, fs * float(gscore)))
            b.append((smi, fs * tfac * sfac))
            c.append((smi, fs * float(gscore) * tfac * sfac))
        rows["a_filter_gen"].append({"predicted": _dedup_top(a, args.max_output), "real": real})
        rows["b_filter_type_site"].append({"predicted": _dedup_top(b, args.max_output), "real": real})
        rows["c_all"].append({"predicted": _dedup_top(c, args.max_output), "real": real})

    report = {"n": len(items), "top_k": args.top_k, "baseline_broad_filter": 0.413, "rankings": {}}
    for key, rws in rows.items():
        m = aggregate_prediction_metrics(rws, KS, match="inchikey_tautomer")
        report["rankings"][key] = {
            "recall@15": round(m.get("top_15_recall", 0.0), 4),
            "recall@5": round(m.get("top_5_recall", 0.0), 4),
            "precision": round(m.get("precision", 0.0), 4),
            "mean_output": round(m.get("mean_output_size", 0.0), 2),
        }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n=== hybrid re-ranking on the broad pool (same substrates) ===", flush=True)
    for key, v in report["rankings"].items():
        print(f"  {key:20s}: recall@15 {v['recall@15']}  recall@5 {v['recall@5']}  prec {v['precision']}  out {v['mean_output']}", flush=True)
    print(f"  (broad+filter baseline ~0.413; deployed 0.330; SyGMa 0.572)", flush=True)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

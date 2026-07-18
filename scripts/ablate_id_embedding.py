#!/usr/bin/env python3
"""FALSIFIER (test 2): is the id_embedding load-bearing for ranking, or geometric illusion?

Test 1 showed id carries ~82% of cross-rule embedding variance; test 3 showed only a weak (0.277)
tie to usefulness. Neither says whether id is LOAD-BEARING. Zero the id component at inference and
re-run the full ensemble recall@k:
  recall barely moves  -> id not load-bearing; the 82% dominance is a geometric illusion; the
                          'model is a lookup table -> P2' story is FALSIFIED; merging headroom small.
  recall collapses     -> id IS the representation; rule scoring is ~a per-slot lookup; the P2
                          mechanism (untrained id for rare rules = noise) is architecturally CONFIRMED.

Inference-only, no retrain. Same broad pool machinery as budget_matched_frontier.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from rdkit import Chem
from rdkit import RDLogger

from grail_metabolism.config import FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.utils.preparation import _normalize_smiles_cached
from grail_metabolism.workflows.factory import build_filter, build_generator

RDLogger.DisableLog("rdApp.*")
GRAIL_CSV = ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"
DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
DEPLOYED_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
OUT = ROOT / "results" / "ablate_id_embedding.json"
KS = [1, 5, 15, 30]


def _load(path, build_fn):
    s = torch.load(path, map_location="cpu", weights_only=False)
    m = build_fn(s["arch"], s.get("rules"))
    m.load_state_dict(s["state_dict"], strict=False)
    m.calibrated_threshold = s.get("calibrated_threshold")
    m.eval()
    return m


def _taut(s):
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return None


def recall_row(gen, filt, subs, real_ik, gen_thr):
    curve = {k: 0.0 for k in KS}
    n = 0
    t0 = time.time()
    for i, sub in enumerate(subs, 1):
        if i % 25 == 0 or i == len(subs):
            print(f"    {i}/{len(subs)} ({time.time()-t0:.0f}s)", flush=True)
        rk = real_ik.get(sub)
        if not rk:
            continue
        n += 1
        mol = Chem.MolFromSmiles(sub)
        detailed = gen.generate_scored_with_details(sub, top_k=300, threshold=gen_thr, compute_sites=False) if mol is not None else []
        smis = [_normalize_smiles_cached(d[0], "canonical") for d in detailed]
        if not smis:
            continue
        fs = filt.score_batch(sub, smis)
        scored = sorted(zip(smis, (float(f) * float(d[1]) for f, d in zip(fs, detailed))), key=lambda x: -x[1])
        ranked, seen = [], set()
        for s, _ in scored:
            k = _taut(s)
            if k and k not in seen:
                seen.add(k)
                ranked.append(k)
        for k in KS:
            hit = len(set(ranked[:k]) & rk)
            curve[k] += hit / len(rk)
    return {k: round(v / n, 4) for k, v in curve.items()}, n


def main() -> int:
    torch.set_num_threads(6)
    real_ik = {}
    with open(GRAIL_CSV) as fh:
        for row in csv.DictReader(fh):
            rk = {k for k in (_taut(r) for r in row.get("real", "").split("|") if r) if k}
            if rk:
                real_ik[row["substrate"]] = rk
    subs = sorted(real_ik)
    print(f"substrates: {len(subs)}", flush=True)

    gen = _load(DEPLOYED_GEN, lambda a, r: build_generator(GeneratorConfig(**a), r))
    gen.gen_normalization = "canonical"
    gen_thr = getattr(gen, "calibrated_threshold", None)
    filt = _load(DEPLOYED_FILTER, lambda a, r: build_filter(FilterConfig(**a)))

    print("  [baseline] id ON", flush=True)
    base, n = recall_row(gen, filt, subs, real_ik, gen_thr)

    # zero the id component, clear the rule-embedding cache, re-run
    with torch.no_grad():
        gen.parser.id_embedding.weight.zero_()
    gen._rule_embedding_cache = None
    gen.parser._rule_embedding_cache = None if hasattr(gen.parser, "_rule_embedding_cache") else None
    print("  [ablation] id ZEROED", flush=True)
    abl, _ = recall_row(gen, filt, subs, real_ik, gen_thr)

    report = {"n": n, "ks": KS, "recall_id_on": base, "recall_id_zeroed": abl,
              "delta": {f"@{k}": round(abl[k] - base[k], 4) for k in KS}}
    OUT.write_text(__import__("json").dumps(report, indent=2))
    print("\n=== TEST 2: zero-id ablation (ensemble recall@k) ===", flush=True)
    print(f"{'k':>4} | {'id ON':>8} | {'id ZERO':>8} | {'delta':>8}", flush=True)
    for k in KS:
        print(f"{k:>4} | {base[k]:>8} | {abl[k]:>8} | {abl[k]-base[k]:>+8.4f}", flush=True)
    verdict = ("id NOT load-bearing (dominance is geometric illusion; P2-lookup story FALSIFIED)"
               if abs(abl[15] - base[15]) < 0.02 else
               "id IS load-bearing (rule scoring ~ per-slot lookup; P2 mechanism architecturally supported)")
    print(f"VERDICT @15: {verdict}", flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

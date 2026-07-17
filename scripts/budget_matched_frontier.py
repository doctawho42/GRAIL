#!/usr/bin/env python3
"""Budget-matched recall@k frontier: GRAIL vs SyGMa on the shared pool (the falsification test).

The reframe's whole claim reduces to: *at equal output budget GRAIL dominates, and the leaderboard
does not measure at equal budget*. So measure at equal budget. This computes the recall@k-vs-k
curve for both methods on the shared n=150 (where SyGMa raw pools exist), under tautomer-InChIKey:

  - GRAIL: its ranked candidate pool (broad top_k=300 generator pool, ranked by filter x generator,
    tautomer-deduped) truncated to each k -- i.e. GRAIL emitting k outputs.
  - SyGMa: its raw dumped pool (json order, tautomer-deduped) truncated to each k.

Decisive cells:
  - GRAIL@64 vs SyGMa@64 (equal budget = SyGMa's mean output). If GRAIL@64 >= SyGMa@64, the reframe
    is proven in one cell (GRAIL dominates at matched budget; the field's k=15 point hides it).
    If GRAIL@64 < SyGMa@64 despite GRAIL's bank ceiling (0.70) > SyGMa's pool (0.52), then GRAIL's
    selector really destroys reachable coverage -- that is P1 (weak selector), NOT "the metric
    rewards not-selecting", and the reframe does not hold.
  - The crossover k where SyGMa's (near-linear, unranked) curve overtakes GRAIL's (front-loaded)
    curve: the field reports k=15, which lies on non-comparable parts of the two curves.
"""
from __future__ import annotations

import csv
import json
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
from grail_metabolism.utils.preparation import _normalize_smiles_cached
from grail_metabolism.workflows.factory import build_filter, build_generator

GRAIL_CSV = ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"
SYGMA_RAW = ROOT / "results" / "match_sens_cache" / "sygma_preds_a7bf90dad9de0e5b.json"
DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
DEPLOYED_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
OUT = ROOT / "results" / "budget_matched_frontier.json"
KS = [1, 2, 4, 8, 15, 32, 64]


def _load(path, build_fn):
    s = torch.load(path, map_location="cpu", weights_only=False)
    m = build_fn(s["arch"], s.get("rules"))
    m.load_state_dict(s["state_dict"], strict=False)
    m.calibrated_threshold = s.get("calibrated_threshold")
    m.eval()
    return m


def _dedup_taut(smiles_list):
    """Order-preserving tautomer-InChIKey dedup -> list of representative SMILES."""
    out, seen = [], set()
    for s in smiles_list:
        if not s:
            continue
        try:
            k = _tautomer_inchikey(s)
        except Exception:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _real_ik(real_smiles):
    out = set()
    for s in real_smiles:
        try:
            out.add(_tautomer_inchikey(s))
        except Exception:
            continue
    return out


def recall_at(ranked_smiles, real_ik, k):
    if not real_ik:
        return None
    keys = {_tautomer_inchikey(s) for s in ranked_smiles[:k]}
    return len(keys & real_ik) / len(real_ik)


def main() -> int:
    torch.set_num_threads(6)
    # ground truth + shared substrates
    grail_real = {}
    with open(GRAIL_CSV) as fh:
        for row in csv.DictReader(fh):
            grail_real[row["substrate"]] = [r for r in row.get("real", "").split("|") if r]
    sygma = json.loads(SYGMA_RAW.read_text())
    subs = sorted(set(sygma) & set(grail_real))
    print(f"shared substrates: {len(subs)}", flush=True)

    generator = _load(DEPLOYED_GEN, lambda a, r: build_generator(GeneratorConfig(**a), r))
    assert float(generator.rule_prior_logits.std()) > 0.1, "degenerate prior; use full5000_priors"
    generator.gen_normalization = "canonical"
    gen_thr = getattr(generator, "calibrated_threshold", None)
    filt = _load(DEPLOYED_FILTER, lambda a, r: build_filter(FilterConfig(**a)))

    grail_curve = {k: 0.0 for k in KS}
    sygma_curve = {k: 0.0 for k in KS}
    n = 0
    t0 = time.time()
    for i, sub in enumerate(subs, 1):
        if i % 25 == 0 or i == len(subs):
            print(f"  {i}/{len(subs)} ({time.time()-t0:.0f}s)", flush=True)
        real_ik = _real_ik(grail_real[sub])
        if not real_ik:
            continue
        n += 1
        # GRAIL ranked pool: broad top_k=300, ranked by filter x generator, tautomer-deduped
        mol = Chem.MolFromSmiles(sub)
        detailed = generator.generate_scored_with_details(sub, top_k=300, threshold=gen_thr, compute_sites=False) if mol is not None else []
        smis = [_normalize_smiles_cached(d[0], "canonical") for d in detailed]
        fs = filt.score_batch(sub, smis) if smis else []
        scored = sorted(zip(smis, (float(f) * float(d[1]) for f, d in zip(fs, detailed))), key=lambda x: -x[1])
        grail_ranked = _dedup_taut([s for s, _ in scored])
        # SyGMa raw pool in its own (unranked) order, tautomer-deduped
        sygma_ranked = _dedup_taut(sygma.get(sub, []))
        for k in KS:
            grail_curve[k] += recall_at(grail_ranked, real_ik, k)
            sygma_curve[k] += recall_at(sygma_ranked, real_ik, k)

    grail_curve = {k: round(v / n, 4) for k, v in grail_curve.items()}
    sygma_curve = {k: round(v / n, 4) for k, v in sygma_curve.items()}
    crossover = next((k for k in KS if sygma_curve[k] > grail_curve[k]), None)
    report = {
        "n": n, "match": "inchikey_tautomer", "ks": KS,
        "grail_recall_at_k": grail_curve,
        "sygma_recall_at_k": sygma_curve,
        "budget_matched": {
            "grail@8": grail_curve[8], "sygma@8": sygma_curve[8],
            "grail@64": grail_curve[64], "sygma@64": sygma_curve[64],
            "grail_ge_sygma_at_64": grail_curve[64] >= sygma_curve[64],
        },
        "crossover_k_sygma_overtakes_grail": crossover,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\n=== budget-matched recall@k frontier (shared n={n}, tautomer) ===", flush=True)
    print(f"{'k':>4} | {'GRAIL':>7} | {'SyGMa':>7}", flush=True)
    for k in KS:
        mark = "  <- field reports here" if k == 15 else ("  <- SyGMa overtakes" if k == crossover else "")
        print(f"{k:>4} | {grail_curve[k]:>7} | {sygma_curve[k]:>7}{mark}", flush=True)
    bm = report["budget_matched"]
    print(f"\nbudget-matched: GRAIL@64 {bm['grail@64']} vs SyGMa@64 {bm['sygma@64']} -> "
          f"{'GRAIL DOMINATES at matched budget (reframe holds)' if bm['grail_ge_sygma_at_64'] else 'GRAIL < SyGMa at matched budget (this is P1, NOT the reframe)'}", flush=True)
    print(f"crossover (SyGMa overtakes GRAIL): k={crossover}", flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

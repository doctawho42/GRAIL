#!/usr/bin/env python3
"""GLORYx pool-oracle diagnostic: is the SyGMa gap COVERAGE or RANKING?

For each of the 37 GLORYx parents, build a candidate pool with the trained generator
and compute:
  - pool oracle recall@15 = mean over 37 of min(|true ∩ pool|, 15) / |true|
    (hits-first ceiling: what a perfect ranker *on this pool* could achieve)
  - pool coverage        = mean over 37 of |true ∩ pool| / |true|
    (total fraction of reference metabolites the generator ever put in the pool)

Tried at two pool sizes:
  - small:  top_k=100,  max_pool=80
  - large:  top_k=200,  max_pool=150

Verdict:
  oracle@15 ≈ reranker (0.333)   -> COVERAGE gap: rule bank doesn't reach GLORYx
                                     metabolites; scaling reranker training WON'T close
                                     the SyGMa gap (need more rules / multi-step).
  oracle@15 >> reranker (0.333)  -> RANKING headroom: scaling the reranker can help.

Usage:
    python scripts/diagnose_gloryx_oracle.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.config import GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey
from grail_metabolism.model.grail import _read_checkpoint
from grail_metabolism.workflows.factory import build_generator
from grail_metabolism.workflows.reranker import build_pool

GEN_CKPT = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
GLORYX_JSON = ROOT / "docs" / "benchmark" / "data" / "gloryx_test.json"


def _flatten(mets):
    out = []
    for m in mets or []:
        if m.get("smiles"):
            out.append(m["smiles"])
        out.extend(_flatten(m.get("metabolites", [])))
    return out


def load_gloryx(path: Path):
    """Load GLORYx JSON using the same escape-fix + flatten as eval_on_gloryx.py."""
    raw = path.read_text()
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
    data = json.loads(fixed)
    return {p["smiles"]: _flatten(p.get("metabolites", [])) for p in data if p.get("smiles")}


def _load_generator():
    state = _read_checkpoint(GEN_CKPT)
    if state is None or "arch" not in state or "rules" not in state:
        raise SystemExit(f"Generator checkpoint missing arch/rules: {GEN_CKPT}")
    gen = build_generator(GeneratorConfig(**state["arch"]), state["rules"])
    gen.load_state_dict(state["state_dict"], strict=False)
    gen.eval()
    return gen


def oracle_metrics(
    generator,
    reals: dict,
    top_k: int,
    max_pool: int,
    label: str,
) -> dict:
    """Compute pool oracle recall@15 and pool coverage for all GLORYx parents."""
    parents = [p for p in reals if reals[p]]  # only parents with reference metabolites
    print(
        f"\n[oracle] {label}: top_k={top_k} max_pool={max_pool}  "
        f"({len(parents)} parents with reference metabolites)",
        flush=True,
    )
    oracle15_vals = []
    coverage_vals = []
    per_parent = []
    t0 = time.time()

    for i, parent in enumerate(parents, 1):
        if i == 1 or i % 10 == 0 or i == len(parents):
            print(f"  {i}/{len(parents)}  ({time.time()-t0:.0f}s)", flush=True)

        pool = build_pool(generator, parent, top_k=top_k, max_pool=max_pool)
        pool_keys = {_tautomer_inchikey(s) for s, _, _ in pool}

        true_keys = set()
        for smi in reals[parent]:
            try:
                true_keys.add(_tautomer_inchikey(smi))
            except Exception:
                pass

        if not true_keys:
            continue

        hits_in_pool = len(pool_keys & true_keys)
        # Oracle recall@15: as if a perfect ranker places all hits at the top
        oracle15 = min(hits_in_pool, 15) / len(true_keys)
        coverage = hits_in_pool / len(true_keys)

        oracle15_vals.append(oracle15)
        coverage_vals.append(coverage)
        per_parent.append({
            "parent": parent,
            "n_true": len(true_keys),
            "hits_in_pool": hits_in_pool,
            "pool_size": len(pool),
            "oracle15": round(oracle15, 4),
            "coverage": round(coverage, 4),
        })

    macro_oracle15 = sum(oracle15_vals) / max(len(oracle15_vals), 1)
    macro_coverage = sum(coverage_vals) / max(len(coverage_vals), 1)
    print(
        f"  -> oracle@15={macro_oracle15:.4f}  coverage={macro_coverage:.4f}  "
        f"(n={len(oracle15_vals)})",
        flush=True,
    )
    return {
        "label": label,
        "top_k": top_k,
        "max_pool": max_pool,
        "n_parents": len(oracle15_vals),
        "macro_oracle_recall_at_15": round(macro_oracle15, 4),
        "macro_coverage": round(macro_coverage, 4),
        "per_parent": per_parent,
    }


def main() -> None:
    print("[gloryx_oracle] loading GLORYx ...", flush=True)
    reals = load_gloryx(GLORYX_JSON)
    print(
        f"[gloryx_oracle] {len(reals)} parents, "
        f"{sum(len(v) for v in reals.values())} reference metabolites total",
        flush=True,
    )

    print("[gloryx_oracle] loading generator ...", flush=True)
    generator = _load_generator()
    print(f"[gloryx_oracle] generator loaded; num_rules={generator.num_rules}", flush=True)

    results = {}

    # Small pool
    small = oracle_metrics(
        generator, reals,
        top_k=100, max_pool=80,
        label="small (top_k=100, max_pool=80)",
    )
    results["small"] = small

    # Large pool
    large = oracle_metrics(
        generator, reals,
        top_k=200, max_pool=150,
        label="large (top_k=200, max_pool=150)",
    )
    results["large"] = large

    # Interpret verdict
    reranker_r15 = 0.333
    sygma_r15 = 0.498
    small_oracle = small["macro_oracle_recall_at_15"]
    large_oracle = large["macro_oracle_recall_at_15"]

    if large_oracle < reranker_r15 + 0.05:
        verdict = "COVERAGE: oracle barely above reranker -- rule bank doesn't reach GLORYx metabolites; scaling the reranker WON'T close the SyGMa gap. Need more rules or multi-step enumeration."
    elif large_oracle > sygma_r15 - 0.05:
        verdict = "RANKING: oracle near or above SyGMa -- the generator pool is rich enough; scaling the reranker CAN close the gap."
    else:
        verdict = f"PARTIAL: oracle ({large_oracle:.3f}) between reranker ({reranker_r15}) and SyGMa ({sygma_r15}). Some ranking headroom exists but coverage is also a bottleneck."

    results["interpretation"] = {
        "reranker_r15_reference": reranker_r15,
        "sygma_r15_reference": sygma_r15,
        "small_oracle_at_15": small_oracle,
        "large_oracle_at_15": large_oracle,
        "verdict": verdict,
    }

    out_path = ROOT / "results" / "gloryx_oracle.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[gloryx_oracle] results -> {out_path}", flush=True)

    # Summary table
    print("\n==== GLORYx pool-oracle summary ====")
    print(f"{'pool config':<35}  oracle@15   coverage")
    for key, r in [("small (top_k=100, max_pool=80)", small), ("large (top_k=200, max_pool=150)", large)]:
        print(f"  {key:<33}  {r['macro_oracle_recall_at_15']:.4f}    {r['macro_coverage']:.4f}")
    print(f"\nReranker recall@15 (reported): {reranker_r15}")
    print(f"SyGMa   recall@15 (reported): {sygma_r15}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()

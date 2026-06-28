#!/usr/bin/env python3
"""Evaluate GRAIL + SyGMa on the GLORYx external set (37 drugs / 136 metabolites) -- the
literature's de-facto shared test set -- under our standardized matching protocol.

This anchors GRAIL to published numbers (GLORYx 0.77, SyGMa 0.68, MetaPredictor 0.47, LAGOM
0.43, MetaTrans 0.35 at their k, from LAGOM Table 2) and produces raw predictions that feed
the match-sensitivity engine. GLORYx JSON: christinadebruynkops/GLORYx.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

from grail_metabolism.config import FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.workflows.factory import build_filter, build_generator

MODES = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
KS = [5, 10, 12, 15]
# Published recall on this set at each method's reported k (LAGOM 2025, Table 2) -- context.
PUBLISHED = {"GLORYx": 0.77, "SyGMa": 0.68, "MetaPredictor": 0.47, "LAGOM": 0.43, "MetaTrans": 0.35}


def _flatten(mets: List[dict]) -> List[str]:
    out = []
    for m in mets or []:
        if m.get("smiles"):
            out.append(m["smiles"])
        out.extend(_flatten(m.get("metabolites", [])))
    return out


def load_gloryx(path: Path) -> Dict[str, List[str]]:
    raw = path.read_text()
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)  # escape stray SMILES backslashes
    data = json.loads(fixed)
    return {p["smiles"]: _flatten(p.get("metabolites", [])) for p in data if p.get("smiles")}


def _load(path, build_fn):
    s = torch.load(path, map_location="cpu", weights_only=False)
    m = build_fn(s["arch"], s.get("rules"))
    m.load_state_dict(s["state_dict"], strict=False)
    m.calibrated_threshold = s.get("calibrated_threshold")
    return m


def _dedup_cap(smiles, mo):
    out, seen = [], set()
    for x in smiles:
        try:
            k = _tautomer_inchikey(x)
        except Exception:
            k = x
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
        if len(out) >= mo:
            break
    return out


def grail_predictions(parents, ckpt_dir, prior_strength, cap, mo) -> Dict[str, List[str]]:
    gen = _load(Path(ckpt_dir) / "generator.pt", lambda a, r: build_generator(GeneratorConfig(**a), r))
    gen.gen_normalization = "canonical"
    gen.prior_strength = float(prior_strength)
    filt = _load(Path(ckpt_dir) / "filter.pt", lambda a, r: build_filter(FilterConfig(**a)))
    thr = getattr(gen, "calibrated_threshold", None)
    out = {}
    t = time.perf_counter()
    for i, p in enumerate(parents, 1):
        if i == 1 or i % 10 == 0 or i == len(parents):
            print(f"  [grail] {i}/{len(parents)} ({time.perf_counter()-t:.0f}s)", flush=True)
        scored = gen.generate_scored(p, top_k=30, threshold=thr)[:cap]
        cands = [s for s, _ in scored]
        fs = filt.score_batch(p, cands) if cands else []
        combined = sorted(((s, float(f) * float(g)) for (s, g), f in zip(scored, fs)), key=lambda x: -x[1])
        out[p] = _dedup_cap([s for s, _ in combined], mo)
    return out


def sygma_predictions(parents) -> Dict[str, List[str]]:
    import sygma
    from rdkit import Chem

    sc = sygma.Scenario([[sygma.ruleset["phase1"], 1], [sygma.ruleset["phase2"], 1]])
    out = {}
    for p in parents:
        mol = Chem.MolFromSmiles(p)
        if mol is None:
            out[p] = []
            continue
        try:
            tree = sc.run(mol)
            tree.calc_scores()
            ranked = [e[0] for e in tree.to_smiles()]
        except Exception:
            ranked = []
        pk = _tautomer_inchikey(p)
        out[p] = [s for s in ranked if _tautomer_inchikey(s) != pk]
    return out


def score(preds, reals, ks):
    subs = [s for s in reals if reals[s]]
    return {
        mode: {f"recall@{k}": round(aggregate_prediction_metrics(
            [{"predicted": preds.get(s, []), "real": reals[s]} for s in subs], ks, match=mode).get(f"top_{k}_recall", 0.0), 3) for k in ks}
        for mode in MODES
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gloryx", default=str(ROOT / "docs" / "benchmark" / "data" / "gloryx_test.json"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "artifacts" / "full5000_priors" / "checkpoints"))
    ap.add_argument("--prior-strength", type=float, default=8.0)
    ap.add_argument("--filter-cap", type=int, default=32)
    ap.add_argument("--max-output", type=int, default=15)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--extra", nargs="*", default=[], help="Name=path.json files {parent:[preds]} (e.g. BioTransformer)")
    ap.add_argument("--no-grail", action="store_true")
    ap.add_argument("--no-sygma", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    reals = load_gloryx(Path(args.gloryx))
    parents = list(reals)
    print(f"GLORYx: {len(parents)} parents, {sum(len(v) for v in reals.values())} reference metabolites", flush=True)

    methods = {}
    if not args.no_grail:
        methods["GRAIL"] = grail_predictions(parents, args.ckpt_dir, args.prior_strength, args.filter_cap, args.max_output)
    if not args.no_sygma:
        methods["SyGMa"] = sygma_predictions(parents)
    for spec in args.extra:
        name, path = spec.split("=", 1)
        methods[name] = json.loads(Path(path).read_text())
    report = {"set": "GLORYx-37", "n_parents": len(parents), "prior_strength": args.prior_strength,
              "by_method": {m: score(p, reals, KS) for m, p in methods.items()}, "published_at_paper_k": PUBLISHED}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "gloryx_eval.json").write_text(json.dumps(report, indent=2))

    print("\n==== GLORYx-37: recall@k under our protocol (tautomer-InChIKey) ====", flush=True)
    print(f"{'method':<8} | " + " | ".join(f"r@{k}" for k in KS), flush=True)
    for m in methods:
        r = report["by_method"][m]["inchikey_tautomer"]
        print(f"{m:<8} | " + " | ".join(f"{r[f'recall@{k}']:.3f}" for k in KS), flush=True)
    print("\nmatch-protocol sensitivity (recall@15):", flush=True)
    print(f"{'method':<8} | " + " | ".join(f"{m[:9]:>9}" for m in MODES), flush=True)
    for me in methods:
        bm = report["by_method"][me]
        print(f"{me:<8} | " + " | ".join(f"{bm[mode]['recall@15']:>9.3f}" for mode in MODES), flush=True)
    print(f"\npublished (paper's own k/match): {PUBLISHED}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

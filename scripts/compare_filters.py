#!/usr/bin/env python3
"""Single-encoder vs MCS-aware pair filter — a paired, apples-to-apples comparison.

Both filters are scored on the IDENTICAL broad candidate pool produced by the same deployed
generator (full5000_priors), ranked the same way (`filter_score × generator_score`), on the same
full clean test split. The only variable is the filter architecture:

  single : artifacts/full5000_single  (mode='single', two independent 16-dim encoders + dual Morgan)
  pair   : artifacts/full5000_pair    (mode='pair',   one merged 18-dim MCS graph + dual Morgan)

The pair filter (scripts/train_pair_filter.py) was trained on the same data, splits, seed, and
filter_optim as the deployed single filter. Reports recall@{5,10,15} + precision for each and a
paired bootstrap CI on the per-substrate recall@15 delta (pair − single).

Shardable like eval_hybrid_rerank.py: `--start/--end/--rows-out` dump raw rows for a slice; then
`--merge 'shard_*.json'` combines them into the headline. Without sharding it evaluates and
aggregates in one process.
"""
from __future__ import annotations

import argparse
import glob
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
from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.stats import paired_diff_bootstrap_ci
from grail_metabolism.utils.transform import from_rdmol
from grail_metabolism.workflows.factory import build_filter, build_generator
from scripts.run_benchmark import load_test_map

DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
SINGLE_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
PAIR_FILTER = ROOT / "artifacts" / "full5000_pair" / "checkpoints" / "filter.pt"
KS = [5, 10, 15]
MAX_OUTPUT = 15


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    model.eval()
    return model


def _dedup_top(cands_scored, mo):
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


def _r15(pred, real):
    tr = {_tautomer_inchikey(s) for s in real}
    if not tr:
        return 0.0
    tp = {_tautomer_inchikey(s) for s in pred[:MAX_OUTPUT]}
    return len(tp & tr) / len(tr)


def _report(rows, n_shards, top_k):
    report = {"n": len(rows["single"]), "top_k": top_k, "n_shards": n_shards, "rankings": {}}
    for key, rws in rows.items():
        m = aggregate_prediction_metrics(rws, KS, match="inchikey_tautomer")
        report["rankings"][key] = {
            "recall@15": round(m.get("top_15_recall", 0.0), 4),
            "recall@5": round(m.get("top_5_recall", 0.0), 4),
            "precision": round(m.get("precision", 0.0), 4),
            "mean_output": round(m.get("mean_output_size", 0.0), 2),
        }
    deltas = [_r15(p["predicted"], p["real"]) - _r15(s["predicted"], s["real"])
              for s, p in zip(rows["single"], rows["pair"])]
    pt, lo, hi = paired_diff_bootstrap_ci(deltas)
    report["paired_delta_recall15_pair_minus_single"] = {"point": round(pt, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
    return report


def _print(report):
    print(f"\n=== single vs pair filter (same generator + pool, n={report['n']}) ===", flush=True)
    for key, v in report["rankings"].items():
        print(f"  {key:8s}: recall@15 {v['recall@15']}  recall@5 {v['recall@5']}  prec {v['precision']}  out {v['mean_output']}", flush=True)
    d = report["paired_delta_recall15_pair_minus_single"]
    sig = "SIGNIFICANT (>0)" if d["lo"] > 0 else ("SIGNIFICANT (<0)" if d["hi"] < 0 else "n.s. (CI includes 0)")
    print(f"  paired pair−single recall@15: {d['point']:+.4f}  95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]  -> {sig}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = to the end")
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--filter-cap", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--single-filter", default=str(SINGLE_FILTER), help="path to the single-mode filter checkpoint")
    ap.add_argument("--pair-filter", default=str(PAIR_FILTER), help="path to the pair-mode filter checkpoint")
    ap.add_argument("--rows-out", default="", help="dump raw per-substrate rows for this slice and skip aggregation")
    ap.add_argument("--merge", default="", help="glob of shard rows files to merge into the headline")
    ap.add_argument("--out", default=str(ROOT / "results" / "filter_compare_full1170.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.merge:
        paths = sorted({p for pat in [args.merge] for p in glob.glob(pat)})
        if not paths:
            print("no shard files matched", flush=True)
            return 1
        rows = {"single": [], "pair": []}
        top_k = None
        for p in paths:
            blob = json.loads(Path(p).read_text())
            top_k = blob.get("top_k")
            for k in rows:
                rows[k].extend(blob["rows"][k])
            print(f"  + {Path(p).name}: {blob.get('n')} substrates", flush=True)
        report = _report(rows, len(paths), top_k)
        Path(args.out).write_text(json.dumps(report, indent=2))
        _print(report)
        print(f"Wrote {args.out}", flush=True)
        return 0

    generator = _load(DEPLOYED_GEN, lambda a, r: build_generator(GeneratorConfig(**a), r))
    assert float(generator.rule_prior_logits.std()) > 0.1, "degenerate prior; use full5000_priors"
    generator.gen_normalization = "canonical"
    gen_threshold = getattr(generator, "calibrated_threshold", None)
    single = _load(args.single_filter, lambda a, r: build_filter(FilterConfig(**a)))
    pair = _load(args.pair_filter, lambda a, r: build_filter(FilterConfig(**a)))
    assert single.mode == "single" and pair.mode == "pair", f"unexpected modes {single.mode}/{pair.mode}"

    test_map = load_test_map(None, 42)
    all_items = list(test_map.items())
    items = all_items[args.start : (args.end or None)] if (args.start or args.end) else all_items
    print(f"substrates: {len(items)} [{args.start}:{args.end or len(all_items)}]  top_k={args.top_k}", flush=True)

    rows = {"single": [], "pair": []}
    t0 = time.time()
    for i, (sub, prods) in enumerate(items, 1):
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        real = sorted(prods)
        sub_mol = Chem.MolFromSmiles(sub)
        data = from_rdmol(sub_mol) if sub_mol is not None else None
        if data is None or data.x.size(0) == 0:
            rows["single"].append({"predicted": [], "real": real})
            rows["pair"].append({"predicted": [], "real": real})
            continue
        detailed = generator.generate_scored_with_details(sub, top_k=args.top_k, threshold=gen_threshold, compute_sites=False)[: args.filter_cap]
        if not detailed:
            rows["single"].append({"predicted": [], "real": real})
            rows["pair"].append({"predicted": [], "real": real})
            continue
        smis = [d[0] for d in detailed]
        gscores = [float(d[1]) for d in detailed]
        s_scores = single.score_batch(sub, smis)
        p_scores = pair.score_batch(sub, smis)
        s_ranked = [(smi, float(fs) * g) for smi, g, fs in zip(smis, gscores, s_scores)]
        p_ranked = [(smi, float(fs) * g) for smi, g, fs in zip(smis, gscores, p_scores)]
        rows["single"].append({"predicted": _dedup_top(s_ranked, MAX_OUTPUT), "real": real})
        rows["pair"].append({"predicted": _dedup_top(p_ranked, MAX_OUTPUT), "real": real})

    if args.rows_out:
        Path(args.rows_out).write_text(json.dumps({"n": len(items), "top_k": args.top_k, "rows": rows}))
        print(f"\nShard done: {len(items)} substrates -> {args.rows_out}", flush=True)
        return 0

    report = _report(rows, 1, args.top_k)
    Path(args.out).write_text(json.dumps(report, indent=2))
    _print(report)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

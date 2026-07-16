#!/usr/bin/env python3
"""D3 joint-training eval: bolt-on vs joint factorized re-ranker on the identical pool (paired).

Scores one broad candidate pool per substrate (deployed generator + filter) and ranks it three ways
with the SAME deployed FactorizedReranker code path, changing only the factorized checkpoint:

  a  filter*gen                     -- baseline (no factorized signal)
  b  filter*gen*type*site [v1]      -- the shipped bolt-on re-ranker (artifacts/factorized_v1)
  c  filter*gen*type*site [joint]   -- the joint-trained heads (artifacts/factorized_joint)

Reports recall@{5,10,15} + precision for each and paired bootstrap CIs on the per-substrate
recall@15 deltas c-b (joint vs bolt-on -- the D3 question), b-a and c-a. Shardable
(`--start/--end/--rows-out`) with a `--merge` headline mode, like scripts/compare_filters.py.
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
from grail_metabolism.model.factorized_infer import FactorizedReranker
from grail_metabolism.stats import paired_diff_bootstrap_ci
from grail_metabolism.utils.transform import from_rdmol
from grail_metabolism.workflows.factory import build_filter, build_generator
from scripts.run_benchmark import load_test_map

DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
DEPLOYED_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
BOLTON = ROOT / "artifacts" / "factorized_v1" / "checkpoints" / "factorized.pt"
JOINT = ROOT / "artifacts" / "factorized_joint" / "checkpoints" / "factorized.pt"
VOCAB = ROOT / "grail_metabolism" / "resources" / "coarse_type_vocab.json"
KS = [5, 10, 15]
MAX_OUTPUT = 15


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    model.eval()
    return model


def _dedup_top(cands, mo):
    out, seen = [], set()
    for smi, _ in sorted(cands, key=lambda x: -x[1]):
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
    report = {"n": len(rows["a"]), "top_k": top_k, "n_shards": n_shards, "rankings": {}}
    labels = {"a": "filter_gen", "b": "boltclon_v1", "c": "joint"}
    for key, rws in rows.items():
        m = aggregate_prediction_metrics(rws, KS, match="inchikey_tautomer")
        report["rankings"][labels[key]] = {
            "recall@15": round(m.get("top_15_recall", 0.0), 4),
            "recall@5": round(m.get("top_5_recall", 0.0), 4),
            "precision": round(m.get("precision", 0.0), 4),
            "mean_output": round(m.get("mean_output_size", 0.0), 2),
        }
    for name, (x, y) in (("c_minus_b_joint_vs_bolton", ("c", "b")), ("b_minus_a", ("b", "a")), ("c_minus_a", ("c", "a"))):
        deltas = [_r15(rows[x][i]["predicted"], rows[x][i]["real"]) - _r15(rows[y][i]["predicted"], rows[y][i]["real"])
                  for i in range(len(rows["a"]))]
        pt, lo, hi = paired_diff_bootstrap_ci(deltas)
        report.setdefault("paired_delta_recall15", {})[name] = {"point": round(pt, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
    return report


def _print(report):
    print(f"\n=== joint vs bolt-on re-ranker (same pool, n={report['n']}) ===", flush=True)
    for key, v in report["rankings"].items():
        print(f"  {key:12s}: recall@15 {v['recall@15']}  recall@5 {v['recall@5']}  prec {v['precision']}  out {v['mean_output']}", flush=True)
    for name, v in report["paired_delta_recall15"].items():
        sig = "SIGNIFICANT (>0)" if v["lo"] > 0 else ("SIGNIFICANT (<0)" if v["hi"] < 0 else "n.s.")
        print(f"  paired {name}: {v['point']:+.4f}  95% CI [{v['lo']:+.4f}, {v['hi']:+.4f}]  -> {sig}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--filter-cap", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rows-out", default="")
    ap.add_argument("--merge", default="")
    ap.add_argument("--out", default=str(ROOT / "results" / "joint_rerank.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.merge:
        rows = {"a": [], "b": [], "c": []}
        top_k = None
        for p in sorted(glob.glob(args.merge)):
            blob = json.loads(Path(p).read_text())
            top_k = blob.get("top_k")
            for k in rows:
                rows[k].extend(blob["rows"][k])
            print(f"  + {Path(p).name}: {blob.get('n')} subs", flush=True)
        report = _report(rows, len(glob.glob(args.merge)), top_k)
        Path(args.out).write_text(json.dumps(report, indent=2))
        _print(report)
        print(f"Wrote {args.out}", flush=True)
        return 0

    generator = _load(DEPLOYED_GEN, lambda a, r: build_generator(GeneratorConfig(**a), r))
    assert float(generator.rule_prior_logits.std()) > 0.1, "degenerate prior; use full5000_priors"
    generator.gen_normalization = "canonical"
    gen_threshold = getattr(generator, "calibrated_threshold", None)
    filt = _load(DEPLOYED_FILTER, lambda a, r: build_filter(FilterConfig(**a)))
    bolton = FactorizedReranker.load(BOLTON, VOCAB, generator.rule_names)
    joint = FactorizedReranker.load(JOINT, VOCAB, generator.rule_names)

    all_items = list(load_test_map(None, 42).items())
    items = all_items[args.start:(args.end or None)] if (args.start or args.end) else all_items
    print(f"substrates: {len(items)} [{args.start}:{args.end or len(all_items)}]  top_k={args.top_k}", flush=True)

    rows = {"a": [], "b": [], "c": []}
    t0 = time.time()
    for i, (sub, prods) in enumerate(items, 1):
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        real = sorted(prods)
        sub_mol = Chem.MolFromSmiles(sub)
        data = from_rdmol(sub_mol) if sub_mol is not None else None
        if data is None or data.x.size(0) == 0:
            for k in rows:
                rows[k].append({"predicted": [], "real": real})
            continue
        detailed = generator.generate_scored_with_details(sub, top_k=args.top_k, threshold=gen_threshold, compute_sites=False)[: args.filter_cap]
        if not detailed:
            for k in rows:
                rows[k].append({"predicted": [], "real": real})
            continue
        smis = [d[0] for d in detailed]
        gs = [float(d[1]) for d in detailed]
        fs = filt.score_batch(sub, smis)
        mb = bolton.multipliers(sub_mol, detailed)
        mc = joint.multipliers(sub_mol, detailed)
        a = [(smi, float(f) * g) for smi, g, f in zip(smis, gs, fs)]
        b = [(smi, float(f) * g * m) for smi, g, f, m in zip(smis, gs, fs, mb)]
        c = [(smi, float(f) * g * m) for smi, g, f, m in zip(smis, gs, fs, mc)]
        rows["a"].append({"predicted": _dedup_top(a, MAX_OUTPUT), "real": real})
        rows["b"].append({"predicted": _dedup_top(b, MAX_OUTPUT), "real": real})
        rows["c"].append({"predicted": _dedup_top(c, MAX_OUTPUT), "real": real})

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

#!/usr/bin/env python3
"""Merge sharded eval_hybrid_rerank.py row dumps into one full-test headline.

Each shard was run with `--start S --end E --rows-out shard.json`, producing raw
per-substrate rows for the three rankings (a=filter x gen, b=filter x type x site,
c=filter x gen x type x site) over a disjoint slice of the 1170-substrate test map.

Concatenating the shard row lists reconstructs exactly the `rows` dict a single
full run would have built (order within each shard is preserved, and a[i]/b[i]/c[i]
stay aligned to the same substrate), so the aggregate metrics and the paired
bootstrap CI computed here are identical to the non-sharded path in
eval_hybrid_rerank.main(). Pass the shard files in any order.

    python scripts/merge_hybrid_shards.py results/hybrid_shard_*.json --out results/hybrid_rerank_full1170.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.stats import paired_diff_bootstrap_ci

KS = [5, 10, 15]
MAX_OUTPUT = 15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="shard rows JSON files (globs allowed)")
    ap.add_argument("--out", default=str(ROOT / "results" / "hybrid_rerank_full1170.json"))
    args = ap.parse_args()

    paths = sorted({p for pat in args.shards for p in glob.glob(pat)})
    if not paths:
        print("no shard files matched", flush=True)
        return 1

    merged = {"a_filter_gen": [], "b_filter_type_site": [], "c_all": []}
    top_ks = set()
    for p in paths:
        blob = json.loads(Path(p).read_text())
        top_ks.add(blob.get("top_k"))
        rows = blob["rows"]
        for key in merged:
            merged[key].extend(rows[key])
        print(f"  + {Path(p).name}: {blob.get('n')} substrates", flush=True)

    n = len(merged["a_filter_gen"])
    assert n == len(merged["b_filter_type_site"]) == len(merged["c_all"]), "ranking row-count mismatch"
    if len(top_ks) != 1:
        print(f"WARNING: shards used different top_k values: {top_ks}", flush=True)

    report = {"n": n, "top_k": sorted(top_ks)[-1], "baseline_broad_filter": 0.413,
              "n_shards": len(paths), "rankings": {}}
    for key, rws in merged.items():
        m = aggregate_prediction_metrics(rws, KS, match="inchikey_tautomer")
        report["rankings"][key] = {
            "recall@15": round(m.get("top_15_recall", 0.0), 4),
            "recall@5": round(m.get("top_5_recall", 0.0), 4),
            "precision": round(m.get("precision", 0.0), 4),
            "mean_output": round(m.get("mean_output_size", 0.0), 2),
        }

    def _r15(pred, real):
        tr = {_tautomer_inchikey(s) for s in real}
        if not tr:
            return 0.0
        tp = {_tautomer_inchikey(s) for s in pred[:MAX_OUTPUT]}
        return len(tp & tr) / len(tr)

    a_rows, b_rows, c_rows = merged["a_filter_gen"], merged["b_filter_type_site"], merged["c_all"]
    deltas_ca = [_r15(c["predicted"], c["real"]) - _r15(a["predicted"], a["real"]) for a, c in zip(a_rows, c_rows)]
    deltas_ba = [_r15(b["predicted"], b["real"]) - _r15(a["predicted"], a["real"]) for a, b in zip(a_rows, b_rows)]
    for name, diffs in (("c_minus_a", deltas_ca), ("b_minus_a", deltas_ba)):
        pt, lo, hi = paired_diff_bootstrap_ci(diffs)
        report.setdefault("paired_delta_recall15", {})[name] = {"point": round(pt, 4), "lo": round(lo, 4), "hi": round(hi, 4)}

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n=== merged full-test hybrid re-ranking (n={n}, {len(paths)} shards) ===", flush=True)
    for key, v in report["rankings"].items():
        print(f"  {key:20s}: recall@15 {v['recall@15']}  recall@5 {v['recall@5']}  prec {v['precision']}  out {v['mean_output']}", flush=True)
    for name, v in report["paired_delta_recall15"].items():
        sig = "SIGNIFICANT (>0)" if v["lo"] > 0 else "n.s. (CI includes 0)"
        print(f"  paired {name} recall@15: {v['point']:+.4f}  95% CI [{v['lo']:+.4f}, {v['hi']:+.4f}]  -> {sig}", flush=True)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Recall<->precision frontier via calibrated abstention (D3).

The deployed ranking is rank-only (`filter*gen`, top-15, never gates) because a hard filter gate
hurts recall@k. But precision is currently just a pessimistic lower bound; abstention lets the
operating point be *chosen* deliberately. Abstention gates a candidate when its filter score is
below a threshold tau, then ranks the survivors by `filter*gen` and takes the top max_output. As
tau rises the output set shrinks: precision up, recall down -- a frontier.

tau only changes WHICH candidates survive, not their scores, so we score each substrate's ranked
candidates + filter scores ONCE (`--split {val,test} --rows-out`, shardable) and then sweep the
whole tau grid analytically (`--frontier`). Operating points are selected on VAL (max-recall = no
gate; F1-max; precision>=target) and reported on TEST -- the select-on-val invariant.
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

from grail_metabolism.config import FilterConfig, GeneratorConfig
from grail_metabolism.metrics import _tautomer_inchikey, aggregate_prediction_metrics
from grail_metabolism.utils.preparation import _normalize_smiles_cached
from grail_metabolism.workflows.factory import build_filter, build_generator
from scripts.run_benchmark import _load_ids_to_smiles, _read_positives

DATA = ROOT / "grail_metabolism" / "data"
DEPLOYED_GEN = ROOT / "artifacts" / "full5000_priors" / "checkpoints" / "generator.pt"
DEPLOYED_FILTER = ROOT / "artifacts" / "full5000_single" / "checkpoints" / "filter.pt"
KS = [5, 10, 15]
MAX_OUTPUT = 15
TAU_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _load(path, build_fn):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_fn(state["arch"], state.get("rules"))
    model.load_state_dict(state["state_dict"], strict=False)
    model.calibrated_threshold = state.get("calibrated_threshold")
    model.eval()
    return model


def load_split_map(split: str):
    triples = DATA / f"{split}_triples_clean.txt"
    if not triples.exists():
        triples = DATA / f"{split}_triples.txt"
    positives = _read_positives(triples)
    ids = {a for a, _ in positives} | {b for _, b in positives}
    id2smi = _load_ids_to_smiles(DATA / f"{split}.sdf", ids)
    smap = {}
    for s_id, p_id in positives:
        s, p = id2smi.get(s_id), id2smi.get(p_id)
        if s and p:
            smap.setdefault(s, set()).add(p)
    return smap


def _dedup_top(gated, mo):
    """gated: list[(smi, combined)] -> tautomer-deduped top-mo smiles by combined desc."""
    out, seen = [], set()
    for smi, _ in sorted(gated, key=lambda x: -x[1]):
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


def _metrics_at_tau(rows, tau):
    """rows: list of {ranked:[(smi,fscore,comb)], real:[...]}. Gate by fscore>=tau, rank by comb."""
    pred_rows = []
    for r in rows:
        gated = [(smi, comb) for smi, fscore, comb in r["ranked"] if fscore >= tau]
        pred_rows.append({"predicted": _dedup_top(gated, MAX_OUTPUT), "real": r["real"]})
    m = aggregate_prediction_metrics(pred_rows, KS, match="inchikey_tautomer")
    return {
        "tau": round(tau, 3),
        "recall@15": round(m.get("top_15_recall", 0.0), 4),
        "recall@5": round(m.get("top_5_recall", 0.0), 4),
        "precision": round(m.get("precision", 0.0), 4),
        "f1": round(m.get("f1", 0.0), 4),
        "mean_output": round(m.get("mean_output_size", 0.0), 2),
    }


def score_rows(items, generator, filt, gen_threshold, top_k, cap):
    rows = []
    t0 = time.time()
    for i, (sub, prods) in enumerate(items, 1):
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        real = sorted(prods)
        scored = generator.generate_scored(sub, top_k=top_k, threshold=gen_threshold)[:cap]
        if not scored:
            rows.append({"ranked": [], "real": real})
            continue
        smis = [_normalize_smiles_cached(s, "canonical") for s, _ in scored]
        fscores = filt.score_batch(sub, smis)
        ranked = [(smi, float(fs), float(fs) * float(g)) for (raw, g), smi, fs in zip(scored, smis, fscores)]
        rows.append({"ranked": ranked, "real": real})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--filter-cap", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rows-out", default="")
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--val", default="", help="glob of val rows shards (frontier mode)")
    ap.add_argument("--test", default="", help="glob of test rows shards (frontier mode)")
    ap.add_argument("--out", default=str(ROOT / "results" / "abstention_frontier.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.frontier:
        def _load_rows(pat):
            rows = []
            for p in sorted(glob.glob(pat)):
                rows.extend(json.loads(Path(p).read_text())["rows"])
            return rows
        val_rows, test_rows = _load_rows(args.val), _load_rows(args.test)
        print(f"val {len(val_rows)} substrates, test {len(test_rows)} substrates", flush=True)
        val_curve = [_metrics_at_tau(val_rows, t) for t in TAU_GRID]
        test_curve = [_metrics_at_tau(test_rows, t) for t in TAU_GRID]

        # Operating points selected on VAL, reported on TEST.
        def _test_at(tau):
            return _metrics_at_tau(test_rows, tau)
        points = {}
        points["max_recall"] = {"val_selected_tau": 0.0, "test": _test_at(0.0), "rule": "no gate (rank-only)"}
        f1_best = max(val_curve, key=lambda r: r["f1"])
        points["f1_max"] = {"val_selected_tau": f1_best["tau"], "val_f1": f1_best["f1"], "test": _test_at(f1_best["tau"])}
        for target in (0.2, 0.3, 0.5):
            clearing = [r for r in val_curve if r["precision"] >= target]
            if clearing:
                best = max(clearing, key=lambda r: r["recall@15"])  # highest-recall tau clearing target
                points[f"precision_ge_{target}"] = {"val_selected_tau": best["tau"], "val_precision": best["precision"], "test": _test_at(best["tau"])}
            else:
                points[f"precision_ge_{target}"] = {"val_selected_tau": None, "note": "no tau on val reaches this precision"}

        report = {"n_val": len(val_rows), "n_test": len(test_rows), "tau_grid": TAU_GRID,
                  "val_curve": val_curve, "test_curve": test_curve, "operating_points": points}
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("\n=== recall<->precision frontier (test) ===", flush=True)
        for r in test_curve:
            print(f"  tau {r['tau']:.2f}: recall@15 {r['recall@15']}  prec {r['precision']}  f1 {r['f1']}  out {r['mean_output']}", flush=True)
        print("\n=== operating points (val-selected -> test) ===", flush=True)
        for name, pt in points.items():
            t = pt.get("test")
            if t:
                print(f"  {name:16s} tau={pt.get('val_selected_tau')}: test recall@15 {t['recall@15']}  prec {t['precision']}  f1 {t['f1']}  out {t['mean_output']}", flush=True)
            else:
                print(f"  {name:16s}: {pt.get('note')}", flush=True)
        print(f"Wrote {args.out}", flush=True)
        return 0

    # scoring mode
    generator = _load(DEPLOYED_GEN, lambda a, r: build_generator(GeneratorConfig(**a), r))
    assert float(generator.rule_prior_logits.std()) > 0.1, "degenerate prior; use full5000_priors"
    generator.gen_normalization = "canonical"
    filt = _load(DEPLOYED_FILTER, lambda a, r: build_filter(FilterConfig(**a)))
    gen_threshold = getattr(generator, "calibrated_threshold", None)

    smap = load_split_map(args.split)
    all_items = sorted(smap.items())
    items = all_items[args.start:(args.end or None)] if (args.start or args.end) else all_items
    print(f"{args.split}: {len(items)} substrates [{args.start}:{args.end or len(all_items)}]  top_k={args.top_k}", flush=True)
    rows = score_rows(items, generator, filt, gen_threshold, args.top_k, args.filter_cap)
    out = args.rows_out or str(ROOT / "results" / f"abstention_rows_{args.split}.json")
    Path(out).write_text(json.dumps({"split": args.split, "n": len(rows), "rows": rows}))
    print(f"Wrote {out} ({len(rows)} substrates)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

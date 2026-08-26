"""The H8 check: the group scorer against rank fusion, on the 291.

H8 predicts the scorer beats rank fusion by at least +0.05 of micro recall@15, with a paired
bootstrap excluding zero, and registers an operational corollary: that the gap to MetaTox turns
positive at the same budget with its interval excluding zero. Both are reported whichever way
each falls.

The model is loaded, not fitted here. Its architecture and every hyperparameter were selected on
validation by train_group_scorer.py, and this reads the 291 once.

One bound the registration did not anticipate and the artifact records: the group oracle on this
pool reaches 0.6752 against fusion's 0.5023, so +0.1729 is the most any reordering of groups can
buy. A margin near that is the scorer approaching the ceiling; a margin far below it is the
scorer failing to, and the two readings should not be confused.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from bank_without_selection import _dedup, _load  # noqa: E402
from grail_metabolism.config import GeneratorConfig  # noqa: E402
from grail_metabolism.workflows.factory import build_generator  # noqa: E402
from group_scorer import Featuriser, GroupScorer, build_examples  # noqa: E402

THRESHOLD, N_BOOT, SEED = 0.05, 10000, 0
KS = (1, 5, 10, 15, 20, 30, 50)
METATOX = ROOT / "results/metatox_smirks_preds.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="results/widepools_implicit/w*.json")
    ap.add_argument("--model", default=str(ROOT / "artifacts/group_scorer.pt"))
    ap.add_argument("--gen-ckpt",
                    default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/h8_verdict.json"))
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    pools, refs = {}, {}
    for f in sorted(glob.glob(args.pools)) or [args.pools]:
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    generator.eval()
    feat = Featuriser(generator)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = GroupScorer(ck["in_dim"], ck["config"]["hidden"], ck["config"]["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ex = build_examples(pools, refs, feat)
    subs = [e["sub"] for e in ex]
    mtx = json.loads(METATOX.read_text())["predictions"]
    mt = {s: _dedup(mtx.get(s, []), max(KS)) for s in subs}

    h_sc = {k: [] for k in KS}
    h_fu = {k: [] for k in KS}
    h_or = {k: [] for k in KS}
    h_mt = {k: [] for k in KS}
    n_ref = []
    with torch.no_grad():
        for e in ex:
            real = e["real"]
            n_ref.append(len(real))
            s = model(torch.from_numpy(e["X"])).numpy()
            by_model = [c["key"] for i in np.argsort(-s)
                        for c in e["by_g"][e["names"][i]]]
            by_fusion = [c["key"] for g in e["names"] for c in e["by_g"][g]]
            hit_first = sorted(range(len(e["names"])),
                               key=lambda i: (not any(c["key"] in real
                                                      for c in e["by_g"][e["names"][i]]), i))
            by_oracle = [c["key"] for i in hit_first for c in e["by_g"][e["names"][i]]]
            for k in KS:
                h_sc[k].append(len(set(by_model[:k]) & real))
                h_fu[k].append(len(set(by_fusion[:k]) & real))
                h_or[k].append(len(set(by_oracle[:k]) & real))
                h_mt[k].append(len(set(mt[e["sub"]][:k]) & real))

    U = np.array(n_ref, dtype=float)
    N = float(U.sum())
    A = {n: {k: np.array(v[k], dtype=float) for k in KS}
         for n, v in (("scorer", h_sc), ("fusion", h_fu), ("oracle", h_or), ("metatox", h_mt))}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ex), (N_BOOT, len(ex)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def contrast(a, b):
        d = a - b
        bt = d[idx].sum(axis=1) / denom
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(d.sum() / N), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    kk = args.k
    primary = contrast(A["scorer"][kk], A["fusion"][kk])
    corollary = contrast(A["scorer"][kk], A["metatox"][kk])
    headroom = float((A["oracle"][kk].sum() - A["fusion"][kk].sum()) / N)

    verdict = "supported" if (primary["gap"] >= THRESHOLD and primary["excludes_zero"]) \
        else "failed"
    out = {"provenance": stamp(__file__), "hypothesis": "H8",
           "registered_threshold": THRESHOLD, "k": kk,
           "population": {"n": len(ex), "n_references": N,
                          "source": "the 291 of results/four_method_291.json"},
           "aggregation": "micro, ratio of sums",
           "model": {"selected_on": "validation", **ck["config"],
                     "val_recall@15": ck.get("val_recall@15"),
                     "val_fusion_baseline": ck.get("fusion_baseline")},
           "recall_micro": {str(k): {n: round(float(A[n][k].sum() / N), 4) for n in A}
                            for k in KS},
           "primary_scorer_minus_fusion": primary,
           "corollary_scorer_minus_metatox": corollary,
           "group_oracle_headroom_over_fusion": round(headroom, 4),
           "share_of_headroom_taken": round(primary["gap"] / headroom, 4) if headroom else None,
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"H8 on {len(ex)} substrates, {N:.0f} references")
    print(f"\n{'k':>4}{'scorer':>10}{'fusion':>10}{'oracle':>10}{'metatox':>10}")
    for k in KS:
        r = out["recall_micro"][str(k)]
        print(f"{k:>4}{r['scorer']:>10.4f}{r['fusion']:>10.4f}"
              f"{r['oracle']:>10.4f}{r['metatox']:>10.4f}")
    print(f"\n  primary   scorer - fusion  {primary['gap']:+.4f} "
          f"[{primary['ci95'][0]:+.4f}, {primary['ci95'][1]:+.4f}]"
          f"{'*' if primary['excludes_zero'] else ''}   threshold +{THRESHOLD}")
    print(f"  corollary scorer - metatox {corollary['gap']:+.4f} "
          f"[{corollary['ci95'][0]:+.4f}, {corollary['ci95'][1]:+.4f}]"
          f"{'*' if corollary['excludes_zero'] else ''}")
    print(f"  headroom the group oracle leaves: {headroom:+.4f}, "
          f"of which the scorer takes {out['share_of_headroom_taken']}")
    print(f"  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

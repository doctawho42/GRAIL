"""The H14 check: the group signal as a gate on the cap rather than a third ranking.

H14 predicts the gate beats the H9 configuration -- the top hundred by generator score, then the
H7 fusion -- by at least +0.02 of micro recall@15 on the 291, with the paired bootstrap excluding
zero. The model is the one H8 registered, with hyperparameters re-selected on validation under
this composition, and this reads the 291 once.

Four arms. The gate and the cap are the registered contrast; the two others say what kind of
result it is. `gate_oracle` admits groups in the order of a perfect group ranking, which bounds
what any scorer could reach through this gate. `three_way` is H12's composition on the same
pools, so the two ways of spending the group signal can be read against each other.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402
from bank_without_selection import _load  # noqa: E402
from grail_metabolism.config import GeneratorConfig  # noqa: E402
from grail_metabolism.workflows.factory import build_generator  # noqa: E402
from group_scorer import (Featuriser, GroupScorer, Standardiser, cap_order,  # noqa: E402
                          gate_order, three_way_order)

THRESHOLD, BUDGET = 0.02, 100
N_BOOT, SEED = 10000, 0
KS = (1, 5, 10, 15, 20, 30, 50)
ARMS = ("gate", "cap", "gate_oracle", "three_way")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="results/widepools_implicit/w*.json")
    ap.add_argument("--model", default=str(ROOT / "artifacts/group_scorer_h14.pt"))
    ap.add_argument("--three-way-model", default=str(ROOT / "artifacts/group_scorer_h12.pt"))
    ap.add_argument("--gen-ckpt",
                    default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/h14_verdict.json"))
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    pools, refs = {}, {}
    for f in sorted(glob.glob(args.pools)) or [args.pools]:
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    subs = sorted(s for s in pools if refs.get(s))

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    generator.eval()
    feat = Featuriser(generator)

    def load_model(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = GroupScorer(ck["in_dim"], ck["config"]["hidden"], ck["config"]["dropout"])
        m.load_state_dict(ck["state_dict"]); m.eval()
        return m, Standardiser.load(ck["standardiser"]), ck

    model, scaler, ck = load_model(args.model)
    m12, sc12, _ = load_model(args.three_way_model)

    hits = {a: {k: [] for k in KS} for a in ARMS}
    n_ref = []
    with torch.no_grad():
        for n, s in enumerate(subs, 1):
            if n % 50 == 0:
                print(f"  {n}/{len(subs)}", file=sys.stderr, flush=True)
            real = set(refs[s])
            n_ref.append(len(real))
            by_g = defaultdict(list)
            for c in rrf_order(pools[s]):
                by_g[feat.formula(c["smiles"])].append(c)
            e = {"names": list(by_g), "by_g": by_g, "real": real}

            cap = cap_order(e, BUDGET)
            if len(e["names"]) >= 2:
                _, _, X = feat.features(s, pools[s])
                sc = model(torch.from_numpy(
                    ((X - scaler.mean) / scaler.std).astype(np.float32))).numpy()
                gate = gate_order(sc, e, BUDGET)
                truth = np.array([1.0 if any(c["key"] in real for c in by_g[g]) else 0.0
                                  for g in e["names"]], dtype=np.float32)
                orc = gate_order(truth, e, BUDGET)
                s12 = m12(torch.from_numpy(
                    ((X - sc12.mean) / sc12.std).astype(np.float32))).numpy()
                tw = three_way_order(s12, e)
            else:
                gate = orc = tw = cap

            seq = {"gate": gate, "cap": cap, "gate_oracle": orc, "three_way": tw}
            for a in ARMS:
                for k in KS:
                    hits[a][k].append(len(set(seq[a][:k]) & real))

    U = np.array(n_ref, dtype=float)
    N = float(U.sum())
    H = {a: {k: np.array(v[k], dtype=float) for k in KS} for a, v in hits.items()}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    den = np.maximum(U[idx].sum(axis=1), 1)

    def contrast(a, b):
        d = a - b
        bt = d[idx].sum(axis=1) / den
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(d.sum() / N), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    kk = args.k
    primary = contrast(H["gate"][kk], H["cap"][kk])
    ceiling = contrast(H["gate_oracle"][kk], H["cap"][kk])
    vs12 = contrast(H["gate"][kk], H["three_way"][kk])
    verdict = "supported" if (primary["gap"] >= THRESHOLD and primary["excludes_zero"]) \
        else "failed"

    out = {"provenance": stamp(__file__), "hypothesis": "H14",
           "registered_threshold": THRESHOLD, "budget": BUDGET, "k": kk,
           "population": {"n": len(subs), "n_references": N,
                          "source": "the 291 of results/four_method_291.json"},
           "aggregation": "micro, ratio of sums",
           "model": {"selected_on": "validation under this composition", **ck["config"],
                     "val_recall@15": ck.get("val_recall@15"),
                     "val_cap_baseline": ck.get("fusion_baseline")},
           "recall_micro": {str(k): {a: round(float(H[a][k].sum() / N), 4) for a in ARMS}
                            for k in KS},
           "primary_gate_minus_cap": primary,
           "ceiling_of_this_gate": ceiling,
           "gate_minus_three_way": vs12,
           # A share of a ceiling is only a share while the ceiling is a gain. When a perfect
           # group ranking is itself worse than the arm it is meant to bound, the ratio of two
           # negative numbers reads as four-fifths of an achievement and describes none.
           "share_of_ceiling_taken": (round(primary["gap"] / ceiling["gap"], 4)
                                      if ceiling["gap"] > 0 else None),
           "ceiling_is_negative": bool(ceiling["gap"] <= 0),
           "ceiling_note": ("a perfect group ranking used as this gate is itself worse than the "
                            "H9 cap, so the design is bounded below the thing it must beat and "
                            "no scorer reaches the threshold through it"
                            if ceiling["gap"] <= 0 else None),
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"\nH14 on {len(subs)} substrates, {N:.0f} references\n")
    print(f"{'k':>4}" + "".join(f"{a:>14}" for a in ARMS))
    for k in KS:
        r = out["recall_micro"][str(k)]
        print(f"{k:>4}" + "".join(f"{r[a]:>14.4f}" for a in ARMS))
    for name, c in (("primary  gate - cap", primary),
                    ("ceiling  oracle - cap", ceiling),
                    ("gate - three_way", vs12)):
        print(f"  {name:<24}{c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"{'*' if c['excludes_zero'] else ' '}")
    if out["ceiling_is_negative"]:
        print(f"  threshold +{THRESHOLD}   the gate's own oracle is BELOW the cap it must beat, "
              f"so no share is reported")
    else:
        print(f"  threshold +{THRESHOLD}   share of this gate's ceiling: "
              f"{out['share_of_ceiling_taken']}")
    print(f"\n  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

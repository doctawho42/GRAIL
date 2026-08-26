"""The H8 check: the group scorer against rank fusion, on the 291.

H8 predicts the scorer beats rank fusion by at least +0.05 of micro recall@15 with a paired
bootstrap excluding zero, and registers a corollary that the gap to MetaTox turns positive at
the same budget. The model is loaded, not fitted here: its architecture and every hyperparameter
were selected on validation, and this reads the 291 once.

Four arms are reported because two of them are needed to read the other two.

  fusion            rank fusion as H7 registers it, groups interleaved -- what H8 must beat
  fusion, blocked   the same order with every member of a formula group kept adjacent
  scorer            the model's group order, members inside a group in fusion order
  oracle            groups holding a reference first, members in fusion order

H8's design fixes that the scorer sets the order of groups and leaves the order inside one
alone, which forces the blocked form. Blocking is not free, and the artifact measures its price
rather than folding it into the model's result: the scorer has to pay that back before any of
its margin counts against the registered threshold.

Every substrate carrying a reference is scored, including those whose pool holds none of them
and those with a single group. Dropping either would change the population the prediction was
made about.
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
from bank_without_selection import _dedup, _load  # noqa: E402
from grail_metabolism.config import GeneratorConfig  # noqa: E402
from grail_metabolism.workflows.factory import build_generator  # noqa: E402
from group_scorer import Featuriser, GroupScorer, Standardiser  # noqa: E402

THRESHOLD, N_BOOT, SEED = 0.05, 10000, 0
KS = (1, 5, 10, 15, 20, 30, 50)
METATOX = ROOT / "results/metatox_smirks_preds.json"
ARMS = ("scorer", "fusion", "fusion_blocked", "oracle", "metatox")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="results/widepools_implicit/w*.json")
    ap.add_argument("--model", default=str(ROOT / "artifacts/group_scorer_wide.pt"))
    ap.add_argument("--gen-ckpt",
                    default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/h8_verdict.json"))
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--big-threshold", type=int, default=40,
                    help="heavy atoms above which no training substrate existed")
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    pools, refs = {}, {}
    for f in sorted(glob.glob(args.pools)) or [args.pools]:
        d = json.loads(Path(f).read_text())
        pools.update(d["pools"]); refs.update(d["references"])
    subs = sorted(s for s in pools if refs.get(s))

    generator = _load(Path(args.gen_ckpt), lambda a, r: build_generator(GeneratorConfig(**a), r))
    generator.eval()
    feat = Featuriser(generator)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = GroupScorer(ck["in_dim"], ck["config"]["hidden"], ck["config"]["dropout"])
    model.load_state_dict(ck["state_dict"]); model.eval()
    scaler = Standardiser.load(ck["standardiser"])

    mtx = json.loads(METATOX.read_text())["predictions"]
    hits = {a: {k: [] for k in KS} for a in ARMS}
    n_ref, heavy = [], []

    with torch.no_grad():
        for n, s in enumerate(subs, 1):
            if n % 50 == 0:
                print(f"  {n}/{len(subs)}", file=sys.stderr, flush=True)
            real = set(refs[s])
            n_ref.append(len(real))
            m = Chem.MolFromSmiles(s)
            heavy.append(m.GetNumHeavyAtoms() if m else 0)

            fused = rrf_order(pools[s])
            plain = [c["key"] for c in fused]
            by_g = defaultdict(list)
            for c in fused:
                by_g[feat.formula(c["smiles"])].append(c)
            names = list(by_g)                    # first appearance = fusion order of best member
            blocked = [c["key"] for g in names for c in by_g[g]]
            hit_first = sorted(names, key=lambda g: not any(c["key"] in real for c in by_g[g]))
            oracle = [c["key"] for g in hit_first for c in by_g[g]]

            if len(names) >= 2:
                _, _, X = feat.features(s, pools[s])
                X = ((X - scaler.mean) / scaler.std).astype(np.float32)
                order = np.argsort(-model(torch.from_numpy(X)).numpy())
                scored = [c["key"] for i in order for c in by_g[names[i]]]
            else:
                scored = blocked                  # one group: nothing to reorder

            seq = {"scorer": scored, "fusion": plain, "fusion_blocked": blocked,
                   "oracle": oracle, "metatox": _dedup(mtx.get(s, []), max(KS))}
            for a in ARMS:
                for k in KS:
                    hits[a][k].append(len(set(seq[a][:k]) & real))

    U = np.array(n_ref, dtype=float)
    N = float(U.sum())
    H = {a: {k: np.array(v[k], dtype=float) for k in KS} for a, v in hits.items()}
    big = np.array([h > args.big_threshold for h in heavy])

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def contrast(a, b):
        d = a - b
        bt = d[idx].sum(axis=1) / denom
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"gap": round(float(d.sum() / N), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    kk = args.k
    primary = contrast(H["scorer"][kk], H["fusion"][kk])
    corollary = contrast(H["scorer"][kk], H["metatox"][kk])
    design_cost = contrast(H["fusion_blocked"][kk], H["fusion"][kk])
    model_over_blocked = contrast(H["scorer"][kk], H["fusion_blocked"][kk])

    def sub_micro(mask, a, k):
        u = U[mask].sum()
        return round(float(H[a][k][mask].sum() / u), 4) if u else None

    verdict = "supported" if (primary["gap"] >= THRESHOLD and primary["excludes_zero"]) \
        else "failed"
    out = {"provenance": stamp(__file__), "hypothesis": "H8",
           "registered_threshold": THRESHOLD, "k": kk,
           "population": {"n": len(subs), "n_references": N,
                          "source": "the 291 of results/four_method_291.json",
                          "note": "every substrate carrying a reference, including those whose "
                                  "pool holds none and those with one group"},
           "aggregation": "micro, ratio of sums",
           "model": {"selected_on": "validation", **ck["config"],
                     "val_recall@15": ck.get("val_recall@15"),
                     "val_fusion_baseline": ck.get("fusion_baseline")},
           "recall_micro": {str(k): {a: round(float(H[a][k].sum() / N), 4) for a in ARMS}
                            for k in KS},
           "primary_scorer_minus_fusion": primary,
           "corollary_scorer_minus_metatox": corollary,
           "what_the_design_costs_before_the_model_acts": design_cost,
           "what_the_model_adds_given_the_design": model_over_blocked,
           "distribution_shift": {
               "threshold_heavy_atoms": args.big_threshold,
               "substrates_above": int(big.sum()),
               "references_above": float(U[big].sum()),
               "scorer_above": sub_micro(big, "scorer", kk),
               "fusion_above": sub_micro(big, "fusion", kk),
               "scorer_at_or_below": sub_micro(~big, "scorer", kk),
               "fusion_at_or_below": sub_micro(~big, "fusion", kk)},
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"\nH8 on {len(subs)} substrates, {N:.0f} references")
    print(f"\n{'k':>4}" + "".join(f"{a:>17}" for a in ARMS))
    for k in KS:
        r = out["recall_micro"][str(k)]
        print(f"{k:>4}" + "".join(f"{r[a]:>17.4f}" for a in ARMS))
    for name, c in (("primary   scorer - fusion", primary),
                    ("corollary scorer - metatox", corollary),
                    ("the design: blocked - fusion", design_cost),
                    ("the model: scorer - blocked", model_over_blocked)):
        print(f"  {name:<30}{c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"{'*' if c['excludes_zero'] else ' '}")
    d = out["distribution_shift"]
    print(f"\n  above {d['threshold_heavy_atoms']} heavy atoms ({d['substrates_above']} "
          f"substrates, {d['references_above']:.0f} references): "
          f"scorer {d['scorer_above']} fusion {d['fusion_above']}")
    print(f"  at or below: scorer {d['scorer_at_or_below']} fusion {d['fusion_at_or_below']}")
    print(f"\n  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

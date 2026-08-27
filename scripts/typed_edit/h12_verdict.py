"""The H12 check: the group score as a third ranking, on the 291.

H12 predicts the three-way fusion beats the two-way by at least +0.02 of micro recall@15 with a
paired bootstrap excluding zero. The model is the one H8 registered and is loaded, not fitted;
its hyperparameters were re-selected on validation under this composition because the old
selection was for a property that no longer decides anything.

Four arms, and the ceiling is the one that belongs to this composition rather than H8's. Entering
a perfect group ranking as the third component is what H12 could reach at best; H8's blocked
oracle of 0.6752 is not that number and is not comparable to it, because the blocked form spends
0.0647 that this form does not.

  two_way        the H7 rule, which H12 must beat
  three_way      the same with the scorer's group ranking added
  oracle_third   the same with a perfect group ranking added -- this composition's ceiling
  metatox        read at the same budgets
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
from group_scorer import (Featuriser, GroupScorer, Standardiser, build_examples,  # noqa: E402
                          three_way_order, two_way_order)

THRESHOLD, N_BOOT, SEED = 0.02, 10000, 0
KS = (1, 5, 10, 15, 20, 30, 50)
METATOX = ROOT / "results/metatox_smirks_preds.json"
ARMS = ("three_way", "two_way", "oracle_third", "metatox")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="results/widepools_implicit/w*.json")
    ap.add_argument("--model", default=str(ROOT / "artifacts/group_scorer_h12.pt"))
    ap.add_argument("--gen-ckpt",
                    default=str(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/h12_verdict.json"))
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
    model.load_state_dict(ck["state_dict"]); model.eval()
    scaler = Standardiser.load(ck["standardiser"])

    # every substrate carrying a reference, including those the pool cannot reach and those with
    # one group; dropping either would change the population the prediction was made about
    subs = sorted(s for s in pools if refs.get(s))
    mtx = json.loads(METATOX.read_text())["predictions"]
    hits = {a: {k: [] for k in KS} for a in ARMS}
    n_ref = []

    with torch.no_grad():
        for n, s in enumerate(subs, 1):
            if n % 50 == 0:
                print(f"  {n}/{len(subs)}", file=sys.stderr, flush=True)
            real = set(refs[s])
            n_ref.append(len(real))
            fused = rrf_order(pools[s])
            by_g = defaultdict(list)
            for c in fused:
                by_g[feat.formula(c["smiles"])].append(c)
            names = list(by_g)
            e = {"names": names, "by_g": by_g, "real": real}

            two = two_way_order(e)
            if len(names) >= 2:
                _, _, X = feat.features(s, pools[s])
                X = ((X - scaler.mean) / scaler.std).astype(np.float32)
                sc = model(torch.from_numpy(X)).numpy()
                three = three_way_order(sc, e)
                truth = np.array([1.0 if any(c["key"] in real for c in by_g[g]) else 0.0
                                  for g in names], dtype=np.float32)
                orc = three_way_order(truth, e)
            else:
                three = orc = two

            seq = {"three_way": three, "two_way": two, "oracle_third": orc,
                   "metatox": _dedup(mtx.get(s, []), max(KS))}
            for a in ARMS:
                for k in KS:
                    hits[a][k].append(len(set(seq[a][:k]) & real))

    U = np.array(n_ref, dtype=float)
    N = float(U.sum())
    H = {a: {k: np.array(v[k], dtype=float) for k in KS} for a, v in hits.items()}
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
    primary = contrast(H["three_way"][kk], H["two_way"][kk])
    ceiling = contrast(H["oracle_third"][kk], H["two_way"][kk])

    # A comparator with no predictions on this population is absent, not beaten. MetaTox was run
    # on the 291 and nowhere else, so on any other split every one of its lists is empty and a
    # contrast against it returns this arm's own recall wearing a gap's clothes. Refuse it.
    covered = sum(1 for s in subs if mtx.get(s))
    vs_mtx = (contrast(H["three_way"][kk], H["metatox"][kk]) if covered
              else {"unavailable": "MetaTox has no predictions for this population; it was run "
                                   "on the 291 and a contrast here would be this arm's recall "
                                   "against an empty list"})

    verdict = "supported" if (primary["gap"] >= THRESHOLD and primary["excludes_zero"]) \
        else "failed"
    out = {"provenance": stamp(__file__), "hypothesis": "H12",
           "registered_threshold": THRESHOLD, "k": kk,
           "population": {"n": len(subs), "n_references": N,
                          "source": "the 291 of results/four_method_291.json"},
           "aggregation": "micro, ratio of sums",
           "model": {"selected_on": "validation under this composition", **ck["config"],
                     "val_recall@15": ck.get("val_recall@15"),
                     "val_two_way_baseline": ck.get("fusion_baseline")},
           "recall_micro": {str(k): {a: round(float(H[a][k].sum() / N), 4) for a in ARMS}
                            for k in KS},
           "primary_three_way_minus_two_way": primary,
           "three_way_minus_metatox": vs_mtx,
           "metatox_substrates_covered": covered,
           "ceiling_of_this_composition": ceiling,
           "share_of_ceiling_taken": round(primary["gap"] / ceiling["gap"], 4)
           if ceiling["gap"] else None,
           "ceiling_note": "oracle_third ranks groups binarily, so it orders nothing among the "
                           "groups holding no reference; a graded ranking can and does exceed "
                           "it, and a share above one means that rather than the model beating "
                           "an upper bound",
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"\nH12 on {len(subs)} substrates, {N:.0f} references")
    print(f"\n{'k':>4}" + "".join(f"{a:>15}" for a in ARMS))
    for k in KS:
        r = out["recall_micro"][str(k)]
        print(f"{k:>4}" + "".join(f"{r[a]:>15.4f}" for a in ARMS))
    for name, c in (("primary  three - two", primary),
                    ("three_way - metatox", vs_mtx),
                    ("ceiling  oracle - two", ceiling)):
        if "unavailable" in c:
            print(f"  {name:<24}not computed: {c['unavailable']}")
            continue
        print(f"  {name:<24}{c['gap']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"{'*' if c['excludes_zero'] else ' '}")
    print(f"  threshold +{THRESHOLD}   share of this composition's ceiling taken: "
          f"{out['share_of_ceiling_taken']}")
    print(f"\n  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Queue point D: does curbing the generator's multiplicity bonus lift recall?

The generator adds match_scale * log_counts to every rule logit, where log_counts grows with the
number of sites a rule matches on the substrate. It is a systematic bonus for promiscuous, many-site
rules (aromatic C-H hydroxylation fires at many positions) over few-site ones (a desaturation or
dehydrogenation fires at few), so it can over-rank the products of promiscuous rules and push a
few-site true metabolite past the budget. match_scale is a learned parameter fixed at 0.25 with no
inference knob.

This is the one survivor that touches the GENERATOR's scores -- the frontier the other three levers
only slide along (results/rulegate_survivors_summary.json). It needs no retraining: match_scale is
overridden at inference on the DEPLOYED generator, the candidate set is unchanged (the whole bank is
applied, so which products exist does not depend on the score), only the generator score and thus the
ranking change. The filter score is a function of the pair, not of match_scale, so it is joined from
the frozen pools rather than recomputed.

Selection is on validation, and the deployed value 0.25 must reproduce the registered validation arm
(0.4748 at k=15), which certifies the regeneration is on the deployed model.

    python scripts/typed_edit/match_scale_sweep.py --sample 8 --values 0.25   # speed test
    python scripts/typed_edit/match_scale_sweep.py --values 0 0.1 0.25 0.5 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402
from _rrf import rrf_order  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
N_BOOT, SEED, CAP = 10000, 0, 100
DEPLOYED_MATCH_SCALE = 0.25
# val_pools.json keys its pools by substrate SMILES (needed to run the generator); val_pools_keyed
# hides the SMILES behind the substrate's own InChIKey. This is the population the registered
# validation arm (0.4748) is computed on, per aggregation_ablation.POPULATIONS["validation"].
VALPOOLS = ROOT / "results" / "val_pools.json"
GEN_CKPT = ROOT / "artifacts" / "full5000_implicit" / "checkpoints" / "generator.pt"
# registered validation arm the deployed value must reproduce
REGISTERED = ("results/aggregation_ablation_validation.json",
              lambda d: d["by_rule"]["noisy_or"]["recall"])


def deployed_order(cands, parent_key):
    cands = sorted(cands, key=lambda c: -(c["filter"] * c["generator"]))
    seen, dedup = set(), []
    for c in cands:
        if not c["key"] or c["key"] in seen:
            continue
        seen.add(c["key"])
        dedup.append(c)
    keep = sorted(dedup, key=lambda c: -c["generator"])[:CAP]
    return [c["key"] for c in rrf_order(keep) if c["key"] != parent_key]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--sample", type=int, default=0, help="0 = all val substrates")
    ap.add_argument("--top-k", type=int, default=7581, help="rules applied; deployed used the whole bank")
    ap.add_argument("--out", default=str(ROOT / "results" / "match_scale_sweep.json"))
    args = ap.parse_args()

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from bank_without_selection import _load, _key as tautkey
    from grail_metabolism.config import GeneratorConfig
    from grail_metabolism.workflows.factory import build_generator

    blob = json.loads(VALPOOLS.read_text())
    pools, refs = blob["pools"], blob["references"]
    subs = sorted(s for s in pools if refs.get(s))
    if args.sample:
        subs = subs[: args.sample]
    # The regenerated candidate SMILES do not string-match the frozen pool's canonicalisation, but
    # the candidate SET is the same structures, so join by tautomer key: frozen filter score keyed by
    # the same tautomer InChIKey the pool already carries (a duplicate key keeps the last, which
    # deployed_order would dedup to one anyway).
    fkey = {s: {c["key"]: float(c["filter"]) for c in pools[s] if c["key"]} for s in subs}
    real = {s: set(refs[s]) for s in subs}
    parent = {s: tautkey(s) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)

    gen = _load(GEN_CKPT, lambda a, r: build_generator(GeneratorConfig(**a), r))
    gen.eval()

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    by_value, hitvecs = {}, {}
    for v in args.values:
        gen.match_scale.data.fill_(float(v))
        order = {}
        t0 = time.time()
        for i, s in enumerate(subs, 1):
            cands = []
            for smiles, gscore, _rid, _sites in gen.generate_scored_with_details(
                    s, top_k=args.top_k, threshold=None, compute_sites=False):
                k = tautkey(smiles)
                if k not in fkey[s]:  # a candidate the frozen pool did not carry has no filter score
                    continue
                cands.append({"smiles": smiles, "generator": float(gscore),
                              "filter": fkey[s][k], "key": k})
            order[s] = deployed_order(cands, parent[s])
            if i % 50 == 0:
                print(f"  v={v}: {i}/{len(subs)} ({time.time()-t0:.0f}s)", flush=True)

        def hitvec(k):
            return np.array([len(set(order[s][:k]) & real[s]) for s in subs], dtype=float)

        hitvecs[f"{v}"] = {k: hitvec(k) for k in KS}
        rec = {str(k): round(float(hitvecs[f'{v}'][k].sum() / U.sum()), 4) for k in KS}
        by_value[f"{v}"] = {"recall": rec,
                            "mean_ranked": round(float(np.mean([len(order[s]) for s in subs])), 1)}
        print(f"  match_scale {v}: r@15 {rec['15']}  r@5 {rec['5']}  r@1 {rec['1']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # delta vs deployed with bootstrap CI, per swept value and budget
    dk = f"{DEPLOYED_MATCH_SCALE}"
    if dk in hitvecs:
        for v in by_value:
            if v == dk:
                continue
            dd = {}
            for k in KS:
                d = hitvecs[v][k] - hitvecs[dk][k]
                bt = d[idx].sum(axis=1) / denom
                lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
                dd[str(k)] = {"delta": round(float(d.sum() / U.sum()), 4),
                              "ci95": [round(lo, 4), round(hi, 4)], "excludes_zero": bool(lo > 0 or hi < 0)}
            by_value[v]["vs_deployed"] = dd

    # deltas vs deployed, with bootstrap CI on the difference at each k
    dep_key = f"{DEPLOYED_MATCH_SCALE}"
    report = {"provenance": stamp(__file__), "inputs": record_inputs([str(VALPOOLS)]),
              "generator": "artifacts/full5000_implicit (deployed), match_scale overridden at inference",
              "population": {"split": "validation", "n_substrates": len(subs), "n_references": int(U.sum())},
              "bootstrap": {"n": N_BOOT, "seed": SEED}, "cap": CAP, "top_k": args.top_k,
              "by_match_scale": by_value}

    if dep_key in by_value:
        # recompute per-substrate hit arrays for CI (only if deployed value was swept)
        report["reproduces_registered_arm"] = None
        try:
            reg = REGISTERED[1](json.loads((ROOT / REGISTERED[0]).read_text()))
            checked = {k: reg[k] for k in by_value[dep_key]["recall"] if k in reg}
            mism = {k: (by_value[dep_key]["recall"][k], reg[k]) for k in checked
                    if abs(by_value[dep_key]["recall"][k] - float(reg[k])) > 1e-4}
            report["reproduces_registered_arm"] = {"reference": REGISTERED[0], "matches": not mism,
                                                   "mismatches": mism}
        except Exception as e:
            report["reproduces_registered_arm"] = {"error": str(e)}

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

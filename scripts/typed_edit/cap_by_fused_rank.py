"""Does capping the pool by the fused rank instead of the generator score recover the refs the cap drops?

The deployed ranking caps the deduplicated pool at 100 by GENERATOR score and only then fuses the
two component scores by reciprocal rank. The cap is not free: it deletes candidates the filter
strongly favours whenever the generator ranks them past 100, and pool_cap_cost.json measures 39
in-pool references (5.9%) lost exactly there. This does NOT propose removing the cap -- the cap
sweep already shows the uncapped pool is worse at k=15 (0.5038 vs 0.5353 on the comparison set),
because reciprocal-rank fusion over thousands of noisy candidates dilutes. It proposes keeping the
SAME budget of 100 but selecting those 100 by the fused rank, so a filter-favoured reference the
generator ranks 101+ is kept, at the cost of dropping some generator-favoured candidate. Whether
that trade is net positive is unknown a priori, which is why it is measured rather than asserted.

Everything except the cap-selection criterion is the deployed pipeline, taken from
aggregation_ablation.merge: dedup by match key in descending product order, then the criterion,
then reciprocal rank fusion, parent dropped. The candidate "generator" field in the keyed pools is
already the deployed noisy-or aggregation, so the arm is read off frozen pools with no model.

Selection is on VALIDATION; the comparison set is confirmatory and is not where the choice is made.
Both arms carry a reproduction gate: the deployed cap must reproduce the registered recall of that
population (validation 0.4748 at k=15 from aggregation_ablation_validation.json; comparison 0.5353
from pool_cap_cost.json cap 100), which is what certifies the re-run is on the deployed model.

    python scripts/typed_edit/cap_by_fused_rank.py            # both populations
    python scripts/typed_edit/cap_by_fused_rank.py --only validation
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
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
N_BOOT, SEED = 10000, 0
CAP = 100

# population -> (glob for pools+refs, reproduction reference)
POPULATIONS = {
    "validation": {
        "pools": "results/val_pools_keyed.json",
        "reproduce": ("results/aggregation_ablation_validation.json",
                      lambda d: d["by_rule"]["noisy_or"]["recall"]),
    },
    "comparison": {
        "pools": "results/widepools_implicit/w*.json",
        # deployment_table.json is the authoritative deployed arm aggregation_ablation checks
        # against; pool_cap_cost.json is a rougher cap-sweep artifact that differs by ~0.002 off k=15.
        "reproduce": ("results/deployment_table.json",
                      lambda d: {k: v["whole bank"] for k, v in d["recall_micro"].items()}),
    },
}


def load_pop(name):
    pools, refs, files = {}, {}, []
    for f in sorted(glob.glob(str(ROOT / POPULATIONS[name]["pools"]))) or [str(ROOT / POPULATIONS[name]["pools"])]:
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"])
        refs.update(blob["references"])
        files.append(f)
    return pools, refs, files


def _dedup_in_product_order(pool):
    """The deployed dedup: keep the first candidate per match key in descending product order."""
    cands = sorted(pool, key=lambda c: -(c["filter"] * c["generator"]))
    seen, out = set(), []
    for c in cands:
        if not c["key"] or c["key"] in seen:
            continue
        seen.add(c["key"])
        out.append(c)
    return out


def order_generator_cap(pool, parent_key):
    """Deployed: dedup, cap top-100 by generator score, fuse, drop parent."""
    dedup = _dedup_in_product_order(pool)
    keep = sorted(dedup, key=lambda c: -c["generator"])[:CAP]
    return [c["key"] for c in rrf_order(keep) if c["key"] != parent_key]


def order_fused_cap(pool, parent_key):
    """Variant: dedup, cap top-100 by the FUSED rank, fuse, drop parent."""
    dedup = _dedup_in_product_order(pool)
    keep = rrf_order(dedup)[:CAP]
    return [c["key"] for c in rrf_order(keep) if c["key"] != parent_key]


def measure(name):
    from bank_without_selection import _key as tautkey

    pools, refs, files = load_pop(name)
    subs = sorted(s for s in pools if refs.get(s))
    real = {s: set(refs[s]) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)
    parent = {s: tautkey(s) for s in subs}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    dep = {s: order_generator_cap(pools[s], parent[s]) for s in subs}
    var = {s: order_fused_cap(pools[s], parent[s]) for s in subs}

    def hits(order, k):
        return np.array([len(set(order[s][:k]) & real[s]) for s in subs], dtype=float)

    out = {"n_substrates": len(subs), "n_references": int(U.sum()),
           "mean_ranked": {"generator_cap": round(float(np.mean([len(dep[s]) for s in subs])), 1),
                           "fused_cap": round(float(np.mean([len(var[s]) for s in subs])), 1)},
           "deployed_generator_cap": {}, "variant_fused_cap": {}, "delta_variant_minus_deployed": {}}
    for k in KS:
        hd, hv = hits(dep, k), hits(var, k)
        out["deployed_generator_cap"][str(k)] = round(float(hd.sum() / U.sum()), 4)
        out["variant_fused_cap"][str(k)] = round(float(hv.sum() / U.sum()), 4)
        d = hv - hd
        bt = d[idx].sum(axis=1) / denom
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        out["delta_variant_minus_deployed"][str(k)] = {
            "difference": round(float(d.sum() / U.sum()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}

    # Reproduction gate: the deployed cap must reproduce the registered recall of this population.
    ref_path, extract = POPULATIONS[name]["reproduce"]
    published = extract(json.loads((ROOT / ref_path).read_text()))
    checked = {k: v for k, v in published.items() if k in out["deployed_generator_cap"]}
    mism = {k: (out["deployed_generator_cap"][k], v) for k, v in checked.items()
            if abs(out["deployed_generator_cap"][k] - float(v)) > 1e-4}
    out["reproduces_registered_arm"] = {"reference": ref_path, "checked_budgets": sorted(checked, key=int),
                                        "matches": not mism, "mismatches": mism}
    return out, files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["validation", "comparison"], default="")
    ap.add_argument("--out", default=str(ROOT / "results" / "cap_by_fused_rank.json"))
    args = ap.parse_args()

    names = [args.only] if args.only else ["validation", "comparison"]
    report = {"provenance": stamp(__file__),
              "question": ("whether capping the deduplicated pool at 100 by the fused rank, rather "
                           "than by the generator score alone, recovers the filter-favoured "
                           "references the generator cap drops (pool_cap_cost.json: 39, 5.9%)"),
              "cap": CAP, "bootstrap": {"n": N_BOOT, "seed": SEED},
              "selection": "validation is decisive; comparison is confirmatory",
              "by_population": {}}
    all_inputs = []
    gate_ok = True
    for name in names:
        res, files = measure(name)
        report["by_population"][name] = res
        all_inputs += files
        rep = res["reproduces_registered_arm"]
        tag = "OK" if rep["matches"] else "FAIL"
        gate_ok = gate_ok and rep["matches"]
        d15 = res["delta_variant_minus_deployed"]["15"]
        print(f"[{name}] reproduces deployed arm: {tag}", flush=True)
        print(f"  deployed  r@15 = {res['deployed_generator_cap']['15']}", flush=True)
        print(f"  fused-cap r@15 = {res['variant_fused_cap']['15']}  "
              f"(delta {d15['difference']:+}, ci {d15['ci95']}, "
              f"{'SIGNIFICANT' if d15['excludes_zero'] else 'n.s.'})", flush=True)
        if not rep["matches"]:
            print(f"  MISMATCH vs {rep['reference']}: {rep['mismatches']}", flush=True)

    report["inputs"] = record_inputs(all_inputs)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}", flush=True)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

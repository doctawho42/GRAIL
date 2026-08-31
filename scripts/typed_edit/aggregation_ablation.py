#!/usr/bin/env python3
"""The scoring assumption the deployed system makes about its own rules, measured.

The generator hands one score to a candidate that several templates reach, and the rule it uses
treats those templates as independent witnesses: one minus the product of one minus each score.
The bank does not satisfy that. It holds published rule sets verbatim and its mined half is full
of near-duplicates, so a candidate reached by twenty templates saying the same thing is scored as
though twenty witnesses agreed. Every other undeclared choice in this work is swept; this one was
named in the Supporting Information and left, because the released pools carry the aggregate and
not the per-template scores a different rule would need.

This runs the generator side again and keeps the scores rather than their aggregate, so any
aggregation rule can be evaluated afterwards without another pass. The filter is not re-run: a
filter score is a function of the substrate and the product and does not depend on how the
generator combined its templates, so the scores already frozen in the wide pools are joined by
candidate structure. What changes between arms is one number per candidate, and the ranking that
number feeds.

    python scripts/typed_edit/aggregation_ablation.py --shard 0 --shards 8   # one worker
    python scripts/typed_edit/aggregation_ablation.py --merge                # then this
"""
from __future__ import annotations

import argparse
import glob
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

from _provenance import stamp  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
N_BOOT, SEED = 10000, 0
CAP = 100
SHARD_DIR = ROOT / "results" / "aggregation_shards"

# The rules compared. "noisy_or" is what the released checkpoint runs and what Equation 1 states;
# "max" is the one a bank of near-duplicates argues for, since it counts a candidate once however
# many templates restate the same transformation. "mean" and "hybrid" are the other two the
# implementation already offers and are swept because a knob with four settings should be reported
# at all four rather than at the two that make a point.
def aggregate(rule: str, scores: list[float]) -> float:
    if not scores:
        return 0.0
    arr = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if rule == "max":
        return float(arr.max())
    if rule == "mean":
        return float(arr.mean())
    noisy_or = float(1.0 - np.prod(1.0 - arr))
    if rule == "noisy_or":
        return noisy_or
    if rule == "hybrid":
        return float(0.65 * float(arr.max()) + 0.35 * noisy_or)
    raise KeyError(rule)


RULES = ("noisy_or", "max", "mean", "hybrid")


def wide_pools():
    pools, refs = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"]); refs.update(blob["references"])
    return pools, refs


def collect(shard: int, shards: int, out: Path) -> int:
    """Per-template scores for every candidate of one shard of the comparison set."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from bank_without_selection import _load
    from grail_metabolism.config import GeneratorConfig
    from grail_metabolism.utils.preparation import safe_run_reactants
    from grail_metabolism.model.generator import _normalize_smiles_cached
    from grail_metabolism.workflows.factory import build_generator

    pools, refs = wide_pools()
    subs = sorted(s for s in pools if refs.get(s))
    mine = subs[shard::shards]
    generator = _load(ROOT / "artifacts/full5000_implicit/checkpoints/generator.pt",
                      lambda a, r: build_generator(GeneratorConfig(**a), r))

    rows, t0 = {}, time.perf_counter()
    for n, s in enumerate(mine, 1):
        mol, scores, ranked = generator._prepare_generation(s, 7581, None)
        per_candidate: dict[str, list[float]] = {}
        if mol is not None:
            # The enumeration of generate_scored_with_details, with the score list kept instead
            # of collapsed. Same reaction order, same normalisation, same per-template dedup.
            for index in ranked:
                if index >= len(generator.rule_reactions):
                    continue
                reaction = generator.rule_reactions[index]
                if reaction is None:
                    continue
                rule_score = float(scores[index])
                seen: set = set()
                for product_tuple in safe_run_reactants(reaction, mol):
                    for product in product_tuple:
                        try:
                            smiles = Chem.MolToSmiles(product)
                        except Exception:
                            continue
                        for fragment in smiles.split("."):
                            fragment = fragment.strip()
                            if not fragment:
                                continue
                            try:
                                normalized = _normalize_smiles_cached(
                                    fragment, generator.gen_normalization)
                            except Exception:
                                continue
                            if normalized in seen:
                                continue
                            seen.add(normalized)
                            per_candidate.setdefault(normalized, []).append(rule_score)
        rows[s] = per_candidate
        print(f"  shard {shard}: {n}/{len(mine)} ({time.perf_counter() - t0:.0f}s) "
              f"{len(per_candidate)} candidates", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"shard": shard, "shards": shards, "rows": rows}))
    print(f"wrote {out}")
    return 0


def merge(out: str) -> int:
    from _rrf import rrf_order
    from bank_without_selection import _key as tautkey

    shard_files = sorted(glob.glob(str(SHARD_DIR / "s*.json")))
    if not shard_files:
        print("no shard written yet", file=sys.stderr)
        return 1
    rows, declared = {}, None
    for f in shard_files:
        blob = json.loads(Path(f).read_text())
        declared = declared or blob["shards"]
        rows.update(blob["rows"])
    if len(shard_files) != declared:
        print(f"FAIL: {len(shard_files)} shards on disk, {declared} were declared; a partial "
              f"collection would silently narrow the population", file=sys.stderr)
        return 1

    pools, refs = wide_pools()
    subs = sorted(s for s in pools if refs.get(s))
    missing = [s for s in subs if s not in rows]
    if missing:
        print(f"FAIL: {len(missing)} of {len(subs)} substrates were not collected",
              file=sys.stderr)
        return 1

    real = {s: set(refs[s]) for s in subs}
    U = np.array([len(real[s]) for s in subs], dtype=float)
    parent = {s: tautkey(s) for s in subs}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    # The filter score, joined by candidate structure from the frozen pools. A candidate the
    # re-run reaches and the pool does not cannot be scored without running the filter, so it is
    # counted and dropped rather than given a default that would flatter or punish an arm.
    filt = {s: {c["smiles"]: float(c["filter"]) for c in pools[s]} for s in subs}
    unjoined = sum(1 for s in subs for c in rows[s] if c not in filt[s])
    joined = sum(1 for s in subs for c in rows[s] if c in filt[s])

    # The match key of a candidate does not depend on the aggregation rule, so it is computed
    # once per candidate and not once per candidate per rule. The pools already carry it for
    # every candidate they hold, which is every candidate this join keeps.
    key_of = {s: {c["smiles"]: c["key"] for c in pools[s]} for s in subs}

    orders, sizes = {}, {}
    for rule in RULES:
        order = {}
        for s in subs:
            cands = [{"smiles": c, "generator": aggregate(rule, v), "filter": filt[s][c],
                      "key": key_of[s][c]}
                     for c, v in rows[s].items() if c in filt[s]]
            # The deployed pipeline's order of operations, which is not interchangeable with any
            # other: the pool is deduplicated by match key in descending order of the product of
            # the two component scores, and only then capped by generator score and fused.
            # Capping first lets duplicate keys consume slots, which costs recall and would have
            # been charged to the aggregation rule rather than to the order of two steps.
            cands.sort(key=lambda c: -(c["filter"] * c["generator"]))
            seen_key, pool = set(), []
            for c in cands:
                if not c["key"] or c["key"] in seen_key:
                    continue
                seen_key.add(c["key"])
                pool.append(c)
            keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
            order[s] = [c["key"] for c in rrf_order(keep) if c["key"] != parent[s]]
        orders[rule] = order
        sizes[rule] = round(float(np.mean([len(order[s]) for s in subs])), 1)

    def hits(order, k):
        return np.array([len(set(order[s][:k]) & real[s]) for s in subs], dtype=float)

    by_rule = {}
    for rule in RULES:
        row = {"mean_ranked": sizes[rule],
               "recall": {str(k): round(float(hits(orders[rule], k).sum() / U.sum()), 4)
                          for k in KS}}
        if rule != "noisy_or":
            for k in KS:
                d = hits(orders[rule], k) - hits(orders["noisy_or"], k)
                bt = d[idx].sum(axis=1) / denom
                lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
                row.setdefault("minus_noisy_or", {})[str(k)] = {
                    "difference": round(float(d.sum() / U.sum()), 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "excludes_zero": bool(lo > 0 or hi < 0)}
        by_rule[rule] = row

    # The gate this ablation needs and did not have. The deployed rule, re-derived here from
    # per-template scores, has to reproduce the arm the comparison table reports. It did not the
    # first time this ran: the script took its generator from the checkpoint directory that holds
    # the FILTER's run, and the arm it produced trailed the paper's by nine points at a budget of
    # thirty. Nothing in the pools recorded which checkpoint wrote them, so the only way to find
    # that was to check a number against another number, which is what this now does on every run.
    reference = json.loads((ROOT / "results/deployment_table.json").read_text())
    published = {k: v["whole bank"] for k, v in reference["recall_micro"].items()}
    reproduces = {k: (abs(by_rule["noisy_or"]["recall"][k] - published[k]) <= 1e-4)
                  for k in by_rule["noisy_or"]["recall"] if k in published}
    if not all(reproduces.values()):
        bad = [k for k, ok in reproduces.items() if not ok]
        print("FAIL: the deployed aggregation does not reproduce the published arm at budgets "
              + ", ".join(bad) + "; this re-run is not on the deployed model", file=sys.stderr)
        for k in bad:
            print(f"  k={k}: here {by_rule['noisy_or']['recall'][k]}, published {published[k]}",
                  file=sys.stderr)
        return 1

    separating = sorted(
        rule for rule in RULES if rule != "noisy_or"
        and any(c["excludes_zero"] for c in by_rule[rule]["minus_noisy_or"].values()))

    report = {
        "provenance": stamp(__file__),
        "question": ("whether the deployed noisy-or aggregation, whose independence assumption "
                     "this bank violates by construction, changes the comparison against the "
                     "alternatives the implementation offers"),
        "population": {"n_substrates": len(subs), "n_references": int(U.sum())},
        "join": {"candidates_scored_by_both": joined,
                 "candidates_the_pool_does_not_carry": unjoined,
                 "note": ("the filter is a function of substrate and product and does not depend "
                          "on the aggregation, so its frozen scores are joined by structure "
                          "rather than recomputed; a candidate absent from the pool is dropped "
                          "from every arm alike")},
        "ranking": ("reciprocal rank fusion of the two component scores over the pool capped at "
                    f"{CAP} by generator score, parent dropped, as everywhere else"),
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "deployed": "noisy_or",
        "reproduces_the_published_arm": ("the deployed rule re-derived here matches the whole-bank "
                                         "column of results/deployment_table.json at every budget, "
                                         "which is what certifies the re-run is on the deployed "
                                         "model and not a neighbouring checkpoint"),
        "by_rule": by_rule,
        "rules_that_separate_from_the_deployed_one_at_any_budget": separating,
        "reading": (
            "The independence the deployed rule assumes is false in this bank, so the question "
            "is not whether the assumption holds but whether anything turns on it. A maximum "
            "counts a candidate once however many near-duplicate templates restate it, which is "
            "the correction the violation argues for."),
    }
    Path(out).write_text(json.dumps(report, indent=1))

    print(f"\n{'rule':10s} {'ranked':>7s} " + " ".join(f"r@{k:<4d}" for k in (5, 15, 30, 50))
          + "   minus the deployed rule at 30")
    for rule in RULES:
        row = by_rule[rule]
        cell = row.get("minus_noisy_or", {}).get("30")
        tail = ("  <- deployed" if rule == "noisy_or" else
                f"   {cell['difference']:+.4f} [{cell['ci95'][0]:+.4f}, {cell['ci95'][1]:+.4f}]"
                + ("  separates" if cell["excludes_zero"] else ""))
        print(f"{rule:10s} {row['mean_ranked']:7.1f} "
              + " ".join(f"{row['recall'][str(k)]:6.4f}" for k in (5, 15, 30, 50)) + tail)
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "aggregation_ablation.json"))
    args = ap.parse_args()
    if args.merge:
        return merge(args.out)
    if args.shard < 0:
        print("give --shard N (with --shards M), or --merge", file=sys.stderr)
        return 2
    return collect(args.shard, args.shards, SHARD_DIR / f"s{args.shard}.json")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Does averaging the two component scores across training seeds move the ranking?

Every calibration-family idea this queue considered died to one argument: the release ranking fuses
competition RANKS, so any monotone per-axis rescaling (temperature, Platt, isotonic, shrinkage)
cannot reorder anything. Averaging across independently trained seeds is not a monotone rescaling
of one axis -- it is a different function of several -- so it is the one statistical lever the rank
argument does not close, and it is the only one left that needs no new training.

The three seed pools at the deployed configuration make it free to ask. They carry, per substrate,
the whole-bank candidate set with both component scores, and their candidate sets are identical
(generation is deterministic given bank and substrate), so the join needs no missing-value policy:
the seeds disagree only about the scores.

WHAT THIS IS NOT. These pools are on the comparison population, which is the population the
manuscript reports. Selecting a configuration on it would be selecting on the reported set, which
this project does not do. So this is a descriptive probe and explicitly not a selection: if it is
flat, the lever is closed and no compute was spent; if it lifts, the honest next step is to
regenerate three seed pools on VALIDATION and decide there.

The reproduction gate below is the part that makes the comparison meaningful. The probe recomputes
each seed's recall with its own code and refuses to report an ensemble unless those match the
registered per-seed numbers in results/retraining_spread.json.

    python scripts/typed_edit/seed_ensemble_probe.py
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
CAP = 100          # H9, as in the release ranking
SEEDS = (0, 1, 2)
ARM = "exhaustive"  # the whole-bank arm, the authoritative one
BOOTSTRAP = 2000


def _order(pool, parent_key, rrf_order):
    """The producer's ordering, verbatim: cap by generator, fuse by reciprocal rank, drop parent.

    The stored pools are already deduplicated by match key, so the dedup step of the release
    ranking has been applied upstream and is not repeated here.
    """
    keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
    return [k for k in (c["key"] for c in rrf_order(keep)) if k and k != parent_key]


def _recall(order, real, universe):
    return {str(k): round(sum(len(set(order[s][:k]) & real[s]) for s in order) / max(universe, 1), 4)
            for k in KS}


def _hits_at(order, real, k):
    """Per-substrate hit counts at budget k, the unit a paired bootstrap resamples."""
    return {s: len(set(order[s][:k]) & real[s]) for s in order}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "seed_ensemble_probe.json"))
    ap.add_argument("--k", type=int, default=15, help="budget the paired bootstrap is run at")
    ap.add_argument("--pools", default=f"results/seedpools/{ARM}_seed{{seed}}.json",
                    help="pool file per seed, with {seed} standing for 0, 1, 2")
    ap.add_argument("--gate", choices=("registered", "structural"), default="registered",
                    help="registered: each seed's recall must match the value recorded in "
                         "retraining_spread.json, which only exists for the comparison pools. "
                         "structural: no registered value to match, so instead every pool must "
                         "be stamped with the checkpoint of the seed it is taken for and the "
                         "three must enumerate the same candidates.")
    args = ap.parse_args()

    from _rrf import rrf_order
    from bank_without_selection import _key as tautkey

    paths = [ROOT / args.pools.format(seed=s) for s in SEEDS]
    blobs = {}
    for s, p in zip(SEEDS, paths):
        if not p.exists():
            print(f"REFUSING: missing seed pool {p}", file=sys.stderr)
            return 1
        blobs[s] = json.loads(p.read_text())

    # The population: substrates with references, shared by all three seeds.
    subs = sorted(set.intersection(*[{x for x in blobs[s]["pools"] if blobs[s]["references"].get(x)}
                                     for s in SEEDS]))
    real = {s: set(blobs[SEEDS[0]]["references"][s]) for s in subs}
    parent = {s: tautkey(s) for s in subs}
    universe = float(sum(len(real[s]) for s in subs))

    # The join is only well posed if the seeds agree about WHICH candidates exist. Checked, not
    # assumed: a seed that enumerated a different set would make an average over seeds a different
    # quantity per candidate, and the probe would be comparing pools of different composition.
    sets = {s: {x: {c["key"] for c in blobs[s]["pools"][x]} for x in subs} for s in SEEDS}
    identical = all(sets[SEEDS[0]][x] == sets[s][x] for x in subs for s in SEEDS[1:])
    union_mean = st.mean(len(set.union(*[sets[s][x] for s in SEEDS])) for x in subs)
    shared_mean = st.mean(len(set.intersection(*[sets[s][x] for s in SEEDS])) for x in subs)

    # --- the per-seed arms, which are also the reproduction gate -------------------------------
    per_seed_order, per_seed = {}, {}
    for s in SEEDS:
        per_seed_order[s] = {x: _order(blobs[s]["pools"][x], parent[x], rrf_order) for x in subs}
        per_seed[f"seed{s}"] = _recall(per_seed_order[s], real, universe)

    if args.gate == "registered":
        registered = json.loads((ROOT / "results/retraining_spread.json").read_text())
        reg_seeds = registered["by_arm"][ARM]["seeds"]
        gate = {}
        for s in SEEDS:
            name = f"{ARM}_seed{s}"
            want = reg_seeds.get(name, {}).get("recall", {})
            got = per_seed[f"seed{s}"]
            gate[name] = {"recall15_here": got["15"], "recall15_registered": want.get("15"),
                          "max_abs_diff_over_budgets": round(max(
                              abs(got[str(k)] - want[str(k)]) for k in KS if str(k) in want), 4)
                          if want else None}
        # `or 1.0` here would be a gate that cannot pass: a perfect reproduction is 0.0, which is
        # falsy, so it would be replaced by the failure sentinel. Missing evidence (None) is the
        # only thing that may fail open.
        worst = max(1.0 if v["max_abs_diff_over_budgets"] is None
                    else v["max_abs_diff_over_budgets"] for v in gate.values())
        if worst > 0.0002:
            print("REFUSING: this probe does not reproduce the registered per-seed recalls "
                  f"(worst budget differs by {worst}); the ordering or the population is not the "
                  "producer's, so an ensemble measured here would not be comparable",
                  file=sys.stderr)
            print(json.dumps(gate, indent=1), file=sys.stderr)
            return 1
    else:
        # No registered per-seed value exists for these pools, so the gate cannot be a
        # reproduction. What it can still establish is that the three files are what they are
        # taken for: each stamped with its own seed's checkpoint, and all three enumerating the
        # same candidates. A pool scored by a checkpoint nobody thinks it was scored by is a
        # defect this repository has already had once.
        gate = {}
        for s, p in zip(SEEDS, paths):
            stamped = ((blobs[s].get("checkpoints") or {}).get("generator") or {}).get("path", "")
            gate[p.name] = {"generator_checkpoint": stamped,
                            "names_this_seed": f"seed{s}" in stamped,
                            "recall15_here": per_seed[f"seed{s}"]["15"]}
        wrong = [k for k, v in gate.items() if not v["names_this_seed"]]
        if wrong or not identical:
            print("REFUSING: the pools are not what they are taken for "
                  f"(checkpoint stamp mismatch: {wrong}; identical candidate sets: {identical})",
                  file=sys.stderr)
            print(json.dumps(gate, indent=1), file=sys.stderr)
            return 1
        worst = None

    # --- the ensembles -------------------------------------------------------------------------
    # Score-level: average (and median) each axis over the three seeds, then rank exactly as the
    # release does. Rank-level: fuse all six axes (three seeds x two components) by reciprocal
    # rank, which is what the pipeline already does for two axes.
    def pooled(x, reduce):
        by_key = {}
        for s in SEEDS:
            for c in blobs[s]["pools"][x]:
                by_key.setdefault(c["key"], {"g": [], "f": []})
                by_key[c["key"]]["g"].append(c["generator"])
                by_key[c["key"]]["f"].append(c["filter"])
        return [{"key": k, "generator": reduce(v["g"]), "filter": reduce(v["f"])}
                for k, v in by_key.items()]

    arms = {}
    arms["mean_scores"] = {x: _order(pooled(x, st.mean), parent[x], rrf_order) for x in subs}
    arms["median_scores"] = {x: _order(pooled(x, st.median), parent[x], rrf_order) for x in subs}

    def six_axis(x):
        """Cap by the mean generator (so the candidate pool matches the score-level arms), then
        fuse the six per-seed axes by reciprocal rank with the same K."""
        from grail_metabolism.model.ranking import RRF_K, competition_ranks
        keep = sorted(pooled(x, st.mean), key=lambda c: -c["generator"])[:CAP]
        keys = [c["key"] for c in keep]
        axes = []
        for s in SEEDS:
            score = {c["key"]: c for c in blobs[s]["pools"][x]}
            for field in ("generator", "filter"):
                axes.append(competition_ranks(keys, lambda k, sc=score, f=field: sc[k][f]))
        fused = sorted(range(len(keys)),
                       key=lambda i: -sum(1.0 / (RRF_K + axis[i]) for axis in axes))
        return [keys[i] for i in fused if keys[i] != parent[x]]

    arms["six_axis_rrf"] = {x: six_axis(x) for x in subs}

    results = {name: _recall(order, real, universe) for name, order in arms.items()}

    # --- against the bar --------------------------------------------------------------------
    seed_values = {str(k): [per_seed[f"seed{s}"][str(k)] for s in SEEDS] for k in KS}
    bar = {str(k): {"seed_mean": round(st.mean(seed_values[str(k)]), 4),
                    "seed_best": max(seed_values[str(k)]),
                    "seed_sd": round(st.stdev(seed_values[str(k)]), 4)} for k in KS}
    deltas = {name: {str(k): {"minus_seed_mean": round(rec[str(k)] - bar[str(k)]["seed_mean"], 4),
                              "minus_seed_best": round(rec[str(k)] - bar[str(k)]["seed_best"], 4)}
                     for k in KS} for name, rec in results.items()}

    # Paired bootstrap over substrates at one budget: the ensemble against each single seed, on
    # the same substrates, so the interval is about the ranking and not about the population.
    k = args.k
    rng = random.Random(0)
    boot = {}
    for name, order in arms.items():
        ens = _hits_at(order, real, k)
        for s in SEEDS:
            base = _hits_at(per_seed_order[s], real, k)
            diffs = []
            for _ in range(BOOTSTRAP):
                sample = [subs[rng.randrange(len(subs))] for _ in subs]
                u = float(sum(len(real[x]) for x in sample)) or 1.0
                diffs.append((sum(ens[x] for x in sample) - sum(base[x] for x in sample)) / u)
            diffs.sort()
            boot[f"{name}_vs_seed{s}"] = {
                "point": round((sum(ens[x] for x in subs) - sum(base[x] for x in subs))
                               / max(universe, 1), 4),
                "ci95": [round(diffs[int(0.025 * BOOTSTRAP)], 4),
                         round(diffs[int(0.975 * BOOTSTRAP)], 4)]}

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs(paths + [ROOT / "results/retraining_spread.json"]),
        "question": ("whether averaging the component scores over training seeds reorders the "
                     "release ranking enough to move recall, which is the one statistical lever "
                     "the rank-invariance argument does not close"),
        "why_the_rank_argument_does_not_close_it": (
            "the fusion consumes competition ranks, so a monotone rescaling of one axis cannot "
            "reorder; an average over seeds is not a rescaling of one axis and does reorder"),
        "population": "the comparison set, whole-bank arm, as in results/retraining_spread.json",
        "n_substrates": len(subs), "n_references": int(universe),
        "criterion": "tautomer-aware InChIKey",
        "ranking": "cap 100 by generator, reciprocal rank fusion, parent dropped",
        "candidate_sets_identical_across_seeds": bool(identical),
        "mean_pool_union": round(union_mean, 1), "mean_pool_shared_by_all_seeds": round(shared_mean, 1),
        "gate_kind": args.gate,
        **({"reproduces_the_registered_per_seed_recalls": gate} if args.gate == "registered"
           else {"the_pools_are_stamped_with_the_seed_they_are_taken_for": gate}),
        "pool_files": [str(p.relative_to(ROOT)) for p in paths],
        "per_seed": per_seed,
        "bar": bar,
        "by_arm": results,
        "deltas": deltas,
        f"paired_bootstrap_at_{k}": boot,
        "this_is_not_a_selection": (
            "the pools are on the reported comparison population, so nothing here may be adopted "
            "on this evidence; a lift would have to be confirmed on validation pools, which do "
            "not yet exist for the three seeds, before any release decision"),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"population: {len(subs)} substrates, {int(universe)} references | "
          f"candidate sets identical across seeds: {identical}")
    if args.gate == "registered":
        print(f"gate: worst per-seed budget differs from the registered value by {worst}")
    else:
        print("gate: every pool is stamped with its own seed's checkpoint and the three "
              "enumerate the same candidates (no registered value exists for these pools)")
    print(f"\n  {'arm':<16s}" + "".join(f"{f'r@{k}':>9s}" for k in (1, 5, 15, 30)))
    for name in ("seed0", "seed1", "seed2"):
        print(f"  {name:<16s}" + "".join(f"{per_seed[name][str(k)]:9.4f}" for k in (1, 5, 15, 30)))
    print(f"  {'seed mean':<16s}" + "".join(f"{bar[str(k)]['seed_mean']:9.4f}" for k in (1, 5, 15, 30)))
    print(f"  {'seed best':<16s}" + "".join(f"{bar[str(k)]['seed_best']:9.4f}" for k in (1, 5, 15, 30)))
    for name, rec in results.items():
        print(f"  {name:<16s}" + "".join(f"{rec[str(k)]:9.4f}" for k in (1, 5, 15, 30)))
    print(f"\ndelta vs seed mean / vs seed best, at k={k}:")
    for name in results:
        d = deltas[name][str(k)]
        print(f"  {name:<16s}{d['minus_seed_mean']:+.4f} / {d['minus_seed_best']:+.4f}")
    print(f"\npaired bootstrap at k={k} (ensemble minus a single seed):")
    for name, cell in boot.items():
        print(f"  {name:<28s}{cell['point']:+.4f}  95% CI [{cell['ci95'][0]:+.4f}, {cell['ci95'][1]:+.4f}]")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Two settings whose sign changes with the output budget, scheduled instead of fixed.

The released system fixes one pool cap and one aggregation rule for every budget it is read at.
Both were measured once, at one budget, and both turn out to change sign along the budget axis: the
cap costs recall at the head of the list and buys it at depth, and the aggregation is the other way
round. Fixing either at a single value therefore runs the worse setting over half the range.

The obvious way to measure this is the wrong one. results/pool_cap_cost.json sweeps eight caps on
the comparison set and says so in its own status field: the best was read off the population it is
reported on, which is an argmax over the evaluation set and not a result. So the schedules here are
chosen on the validation draw and spent once on the comparison set, and what the comparison set
says is reported whether or not it agrees with what validation promised. The gap between the two
is printed, because a schedule that does not transfer is the outcome this design exists to expose.

What is not measured: the two axes together. The aggregation ablation fixes the cap at the deployed
value and this file's cap sweep fixes the aggregation at the deployed rule, so each schedule is
read with the other axis held. Their joint effect needs the per-template scores under a varying
cap, which no artifact holds.

Both populations are the released pair, verified 120 of 120 per stage by
scripts/typed_edit/pool_checkpoints.py, so a schedule carried between them is carried between two
readings of one model.

    python scripts/budget_dependent_schedules.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _rrf import RRF_K, competition_ranks  # noqa: E402

BUDGETS = [1, 3, 5, 8, 10, 15, 20, 30, 50]
CAPS = [50, 100, 250, 500, 1000, 2000, 0]   # 0 is uncapped
DEPLOYED_CAP, DEPLOYED_RULE = 100, "noisy_or"
POOLS = {"validation": "results/val_pools.json",
         "comparison": "results/widepools_implicit/w*.json"}
AGG = {"validation": "results/aggregation_ablation_validation.json",
       "comparison": "results/aggregation_ablation.json"}


def load(population):
    pools, refs = {}, {}
    for f in sorted(glob.glob(str(ROOT / POOLS[population]))) or [str(ROOT / POOLS[population])]:
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"])
        refs.update(blob["references"])
    subs = sorted(s for s in pools if refs.get(s))
    return pools, {s: set(refs[s]) for s in subs}, subs


def ranked_keys(cands, cap, limit, self_key=None):
    """The deployed ranking: cap by generator score, fuse by rank, unique keys in order.

    The parent drop belongs here and not to the caller: a prediction whose key equals the
    substrate's is discarded before the budget, for every arm alike, which is what
    four_method_291.py does. Omitting it read the deployed cap's widest column as 0.7053 against
    the manuscript's 0.7023, and the gate below is what caught that.
    """
    kept = cands if not cap or len(cands) <= cap else \
        sorted(cands, key=lambda c: -c["generator"])[:cap]
    rf = competition_ranks(kept, lambda c: c["filter"])
    rg = competition_ranks(kept, lambda c: c["generator"])
    order = sorted(range(len(kept)),
                   key=lambda i: -(1.0 / (RRF_K + rf[i]) + 1.0 / (RRF_K + rg[i])))
    out, seen = [], set()
    for i in order:
        k = kept[i]["key"]
        if k in seen or k == self_key:
            continue
        seen.add(k)
        out.append(k)
        if len(out) >= limit:
            break
    return out


def cap_table(population):
    """{cap: {budget: micro recall}} under the deployed aggregation."""
    import bank_without_selection as B
    pools, refs, subs = load(population)
    self_keys = {s: B._key(s) for s in subs}
    U = sum(len(refs[s]) for s in subs)
    lim = max(BUDGETS)
    out = {}
    for cap in CAPS:
        order = {s: ranked_keys(pools[s], cap, lim, self_keys[s]) for s in subs}
        out[cap] = {b: round(sum(len(set(order[s][:b]) & refs[s]) for s in subs) / U, 4)
                    for b in BUDGETS}
        print(f"    cap {cap or 'none':>4}: " +
              " ".join(f"{b}:{out[cap][b]:.4f}" for b in (1, 5, 15, 50)), flush=True)
    return out, len(subs), U


def schedule(table, deployed_key):
    """Per budget, the setting validation prefers, and what it promised over the deployed one."""
    return {b: {"choice": max(table, key=lambda c: table[c][b]),
                "validation_gain": round(max(table[c][b] for c in table)
                                         - table[deployed_key][b], 4)}
            for b in BUDGETS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "budget_dependent_schedules.json"))
    args = ap.parse_args()

    rep = {"config": {"budgets": BUDGETS, "caps": CAPS,
                      "deployed": {"cap": DEPLOYED_CAP, "aggregation": DEPLOYED_RULE},
                      "selected_on": "validation", "evaluated_on": "comparison",
                      "aggregation": "micro, ratio of sums",
                      "not_measured": "the two axes jointly; each is read with the other held "
                                      "at its deployed value"}}

    print("  cap sweep on validation (selection)")
    val_caps, val_n, val_u = cap_table("validation")
    print("  cap sweep on comparison (evaluation, read once)")
    cmp_caps, cmp_n, cmp_u = cap_table("comparison")

    cap_sched = schedule(val_caps, DEPLOYED_CAP)
    agg = {p: json.loads((ROOT / AGG[p]).read_text())["by_rule"] for p in AGG}
    agg_val = {r: {b: agg["validation"][r]["recall"][str(b)] for b in BUDGETS} for r in agg["validation"]}
    agg_cmp = {r: {b: agg["comparison"][r]["recall"][str(b)] for b in BUDGETS} for r in agg["comparison"]}
    agg_sched = schedule(agg_val, DEPLOYED_RULE)

    rep["populations"] = {"validation": {"n": val_n, "references": val_u},
                          "comparison": {"n": cmp_n, "references": cmp_u}}
    rep["cap"] = {"validation_table": {str(c): val_caps[c] for c in val_caps},
                  "comparison_table": {str(c): cmp_caps[c] for c in cmp_caps},
                  "schedule": {str(b): cap_sched[b] for b in BUDGETS}, "by_budget": {}}
    rep["aggregation"] = {"validation_table": agg_val, "comparison_table": agg_cmp,
                          "schedule": {str(b): agg_sched[b] for b in BUDGETS}, "by_budget": {}}

    # The aggregation ablation already carries a paired interval on each rule's difference from
    # the deployed one. A delivered gain without one is a point estimate, so it is carried here
    # rather than left in the artifact this file reads.
    agg_ci = json.loads((ROOT / AGG["comparison"]).read_text())["by_rule"]

    def spend(name, sched, cmp_table, deployed_key):
        rows = {}
        for b in BUDGETS:
            pick = sched[b]["choice"]
            got, base = cmp_table[pick][b], cmp_table[deployed_key][b]
            rows[str(b)] = {"chosen_on_validation": pick, "deployed": base, "scheduled": got,
                            "delivered": round(got - base, 4),
                            "validation_promised": sched[b]["validation_gain"],
                            "transfer_gap": round((got - base) - sched[b]["validation_gain"], 4)}
            if name == "aggregation" and pick != deployed_key:
                ci = agg_ci[pick].get("minus_noisy_or", {}).get(str(b))
                if ci:
                    rows[str(b)]["ci95"] = ci["ci95"]
                    rows[str(b)]["excludes_zero"] = ci["excludes_zero"]
        rep[name]["by_budget"] = rows
        return rows

    cap_rows = spend("cap", cap_sched, cmp_caps, DEPLOYED_CAP)
    agg_rows = spend("aggregation", agg_sched, agg_cmp, DEPLOYED_RULE)

    # The gate: the flat deployed settings have to be the manuscript's own column.
    col = json.loads((ROOT / "results/deployment_table.json").read_text())["recall_micro"]
    mism = [f"k={b}: cap sweep {cmp_caps[DEPLOYED_CAP][b]} vs manuscript {col[str(b)]['whole bank']}"
            for b in BUDGETS
            if str(b) in col and abs(cmp_caps[DEPLOYED_CAP][b] - col[str(b)]["whole bank"]) > 5e-4]
    rep["gate"] = {"deployed_cap_reproduces_the_manuscript_column": not mism, "mismatches": mism}
    print(f"\n  gate: the deployed cap reproduces the manuscript's column: {not mism}")
    for m in mism:
        print(f"    {m}")

    for name, rows in (("pool cap", cap_rows), ("aggregation", agg_rows)):
        print(f"\n  {name}: chosen on validation, spent once on the comparison set")
        print(f"  {'k':>3} | {'choice':>10} | {'deployed':>8} | {'scheduled':>9} | "
              f"{'delivered':>9} | {'promised':>8} | {'transfer':>8}")
        for b in BUDGETS:
            r = rows[str(b)]
            print(f"  {b:>3} | {str(r['chosen_on_validation']):>10} | {r['deployed']:>8.4f} | "
                  f"{r['scheduled']:>9.4f} | {r['delivered']:>+9.4f} | "
                  f"{r['validation_promised']:>+8.4f} | {r['transfer_gap']:>+8.4f}"
                  + (f" | {'sep' if r.get('excludes_zero') else 'n.s.'} {r['ci95']}"
                     if "ci95" in r else ""))

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

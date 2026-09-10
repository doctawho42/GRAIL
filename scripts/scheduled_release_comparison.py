#!/usr/bin/env python3
"""The released arm with the blend at the head of the list, against every comparator.

ModelWrapper.generate now picks the candidate aggregation from the budget it is asked for: the
blend at ten and below, the noisy-or above. This measures that arm on the comparison set, against
all five published predictors, at the budgets the manuscript reads.

Nothing is re-run. The per-template generator scores are in results/aggregation_shards/s*.json and
the filter score and tautomer key for each candidate are in results/widepools_implicit/w*.json,
joined by structure: a filter score is a function of the substrate and the product and does not
depend on how the generator combined its templates.

The aggregation is the one in grail_metabolism/model/generator.py rather than a second spelling of
it, so a change to the model changes this measurement instead of drifting from it.

Two gates. Above the switch point the scheduled arm is the released arm, so it has to reproduce the
manuscript's exhaustive column exactly at fifteen, twenty, thirty and fifty; a difference there
means the join or the ranking is not the deployed one and nothing below the switch is trustworthy
either. And MetaTox has to reproduce results/four_method_291.json, which is what says the
population and the conventions are the manuscript's.

    python scripts/scheduled_release_comparison.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "typed_edit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bank_without_selection as B  # noqa: E402
from _rrf import RRF_K, competition_ranks  # noqa: E402
from grail_metabolism.model.generator import Generator  # noqa: E402
from grail_metabolism.model.wrapper import ModelWrapper  # noqa: E402

BUDGETS = [1, 3, 5, 8, 10, 15, 20, 30, 50]
N_BOOT, SEED, CAP = 10000, 0, 100
COMPARATORS = {
    "MetaTox":        ("results/metatox_smirks_preds.json", "predictions"),
    "GLORYx":         ("results/gloryx_service_preds.json", "predictions"),
    "MetaPredictor":  ("artifacts/tier2_1170/metapredictor_preds.json", None),
    "SyGMa":          ("results/sygma_fulltest_predictions.json", None),
    "BioTransformer": ("results/biotransformer_fulltest_preds.json", None),
}


def aggregate(scores, rule):
    """The model's own aggregation, called rather than reimplemented."""
    g = Generator.__new__(Generator)
    g.candidate_aggregation = rule
    return g._aggregate_candidate_scores(scores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "scheduled_release_comparison.json"))
    args = ap.parse_args()

    # Both arms of the released system: the same pipeline at two rule budgets. The interactive
    # arm's per-template scores did not exist until they were collected for this, which is why
    # every earlier reading of the aggregation covers one arm only.
    ARMS = {"whole bank": "results/aggregation_shards/s*.json",
            "trained budget": "results/aggregation_shards_k30/s*.json"}
    per_template = {}
    for arm, spec in ARMS.items():
        rows = {}
        for f in sorted(glob.glob(str(ROOT / spec))):
            rows.update(json.loads(Path(f).read_text())["rows"])
        per_template[arm] = rows
    pools, refs_raw = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/widepools_implicit/w*.json"))):
        blob = json.loads(Path(f).read_text())
        pools.update(blob["pools"])
        refs_raw.update(blob["references"])

    comp_raw = {n: (json.loads((ROOT / rel).read_text())[k] if k
                    else json.loads((ROOT / rel).read_text()))
                for n, (rel, k) in COMPARATORS.items()}
    subs = sorted(set(pools) & set(refs_raw)
                  & set.intersection(*(set(r) for r in per_template.values()))
                  & set.intersection(*(set(p) for p in comp_raw.values())))
    if len(subs) != 291:
        raise SystemExit(f"population is {len(subs)}, not the committed 291")
    refs = {s: set(refs_raw[s]) for s in subs}
    print(f"population: {len(subs)} substrates, {sum(map(len, refs.values()))} references",
          flush=True)

    from multiprocessing import Pool
    kp = Pool(6)
    self_key = {s: B._key(s) for s in subs}
    lim = max(BUDGETS)
    comp = {n: {s: [k for k in B._dedup(p.get(s, []), None, kp) if k != self_key[s]][:lim]
                for s in subs} for n, p in comp_raw.items()}
    kp.close()

    # The filter score and the key come from the pool, the per-template scores from the shard.
    side = {s: {c["smiles"]: (c["filter"], c["key"]) for c in pools[s]} for s in subs}
    join = {}
    for arm, rows in per_template.items():
        j = sum(1 for s in subs for x in rows[s] if x in side[s])
        u = sum(1 for s in subs for x in rows[s] if x not in side[s])
        join[arm] = {"joined": j, "unjoined": u, "share": round(u / max(j + u, 1), 4)}
        print(f"  {arm}: candidates joined {j}, unjoined {u} ({join[arm]['share']:.4f})",
              flush=True)

    def ranked(s, rule, arm):
        """The deployed order of operations, which is not interchangeable with any other.

        Deduplicate by match key in descending order of the product of the two component scores,
        then cap by generator score, then fuse. Capping first lets duplicate keys consume cap
        slots, which costs recall and would be charged to the aggregation rule rather than to the
        order of two steps: doing it that way read 0.5368 at a budget of fifteen where the
        manuscript reads 0.5353, and the gate below is what caught it.
        """
        cands = [(x, aggregate(v, rule), *side[s][x])
                 for x, v in per_template[arm][s].items() if x in side[s]]
        cands.sort(key=lambda c: -(c[2] * c[1]))
        seen, pool = set(), []
        for c in cands:
            if not c[3] or c[3] in seen:
                continue
            seen.add(c[3])
            pool.append(c)
        kept = sorted(pool, key=lambda c: -c[1])[:CAP]
        rf = competition_ranks(kept, lambda c: c[2])
        rg = competition_ranks(kept, lambda c: c[1])
        order = sorted(range(len(kept)),
                       key=lambda i: -(1.0 / (RRF_K + rf[i]) + 1.0 / (RRF_K + rg[i])))
        return [kept[i][3] for i in order if kept[i][3] != self_key[s]][:lim]

    by_arm = {arm: {r: {s: ranked(s, r, arm) for s in subs} for r in ("noisy_or", "hybrid")}
              for arm in per_template}
    U = np.array([len(refs[s]) for s in subs], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    denom = np.maximum(U[idx].sum(axis=1), 1)

    def hits(order, b):
        return np.array([len(set(order[s][:b]) & refs[s]) for s in subs], dtype=float)

    rep = {"config": {**B._code_version(), "population": len(subs), "budgets": BUDGETS,
                      "switch": {"blend_at_or_below": ModelWrapper.BLEND_BUDGET,
                                 "rule_at_the_head": "hybrid", "rule_above": "noisy_or"},
                      "cap": CAP, "aggregation": "micro, ratio of sums", "join": join},
           "comparators": {}, "arms": {}}
    ch = {n: {b: hits(comp[n], b) for b in BUDGETS} for n in comp}
    for n in ch:
        rep["comparators"][n] = {str(b): round(float(ch[n][b].sum() / U.sum()), 4) for b in BUDGETS}

    for arm in by_arm:
        rep["arms"][arm] = {}
        for b in BUDGETS:
            rule = "hybrid" if b <= ModelWrapper.BLEND_BUDGET else "noisy_or"
            h = hits(by_arm[arm][rule], b)
            base = hits(by_arm[arm]["noisy_or"], b)
            row = {"rule": rule, "micro": round(float(h.sum() / U.sum()), 4),
                   "released_rule_everywhere": round(float(base.sum() / U.sum()), 4), "vs": {}}
            d = h - base
            bt = d[idx].sum(axis=1) / denom
            lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            row["gain_over_the_released_rule"] = {
                "difference": round(float(d.sum() / U.sum()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
            for n in ch:
                dd = h - ch[n][b]
                bb = dd[idx].sum(axis=1) / denom
                l2, h2 = float(np.quantile(bb, .025)), float(np.quantile(bb, .975))
                row["vs"][n] = {"gap": round(float(dd.sum() / U.sum()), 4),
                                "ci95": [round(l2, 4), round(h2, 4)],
                                "excludes_zero": bool(l2 > 0 or h2 < 0)}
            rep["arms"][arm][str(b)] = row

    col = json.loads((ROOT / "results/deployment_table.json").read_text())["recall_micro"]
    above = [b for b in BUDGETS if b > ModelWrapper.BLEND_BUDGET]
    m1 = [f"{arm} k={b}: {rep['arms'][arm][str(b)]['micro']} vs manuscript {col[str(b)][arm]}"
          for arm in rep["arms"] for b in above
          if abs(rep["arms"][arm][str(b)]["micro"] - col[str(b)][arm]) > 5e-4]
    four = json.loads((ROOT / "results/four_method_291.json").read_text())["per_method"]["MetaTox"]["recall"]
    m2 = [f"k={b}: {rep['comparators']['MetaTox'][str(b)]} vs {four[str(b)]}"
          for b in BUDGETS if str(b) in four
          and abs(rep["comparators"]["MetaTox"][str(b)] - four[str(b)]) > 1e-9]
    rep["gates"] = {"above_the_switch_reproduces_the_manuscript": not m1, "mismatches": m1,
                    "metatox_reproduces_four_method_291": not m2, "metatox_mismatches": m2}
    print(f"  gate: above the switch reproduces the manuscript: {not m1}")
    for m in m1:
        print(f"    {m}")
    print(f"  gate: MetaTox reproduces the committed artifact: {not m2}")

    for arm in rep["arms"]:
        print(f"\n  {arm}")
        print(f"  {'k':>3} | {'rule':>8} | {'scheduled':>9} | {'released':>8} | {'gain':>14} | "
              f"{'best rival':>14} | {'gap':>14}")
        for b in BUDGETS:
            _print_row(rep, arm, b, ch)

    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nWrote {args.out}")
    return 0


def _print_row(rep, arm, b, ch):
        r = rep["arms"][arm][str(b)]
        g = r["gain_over_the_released_rule"]
        rival = max(ch, key=lambda n: rep["comparators"][n][str(b)])
        v = r["vs"][rival]
        print(f"  {b:>3} | {r['rule']:>8} | {r['micro']:>9.4f} | "
              f"{r['released_rule_everywhere']:>8.4f} | "
              f"{g['difference']:>+8.4f} {'sep' if g['excludes_zero'] else 'n.s.':>4} | "
              f"{rival[:13]:>14} | {v['gap']:>+8.4f} "
              f"{'sep' if v['excludes_zero'] else 'n.s.':>4}")


if __name__ == "__main__":
    raise SystemExit(main())

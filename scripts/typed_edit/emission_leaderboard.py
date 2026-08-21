#!/usr/bin/env python3
"""Does one declared emission rule make GRAIL the leader on macro F1?

The five-method table is read at recall@15, where SyGMa reaches 0.554 over 74 emitted candidates
and GRAIL 0.365 over 8.7. Read at macro F1 the ordering is different, and the spread between
GRAIL and the leader is 0.034 while the pool-relative emission rule of the emission appendix is
worth about 0.073 over the deployed budget. That arithmetic says the rule alone could reorder
the board, and it is computable on files that already exist.

What is compared, declared rather than implied. Every method is scored at its OWN emission
policy, because how many candidates a method emits is part of the method: SyGMa's 74, and
GRAIL's either the deployed budget or the pool-relative rule. That asymmetry is the comparison,
not a flaw in it, but it is an asymmetry and the budget-matched view is reported beside it.

alpha is not fitted here. It is the value the appendix reports, and `emission_crossfit.py`
already tested it out of sample; refitting it on this table would make the answer its own
argmax.

Nothing is retrained. GRAIL's arms come from the released per-candidate score dump, and the
deployed arm is gated against the committed five-method table before anything else is read: if
it does not reproduce that row, this is not the same population or the same matcher and no
comparison it makes is worth anything.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from grail_metabolism.metrics import aggregate_prediction_metrics  # noqa: E402
from scripts.run_benchmark import load_test_map  # noqa: E402
from scripts.run_match_sensitivity import _dedup_canon  # noqa: E402

SHARED = ROOT / "artifacts" / "tier2" / "substrates.json"
DUMP = ROOT / "results" / "scored_predictions.json"
COMMITTED = ROOT / "results" / "budget_matched_leaderboard.json"
PREDS = {"BioTransformer": ROOT / "artifacts/tier2/biotransformer_preds.json",
         "MetaPredictor": ROOT / "artifacts/tier2/metapredictor_preds.json",
         "MetaTrans": ROOT / "artifacts/tier2/metatrans_preds.json"}
SYGMA = ROOT / "results" / "sygma_fulltest_predictions.json"
MATCH = "inchikey_tautomer"
KS = [5, 10, 15]


def evaluate(preds, reals, subs):
    """One pass: per-substrate metrics, and the macro table as their mean.

    Macro here is the mean over substrates, which is what `aggregate_prediction_metrics` does
    internally, so the table and the per-substrate vector the bootstrap resamples are the same
    numbers rather than two computations that could disagree.
    """
    # the committed table deduplicates by canonical SMILES before scoring, which leaves recall
    # and precision alone -- the matcher dedups anyway -- and changes mean_output. Scoring the
    # raw lists reproduced four values of six and was caught on the other two.
    f1s, rows = [], []
    for s in subs:
        m = aggregate_prediction_metrics(
            [{"predicted": _dedup_canon(list(preds.get(s, []))), "real": list(reals[s])}],
            KS, match=MATCH)
        f1s.append(m["f1"])
        rows.append((m["top_15_recall"], m["precision"], m["mean_output_size"]))
    n = max(len(rows), 1)
    table = {"recall@15": round(sum(r[0] for r in rows) / n, 3),
             "precision": round(sum(r[1] for r in rows) / n, 3),
             "f1": round(sum(f1s) / n, 3),
             "mean_output": round(sum(r[2] for r in rows) / n, 2)}
    return table, f1s


def paired_ci(a, b, n_boot=10000, seed=0):
    """Paired bootstrap on the per-substrate difference, rows sorted so a rerun reproduces."""
    import numpy as np

    order = sorted(range(len(a)), key=lambda i: (a[i] - b[i], i))
    d = np.array([a[i] - b[i] for i in order], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "emission_leaderboard.json"))
    args = ap.parse_args()

    shared = json.loads(SHARED.read_text())
    tm = load_test_map(None, 42)
    dump = {r["sub"]: r["candidates"] for r in json.loads(DUMP.read_text())["rows"]}
    subs = sorted(s for s in shared if s in tm and tm[s] and s in dump)

    arms = {
        "GRAIL, deployed budget": {s: [c["smiles"] for c in dump[s][:15]] for s in subs},
        f"GRAIL, pool-relative alpha={args.alpha}": {
            s: [c["smiles"] for c in dump[s]
                if dump[s] and c["combined"] >= args.alpha * dump[s][0]["combined"]]
            for s in subs},
    }
    for name, path in PREDS.items():
        if path.exists():
            arms[name] = json.loads(path.read_text())
    if SYGMA.exists():
        sy = json.loads(SYGMA.read_text())
        arms["SyGMa"] = sy if isinstance(sy, dict) else {}

    table, f1_by_arm = {}, {}
    for name, p in arms.items():
        print(f"  scoring {name} ...", file=sys.stderr, flush=True)
        table[name], f1_by_arm[name] = evaluate(p, tm, subs)

    # the gate: the deployed arm has to reproduce the committed row, or this is a different
    # population or a different matcher and nothing below means what it says
    committed = json.loads(COMMITTED.read_text())["by_method"]
    got = table["GRAIL, deployed budget"]
    want = committed["GRAIL"]
    mismatches = [k for k in ("recall@15", "precision", "mean_output")
                  if abs(got[k] - want[k]) > 1e-9]
    for name in PREDS:
        if name in table:
            mismatches += [f"{name}.{k}" for k in ("recall@15", "precision", "mean_output")
                           if abs(table[name][k] - committed[name][k]) > 1e-9]

    ranked = sorted(table.items(), key=lambda kv: -kv[1]["f1"])
    rule = f"GRAIL, pool-relative alpha={args.alpha}"
    contrasts = {name: paired_ci(f1_by_arm[rule], f1_by_arm[name], args.n_boot, args.seed)
                 for name in table if name != rule}

    rep = {
        "population": {"source": str(SHARED.relative_to(ROOT)), "n": len(subs)},
        "match": MATCH, "alpha": args.alpha,
        "alpha_note": "the value the emission appendix reports, not refitted here",
        "declared": "each method at its own emission policy; the budget-matched view is "
                    "results/budget_matched_leaderboard.json",
        "gate": {"reproduces_committed_table": not mismatches, "mismatches": mismatches},
        "table": table,
        "f1_order": [n for n, _ in ranked],
        "leader": ranked[0][0],
        "paired_f1_contrasts_vs_rule": {k: {"delta": v[0], "ci95": [v[1], v[2]]}
                                        for k, v in contrasts.items()},
        "n_boot": args.n_boot, "seed": args.seed,
    }
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\npopulation {len(subs)} shared substrates, match {MATCH}\n")
    print(f"{'arm':<34}{'F1':>7}{'recall@15':>11}{'precision':>11}{'output':>8}")
    for name, r in ranked:
        print(f"{name:<34}{r['f1']:>7.3f}{r['recall@15']:>11.3f}{r['precision']:>11.3f}"
              f"{r['mean_output']:>8.2f}")
    print(f"\nF1 leader: {rep['leader']}")
    print("\npaired F1, the rule minus each other arm:")
    for k, v in rep["paired_f1_contrasts_vs_rule"].items():
        star = "" if v["ci95"][0] <= 0 <= v["ci95"][1] else "  separated from zero"
        print(f"  {k:<34}{v['delta']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]{star}")
    if mismatches:
        print(f"\nGATE FAILED, {len(mismatches)} values do not reproduce: {mismatches}")
        return 1
    print("\ngate: the deployed arm and the frozen methods reproduce the committed table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

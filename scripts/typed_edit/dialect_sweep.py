#!/usr/bin/env python3
"""The comparison recomputed with the substrate drawn the other way.

The corpus stores amides as imidic acids and cytosine as its lactim, and the rules are matched
against that drawing. This paper sweeps the matching criterion and the output budget and reports
the whole grid; the substrate's presentation is a choice of the same kind and was not swept. It is
here.

Both GRAIL arms are rebuilt on the identical substrates, presented as the declared standardiser
draws them, with the same checkpoints, the same rule budgets, the same pool cap and the same
ranking. The pools stay keyed by the corpus string, so the annotation is one annotation and the
two dialects are paired substrate by substrate.

The comparators cannot all move with them, and that is stated rather than hidden. SyGMa is
re-runnable and is swept separately. MetaPredictor's predictions are a frozen delivery and
MetaTox's come from a web service that is not re-runnable here, so both are held fixed; a
difference against them therefore measures how much of our own margin was the drawing, which is
the question that matters for reading the comparison.

    python scripts/typed_edit/dialect_sweep.py
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

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402

KS = (1, 3, 5, 8, 10, 15, 20, 30, 50)
CAP = 100
N_BOOT, SEED = 10000, 0
COMPARATORS = {
    "MetaTox": ("results/metatox_smirks_preds.json", "predictions"),
    "SyGMa": ("results/sygma_fulltest_predictions.json", None),
    "MetaPredictor": ("artifacts/tier2_1170/metapredictor_preds.json", None),
}
ARMS = {
    # The exhaustive arm's standardised pools come from the coarse shards where those completed
    # and from a finer re-run of the same ranges where they did not. Both are the same
    # measurement on the same substrates with the same checkpoints; the finer pieces exist only
    # because a shard writes when it finishes, so two ranges holding the largest substrates had
    # hours of work that nothing on disk could show.
    "GRAIL exhaustive": ("results/widepools_implicit/w*.json",
                         ["results/widepools_std/w*.json",
                          "results/widepools_std_fine/p*.json"]),
    "GRAIL interactive": ("results/widepools_k30/all.json", "results/widepools_k30_std/all.json"),
}


def load(pattern):
    """One or several glob patterns, merged. A substrate present in more than one is taken once."""
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    pools, refs = {}, {}
    for spec in patterns:
        for path in sorted(glob.glob(str(ROOT / spec))):
            blob = json.loads(Path(path).read_text())
            for substrate, pool in blob["pools"].items():
                pools.setdefault(substrate, pool)
            for substrate, r in (blob.get("references") or {}).items():
                refs.setdefault(substrate, r)
    if not pools:
        raise SystemExit(f"no pool matched {patterns}")
    return pools, refs


def ranked(pool):
    keep = sorted(pool, key=lambda c: -c["generator"])[:CAP]
    return [c["key"] for c in rrf_order(keep) if c.get("key")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "dialect_sweep.json"))
    ap.add_argument("--null-control", action="store_true",
                    help="point both dialects at the stored pools. Every difference must come "
                         "out exactly zero and no verdict may move; anything else is a defect in "
                         "the pairing rather than a result, and this is the only way to see it "
                         "before the real numbers are there to be believed.")
    args = ap.parse_args()

    arms_spec = dict(ARMS)
    if args.null_control:
        arms_spec = {name: (stored, stored) for name, (stored, _) in ARMS.items()}

    from bank_without_selection import _dedup, _key as tautkey

    arms, refs = {}, {}
    for name, (stored_pattern, drawn_pattern) in arms_spec.items():
        stored, r = load(stored_pattern)
        refs.update(r)
        drawn, _ = load(drawn_pattern)
        arms[name] = {"stored": stored, "standardised": drawn}

    subs = sorted(set.intersection(*[set(d[k]) for d in arms.values() for k in d])
                  & {s for s in refs if refs[s]})
    print(f"population: {len(subs)} substrates held by both dialects of both arms")

    parent = {s: tautkey(s) for s in subs}
    ordered = {}
    for name, dialects in arms.items():
        for dialect, pools in dialects.items():
            ordered[(name, dialect)] = {
                s: [k for k in ranked(pools[s]) if k != parent[s]] for s in subs}

    for name, (rel, field) in COMPARATORS.items():
        blob = json.loads((ROOT / rel).read_text())
        preds = blob[field] if field else blob
        ordered[(name, "stored")] = {
            s: [k for k in _dedup(preds.get(s, []), max(KS) + 20) if k and k != parent[s]]
            for s in subs}

    truth = {s: set(refs[s]) for s in subs}
    U = np.array([len(truth[s]) for s in subs], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    den = np.maximum(U[idx].sum(axis=1), 1)

    def hits(key, k):
        return np.array([len(set(ordered[key][s][:k]) & truth[s]) for s in subs], dtype=float)

    # 1. what the drawing does to each GRAIL arm, paired
    effect = {}
    for name in arms_spec:
        row = {}
        for k in KS:
            a, b = hits((name, "standardised"), k), hits((name, "stored"), k)
            d = a - b
            bt = d[idx].sum(axis=1) / den
            lo, hi = np.percentile(bt, [2.5, 97.5])
            row[str(k)] = {"stored": round(float(b.sum() / U.sum()), 4),
                           "standardised": round(float(a.sum() / U.sum()), 4),
                           "difference": round(float(d.sum() / U.sum()), 4),
                           "ci95": [round(float(lo), 4), round(float(hi), 4)],
                           "separates": bool(lo > 0 or hi < 0)}
        effect[name] = row

    # 2. what it does to every head-to-head verdict the paper claims
    verdicts = {}
    for name in arms_spec:
        for comparator in COMPARATORS:
            row = {}
            for k in KS:
                out = {}
                for dialect in ("stored", "standardised"):
                    d = hits((name, dialect), k) - hits((comparator, "stored"), k)
                    bt = d[idx].sum(axis=1) / den
                    lo, hi = np.percentile(bt, [2.5, 97.5])
                    out[dialect] = {"difference": round(float(d.sum() / U.sum()), 4),
                                    "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                    "separates": bool(lo > 0 or hi < 0),
                                    "sign": int(np.sign(d.sum()))}
                out["verdict_moves"] = bool(
                    out["stored"]["separates"] != out["standardised"]["separates"]
                    or (out["stored"]["separates"] and out["standardised"]["separates"]
                        and out["stored"]["sign"] != out["standardised"]["sign"]))
                row[str(k)] = out
            verdicts[f"{name} vs {comparator}"] = row

    # 3. the coverage ceiling on this population, under both drawings. The exhaustive pool is the
    # whole bank applied without a selector and is not capped, so the share of references that
    # appear anywhere in it IS the ceiling on these substrates. It costs nothing extra here, and
    # it is the same quantity the main text reports on the full split under one drawing.
    ceiling = {}
    for dialect in ("stored", "standardised"):
        pools = arms["GRAIL exhaustive"][dialect]
        reached = np.array([len({c["key"] for c in pools[s] if c.get("key")} & truth[s])
                            for s in subs], dtype=float)
        ceiling[dialect] = {"recovered": int(reached.sum()), "of": int(U.sum()),
                            "coverage": round(float(reached.sum() / U.sum()), 4)}
    a = np.array([len({c["key"] for c in arms["GRAIL exhaustive"]["standardised"][s]
                       if c.get("key")} & truth[s]) for s in subs], dtype=float)
    b = np.array([len({c["key"] for c in arms["GRAIL exhaustive"]["stored"][s]
                       if c.get("key")} & truth[s]) for s in subs], dtype=float)
    bt = (a - b)[idx].sum(axis=1) / den
    lo, hi = np.percentile(bt, [2.5, 97.5])
    ceiling["difference"] = {"value": round(float((a - b).sum() / U.sum()), 4),
                             "ci95": [round(float(lo), 4), round(float(hi), 4)],
                             "separates": bool(lo > 0 or hi < 0)}
    ceiling["note"] = ("measured on the comparison set rather than the full evaluated split, "
                       "because the uncapped exhaustive pool for this population exists in both "
                       "drawings and the full split's does not in the second")

    moved = sum(1 for r in verdicts.values() for k in r if r[k]["verdict_moves"])
    cells = sum(len(r) for r in verdicts.values())

    if args.null_control:
        offenders = [(a, k) for a, row in effect.items() for k, v in row.items()
                     if v["difference"] != 0.0]
        moved_ = sum(1 for r in verdicts.values() for k in r if r[k]["verdict_moves"])
        if offenders or moved_ or ceiling["difference"]["value"] != 0.0:
            print(f"NULL CONTROL FAILED: {len(offenders)} non-zero cells, {moved_} verdicts "
                  f"moved, ceiling difference {ceiling['difference']['value']}")
            return 1
        print("null control: every difference is exactly zero and no verdict moves")
        return 0

    rep = {"provenance": stamp(__file__), "n": len(subs), "budgets": list(KS),
           "match": "inchikey_tautomer", "cap": CAP, "n_boot": N_BOOT, "seed": SEED,
           "effect_on_each_arm": effect,
           "coverage_ceiling": ceiling,
           "verdicts": verdicts,
           "verdict_cells": cells, "verdict_cells_that_move": moved,
           "comparators_held_fixed": ["MetaTox", "MetaPredictor", "SyGMa"],
           "why_held_fixed": ("MetaTox is a web service and MetaPredictor a frozen delivery, so "
                              "neither can be re-run here; SyGMa is swept separately in "
                              "results/sygma_by_dialect.json"),
           "reading": ("difference is standardised minus stored, so a positive value is recall "
                       "the corpus's drawing was denying the arm")}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\n{'arm':<20}" + "".join(f"{k:>9}" for k in KS))
    for name in arms_spec:
        cells_ = "".join(f"{effect[name][str(k)]['difference']:>+9.4f}" for k in KS)
        print(f"{name:<20}{cells_}")
    print(f"\ncoverage ceiling on this population: stored "
          f"{ceiling['stored']['coverage']:.4f} ({ceiling['stored']['recovered']} of "
          f"{ceiling['stored']['of']}), standardised {ceiling['standardised']['coverage']:.4f} "
          f"({ceiling['standardised']['recovered']}); difference "
          f"{ceiling['difference']['value']:+.4f} "
          f"[{ceiling['difference']['ci95'][0]:+.4f}, {ceiling['difference']['ci95'][1]:+.4f}]"
          + ("  separates" if ceiling["difference"]["separates"] else ""))
    print(f"\nverdict cells that move: {moved} of {cells}")
    for label, row in verdicts.items():
        movers = [k for k in row if row[k]["verdict_moves"]]
        if movers:
            print(f"  {label}: k = {', '.join(movers)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

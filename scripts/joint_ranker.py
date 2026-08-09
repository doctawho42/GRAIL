#!/usr/bin/env python3
"""Ordering the joint GRAIL x MetaTox pool, which is where that ensemble's headroom turned out to be.

At a matched budget the ensemble does not beat its better half: rank-interleaving the two reaches
0.499 at k=15 against MetaTox's 0.514 alone. The pool is not the problem. It holds 0.674 of the
references in about forty candidates, and a perfect ranker takes 0.624 of them with five. The whole
deficit is ordering, so this measures how much of it an ordering policy can close.

Untrained policies come first, and not as a formality. A learned ranker here could only be trained
and evaluated on one population -- MetaTox's predictions exist for 291 substrates and no others --
so any learned number is cross-validated rather than held out, and it has to beat something that
needs no fitting at all before it is worth reporting.

  metatox / grail        each method's own order, as a floor
  interleave             rank 1 of each, then rank 2; the naive ensemble already measured
  both-first             candidates proposed by BOTH methods first, then the rest by MetaTox order
  rrf                    reciprocal rank fusion, score = sum over methods of 1/(k0 + rank), the
                         standard untrained fusion; a candidate found twice scores twice, so the
                         agreement signal is used without being tuned
  metatox-score          MetaTox's own likeness score, GRAIL's candidates appended
  grail-filter           GRAIL's filter score over the whole pool where it has one

The diagnostic that decides whether agreement is worth anything is printed first: how often a
candidate proposed by both methods is a reference, against one proposed by either alone. A fusion
that leans on agreement is only sensible if that number says so.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _tautomer_inchikey as _tk

KEYS = ROOT / "results" / "key_tables" / "inchikey_tautomer.json"
BUDGETS = (1, 3, 5, 10, 15, 30)
N_BOOT, SEED = 10000, 0
RRF_K0 = 60          # the value the fusion literature uses; fixed here, not tuned on the outcome


def _code_version() -> dict:
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    return {"script": pathlib.Path(__file__).name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "joint_ranker.json"))
    args = ap.parse_args()

    cache = json.loads(KEYS.read_text()) if KEYS.exists() else {}

    def key(s):
        k = cache.get(s)
        if k is None:
            try:
                k = _tk(s)
            except Exception:
                k = None
            cache[s] = k
        return k

    graw = {r["sub"]: r["candidates"]
            for r in json.loads((ROOT / "results/scored_predictions.json").read_text())["rows"]}
    mt = json.loads((ROOT / "results/metatox_smirks_preds.json").read_text())
    mscored = mt["predictions_with_scores"]
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    subs = sorted(set(truth) & set(graw) & set(mscored))
    print(f"population: {len(subs)} substrates with references and both methods", flush=True)

    # per substrate: key -> {g_rank, m_rank, g_filter, m_score}, plus the reference key set
    per = []
    census = Counter()
    for s in subs:
        refs = {key(y) for y in truth[s]} - {None}
        if not refs:
            continue
        pk, feats = key(s), {}
        for i, c in enumerate(graw[s]):
            k = key(c["smiles"])
            if k is None or k == pk or k in feats:
                continue
            feats[k] = {"g_rank": i, "g_filter": float(c.get("filter") or 0.0),
                        "m_rank": None, "m_score": None}
        for i, row in enumerate(mscored[s]):
            smi, pa = row[0], float(row[1])
            k = key(smi)
            if k is None or k == pk:
                continue
            if k in feats:
                if feats[k]["m_rank"] is None:
                    feats[k].update(m_rank=i, m_score=pa)
            else:
                feats[k] = {"g_rank": None, "g_filter": None, "m_rank": i, "m_score": pa}
        for k, f in feats.items():
            both = f["g_rank"] is not None and f["m_rank"] is not None
            tag = "both" if both else ("grail" if f["g_rank"] is not None else "metatox")
            census[(tag, k in refs)] += 1
        per.append({"sub": s, "refs": refs, "feats": feats})

    print("\n  is agreement worth anything? precision of a candidate by who proposed it")
    for tag in ("both", "grail", "metatox"):
        hit, tot = census[(tag, True)], census[(tag, True)] + census[(tag, False)]
        print(f"    proposed by {tag:8} {hit:>5}/{tot:<6} = {hit / max(tot, 1):.4f}")

    def order(f: dict, policy: str) -> tuple:
        g, m = f["g_rank"], f["m_rank"]
        if policy == "metatox":
            return (0, m) if m is not None else (1, g or 0)
        if policy == "grail":
            return (0, g) if g is not None else (1, m or 0)
        if policy == "interleave":
            return (min(2 * (m if m is not None else 10**6),
                        2 * (g if g is not None else 10**6) + 1),)
        if policy == "both_first":
            return (0 if (g is not None and m is not None) else 1,
                    m if m is not None else 10**6, g if g is not None else 10**6)
        if policy == "rrf":
            s = 0.0
            if g is not None:
                s += 1.0 / (RRF_K0 + g + 1)
            if m is not None:
                s += 1.0 / (RRF_K0 + m + 1)
            return (-s,)
        if policy == "metatox_score":
            return (0, -(f["m_score"] or 0.0)) if m is not None else (1, g or 0)
        if policy == "grail_filter":
            return (-(f["g_filter"] or 0.0), m if m is not None else 10**6)
        raise SystemExit(f"unknown policy {policy}")

    # A learned arm, and the honesty constraint on it. MetaTox's predictions exist for one
    # population and no other, so there is no disjoint split to hold out; the model is therefore
    # fitted and scored under grouped cross-validation over substrates -- every candidate is scored
    # by a model that never saw its substrate -- and reported as cross-validated, not held out.
    # Five features, all of them things the untrained policies use one at a time.
    def _num(x) -> float:
        """A missing score is zero, and so is a NaN -- `x or 0.0` does not catch NaN, which is
        truthy, and sklearn refuses the array rather than silently fitting on it."""
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if v != v else v

    def features(f: dict) -> list[float]:
        g, m = f["g_rank"], f["m_rank"]
        return [1.0 if (g is not None and m is not None) else 0.0,
                1.0 / (1.0 + g) if g is not None else 0.0,
                1.0 / (1.0 + m) if m is not None else 0.0,
                _num(f["g_filter"]), _num(f["m_score"])]

    def learned_scores(folds: int = 5):
        """Per substrate, a score for each candidate from a model fitted without that substrate."""
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception:
            return None
        order_subs = list(range(len(per)))
        out = [None] * len(per)
        for fold in range(folds):
            test = {i for i in order_subs if i % folds == fold}
            Xtr, ytr = [], []
            for i, r in enumerate(per):
                if i in test:
                    continue
                for k, f in r["feats"].items():
                    Xtr.append(features(f))
                    ytr.append(1 if k in r["refs"] else 0)
            if not any(ytr) or all(ytr):
                continue
            model = LogisticRegression(max_iter=2000, class_weight="balanced")
            model.fit(np.array(Xtr), np.array(ytr))
            for i in sorted(test):
                keys = list(per[i]["feats"])
                if not keys:
                    out[i] = {}
                    continue
                pr = model.predict_proba(np.array([features(per[i]["feats"][k]) for k in keys]))[:, 1]
                out[i] = dict(zip(keys, pr))
        return out

    learned = learned_scores()
    POLICIES = ["metatox", "grail", "interleave", "both_first", "rrf", "metatox_score",
                "grail_filter"] + (["learned"] if learned else [])
    U = np.array([len(r["refs"]) for r in per], dtype=float)
    hits = {p: {b: np.zeros(len(per)) for b in BUDGETS} for p in POLICIES}
    oracle = {b: np.zeros(len(per)) for b in BUDGETS}
    for j, r in enumerate(per):
        for p in POLICIES:
            if p == "learned":
                sc = learned[j] or {}
                ranked = sorted(r["feats"], key=lambda k: -sc.get(k, 0.0))
            else:
                ranked = sorted(r["feats"], key=lambda k: order(r["feats"][k], p))
            for b in BUDGETS:
                hits[p][b][j] = len(r["refs"] & set(ranked[:b]))
        present = [k for k in r["feats"] if k in r["refs"]]
        for b in BUDGETS:
            oracle[b][j] = min(len(present), b)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per), (N_BOOT, len(per)))

    def rate(H):
        return round(float(H.sum() / U.sum()), 4)

    def paired(A, B):
        d = A - B
        bt = np.array([d[j].sum() / max(U[j].sum(), 1) for j in idx])
        return {"delta": round(float(d.sum() / U.sum()), 4),
                "ci95": [round(float(np.quantile(bt, .025)), 4),
                         round(float(np.quantile(bt, .975)), 4)]}

    print(f"\n  {'policy':16}" + "".join(f"{('k=' + str(b)):>9}" for b in BUDGETS))
    table = {}
    for p in POLICIES:
        table[p] = {b: rate(hits[p][b]) for b in BUDGETS}
        print(f"  {p:16}" + "".join(f"{table[p][b]:>9}" for b in BUDGETS))
    table["oracle"] = {b: rate(oracle[b]) for b in BUDGETS}
    print(f"  {'oracle':16}" + "".join(f"{table['oracle'][b]:>9}" for b in BUDGETS))

    print(f"\n  against MetaTox alone, paired over substrates, at each budget")
    gains = {}
    for p in POLICIES:
        if p == "metatox":
            continue
        gains[p] = {b: paired(hits[p][b], hits["metatox"][b]) for b in BUDGETS}
        cells = "".join(f"{gains[p][b]['delta']:>+9.4f}" for b in BUDGETS)
        print(f"  {p:16}{cells}")

    rep = {"config": {**_code_version(), "n_substrates": len(per), "references": int(U.sum()),
                      "match": "inchikey_tautomer", "aggregation": "micro, ratio of sums",
                      "budgets": list(BUDGETS), "rrf_k0": RRF_K0, "n_boot": N_BOOT, "seed": SEED,
                      "learned_arm": "logistic regression on five features, grouped 5-fold "
                                       "cross-validation over substrates; there is no disjoint "
                                       "population to hold out, so it is cross-validated not held out"},
           "agreement_precision": {t: {"references": census[(t, True)],
                                       "candidates": census[(t, True)] + census[(t, False)]}
                                   for t in ("both", "grail", "metatox")},
           "recall": table, "gain_over_metatox": gains}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    KEYS.write_text(json.dumps(cache))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

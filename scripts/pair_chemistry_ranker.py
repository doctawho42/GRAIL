#!/usr/bin/env python3
"""Ordering the joint pool by the chemistry of each substrate-product pair, not by who proposed it.

Fusing what the two methods say about their own candidates -- ranks, agreement, their own scores --
closes about a seventh of the available headroom. The pool holds 0.674 of the references and the
best such ordering reaches 0.540 at k=15. That is the ceiling of a family of policies that never
look at a molecule.

GRAIL already contains a model that does: its filter is a binary classifier over the (substrate,
product) pair, built on a merged graph aligned by maximum common substructure. It has never been
applied here. The `grail_filter` policy of the previous run scored only GRAIL's own candidates,
because MetaTox's carried no filter score, and silently gave the rest zero -- which is why it
finished below MetaTox alone rather than testing anything.

So this scores EVERY candidate in the joint pool with that model, including the ones GRAIL never
proposed, and ranks by chemistry. Nothing is trained: the checkpoint is the deployed one, and the
candidates it now sees from MetaTox are out of its training distribution in the way that matters,
being products of a rule bank it does not have.

Two readings are possible and the run distinguishes them. If pair chemistry orders the union, the
ceiling of rank fusion was a property of the features and not of the problem. If it does not, the
ordering deficit survives a model that looks at the molecules, and the second paper's ranker has to
be trained on the union rather than borrowed from one arm of it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.config import FilterConfig
from grail_metabolism.metrics import _tautomer_inchikey as _tk
from grail_metabolism.workflows.factory import build_filter

KEYS = ROOT / "results" / "key_tables" / "inchikey_tautomer.json"
BUDGETS = (1, 3, 5, 10, 15, 30)
N_BOOT, SEED = 10000, 0


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


def load_filter(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_filter(FilterConfig(**state["arch"]))
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-ckpt", default="artifacts/full5000_single/checkpoints/filter.pt")
    ap.add_argument("--out", default=str(ROOT / "results" / "pair_chemistry_ranker.json"))
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
    print(f"population: {len(subs)} substrates", flush=True)

    # one entry per candidate, carrying the SMILES the model needs and the provenance we compare
    per = []
    for s in subs:
        refs = {key(y) for y in truth[s]} - {None}
        if not refs:
            continue
        pk, seen = key(s), {}
        for i, c in enumerate(graw[s]):
            k = key(c["smiles"])
            if k is None or k == pk or k in seen:
                continue
            seen[k] = {"smiles": c["smiles"], "g_rank": i, "m_rank": None,
                       "m_score": 0.0, "g_filter": float(c.get("filter") or 0.0)}
        for i, row in enumerate(mscored[s]):
            k = key(row[0])
            if k is None or k == pk:
                continue
            if k in seen:
                if seen[k]["m_rank"] is None:
                    seen[k].update(m_rank=i, m_score=float(row[1]))
            else:
                seen[k] = {"smiles": row[0], "g_rank": None, "m_rank": i,
                           "m_score": float(row[1]), "g_filter": None}
        per.append({"sub": s, "refs": refs, "cand": seen})

    filt = load_filter(ROOT / args.filter_ckpt)
    total = sum(len(r["cand"]) for r in per)
    print(f"scoring {total} substrate-product pairs with the deployed filter", flush=True)
    t0, done = time.time(), 0
    for r in per:
        keys = list(r["cand"])
        smis = [r["cand"][k]["smiles"] for k in keys]
        with torch.no_grad():
            scores = filt.score_batch(r["sub"], smis) if smis else []
        for k, sc in zip(keys, list(scores)):
            r["cand"][k]["pair_score"] = float(sc)
        done += len(keys)
        if done % 2000 < len(keys):
            print(f"  {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    # precision of the model's own top choice, by who proposed the candidate: the check that the
    # score means the same thing on candidates the model's own generator never produced
    band = {}
    for r in per:
        for k, c in r["cand"].items():
            tag = ("both" if c["g_rank"] is not None and c["m_rank"] is not None
                   else ("grail" if c["g_rank"] is not None else "metatox"))
            b = band.setdefault(tag, {"n": 0, "hit": 0, "score_hit": 0.0, "score_miss": 0.0})
            b["n"] += 1
            if k in r["refs"]:
                b["hit"] += 1
                b["score_hit"] += c["pair_score"]
            else:
                b["score_miss"] += c["pair_score"]
    print("\n  mean filter score on references against non-references, by who proposed the candidate")
    for tag, b in band.items():
        mh = b["score_hit"] / max(b["hit"], 1)
        mm = b["score_miss"] / max(b["n"] - b["hit"], 1)
        print(f"    {tag:8} n={b['n']:>5}  references {mh:.4f}  others {mm:.4f}  "
              f"separation {mh - mm:+.4f}")

    POLICIES = ("metatox", "pair_chemistry", "pair_then_agreement")

    def order(c: dict, policy: str):
        if policy == "metatox":
            return (0, c["m_rank"]) if c["m_rank"] is not None else (1, c["g_rank"] or 0)
        if policy == "pair_chemistry":
            return (-c["pair_score"],)
        if policy == "pair_then_agreement":
            both = c["g_rank"] is not None and c["m_rank"] is not None
            return (0 if both else 1, -c["pair_score"])
        raise SystemExit(policy)

    U = np.array([len(r["refs"]) for r in per], dtype=float)
    hits = {p: {b: np.zeros(len(per)) for b in BUDGETS} for p in POLICIES}
    oracle = {b: np.zeros(len(per)) for b in BUDGETS}
    for j, r in enumerate(per):
        for p in POLICIES:
            ranked = sorted(r["cand"], key=lambda k: order(r["cand"][k], p))
            for b in BUDGETS:
                hits[p][b][j] = len(r["refs"] & set(ranked[:b]))
        present = sum(1 for k in r["cand"] if k in r["refs"])
        for b in BUDGETS:
            oracle[b][j] = min(present, b)

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

    table = {p: {b: rate(hits[p][b]) for b in BUDGETS} for p in POLICIES}
    table["oracle"] = {b: rate(oracle[b]) for b in BUDGETS}
    print(f"\n  {'policy':22}" + "".join(f"{('k=' + str(b)):>9}" for b in BUDGETS))
    for p in list(POLICIES) + ["oracle"]:
        print(f"  {p:22}" + "".join(f"{table[p][b]:>9}" for b in BUDGETS))

    gains = {p: {b: paired(hits[p][b], hits["metatox"][b]) for b in BUDGETS}
             for p in POLICIES if p != "metatox"}
    print(f"\n  against MetaTox alone, paired over substrates")
    for p, g in gains.items():
        print(f"  {p:22}" + "".join(f"{g[b]['delta']:>+9.4f}" for b in BUDGETS))

    rep = {"config": {**_code_version(), "n_substrates": len(per), "references": int(U.sum()),
                      "pairs_scored": total, "filter_ckpt": args.filter_ckpt,
                      "match": "inchikey_tautomer", "aggregation": "micro, ratio of sums",
                      "n_boot": N_BOOT, "seed": SEED,
                      "note": "the deployed filter, applied unchanged to candidates from a rule "
                              "bank it does not have; nothing is trained here"},
           "score_separation_by_provenance": {
               t: {"candidates": b["n"], "references": b["hit"],
                   "mean_on_references": round(b["score_hit"] / max(b["hit"], 1), 4),
                   "mean_on_others": round(b["score_miss"] / max(b["n"] - b["hit"], 1), 4)}
               for t, b in band.items()},
           "recall": table, "gain_over_metatox": gains}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    KEYS.write_text(json.dumps(cache))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

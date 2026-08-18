#!/usr/bin/env python3
"""Does the matching convention reorder a leaderboard outside metabolism? Seven systems, one split.

This is the run the manuscript names and defers. The deferral asked for "at least two comparable
published models whose raw ranked predictions are public", scored under the same criteria; the
EvalRetro release supplies seven that share a test set (scripts/evalretro_ingest.py recovers which
systems those are, because the eleven published files are on three different test sets and only a
group that shares one is a leaderboard at all).

Nothing is trained, generated or tuned here. The predictions are frozen, the split is the one the
seven agree on, and the only thing that varies is the rule for deciding whether a predicted reactant
set equals the recorded one:

  canonical   canonical SMILES, stereochemistry kept
  nostereo    stereochemistry discarded
  inchikey    InChIKey, which normalises more than SMILES does
  tautomer    tautomer-canonical InChIKey, the metabolite paper's default

Two things are reported, and the second is the claim. The first is each system's top-k accuracy
under each convention, which shows how far the criterion moves a number. The second is the paired
INTERACTION for every pair of systems: whether the gap between two systems is itself different under
two conventions. A leaderboard reorders when the criterion moves two systems by different amounts,
so the interaction is the estimand, and a marginal comparison of two accuracies is not it.

Ranks beyond the tenth are dropped. Top-k for k in {1,3,5,10} is what this field publishes and a
prediction at rank 11 cannot change any of them, so keeping them would only pay tautomer
canonicalisation on molecules that no reported number depends on.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import pathlib
import sys
from pathlib import Path

import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from grail_metabolism.metrics import _tautomer_inchikey

RDLogger.DisableLog("rdApp.*")
MODES = ("canonical", "nostereo", "inchikey", "tautomer")
KS = (1, 3, 5, 10)
N_BOOT, SEED = 10000, 0


def _paired_p(d) -> float:
    """The same test ``robust_order._paired_p`` applies, on the same kind of data.

    A bootstrap tail cannot resolve past $1/B$, and Holm compares the $i$-th smallest against
    $\\alpha/(m-i)$. With a family of this size every surviving test sat exactly on that floor, so
    the count was of empty tails rather than of separated effects and moved with $B$ alone. And
    this board scores each reaction $0$ or $1$, so the paired difference is a matched difference of
    proportions whose exact conditional test is the sign test on the discordant pairs; a normal
    approximation errs low exactly in the tail Holm compares against. Both are imported rather than
    restated, so a repair to one is a repair to both.
    """
    from robust_order import _paired_p as _shared  # imported here: robust_order imports this module

    return _shared(d)


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


def _key_one(smiles: str):
    """Every convention's key for one component, computed once and shared across all systems."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return smiles, None
    out = {}
    try:
        out["canonical"] = Chem.MolToSmiles(m)
        n = Chem.Mol(m)
        Chem.RemoveStereochemistry(n)
        out["nostereo"] = Chem.MolToSmiles(n)
        out["inchikey"] = Chem.MolToInchiKey(m)
        out["tautomer"] = _tautomer_inchikey(smiles)
    except Exception:
        return smiles, None
    return smiles, out


def _init():
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")


def build_keys(components: list[str], workers: int) -> dict:
    ctx = mp.get_context("spawn")
    keys = {}
    with ctx.Pool(workers, initializer=_init) as pool:
        for i, (smi, k) in enumerate(pool.imap_unordered(_key_one, components, chunksize=200), 1):
            keys[smi] = k
            if i % 50000 == 0 or i == len(components):
                print(f"    keyed {i}/{len(components)}", flush=True)
    return keys


def set_key(dotjoined: str, mode: str, keys: dict):
    """A reactant set under one convention, or None where any component will not parse."""
    parts = [p for p in (dotjoined or "").split(".") if p]
    if not parts:
        return None
    out = set()
    for p in parts:
        k = keys.get(p)
        if k is None:
            return None
        out.add(k[mode])
    return frozenset(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="cluster0")
    ap.add_argument("--ingest", default="results/evalretro_ingest.json",
                    help="the ingest naming this cluster's systems and test csv")
    ap.add_argument("--dir", default=str(ROOT / "grail_metabolism" / "data" / "evalretro"))
    ap.add_argument("--max-rank", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(ROOT / "results" / f"retro_leaderboard_{args.cluster}.json")

    ingest = json.loads((ROOT / args.ingest).read_text())
    meta = ingest["clusters"][args.cluster]
    systems = meta["systems"]
    rows = list(csv.DictReader(open(ROOT / meta["test_csv"])))
    preds = {n: json.loads((Path(args.dir) / "normalised" / f"{n}.json").read_text())
             for n in systems}
    for n, p in preds.items():
        if len(p) != len(rows):
            raise SystemExit(f"{n}: {len(p)} prediction lists against {len(rows)} reactions")
        bad = sum(1 for a, b in zip(p, rows) if a["product"] != b["PRODUCT"])
        if bad:
            raise SystemExit(f"{n}: {bad} prediction lists are not aligned to the split")
    print(f"{args.cluster}: {len(systems)} systems, {len(rows)} reactions, "
          f"ranks capped at {args.max_rank}", flush=True)

    comps = {c for r in rows for c in r["REACTANT"].split(".") if c}
    for n in systems:
        for p in preds[n]:
            for s in p["preds"][: args.max_rank]:
                comps.update(c for c in s.split(".") if c)
    comps = sorted(comps)
    # Tautomer canonicalisation is the expensive step in this codebase and the components recur
    # across the seven systems, so the keys are computed once and kept. A re-run with a different
    # estimator should cost minutes, not the half hour that would tempt anyone to skip re-running.
    cache_path = Path(args.dir) / f"keys_{args.cluster}_r{args.max_rank}.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [c for c in comps if c not in cached]
    print(f"  {len(comps)} distinct components; {len(cached)} cached, keying {len(todo)} "
          f"on {args.workers} workers", flush=True)
    if todo:
        cached.update(build_keys(todo, args.workers))
        cache_path.write_text(json.dumps(cached))
    keys = cached

    # hit[system][mode][k] is a 0/1 vector over reactions, kept per reaction so a pair of systems
    # can be resampled together -- the interaction is a paired quantity and a marginal is not it
    hit = {n: {m: {k: np.zeros(len(rows), dtype=float) for k in KS} for m in MODES} for n in systems}
    for j, row in enumerate(rows):
        truth = {m: set_key(row["REACTANT"], m, keys) for m in MODES}
        for n in systems:
            pk = {m: [] for m in MODES}
            for s in preds[n][j]["preds"][: args.max_rank]:
                for m in MODES:
                    pk[m].append(set_key(s, m, keys))
            for m in MODES:
                if truth[m] is None:
                    continue
                for k in KS:
                    if any(x is not None and x == truth[m] for x in pk[m][:k]):
                        hit[n][m][k][j] = 1.0
        if (j + 1) % 1000 == 0 or j + 1 == len(rows):
            print(f"    scored {j + 1}/{len(rows)}", flush=True)

    acc = {n: {m: {f"top{k}": round(float(hit[n][m][k].mean()), 4) for k in KS} for m in MODES}
           for n in systems}
    print(f"\n  {'system':16}" + "".join(f"{m[:9]:>11}" for m in MODES) + "   (top-1)")
    for n in systems:
        print(f"  {n:16}" + "".join(f"{acc[n][m]['top1']:>11}" for m in MODES))

    orderings, exchanges = {}, {}
    for m in MODES:
        for k in KS:
            o = tuple(sorted(systems, key=lambda n: -acc[n][m][f"top{k}"]))
            orderings.setdefault(f"top{k}", {})[m] = list(o)
    for k in KS:
        per = orderings[f"top{k}"]
        pairs = set()
        for m1, m2 in itertools.combinations(MODES, 2):
            for a, b in itertools.combinations(systems, 2):
                s1 = acc[a][m1][f"top{k}"] - acc[b][m1][f"top{k}"]
                s2 = acc[a][m2][f"top{k}"] - acc[b][m2][f"top{k}"]
                if s1 * s2 < 0:
                    pairs.add((a, b))
        exchanges[f"top{k}"] = sorted(f"{a} vs {b}" for a, b in pairs)
        print(f"  top{k}: {len({tuple(v) for v in per.values()})} distinct orderings across "
              f"{len(MODES)} conventions, {len(pairs)} pairs exchange", flush=True)

    # The claim: is a pair's gap different under two conventions? Paired over reactions, which is
    # the estimator the metabolite result uses for exactly this contrast.
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))
    interactions, tested = {}, {}
    for k in KS:
        for a, b in itertools.combinations(systems, 2):
            for m1, m2 in itertools.combinations(MODES, 2):
                d = (hit[a][m1][k] - hit[b][m1][k]) - (hit[a][m2][k] - hit[b][m2][k])
                if not d.any():
                    continue
                bt = d[idx].mean(axis=1)
                lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
                # A two-sided p, so the family can be corrected rather than counted: 233 intervals
                # excluding zero out of five hundred tests is not a result. It is NOT the bootstrap
                # tail. That tail is floored at 1/B, and every test the correction used to keep sat
                # exactly on the floor -- so the count was the number of empty tails, identical
                # whatever their true p, and at B=8{,}000 the floor would have exceeded Holm's first
                # threshold and left nothing. The floor is a property of the resample count and was
                # deciding a published number. This is the analytic p that robust_order uses.
                pv = _paired_p(d)
                tested[f"top{k}|{a} vs {b}|{m1} vs {m2}"] = {
                    "delta": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "p": pv}
                if lo > 0 or hi < 0:
                    interactions[f"top{k}|{a} vs {b}|{m1} vs {m2}"] = tested[
                        f"top{k}|{a} vs {b}|{m1} vs {m2}"]

    # Holm over the whole family of interaction tests, which is the correction the paper applies
    # wherever an interval is load-bearing rather than exploratory.
    order = sorted(tested.items(), key=lambda kv: kv[1]["p"])
    n_tests, holm, running = len(order), {}, 0.0
    for i, (name, v) in enumerate(order):
        thresh = 0.05 / (n_tests - i)
        if v["p"] > thresh:
            break
        running = max(running, v["p"])
        holm[name] = {**v, "holm_threshold": round(thresh, 6)}
    print(f"\n  interaction tests run: {n_tests}", flush=True)
    print(f"  intervals excluding zero, uncorrected: {len(interactions)}", flush=True)
    print(f"  surviving Holm at 0.05 across the family: {len(holm)}", flush=True)
    for name, v in list(interactions.items())[:8]:
        print(f"    {name:58} {v['delta']:+.4f} {v['ci95']}")

    rep = {"config": {**_code_version(), "cluster": args.cluster, "systems": systems,
                      "n_reactions": len(rows), "max_rank": args.max_rank,
                      "modes": list(MODES), "ks": list(KS), "n_boot": N_BOOT, "seed": SEED,
                      "source": ingest["config"]["source"],
                      "estimator": "paired bootstrap over reactions; the interaction is the claim"},
           "accuracy": acc, "orderings": orderings, "pairs_that_exchange": exchanges,
           "n_interaction_tests": n_tests,
           "certified_interactions": interactions,
           "holm_survivors": holm}
    Path(out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

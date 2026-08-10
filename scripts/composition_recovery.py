#!/usr/bin/env python3
"""Does a one-step predictor already contain the references a one-step evaluation says it missed?

Metabolism composes. If a method predicts that A becomes B, and separately that B becomes C, then
it has predicted a two-step route from A to C without ever emitting C for A. Every evaluation in
this field scores the method on what it emits for A, so a reference reachable only by composing the
method's own two predictions is counted as a miss.

This measures how large that is, on frozen predictions and with nothing retrained:

    missed              references of A the method does not emit for A
    recovered           of those, the ones it emits for some B it DOES emit for A
    the composition     restricted to intermediates B that are themselves substrates in this split,
                        because only for those does every method have a prediction to compose with

Composing enlarges the emitted set, and a larger set matches more for no good reason, so the
recovery is reported against two controls that enlarge it by the same amount: composing through
randomly drawn substrates instead of predicted ones, and the precision cost, expressed as how many
candidates the composition adds per reference it recovers. A recovery that a random intermediate
reproduces is set size and not chemistry.

The quantity is a property of the method, not of the corpus: two methods with the same recall can
differ in it, and the one that differs more is the one whose one-step output is more nearly closed
under its own dynamics.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import RDLogger

from reference_closure import Keyer, load_predictions

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED, K = 10000, 0, 15


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
    ap.add_argument("--out", default=str(ROOT / "results" / "composition_recovery.json"))
    args = ap.parse_args()

    key = Keyer()
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    preds = load_predictions()
    methods = sorted(preds)
    subs = [s for s in truth if truth[s] and all(s in preds[m] for m in methods)]

    sub_key = {s: key(s) for s in subs}
    by_key = {k: s for s, k in sub_key.items() if k}
    ref_keys = {s: {k for k in (key(y) for y in truth[s]) if k} for s in subs}
    emitted = {m: {s: {k for k in (key(x) for x in preds[m][s][:K]) if k} for s in subs}
               for m in methods}
    key.flush()

    rng = np.random.default_rng(SEED)
    rep = {"config": {**_code_version(), "k": K, "n_boot": N_BOOT, "seed": SEED,
                      "n_substrates": len(subs), "methods": methods,
                      "match": "inchikey_tautomer",
                      "restriction": "intermediates are limited to substrates of this split, the "
                                     "only molecules every method has a prediction for"},
           "per_method": {}}

    curves = {}
    for m in methods:
        rec, elig, added, gained, ctrl = [], 0, [], [], []
        for s in subs:
            missed = ref_keys[s] - emitted[m][s]
            inter = [by_key[k] for k in emitted[m][s] if k in by_key and by_key[k] != s]
            if inter:
                elig += 1
            comp = set().union(*[emitted[m][b] for b in inter]) if inter else set()
            comp -= emitted[m][s]
            got = len(missed & comp)
            rec.append(got / len(missed) if missed else 0.0)
            added.append(len(comp))
            gained.append(got)
            # control: compose through the same number of substrates drawn at random, so the
            # emitted set grows by a comparable amount without the method having chosen them
            if inter:
                draw = [subs[i] for i in rng.integers(0, len(subs), len(inter))]
                cc = set().union(*[emitted[m][b] for b in draw]) - emitted[m][s]
                ctrl.append(len(missed & cc) / len(missed) if missed else 0.0)
            else:
                ctrl.append(0.0)
        curves[m] = {"recovered": np.array(rec), "control": np.array(ctrl)}
        idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
        r_, c_ = curves[m]["recovered"], curves[m]["control"]
        d = r_ - c_
        bt = d[idx].mean(axis=1)
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        rep["per_method"][m] = {
            "substrates_with_a_predicted_intermediate": elig,
            "share_of_missed_references_recovered": round(float(r_.mean()), 4),
            "same_through_random_intermediates": round(float(c_.mean()), 4),
            "over_the_random_control": round(float(d.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)], "separated": bool(lo * hi > 0),
            "candidates_added_per_substrate": round(float(np.mean(added)), 3),
            "references_recovered_per_substrate": round(float(np.mean(gained)), 4),
            "candidates_added_per_reference_recovered": round(
                float(np.sum(added) / max(np.sum(gained), 1)), 2)}
        v = rep["per_method"][m]
        print(f"  {m:14} eligible {elig:4}  recovered {v['share_of_missed_references_recovered']:.4f}"
              f"  random {v['same_through_random_intermediates']:.4f}"
              f"  delta {v['over_the_random_control']:+.4f} {v['ci95']}"
              f"  cost {v['candidates_added_per_reference_recovered']}/ref", flush=True)

    idx = rng.integers(0, len(subs), (N_BOOT, len(subs)))
    rep["differs_by_method"] = {}
    for a, b in itertools.combinations(methods, 2):
        d = ((curves[a]["recovered"] - curves[a]["control"])
             - (curves[b]["recovered"] - curves[b]["control"]))
        bt = d[idx].mean(axis=1)
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        rep["differs_by_method"][f"{a} vs {b}"] = {
            "delta": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "separated": bool(lo * hi > 0)}
        print(f"    {a} vs {b}: {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")

    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

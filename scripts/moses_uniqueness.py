#!/usr/bin/env python3
"""Does the matching criterion move the ordering of published generative models?

MOSES reports Unique@k as distinct canonical SMILES among a model's generations. Canonical SMILES
does not normalise tautomers, so a model that emits both forms of one compound is credited with
two distinct molecules. Models differ in how much they do this, which is the same mechanism that
reorders the metabolite leaderboard.

Frozen published outputs, re-scored; nothing is retrained.
"""
from __future__ import annotations
import itertools
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
KS = [1000, 10000]
N_BOOT, SEED = 10000, 0
OUT = ROOT / "results" / "moses_uniqueness.json"


def main() -> int:
    samples = json.loads((ROOT / "results" / "moses_samples.json").read_text())
    rep = {"models": sorted(samples), "modes": MODES, "ks": KS,
           "n_generated": {m: len(v) for m, v in samples.items()},
           "uniqueness": {}, "ranking": {}}

    # Uniqueness is a functional of the whole generated list, so it cannot be resampled with
    # replacement: drawing n of n items with replacement collides on its own, and the resulting
    # "interval" measures that collision rather than the model's. Every stored interval this
    # produced sat far below its own point estimate, at the 0.85 a with-replacement draw of
    # 10,000 from 30,000 gives whatever the model does. The valid resampling is WITHOUT
    # replacement, at the size reported, from the generations the model actually emitted -- of
    # which there are three times the reported size, so the draw is honest rather than degenerate.
    codes, per = {}, {}
    for mode in MODES:
        table = json.loads((ROOT / "results" / "moses_keys" / f"{mode}.json").read_text())
        rep["uniqueness"][mode] = {}
        for m, mols in samples.items():
            keys = [table.get(s) for s in mols]
            keys = [k for k in keys if k]
            entry = {}
            for k in KS:
                sub = keys[:k]
                entry[f"unique@{k}"] = round(len(set(sub)) / len(sub), 4) if sub else 0.0
            uniq, inv = np.unique(np.array(keys, dtype=object), return_inverse=True)
            codes[(mode, m)] = inv.astype(np.int64)
            rep["uniqueness"][mode][m] = entry
        per[mode] = None

    n = KS[-1]
    rng = np.random.default_rng(SEED)
    # One set of draws per model, shared across criteria, so a criterion difference within a model
    # is paired on the same generations. Two models generate independently and cannot be paired.
    draws = {m: np.array([rng.permutation(len(samples[m]))[:n] for _ in range(N_BOOT)])
             for m in sorted(samples)}
    boot = {}
    for mode in MODES:
        for m in sorted(samples):
            c = codes[(mode, m)]
            idx = draws[m][:, :min(n, len(c))]
            boot[(mode, m)] = np.array([len(np.unique(c[row])) / idx.shape[1] for row in idx])
            lo, hi = np.quantile(boot[(mode, m)], [.025, .975])
            rep["uniqueness"][mode][m][f"ci95_at_{n}"] = [round(float(lo), 4), round(float(hi), 4)]

    # What the criterion does to one model, paired, and what it does differently to two -- which is
    # the estimand a rank exchange belongs to. The exchange this set admits was reported without
    # one, which is the same defect the population axis carried.
    strict = "canonical"
    rep["criterion_effect"] = {}
    rep["interactions"] = {}
    for mode in MODES:
        if mode == strict:
            continue
        for m in sorted(samples):
            d = boot[(strict, m)] - boot[(mode, m)]
            rep["criterion_effect"].setdefault(mode, {})[m] = {
                "delta": round(float(d.mean()), 5),
                "ci95": [round(float(np.quantile(d, .025)), 5),
                         round(float(np.quantile(d, .975)), 5)]}
        for a, b in itertools.combinations(sorted(samples), 2):
            da = boot[(strict, a)] - boot[(mode, a)]
            db = boot[(strict, b)] - boot[(mode, b)]
            inter = da - db
            lo, hi = float(np.quantile(inter, .025)), float(np.quantile(inter, .975))
            pv = 2.0 * min((inter <= 0).mean(), (inter >= 0).mean())
            rep["interactions"].setdefault(mode, {})[f"{a} vs {b}"] = {
                "delta": round(float(inter.mean()), 5), "ci95": [round(lo, 5), round(hi, 5)],
                "excludes_zero": bool(lo * hi > 0),
                "p": round(max(float(pv), 1.0 / N_BOOT), 6)}


    for mode in MODES:
        order = sorted(samples, key=lambda m: -rep["uniqueness"][mode][m][f"unique@{KS[-1]}"])
        rep["ranking"][mode] = order
        print(f"{mode:20} " + "  ".join(f"{m}:{rep['uniqueness'][mode][m][f'unique@{KS[-1]}']:.3f}"
                                        for m in order), flush=True)

    # Compare EVERY criterion against the base, not just the two endpoints: the criteria are not
    # ordered by tolerance in a way that makes canonical-vs-tautomer an upper bound on movement,
    # and the intermediate stereo-blind rung is exactly where this set moves.
    base = rep["ranking"][MODES[0]]
    rep["models_that_moved_by_mode"] = {
        mode: [m for m in base if base.index(m) != rep["ranking"][mode].index(m)]
        for mode in MODES[1:]
    }
    moved = sorted({m for v in rep["models_that_moved_by_mode"].values() for m in v})
    rep["ordering_changed"] = bool(moved)
    rep["models_that_moved"] = moved
    # Holm across every interaction this family contains, at the threshold the paper applies to
    # its other families, so a third domain is not held to a looser standard than the first two.
    flat = [(f"{mode}|{pair}", v) for mode in rep["interactions"]
            for pair, v in rep["interactions"][mode].items()]
    flat.sort(key=lambda kv: kv[1]["p"])
    survivors = []
    stopped = False
    for i, (name, v) in enumerate(flat):
        if not stopped and v["p"] <= 0.05 / (len(flat) - i):
            v["survives_holm"] = True
            survivors.append(name)
        else:
            stopped = True
            v["survives_holm"] = False
    rep["n_interactions"] = len(flat)
    rep["interactions_excluding_zero"] = sum(1 for _, v in flat if v["excludes_zero"])
    rep["holm_survivors"] = len(survivors)
    print(f"\ninteractions: {len(flat)}, excluding zero "
          f"{rep['interactions_excluding_zero']}, surviving Holm {len(survivors)}", flush=True)

    for mode in MODES:
        print(f"{mode:20} {rep['ranking'][mode]}", flush=True)
    print(f"ordering changed: {rep['ordering_changed']}  (moves under: "
          f"{[k for k, v in rep['models_that_moved_by_mode'].items() if v]})", flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

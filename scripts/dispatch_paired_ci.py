#!/usr/bin/env python3
"""Is the one residual that is not zero distinguishable from zero?

results/hydrogen_dispatch.json reports a residual of $+0.016$ for our own bank: dispatching the
hydrogen convention per template reaches further than the better of the two global settings. Its
marginal interval overlaps that setting's heavily, and a marginal comparison is the wrong test
anyway, so the run that produced it cannot say whether the residual is real.

This runs the two arms in one pass over the same substrates and keeps the per-substrate counts, so
the difference can be resampled as a paired quantity -- the estimator the rest of the paper uses for
a contrast, micro, a difference of ratio-of-sums.

Only our bank is worth the pass. SyGMa dispatches nothing, so its residual is zero by construction,
and BioTransformer's dispatch reproduced its global arm exactly, so there is no difference to test.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem, RDLogger

from _population import POPULATIONS, population_items, load_population, tagged_out
from bank_engine_replication import load_bank
from engine_knobs import DEFAULT, apply_with
from hydrogen_dispatch import MAJORITY_CONVENTION, _apply, classify
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
BANK = "grail_full"
# results/hydrogen_dispatch.json and results/bank_engine_replication.json, both on the
# 245-substrate subsample; this pair gates that population and no other.
COMMITTED = {"dispatch": 0.8148, "implicit": 0.7989}
_CTX: dict = {}


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


def _init(rules, wants):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX["rules"], _CTX["wants"] = rules, wants
    _CTX["want"] = [r for r, w in zip(rules, wants) if w]
    _CTX["rest"] = [r for r, w in zip(rules, wants) if not w]


def _worker(item):
    """All three arms of one rule set, from two passes over it, on one substrate.

    The residual is dispatch minus the BETTER of the two global settings, and which of the two is
    better is not the same for every subset -- implicit wins on the whole bank and expanded wins on
    the curated part -- so both have to be measured rather than one assumed. Splitting the rule set
    into the templates that want the expansion and those that do not, and applying each subset under
    both conventions, gives all three arms by union from exactly two passes:

        all-explicit = E(want) u E(rest)     all-implicit = I(want) u I(rest)
        dispatch     = E(want) u I(rest)
    """
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return sub, 0, 0, 0, 0
    substrates = {True: Chem.AddHs(Chem.Mol(mol)), False: Chem.Mol(mol)}
    want, rest = _CTX["want"], _CTX["rest"]
    e_want = set(_apply(substrates, want, [True] * len(want)))
    e_rest = set(_apply(substrates, rest, [True] * len(rest)))
    i_want = set(_apply(substrates, want, [False] * len(want)))
    i_rest = set(_apply(substrates, rest, [False] * len(rest)))
    denom = None
    hits = {}
    for k, pool in (("dispatch", e_want | i_rest), ("explicit", e_want | e_rest),
                    ("implicit", i_want | i_rest)):
        u, h, _ = _tautomer_recovered(trues, sorted(pool), audit=False)
        assert denom is None or u == denom, "the reference denominator must not depend on the arm"
        denom, hits[k] = u, h
    return sub, int(denom), int(hits["dispatch"]), int(hits["explicit"]), int(hits["implicit"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "dispatch_paired_ci.json"))
    ap.add_argument("--subset", default="full", choices=("full", "curated", "mined"),
                    help="the pre-registered partition, docs/PROVENANCE_DISPATCH_PREREGISTRATION.md")
    ap.add_argument("--population", default="clean_test", choices=POPULATIONS,
                    help="subsample245 reproduces the committed artifact; clean_test is the split")
    args = ap.parse_args()
    args.out = tagged_out(args.out, args.population)
    if args.subset != "full":
        q = pathlib.Path(args.out)
        args.out = str(q.with_name(f"{q.stem}__{args.subset}{q.suffix}"))

    bank = load_bank(BANK)
    # The classifier is imported and run on the WHOLE bank, then restricted. Re-running it on a
    # subset would let the unclassifiable policy follow that subset's majority, which the
    # registration closes: the policy is frozen at the whole bank's convention.
    bank_wants = classify(bank, MAJORITY_CONVENTION[BANK])
    if args.subset == "full":
        keep = [True] * len(bank)
    else:
        mined = {l.strip() for l in open(ROOT / "grail_metabolism/resources/mined_only.txt")
                 if l.strip()}
        keep = [(r in mined) if args.subset == "mined" else (r not in mined) for r in bank]
    rules = [r for r, k in zip(bank, keep) if k]
    wants = [w for w, k in zip(bank_wants, keep) if k]
    items = population_items(args.population)
    print(f"{BANK} [{args.subset}]: {len(rules)} of {len(bank)} rules, {sum(wants)} dispatched, "
          f"{len(items)} substrates", flush=True)

    workers = max(1, (os.cpu_count() or 4) - 2)
    with multiprocessing.get_context("spawn").Pool(workers, _init, (rules, wants)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 2), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(items):
                print(f"  {n}/{len(items)}", flush=True)
    rows.sort(key=lambda r: r[0])

    U = np.array([r[1] for r in rows])
    D = np.array([r[2] for r in rows])
    E = np.array([r[3] for r in rows])
    I = np.array([r[4] for r in rows])

    def reach_of(X):
        return round(float(X.sum() / max(U.sum(), 1)), 4)

    reach_d, reach_e, reach_i = reach_of(D), reach_of(E), reach_of(I)
    # the better global setting is measured, not assumed: implicit wins on the whole bank and
    # expanded wins on the curated part, which is the whole point of the registered prediction
    better, better_name = ((E, "all_explicit") if reach_e >= reach_i else (I, "all_implicit"))
    best_global = max(reach_e, reach_i)
    # The committed values were measured on the 245-substrate subsample, so they gate that
    # population and no other. Asserting them against a run on the full split would be a comparison
    # across populations -- the defect this paper names -- dressed as a reproducibility check.
    # ...and the whole bank only: the committed pair is a whole-bank measurement, so asserting it
    # against a subset arm compares two different rule sets as well as, possibly, two populations.
    if args.population == "subsample245" and args.subset == "full":
        print(f"\ngate: dispatch {reach_d} against committed {COMMITTED['dispatch']}, "
              f"implicit {reach_i} against {COMMITTED['implicit']}")
        for got, want, name in ((reach_d, COMMITTED["dispatch"], "dispatch"),
                                (reach_i, COMMITTED["implicit"], "implicit")):
            if abs(got - want) > 1e-4:
                raise SystemExit(f"the {name} arm does not reproduce its committed reach")
    else:
        print(f"\narms measured here: dispatch {reach_d}, explicit {reach_e}, implicit {reach_i} "
              f"(the committed pair gates the 245-substrate subsample only)")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))
    diff = D - better
    micro = float(diff.sum() / max(U.sum(), 1))
    bt = np.array([diff[j].sum() / max(U[j].sum(), 1) for j in idx])
    lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
    macro = (diff / np.maximum(U, 1))
    abt = macro[idx].mean(axis=1)
    alo, ahi = float(np.quantile(abt, .025)), float(np.quantile(abt, .975))

    rep = {"config": {**_code_version(), "population": args.population, "bank": BANK, "n_rules": len(rules),
                      "dispatched": int(sum(wants)), "n_substrates": len(rows),
                      "match": "inchikey_tautomer", "n_boot": N_BOOT, "seed": SEED,
                      "subset": args.subset,
                      "policy": "docs/DISPATCH_PREREGISTRATION.md and "
                                "docs/PROVENANCE_DISPATCH_PREREGISTRATION.md",
                      "residual_is_against": better_name,
                      "gate": "on subsample245 both arms reproduce their committed reach"},
           "references": int(U.sum()),
           "recovered": {"dispatch": int(D.sum()), "all_explicit": int(E.sum()),
                         "all_implicit": int(I.sum())},
           "reach": {"dispatch": reach_d, "all_explicit": reach_e, "all_implicit": reach_i,
                     "best_global": best_global},
           "paired_residual": {"delta": round(micro, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "excludes_zero": bool(lo > 0 or hi < 0), "estimator": "micro",
                               "macro": {"delta": round(float(macro.mean()), 4),
                                         "ci95": [round(alo, 4), round(ahi, 4)],
                                         "excludes_zero": bool(alo > 0 or ahi < 0)}}}
    pr = rep["paired_residual"]
    print(f"\nreferences {int(U.sum())}: dispatch recovers {int(D.sum())}, implicit {int(I.sum())}")
    print(f"paired residual {pr['delta']:+.4f} {pr['ci95']} micro "
          f"({'certified' if pr['excludes_zero'] else 'spans zero'}), "
          f"{pr['macro']['delta']:+.4f} {pr['macro']['ci95']} macro")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

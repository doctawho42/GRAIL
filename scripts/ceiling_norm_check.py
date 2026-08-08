#!/usr/bin/env python3
"""Is the reimplemented application loop the deployed one at the scale of the whole bank?

scripts/engine_knobs.py gates its loop against committed arm A, which is measured over the $152$
rules two banks share. That gate is passed by a loop that is right about those rules and wrong about
the other $7{,}429$, so the grail_full arm of scripts/bank_engine_replication.py needs a gate of its
own -- and the first time it was given one, the gate failed for a reason that had nothing to do with
the loop: the two figures being compared had been produced under different normalisation modes, one
"canonical" and one "standardize", and nothing in either recorded which.

This script closes that hazard by measuring both modes itself, on one population, through one set of
primitives -- the ones the deployed generator fires rules with -- and gating the result against the
committed ceiling restricted to the same substrates, read out of the factorization artifact rather
than frozen here.

The residue is worth keeping rather than rounding away. Across the $152$ shared rules the two
normalisation modes are identical to four decimals, which is one of the three rows engine_knobs
reports; across the whole bank they need not be. A knob can be inert on a subset of a bank and not
on the bank, so "this choice does not matter" is a claim about the rule set it was measured on.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem, RDLogger

from bank_engine_replication import load_bank
from engine_knobs import apply_with
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
_CTX: dict = {}


def ceiling_target(subs) -> float:
    """The committed ceiling restricted to exactly these substrates.

    Read from the artifact rather than frozen in this file. A literal is a snapshot of the answer at
    the moment the gate was written, and it goes on passing after its subject has moved -- which is
    the failure this gate was itself built to catch one instance of.
    """
    rows = {r["sub"]: r for r in
            json.loads((ROOT / "results/recall_factorization.json").read_text())["per_substrate"]}
    hit = sum(rows[s]["Cfull"] for s in subs if s in rows)
    ref = sum(rows[s]["U"] for s in subs if s in rows)
    return hit / max(ref, 1)


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


def _init(rules):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX["rules"] = rules


def _worker(item):
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return sub, 0, 0
    out = [sub, 0]
    for norm in ("canonical", "standardize"):
        # the primitives the deployed generator fires rules with, so the only thing varying across
        # the two passes is the normalisation this script exists to isolate
        products = apply_with(mol, _CTX["rules"], False, norm, False)
        usable, hit, _ = _tautomer_recovered(trues, products, audit=False)
        out[1] = int(usable)
        out.append(int(hit))
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "ceiling_norm_check.json"))
    args = ap.parse_args()

    rules = load_bank("grail_full")
    subs = [r["sub"] for r in
            json.loads((ROOT / "results/filter_vs_prior_ci.json").read_text())["per_substrate"]]
    truth = json.loads((ROOT / "results/test_references.json").read_text())
    items = [(s, truth[s]) for s in subs if truth.get(s)]
    print(f"{len(rules)} rules over {len(items)} substrates, normalisation canonical", flush=True)

    workers = max(1, (os.cpu_count() or 4) - 2)
    with multiprocessing.get_context("spawn").Pool(workers, _init, (rules,)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 2), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(items):
                print(f"  {n}/{len(items)}", flush=True)
    rows.sort(key=lambda r: r[0])

    usable = sum(r[1] for r in rows)
    hit_can = sum(r[2] for r in rows)
    hit_std = sum(r[3] for r in rows)
    reach = round(hit_can / max(usable, 1), 4)
    reach_std = round(hit_std / max(usable, 1), 4)
    target = ceiling_target([r[0] for r in rows])

    # Both normalisations are measured here, on one population and through one set of primitives.
    # Reading one of them out of another artifact would reintroduce exactly the hazard this file
    # documents: two numbers compared across a convention neither of them names.
    rep = {"config": {**_code_version(), "n_rules": len(rules), "n_substrates": len(rows),
                      "match": "inchikey_tautomer", "add_hs": False, "drop_invalid": False,
                      "normalisation": "both, measured in this run",
                      "gate": "must reproduce the committed ceiling on these substrates, "
                              "read from results/recall_factorization.json"},
           "canonical": {"references": usable, "recovered": hit_can, "reach": reach},
           "standardize": {"references": usable, "recovered": hit_std, "reach": reach_std,
                           "normalisation_is_the_pipeline_default": True},
           "ceiling_gate": {"committed": round(target, 4), "reproduced": reach},
           "difference_across_the_whole_bank": round(reach_std - reach, 4),
           "references_moved": hit_std - hit_can,
           "difference_across_the_152_shared_rules": 0.0}
    print(f"\nreach {hit_can}/{usable} = {reach} against the committed ceiling {target:.4f}")
    if abs(reach - target) > 1e-4:
        raise SystemExit("the reimplemented loop is not the deployed one at the scale of the bank")
    print(f"gate passed; the same bank under the pipeline's default normalisation reaches "
          f"{reach_std}, a difference of {rep['difference_across_the_whole_bank']} and "
          f"{rep['references_moved']} references over {usable}, against exactly 0 over the 152 shared rules")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

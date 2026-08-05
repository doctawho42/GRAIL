#!/usr/bin/env python3
"""Is the reimplemented application loop the deployed one at the scale of the whole bank?

scripts/engine_knobs.py gates its loop against committed arm A, which is measured over the $152$
rules two banks share. That gate is passed by a loop that is right about those rules and wrong about
the other $7{,}429$, so the grail_full arm of scripts/bank_engine_replication.py needs a gate of its
own, and when it was first run it failed one: it returned $0.7302$ where the committed ceiling for
the same substrates is $0.7284$.

The failure was the gate's, not the loop's. The committed ceiling comes from
scripts/ceiling_by_provenance.py, which passes normalization_mode="canonical", while the replication
arm ran under "standardize" -- the default of apply_rules_to_molecule itself. Two different
configurations were being compared. This script closes that by running the full bank under the
configuration the committed figure was actually measured in, and it must reproduce it exactly.

The residue is worth keeping rather than rounding away. Across the $152$ shared rules the two
normalisation modes are identical to four decimals, which is one of the three rows engine_knobs
reports; across all $7{,}581$ they differ by a single recovered reference. A knob can be inert on a
subset of a bank and not on the bank, so "this choice does not matter" is a claim about the rule set
it was measured on.
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
# results/ceiling_by_provenance.json -> subsets.full.coverage, measured under normalization_mode
# "canonical" on these same 245 substrates
COMMITTED_CEILING = 0.7284
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
    products = apply_with(mol, _CTX["rules"], True, "canonical", True)
    usable, hit, _ = _tautomer_recovered(trues, products, audit=False)
    return sub, int(usable), int(hit)


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
    hit = sum(r[2] for r in rows)
    reach = round(hit / max(usable, 1), 4)

    # the same bank and substrates under the other normalisation, as already committed
    replication = json.loads((ROOT / "results/bank_engine_replication.json").read_text())
    standardize = replication["banks"]["grail_full"]["deployed"]
    rep = {"config": {**_code_version(), "n_rules": len(rules), "n_substrates": len(rows),
                      "match": "inchikey_tautomer", "add_hs": True, "drop_invalid": True,
                      "normalisation": "canonical",
                      "gate": f"must reproduce the committed ceiling {COMMITTED_CEILING}"},
           "canonical": {"references": usable, "recovered": hit, "reach": reach},
           "standardize": {"reach": standardize["reach"],
                           "source": "results/bank_engine_replication.json banks.grail_full.deployed",
                           "normalisation_is_the_pipeline_default": True},
           "difference_across_the_whole_bank": round(standardize["reach"] - reach, 4),
           "difference_across_the_152_shared_rules": 0.0}
    print(f"\nreach {hit}/{usable} = {reach} against the committed ceiling {COMMITTED_CEILING}")
    if abs(reach - COMMITTED_CEILING) > 1e-4:
        raise SystemExit("the reimplemented loop is not the deployed one at the scale of the bank")
    print(f"gate passed; the same bank under the pipeline's default normalisation reaches "
          f"{standardize['reach']}, a difference of {rep['difference_across_the_whole_bank']} "
          f"over {usable} references, against exactly 0 over the 152 shared rules")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pruning probe: split the bank into ACTIVE / SALVAGEABLE-NARROW / DEAD-WEIGHT.

Correction to an earlier overclaim: "64% of rules unfired => sparsity is irreducible" glued together
two different populations. Merging targets REDUNDANCY (duplicates). But the 64% unfired is a mix of
(i) overfit singletons -- mined narrow templates that fire only on their source-like chemistry and are
never a true label anywhere = noise in the 7,581-way target, safely PRUNABLE (a separate, untested
lever on the very P2 sparsity I called irreducible); and (ii) narrow-but-useful rules that DO explain a
true metabolite = the D1 generalization/coverage population (generalize, don't prune). One population,
two answers: generalize the salvageable, prune the hopeless.

Measurement (distribution-honest, no source-provenance needed):
  breadth  = # of a large random probe pool a rule fires on            (from rule_collapse_cache.json)
  useful   = # annotated test substrates where the rule produces a TRUE metabolite (canonical-SMILES
             match; tautomer would only ADD a few useful rules => this is a lower bound on 'useful',
             i.e. an UPPER bound on 'dead')
Classes:
  ACTIVE            : useful>=1 OR breadth-heavy                 -> keep
  SALVAGEABLE-NARROW: useful>=1 but breadth<=1                   -> generalize (coverage lever)
  DEAD-WEIGHT       : useful==0 AND breadth<=1                   -> prunable noise (selection lever)
Caveat: 'useful' is measured on the annotated TEST set only; a train-inclusive pass would move some
DEAD -> SALVAGEABLE. So DEAD here is an upper bound. Reported as such.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem
from rdkit import RDLogger

from grail_metabolism.utils.preparation import apply_rules_to_molecule, load_default_rules

RDLogger.DisableLog("rdApp.*")
GRAIL_CSV = ROOT / "artifacts" / "full5000_single" / "predictions" / "test_predictions.csv"
POOL_CACHE = ROOT / "results" / "rule_collapse_cache.json"
OUT = ROOT / "results" / "rule_prune_probe.json"


def _canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def main() -> int:
    rules = load_default_rules()
    n_total = len(rules)

    # annotated test substrates + their TRUE metabolites (canonical SMILES sets)
    subs = {}
    with open(GRAIL_CSV) as fh:
        for row in csv.DictReader(fh):
            reals = {c for c in (_canon(r) for r in row.get("real", "").split("|") if r) if c}
            if reals:
                subs[row["substrate"]] = reals
    print(f"annotated test substrates: {len(subs)}  rules: {n_total}", flush=True)

    useful = [0] * n_total   # # test substrates where rule makes a TRUE metabolite
    fire_t = [0] * n_total   # # test substrates where rule fires at all
    t0 = time.time()
    for i, (sub, true_can) in enumerate(subs.items(), 1):
        if i % 20 == 0 or i == len(subs):
            print(f"  {i}/{len(subs)} ({time.time()-t0:.0f}s)", flush=True)
        mol = Chem.MolFromSmiles(sub)
        if mol is None:
            continue
        generated = apply_rules_to_molecule(mol, rules, "canonical")  # {canon_product: {rule_idx}}
        for product, idxs in generated.items():
            is_true = product in true_can
            for r in idxs:
                fire_t[r] += 1
                if is_true:
                    useful[r] += 1

    # breadth on the 1000-random probe pool (cached signatures -> distinct substrates fired on)
    breadth = [0] * n_total
    if POOL_CACHE.exists():
        blob = json.loads(POOL_CACHE.read_text())
        for r, pairs in blob["sig"].items():
            breadth[int(r)] = len({p[0] for p in pairs})

    # classify
    active = salvage = dead = fires_never_useful = 0
    dead_idx = []
    for r in range(n_total):
        u, ft, br = useful[r], fire_t[r], breadth[r]
        if u >= 1:
            if br <= 1:
                salvage += 1
            else:
                active += 1
        else:
            if br <= 1 and ft <= 1:
                dead += 1
                dead_idx.append(r)
            else:
                fires_never_useful += 1

    unfired_random = sum(1 for r in range(n_total) if breadth[r] == 0)
    unfired_and_useful = sum(1 for r in range(n_total) if breadth[r] == 0 and useful[r] >= 1)
    unfired_and_dead = sum(1 for r in range(n_total) if breadth[r] == 0 and useful[r] == 0 and fire_t[r] <= 1)

    report = {
        "n_rules_total": n_total,
        "annotated_test_substrates": len(subs),
        "match": "canonical_smiles (upper bound on DEAD; tautomer would shrink DEAD)",
        "classes": {
            "ACTIVE_useful_or_broad": active,
            "SALVAGEABLE_narrow_but_useful": salvage,
            "FIRES_never_useful": fires_never_useful,
            "DEAD_weight_prunable": dead,
        },
        "useful_rules_total": sum(1 for r in range(n_total) if useful[r] >= 1),
        "unfired_on_random_pool": unfired_random,
        "of_unfired__salvageable_useful_on_test": unfired_and_useful,
        "of_unfired__dead_never_useful": unfired_and_dead,
        "note": "DEAD = never a true metabolite on the annotated test set AND fires on <=1 of ~1000 random "
                "probes + <=1 test substrate. Upper bound (train-inclusive usefulness would reduce it). "
                "Pruning DEAD is reachability-preserving for the annotated data by construction.",
    }
    OUT.write_text(json.dumps(report, indent=2))

    print("\n=== PRUNING PROBE: bank composition ===", flush=True)
    print(f"total rules                    : {n_total}", flush=True)
    print(f"  ACTIVE (useful or broad)     : {active}", flush=True)
    print(f"  SALVAGEABLE (narrow, useful) : {salvage}   <- generalize (coverage lever)", flush=True)
    print(f"  FIRES but never useful       : {fires_never_useful}", flush=True)
    print(f"  DEAD-WEIGHT (prunable, UB)   : {dead}   <- prune (selection lever)", flush=True)
    print(f"useful rules total             : {report['useful_rules_total']}", flush=True)
    print(f"of {unfired_random} unfired-on-random: {unfired_and_useful} salvageable-useful, "
          f"{unfired_and_dead} dead-never-useful", flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

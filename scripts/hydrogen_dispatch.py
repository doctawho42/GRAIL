#!/usr/bin/env python3
"""Can a rule bank's reach be made a property of the bank rather than of a (bank, convention) pair?

results/bank_engine_replication.json shows that one global preprocessing choice moves a bank's reach
by up to a third, and in opposite directions for banks written in opposite conventions. That makes a
published coverage figure the value of a pair. The repair, if there is one, is not to declare the
convention but to stop having one: dispatch it per template, so each rule is applied under the
convention it was written in.

The policy is the one pre-registered in docs/DISPATCH_PREREGISTRATION.md and is not chosen here.
A template gets the expanded substrate iff its reactant side carries a bracket-hydrogen ATOM, by the
same syntactic test the mechanism script uses; recursive SMARTS, which that test cannot see inside,
take their bank's majority convention. Neither branch consults an outcome, which is what separates
this from an oracle that would beat any global setting by construction.

What the run reports is the residual, not the win:

    residual = reach(dispatch) - max(reach(all explicit), reach(all implicit))

Zero says the bank is single-convention and choosing the right global setting is the whole story;
above zero says no global setting can express the bank and its reach is only well defined once the
convention travels with the template. SyGMa is the null -- none of its 175 templates carries a
hydrogen atom, so dispatch must reduce to the identity and reproduce its implicit arm exactly.
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
from engine_knobs import DEFAULT
from explicit_h_mechanism import _ATOM_TOKEN, needs_explicit_hydrogen
from grail_metabolism.utils.preparation import (
    _clean_product_smiles, _iter_reaction_products, _normalize_smiles_cached)
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
# docs/DISPATCH_PREREGISTRATION.md: recursive SMARTS take their bank's majority convention, fixed
# before the run and never set from an outcome.
MAJORITY_CONVENTION = {"grail_full": False, "sygma_175": False, "biotransformer": True}
# results/bank_engine_replication.json, the two global arms this has to beat to mean anything
COMMITTED = {"sygma_175": {True: 0.2205, False: 0.5273},
             "grail_full": {True: 0.7302, False: 0.7989},
             "biotransformer": {True: 0.4727, False: 0.1305}}
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


def classify(rules, majority: bool) -> list[bool]:
    """True where a template needs the substrate expanded, by the pre-registered syntactic test."""
    out = []
    for rule in rules:
        reactants = rule.split(">>")[0]
        tokens = _ATOM_TOKEN.findall(reactants)
        if any(needs_explicit_hydrogen(t) for t in tokens):
            out.append(True)
        elif "$(" in reactants:
            out.append(majority)            # unclassifiable: the bank's convention, frozen
        else:
            out.append(False)
    return out


def _init(rules, wants):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX["rules"], _CTX["wants"] = rules, wants


def _apply(substrates, rules, wants) -> list[str]:
    """The deployed loop with the expansion decided per template rather than per run."""
    seen = set()
    for rule, want in zip(rules, wants):
        for product in _iter_reaction_products(substrates[want], rule):
            try:
                smiles = Chem.MolToSmiles(product)
            except Exception:
                continue
            for fragment in _clean_product_smiles(smiles):
                try:
                    seen.add(_normalize_smiles_cached(fragment, DEFAULT["norm"]))
                except Exception:
                    continue
    return [s for s in seen if s]


def _worker(item):
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return sub, 0, 0
    # both copies once per molecule, not once per rule
    substrates = {True: Chem.AddHs(Chem.Mol(mol)), False: Chem.Mol(mol)}
    products = _apply(substrates, _CTX["rules"], _CTX["wants"])
    usable, hit, _ = _tautomer_recovered(trues, products, audit=False)
    return sub, int(usable), int(hit)


def run_bank(name: str, items, workers: int) -> dict:
    rules = load_bank(name)
    wants = classify(rules, MAJORITY_CONVENTION[name])
    n_exp = sum(wants)
    print(f"\n{name}: {len(rules)} rules, {n_exp} dispatched to the expanded substrate", flush=True)

    with multiprocessing.get_context("spawn").Pool(workers, _init, (rules, wants)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 2), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(items):
                print(f"  {name} {n}/{len(items)}", flush=True)
    rows.sort(key=lambda r: r[0])

    U = np.array([r[1] for r in rows])
    H = np.array([r[2] for r in rows])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))
    bt = np.array([H[j].sum() / max(U[j].sum(), 1) for j in idx])
    reach = round(float(H.sum() / max(U.sum(), 1)), 4)

    globals_ = COMMITTED[name]
    best_global = max(globals_.values())
    out = {"n_rules": len(rules), "dispatched_to_expanded": n_exp,
           "references": int(U.sum()), "recovered": int(H.sum()), "reach": reach,
           "ci95": [round(float(np.quantile(bt, .025)), 4),
                    round(float(np.quantile(bt, .975)), 4)],
           "global_arms": {"all_explicit": globals_[True], "all_implicit": globals_[False]},
           "best_global": best_global,
           "residual_convention_dependence": round(reach - best_global, 4)}
    print(f"  {name}: dispatch {reach}, best global {best_global}, "
          f"residual {out['residual_convention_dependence']:+.4f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", default=["sygma_175", "biotransformer", "grail_full"])
    ap.add_argument("--out", default=str(ROOT / "results" / "hydrogen_dispatch.json"))
    ap.add_argument("--population", default="clean_test", choices=POPULATIONS,
                    help="subsample245 reproduces the committed artifact; clean_test is the split")
    args = ap.parse_args()
    args.out = tagged_out(args.out, args.population)

    items = population_items(args.population)
    workers = max(1, (os.cpu_count() or 4) - 2)

    out_path = Path(args.out)
    banks = json.loads(out_path.read_text()).get("banks", {}) if out_path.exists() else {}
    for name in args.banks:
        banks[name] = run_bank(name, items, workers)
        out_path.write_text(json.dumps(
            {"config": {**_code_version(), "population": args.population, "n_substrates": len(items),
                        "match": "inchikey_tautomer", "n_boot": N_BOOT, "seed": SEED,
                        "policy": "docs/DISPATCH_PREREGISTRATION.md",
                        "recursive_smarts_convention": MAJORITY_CONVENTION,
                        "banks_this_invocation": list(args.banks)},
             "banks": banks}, indent=1))

    # the pre-registered null: SyGMa has no hydrogen-atom template, so dispatch is the identity
    if "sygma_175" in banks:
        got, want = banks["sygma_175"]["reach"], COMMITTED["sygma_175"][False]
        print(f"\nnull: SyGMa dispatch {got} against its implicit arm {want}")
        if banks["sygma_175"]["dispatched_to_expanded"] != 0 or abs(got - want) > 1e-4:
            raise SystemExit("the classifier sends SyGMa templates somewhere it should not; "
                             "nothing else in this run is interpretable")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

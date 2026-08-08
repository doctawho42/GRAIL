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
# results/bank_engine_replication.json, on the 245-substrate subsample. Kept ONLY as a
# reproducibility gate for that population; the arms a run reports are measured in the run.
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
    _CTX["want"] = [r for r, w in zip(rules, wants) if w]
    _CTX["rest"] = [r for r, w in zip(rules, wants) if not w]


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
    """All three arms from two passes over the bank, on one substrate and one population.

    The dispatch arm is only meaningful against the two global arms it is supposed to beat, so the
    three have to be the same measurement with one thing varied. Reading a global arm out of another
    artifact makes the residual a difference between populations, which is the defect this paper is
    about; it is also unnecessary. Splitting the bank into the templates that want the expansion and
    those that do not, and applying each subset under both conventions, costs exactly two passes
    over the bank and yields all three arms by union:

        all-explicit = E(want) u E(rest)      all-implicit = I(want) u I(rest)
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
    arms = {"dispatch": e_want | i_rest,
            "explicit": e_want | e_rest,
            "implicit": i_want | i_rest}
    usable, hits = 0, {}
    for k, pool in arms.items():
        usable, hits[k], _ = _tautomer_recovered(trues, sorted(pool), audit=False)
    return sub, int(usable), int(hits["dispatch"]), int(hits["explicit"]), int(hits["implicit"])


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
    E = np.array([r[3] for r in rows])
    I = np.array([r[4] for r in rows])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))

    def reach_of(X):
        return round(float(X.sum() / max(U.sum(), 1)), 4)

    bt = np.array([H[j].sum() / max(U[j].sum(), 1) for j in idx])
    reach = reach_of(H)
    arms = {"all_explicit": reach_of(E), "all_implicit": reach_of(I)}
    best_global = max(arms.values())
    # the residual is paired: the same substrates carry both arms in the same run, so the
    # difference can be resampled rather than compared across two marginal intervals
    better = E if arms["all_explicit"] >= arms["all_implicit"] else I
    d = H - better
    bt_d = np.array([d[j].sum() / max(U[j].sum(), 1) for j in idx])
    out = {"n_rules": len(rules), "dispatched_to_expanded": n_exp,
           "references": int(U.sum()), "recovered": int(H.sum()), "reach": reach,
           "ci95": [round(float(np.quantile(bt, .025)), 4),
                    round(float(np.quantile(bt, .975)), 4)],
           "global_arms": arms, "best_global": best_global,
           "residual_convention_dependence": round(reach - best_global, 4),
           "residual_ci95": [round(float(np.quantile(bt_d, .025)), 4),
                             round(float(np.quantile(bt_d, .975)), 4)]}
    print(f"  {name}: dispatch {reach}, globals {arms}, "
          f"residual {out['residual_convention_dependence']:+.4f} {out['residual_ci95']}", flush=True)
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
        # the null is now internal: SyGMa's dispatch must reproduce ITS OWN implicit arm from this
        # run, not a literal measured on another population. A gate that compares across
        # populations is the defect this paper names, and it does not stop being one here.
        got, want = banks["sygma_175"]["reach"], banks["sygma_175"]["global_arms"]["all_implicit"]
        print(f"\nnull: SyGMa dispatch {got} against its own implicit arm {want}")
        if banks["sygma_175"]["dispatched_to_expanded"] != 0 or abs(got - want) > 1e-4:
            raise SystemExit("the classifier sends SyGMa templates somewhere it should not; "
                             "nothing else in this run is interpretable")
        if args.population == "subsample245":
            for k, v in COMMITTED.items():
                if k in banks and abs(banks[k]["global_arms"]["all_implicit"] - v[False]) > 1e-4:
                    raise SystemExit(f"{k}: the implicit arm measured here does not reproduce the "
                                     f"published {v[False]} on the population it was published on")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

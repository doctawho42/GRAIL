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

    residual = reach(dispatch) - max over legitimate global settings

The maximum ranges over the two settings someone might choose -- the implicit substrate, and the
expanded one with the product contracted again -- and not over the arm that expands and never
contracts, which is a defect rather than a convention. Measuring against that arm would credit
dispatch with repairing a bug. The same restriction defines the guaranteed reach reported beside
it, the least a bank recovers over the settings it might be run under.

Zero residual says the bank is single-convention and choosing the right global setting is the whole
story; above zero says no global setting can express the bank and its reach is only well defined
once the convention travels with the template. SyGMa is the null -- none of its 175 templates carries a
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
from _contract import contract

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


def _fragments(product, complete: bool):
    """Readable fragments of one product, optionally contracting it first."""
    if complete:
        try:
            product = contract(product)
        except Exception:
            return
    try:
        smiles = Chem.MolToSmiles(product)
    except Exception:
        return
    for fragment in _clean_product_smiles(smiles):
        try:
            key = _normalize_smiles_cached(fragment, DEFAULT["norm"])
        except Exception:
            continue
        if key:
            yield key


def _apply(substrates, rules, wants, both=False):
    """The deployed loop with the expansion decided per template rather than per run.

    Contracting the product before sanitisation is what completes the loop: expanding a substrate
    and leaving the drawn hydrogen on the reacting atom refuses most of the fragments that follow,
    so an arm run without the contraction measures an unfinished loop as well as a convention.

    The two arms differ only in that contraction, which happens after the templates have fired, so
    `both` returns them from a single enumeration. Running them as separate passes would enumerate
    the same reactions twice and cost half as much again for nothing.
    """
    seen, completed = set(), set()
    for rule, want in zip(rules, wants):
        for product in _iter_reaction_products(substrates[want], rule):
            seen.update(_fragments(product, complete=False))
            if both:
                completed.update(_fragments(product, complete=True))
    return (seen, completed) if both else seen


def _worker(item):
    """All four arms from two enumerations of the bank, on one substrate and one population.

    The dispatch arm is only meaningful against the global arms it is supposed to beat, so they have
    to be the same measurement with one thing varied. Reading a global arm out of another artifact
    makes the residual a difference between populations, which is the defect this paper is about; it
    is also unnecessary. Splitting the bank into the templates that want the expansion and those that
    do not, and applying each subset to each presentation of the substrate, yields every arm by
    union, and the contracted arm comes free from the expanded enumeration:

        all-explicit = E(want) u E(rest)      all-implicit  = I(want) u I(rest)
        dispatch     = E(want) u I(rest)      completed     = C(want) u C(rest)
    """
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return sub, 0, 0, 0, 0, 0
    substrates = {True: Chem.AddHs(Chem.Mol(mol)), False: Chem.Mol(mol)}
    want, rest = _CTX["want"], _CTX["rest"]
    e_want, c_want = _apply(substrates, want, [True] * len(want), both=True)
    e_rest, c_rest = _apply(substrates, rest, [True] * len(rest), both=True)
    i_want = _apply(substrates, want, [False] * len(want))
    i_rest = _apply(substrates, rest, [False] * len(rest))
    # Dispatch sends a template to the expanded substrate; it does not thereby inherit the defect of
    # never contracting what comes back. Completing the loop is a correctness requirement and not a
    # convention, so the expanded half of the dispatch arm is the contracted one. Built from the
    # uncontracted half instead, dispatch would be measured under the unfinished loop on exactly the
    # templates it was introduced to help, and its residual would report that rather than the policy.
    arms = {"dispatch": c_want | i_rest,
            "explicit": e_want | e_rest,
            "implicit": i_want | i_rest,
            "explicit_completed": c_want | c_rest}
    usable, hits = 0, {}
    for k, pool in arms.items():
        usable, hits[k], _ = _tautomer_recovered(trues, sorted(pool), audit=False)
    return (sub, int(usable), int(hits["dispatch"]), int(hits["explicit"]),
            int(hits["implicit"]), int(hits["explicit_completed"]))


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
    Ccomp = np.array([r[5] for r in rows])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))

    def reach_of(X):
        return round(float(X.sum() / max(U.sum(), 1)), 4)

    bt = np.array([H[j].sum() / max(U[j].sum(), 1) for j in idx])
    reach = reach_of(H)
    arms = {"all_explicit": reach_of(E), "all_implicit": reach_of(I),
            "all_explicit_completed": reach_of(Ccomp)}
    # The two global arms carry a claim of their own: which of two published banks reaches further
    # is decided by the convention. That claim is a paired difference on the same substrates, so it
    # gets the same treatment as the residual rather than a comparison of two marginal intervals.
    # The recovered counts are recorded here because they are quoted directly; deriving them by
    # multiplying a rounded reach by the reference total is the provenance defect this paper names.
    d_arm = E - I
    bt_arm = np.array([d_arm[j].sum() / max(U[j].sum(), 1) for j in idx])
    arm_lo, arm_hi = (float(np.quantile(bt_arm, .025)), float(np.quantile(bt_arm, .975)))
    arm_detail = {
        "recovered_explicit": int(E.sum()), "recovered_implicit": int(I.sum()),
        "recovered_explicit_completed": int(Ccomp.sum()),
        "explicit_minus_implicit": round(float(d_arm.sum() / max(U.sum(), 1)), 4),
        "ci95": [round(arm_lo, 4), round(arm_hi, 4)],
        "excludes_zero": bool(arm_lo * arm_hi > 0)}
    # The residual is what dispatch adds over the best a single global setting could have done, so
    # "best" has to range over settings someone might actually choose. `all_explicit` expands the
    # substrate and never contracts the product; Section 4 calls that a defect, and measuring
    # against it would credit dispatch with repairing a bug rather than with handling a mixed bank.
    # The comparison is therefore against the two real conventions, and the same restriction gives
    # the guaranteed reach: the least a bank recovers over the settings it might be run under.
    legitimate = {"all_implicit": I, "all_explicit_completed": Ccomp}
    best_name = max(legitimate, key=lambda k: reach_of(legitimate[k]))
    better = legitimate[best_name]
    best_global = reach_of(better)
    worst_global = min(reach_of(v) for v in legitimate.values())
    d = H - better
    bt_d = np.array([d[j].sum() / max(U[j].sum(), 1) for j in idx])
    # Banks are run one at a time and merged into one file, so an entry can outlive the code that
    # produced it. Recording the version per bank makes a mixed artifact detectable instead of
    # leaving it to be noticed, which is how the last two defects survived as long as they did.
    out = {"measured_by": _code_version(),
           "n_rules": len(rules), "dispatched_to_expanded": n_exp,
           "references": int(U.sum()), "recovered": int(H.sum()), "reach": reach,
           "ci95": [round(float(np.quantile(bt, .025)), 4),
                    round(float(np.quantile(bt, .975)), 4)],
           "global_arms": arms, "global_arms_paired": arm_detail,
           "legitimate_global_arms": list(legitimate),
           "best_global": best_global, "best_global_arm": best_name,
           "guaranteed_reach": worst_global,
           "residual_convention_dependence": round(reach - best_global, 4),
           "residual_ci95": [round(float(np.quantile(bt_d, .025)), 4),
                             round(float(np.quantile(bt_d, .975)), 4)]}
    print(f"  {name}: {int(H.sum())}/{int(U.sum())} references under dispatch = {reach}; "
          f"globals {arms}; best legitimate {best_name} {best_global}, guaranteed {worst_global}; "
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

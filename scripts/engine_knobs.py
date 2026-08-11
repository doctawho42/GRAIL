#!/usr/bin/env python3
"""How much of a rule bank's reach is decided by the engine's undocumented choices?

The paper shows that swapping the whole application engine, at fixed rules, moves reach by as much
as the entire gap between two banks. A reader can accept that and still think it is a fact about two
particular programs. It is not. Our own engine makes three discretionary choices before it reports a
product, and each is one a group would make silently:

    explicit hydrogens   whether the substrate is expanded with AddHs before the SMIRKS fire, which
                         decides whether patterns written for explicit H match at all
    product normalisation  tautomer canonicalisation with stereo stripped, or plain canonical SMILES
    validity filter      what counts as a product worth keeping -- here fragments below a heavy-atom
                         floor are dropped

Holding the rules, the substrates and the matcher fixed and toggling one knob at a time gives the
spread a single implementation can produce on its own. That is the quantity the recommendation
"report the engine" is worth, and it needs no second engine to measure.

The loop is reimplemented here rather than imported because `apply_rules_to_molecule` exposes only
the normalisation mode, and reimplementation is exactly what makes a measurement untrustworthy. So
the default configuration is gated against the committed arm-A reach of
results/reach_engine_vs_bank.json, to four decimals, and the comparator arm against arm B.

What that gate does and does not certify is worth stating, because it is easy to overclaim. It binds
the rule set, the substrate set, the matcher and the hydrogen decision, since moving any of those
moves the number it checks. It cannot bind the other two knobs, and for the reason this script
exists: all four configurations with hydrogens explicit return 0.1887 on the nose, so the gate is
passed identically by each. The normalisation row below is the evidence for that, not an assumption
behind it. One consequence is that committed arm A was itself produced with normalisation_mode
"canonical" (reach_engine_vs_bank.py) while the deployed default in apply_rules_to_molecule is
"standardize" -- two different configurations that coincide to four decimals here, which is exactly
what the second row reports.
"""
from __future__ import annotations

import argparse
import itertools
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

import sygma
from grail_metabolism.metrics import _tautomer_inchikey as _tk
from rdkit import Chem, RDLogger
from _contract import contract

from grail_metabolism.utils.preparation import (
    _clean_product_smiles, _iter_reaction_products, _normalize_smiles_cached, iscorrect)
from _population import POPULATIONS, population_items, load_population, tagged_out
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
SYGMA_RULES = Path(os.environ.get("SYGMA_RULES") or (Path(sygma.__file__).parent / "rules"))
# the deployed configuration -- apply_rules_to_molecule's own defaults. Committed arm A was measured
# with norm="canonical"; both reach 0.1887, which is the normalisation row of this measurement.
DEFAULT = {"add_hs": True, "norm": "standardize", "drop_invalid": True}
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


def shared_rules() -> list[str]:
    """The rules both banks hold verbatim, exactly as bank_overlap_sygma.py selects them."""
    bank = set()
    for line in (ROOT / "grail_metabolism/resources/extended_smirks.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            bank.add(line.split("\t")[0].strip())
    out = []
    for f in ("phase1.txt", "phase2.txt"):
        for line in (SYGMA_RULES / f).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smirks = line.split("\t")[0].strip()
            if smirks in bank and smirks not in out:
                out.append(smirks)
    return out


def _init(rules):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _CTX["rules"] = rules


def apply_with(mol, rules, add_hs: bool, norm: str, drop_invalid: bool,
               remove_hs: bool = False) -> list[str]:
    """The deployed loop with each discretionary step exposed as a switch.

    `remove_hs` completes the loop rather than varying it. Expanding a substrate and never
    contracting the product leaves the drawn hydrogen on the reacting atom, and what looked like a
    convention two banks could each defend is then partly a missing call: with the contraction the
    same templates return the products the unexpanded arm returns. The switch exists so the
    difference between an incomplete loop and a complete one can be measured rather than argued.
    """
    substrate = Chem.AddHs(Chem.Mol(mol)) if add_hs else Chem.Mol(mol)
    seen = set()
    for rule in rules:
        for product in _iter_reaction_products(substrate, rule):
            if remove_hs:
                try:
                    product = contract(product)
                except Exception:
                    continue
            try:
                smiles = Chem.MolToSmiles(product)
            except Exception:
                continue
            frags = (_clean_product_smiles(smiles) if drop_invalid
                     else [f.strip() for f in smiles.split(".") if f.strip()])
            for fragment in frags:
                try:
                    seen.add(_normalize_smiles_cached(fragment, norm))
                except Exception:
                    continue
    return [s for s in seen if s]


def _sygma_pool(smiles: str) -> list[str]:
    """The comparator's own engine on the same 152 rules, one step, parent dropped as everywhere."""
    import tempfile
    if "sc" not in _CTX:
        tmp = tempfile.mkdtemp()
        paths = {}
        for f in ("phase1.txt", "phase2.txt"):
            keep = [l for l in (SYGMA_RULES / f).read_text().splitlines()
                    if l.strip() and not l.startswith("#")
                    and l.split("\t")[0].strip() in set(_CTX["rules"])]
            q = Path(tmp) / f
            q.write_text("\n".join(keep) + "\n")
            paths[f] = str(q)
        _CTX["sc"] = [sygma.Scenario([[paths["phase1.txt"], 1]]),
                      sygma.Scenario([[paths["phase2.txt"], 1]])]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    pk = _tk(Chem.MolToSmiles(mol))
    out = []
    for sc in _CTX["sc"]:
        try:
            t = sc.run(Chem.Mol(mol)); t.calc_scores()
            out += [e[0] for e in t.to_smiles() if pk is None or _tk(e[0]) != pk]
        except Exception:
            pass
    return out


def _worker(item):
    sub, trues, cfgs = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return sub, {c: (0, 0) for c in cfgs}
    out = {}
    for c in cfgs:
        if c == "sygma_engine":
            prods = _sygma_pool(sub)
        else:
            add_hs, norm, drop = c
            prods = apply_with(mol, _CTX["rules"], add_hs, norm, drop)
        u, hit, _ = _tautomer_recovered(trues, prods, audit=False)
        out[c] = (int(u), int(hit))
    return sub, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "engine_knobs.json"))
    ap.add_argument("--population", default="clean_test", choices=POPULATIONS,
                    help="subsample245 reproduces the committed artifact; clean_test is the split")
    args = ap.parse_args()
    args.out = tagged_out(args.out, args.population)

    rules = shared_rules()
    print(f"rules held by both banks: {len(rules)}", flush=True)
    if len(rules) != 152:
        raise SystemExit(f"expected the 152 shared rules, selected {len(rules)}")

    items = population_items(args.population)
    if args.limit:
        items = items[: args.limit]
    print(f"substrates: {len(items)}", flush=True)

    cfgs = [(h, n, d) for h, n, d in itertools.product((True, False), ("standardize", "canonical"),
                                                       (True, False))]
    cfgs.append("sygma_engine")     # the comparator's own program, same rules, for the paired test
    work = [(s, t, cfgs) for s, t in items]
    workers = max(1, (os.cpu_count() or 4) - 2)
    with multiprocessing.get_context("spawn").Pool(workers, _init, (rules,)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, work, 4), 1):
            rows.append(r)
            if n % 25 == 0 or n == len(work):
                print(f"  {n}/{len(work)}", flush=True)
    # imap_unordered: fix the order before resampling. The key is the substrate, which is stable;
    # reach_engine_vs_bank.py sorts its rows by their counts instead, so the two scripts resample
    # the same statistic in different orders and their intervals for a shared arm can differ in the
    # fourth decimal. The point estimates are order-invariant, which is what the gate below checks.
    rows.sort(key=lambda r: r[0])

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))
    res = {}
    for c in cfgs:
        U = np.array([r[1][c][0] for r in rows]); H = np.array([r[1][c][1] for r in rows])
        bt = np.array([H[j].sum() / U[j].sum() for j in idx])
        res[c] = {"reach": round(float(H.sum() / U.sum()), 4),
                  "ci95": [round(float(np.quantile(bt, .025)), 4),
                           round(float(np.quantile(bt, .975)), 4)],
                  "_H": H, "_U": U}

    dflt = (DEFAULT["add_hs"], DEFAULT["norm"], DEFAULT["drop_invalid"])
    committed = json.loads((ROOT / "results/reach_engine_vs_bank.json").read_text())
    arm_a = committed["arms"]["A_grail_engine_152_rules"]["point"]
    arm_b = committed["arms"]["B_sygma_engine_152_rules_one_step"]["point"]
    print(f"\ngate: default config {res[dflt]['reach']} against committed arm A {arm_a}")
    print(f"gate: comparator arm {res['sygma_engine']['reach']} against committed arm B {arm_b}")
    # The committed arms were measured on the 245-substrate subsample, so they gate that population
    # and no other. Asserting them against a run on the full split compares two populations, which
    # is the defect this paper names; on any other population the arms are simply reported.
    if args.population == "subsample245":
        if abs(res[dflt]["reach"] - arm_a) > 1e-4:
            raise SystemExit("the reimplemented loop is not the deployed engine -- the other arms "
                             "are not interpretable")
        if abs(res["sygma_engine"]["reach"] - arm_b) > 1e-4:
            raise SystemExit("the comparator arm does not reproduce the committed arm B")
    else:
        print(f"(the committed arms {arm_a} and {arm_b} gate the 245-substrate subsample only)")

    # The committed engine term is a difference of ratio-of-sums (micro); a mean of per-substrate
    # ratios (macro) is a different estimand and the two disagree here by a sixth. Both are reported,
    # each labelled, and the paper quotes the micro one because that is what it is being compared to.
    U0 = res[dflt]["_U"]
    for c in cfgs:
        if not np.array_equal(res[c]["_U"], U0):
            raise SystemExit("the reference denominator differs across arms; the contrasts below "
                             "would not share a denominator")

    def _ci(bt):
        return float(np.quantile(bt, .025)), float(np.quantile(bt, .975))

    def paired(a, b):
        Ha, Hb = res[a]["_H"], res[b]["_H"]
        mic = float((Ha - Hb).sum() / max(U0.sum(), 1))
        mlo, mhi = _ci(np.array([(Ha[j] - Hb[j]).sum() / max(U0[j].sum(), 1) for j in idx]))
        d = (Ha - Hb) / np.maximum(U0, 1)
        alo, ahi = _ci(d[idx].mean(axis=1))
        return {"delta": round(mic, 4), "ci95": [round(mlo, 4), round(mhi, 4)],
                "excludes_zero": bool(mlo > 0 or mhi < 0), "estimator": "micro",
                "macro": {"delta": round(float(d.mean()), 4),
                          "ci95": [round(alo, 4), round(ahi, 4)],
                          "excludes_zero": bool(alo > 0 or ahi < 0)}}

    one_knob = {}
    for i, name in enumerate(("explicit_hydrogens", "product_normalisation", "validity_filter")):
        if i >= 3: break
        flipped = list(dflt); flipped[i] = (not dflt[i]) if i != 1 else "canonical"
        one_knob[name] = {"from": str(dflt[i]), "to": str(flipped[i]),
                          "reach": res[tuple(flipped)]["reach"],
                          "paired_vs_default": paired(tuple(flipped), dflt)}

    # Does flipping the one switch that matters land us on the comparator's own engine?
    no_hs = (False, DEFAULT["norm"], DEFAULT["drop_invalid"])
    rep_vs = {"our_engine_without_explicit_h": res[no_hs]["reach"],
              "comparator_engine_same_rules": res["sygma_engine"]["reach"],
              "paired_difference": paired(no_hs, "sygma_engine")}
    # Both ends of this ratio come from the unrounded hit vectors. Differencing the two ROUNDED
    # arms gives 0.1923 where the committed paired estimate is 0.1922, and dividing a rounded
    # numerator by it turns 1.018 into 1.019 -- re-deriving a quantity from printed numbers is the
    # defect this paper is about, and it is two lines from happening here.
    switch_hits = int((res[no_hs]["_H"] - res[dflt]["_H"]).sum())
    engine_hits = int((res["sygma_engine"]["_H"] - res[dflt]["_H"]).sum())
    rep_vs["engine_term_micro_recomputed"] = round(float(engine_hits / U0.sum()), 4)
    # The engine term this run must reproduce is the one measured on the SAME population. The
    # committed artifact holds the subsample's; a clean_test run reads the clean_test one, which
    # reach_engine_vs_bank writes beside it. Reproducing a term across populations is not a
    # reproducibility check, it is the comparison this paper exists to warn about -- and this is the
    # fourth gate in this family to have been written that way.
    peer = ROOT / f"results/reach_engine_vs_bank{'' if args.population == 'subsample245' else '__' + args.population}.json"
    if not peer.exists():
        raise SystemExit(f"{peer.name} is missing; run reach_engine_vs_bank.py on this population "
                         f"first, or this run has no engine term to reproduce")
    rep_vs["committed_engine_term_micro"] = json.loads(peer.read_text())["contrasts"][
        "engine_at_fixed_rules_B_minus_A"]["point"]
    rep_vs["engine_term_source"] = peer.name
    if abs(rep_vs["engine_term_micro_recomputed"]
           - rep_vs["committed_engine_term_micro"]) > 1e-4:
        raise SystemExit(f"this run does not reproduce the engine term in {peer.name}")
    rep_vs["share_of_engine_term_carried_by_the_switch"] = round(switch_hits / engine_hits, 3)
    rep_vs["hit_counts"] = {"references": int(U0.sum()), "deployed": int(res[dflt]["_H"].sum()),
                            "without_explicit_h": int(res[no_hs]["_H"].sum()),
                            "comparator_engine": int(res["sygma_engine"]["_H"].sum())}
    pd_ = rep_vs["paired_difference"]
    print(f"\nour engine without explicit hydrogens {res[no_hs]['reach']:.4f} against the "
          f"comparator's own {res['sygma_engine']['reach']:.4f}: "
          f"{pd_['delta']:+.4f} {pd_['ci95']} micro, "
          f"{pd_['macro']['delta']:+.4f} {pd_['macro']['ci95']} macro")
    print(f"the switch carries {rep_vs['share_of_engine_term_carried_by_the_switch']:.3f} of the "
          f"engine term {rep_vs['committed_engine_term_micro']:+.4f}, both micro")

    reaches = [v["reach"] for v in res.values() if v is not res.get("sygma_engine")]
    rep = {"config": {**_code_version(), "population": args.population, "n_rules": len(rules), "n_substrates": len(rows),
                      "match": "inchikey_tautomer", "n_boot": N_BOOT, "seed": SEED,
                      "default": DEFAULT, "gate": f"default reproduces arm A ({arm_a})"},
           "default_reach": res[dflt]["reach"],
           "one_knob_at_a_time": one_knob,
           "all_configurations": {f"add_hs={c[0]},norm={c[1]},drop_invalid={c[2]}":
                                  {"reach": res[c]["reach"], "ci95": res[c]["ci95"]}
                                  for c in cfgs if c != "sygma_engine"},
           "spread": {"min": round(min(reaches), 4), "max": round(max(reaches), 4),
                      "range": round(max(reaches) - min(reaches), 4)},
           "against_the_comparator_engine": rep_vs}
    print("\none knob at a time, against the deployed default:")
    for k, v in one_knob.items():
        pv = v["paired_vs_default"]
        print(f"  {k:22} {v['from']:>12} -> {v['to']:<12} reach {v['reach']:.4f}  "
              f"{pv['delta']:+.4f} {pv['ci95']} micro / {pv['macro']['delta']:+.4f} macro "
              f"{'certified' if pv['excludes_zero'] else 'n.s.'}")
    print(f"\nacross all {len(cfgs) - 1} configurations of one engine: "
          f"{rep['spread']['min']} to {rep['spread']['max']}, a range of {rep['spread']['range']}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

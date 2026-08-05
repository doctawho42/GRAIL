#!/usr/bin/env python3
"""SyGMa's reach under GRAIL's own coverage primitive: one uncapped application of the whole bank.

The paper reads 0.735 against 0.542 as a difference in bank breadth. The two numbers are not
measured the same way. GRAIL's 0.735 is one uncapped depth-1 pass over 7,581 rules. SyGMa's 0.542
is its deployed pool, and scripts/sygma_fulltest_predictions.py builds
Scenario([[phase1,1],[phase2,1]]), which runs metabolize_all_nodes(phase2) over a tree that already
contains the phase-1 products -- a composed two-step run. Cfull has to be the deployed pool for the
decomposition's nesting (R(k) subset Pbud subset Pfull) to hold, so decompose_sygma.py is right to
use it; but that same number then cannot also serve as the knowledge base's reach, which is what
Section 4 spends it on.

This measures the missing primitive: every SyGMa rule applied once to the parent and nothing
composed, phase1-once union phase2-once, scored by the same matcher on the same substrates. The
gap between it and 0.542 is what SyGMa's engine contributes on top of its rule set.

Gate: re-running the deployed two-step scenario here must reproduce the frozen pools that
results/decompose_sygma.json was computed from, or the two arms are not the same measurement.
"""
from __future__ import annotations
import json, sys, time
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MATCH, N_BOOT, SEED = "inchikey_tautomer", 10000, 0
OUT = ROOT / "results" / "sygma_depth_matched_reach.json"
_SC = None


def _worker(smiles):
    """parent -> (phase1-once pool, phase2-once pool, deployed two-step pool)."""
    global _SC
    import sygma
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    if _SC is None:
        _SC = (sygma.Scenario([[sygma.ruleset["phase1"], 1]]),
               sygma.Scenario([[sygma.ruleset["phase2"], 1]]),
               sygma.Scenario([[sygma.ruleset["phase1"], 1], [sygma.ruleset["phase2"], 1]]))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, [], [], []
    out = []
    for sc in _SC:
        try:
            tree = sc.run(mol)
            tree.calc_scores()
            # SyGMa's tree is rooted at the substrate and returns it first. Every other entry
            # point drops it on tautomer-InChIKey equality with the parent; do the same here so
            # the reach figures compare like with like. Cost, measured: eight annotated
            # references share a tautomer key with their own substrate and go with it.
            out.append([e[0] for e in tree.to_smiles()])
        except Exception:
            out.append([])
    return smiles, out[0], out[1], out[2]


def main() -> int:
    preds = json.loads((ROOT / "results" / "sygma_fulltest_predictions.json").read_text())
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    table = json.loads((ROOT / "results" / "key_tables" / f"{MATCH}.json").read_text())
    grail = {r["sub"]: r for r in
             json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]}
    subs = sorted(s for s in truth if truth[s] and s in preds)
    print(f"substrates: {len(subs)}", flush=True)

    n_proc = max(1, cpu_count() - 2)
    pools, t0 = {}, time.perf_counter()
    with Pool(n_proc) as pool:
        for i, (s, p1, p2, dep) in enumerate(pool.imap_unordered(_worker, subs, chunksize=8), 1):
            # SyGMa's tree is rooted at the substrate and returns it first; every other entry point
            # drops it on tautomer-InChIKey equality with the parent, so do the same here. Keyed
            # from the frozen table rather than recomputed -- canonicalisation is the bottleneck.
            pk = table.get(s)
            drop = lambda pool_: [x for x in pool_ if pk is None or table.get(x) != pk]
            pools[s] = (drop(p1), drop(p2), drop(dep))
            if i % 300 == 0 or i == len(subs):
                print(f"  {i}/{len(subs)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # Any structure the frozen key table has not seen is keyed here with the same function.
    unseen = sorted({x for s in subs for arm in pools[s] for x in arm if x and x not in table})
    if unseen:
        from grail_metabolism.metrics import _match_keys
        print(f"keying {len(unseen)} structures absent from the frozen table", flush=True)
        for x in unseen:
            table[x] = next(iter(_match_keys([x], MATCH)), None)

    def keys(items):
        return {table[x] for x in items if table.get(x)}

    U, C1, Cdep, Cfrozen, Cg = [], [], [], [], []
    mismatched = 0
    for s in subs:
        refs = keys(truth[s])
        p1, p2, dep = pools[s]
        if set(dep) != set(preds[s]):
            mismatched += 1
        U.append(len(refs))
        C1.append(len(refs & keys(p1 + p2)))
        Cdep.append(len(refs & keys(dep)))
        Cfrozen.append(len(refs & keys(preds[s])))
        Cg.append(grail[s]["Cfull"] if s in grail else 0)
    U, C1, Cdep, Cfrozen, Cg = map(np.array, (U, C1, Cdep, Cfrozen, Cg))
    print(f"substrates whose re-run pool differs from the frozen one: {mismatched}", flush=True)

    n = len(subs)
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]

    def stat(f):
        return round(float(f(np.arange(n))), 4), [round(float(q), 4) for q in
                                                  np.quantile([f(i) for i in idx], [0.025, 0.975])]

    micro = lambda num: (lambda i: num[i].sum() / U[i].sum())
    depth1, depth1_ci = stat(micro(C1))
    deployed, deployed_ci = stat(micro(Cdep))
    frozen, frozen_ci = stat(micro(Cfrozen))
    grail_cov, grail_ci = stat(micro(Cg))
    engine, engine_ci = stat(lambda i: (Cdep[i].sum() - C1[i].sum()) / U[i].sum())
    gap_rep, gap_rep_ci = stat(lambda i: (Cg[i].sum() - Cdep[i].sum()) / U[i].sum())
    gap_matched, gap_matched_ci = stat(lambda i: (Cg[i].sum() - C1[i].sum()) / U[i].sum())

    rep = {
        "match": MATCH, "n_substrates": n, "n_boot": N_BOOT, "seed": SEED,
        "aggregation": "micro (pooled ratio of sums), as in factorize_recall.py and decompose_sygma.py",
        "gate": {"frozen_two_step_reach": frozen, "rerun_two_step_reach": deployed,
                 "decompose_sygma_committed": 0.5422, "substrates_with_differing_pool": mismatched},
        "reach": {
            "sygma_depth1_matched": {"point": depth1, "ci95": depth1_ci,
                                     "definition": "phase1-once union phase2-once on the parent; "
                                                   "nothing composed -- GRAIL's own Cfull primitive"},
            "sygma_deployed_two_step": {"point": deployed, "ci95": deployed_ci,
                                        "definition": "Scenario([[phase1,1],[phase2,1]]): phase II "
                                                      "fires on phase I products"},
            "grail_depth1": {"point": grail_cov, "ci95": grail_ci,
                             "definition": "full 7,581-rule bank, depth-1, uncapped"},
        },
        "engine_contribution": {"point": engine, "ci95": engine_ci,
                                "definition": "deployed two-step minus depth-1 matched, paired"},
        "gap": {
            "as_reported": {"point": gap_rep, "ci95": gap_rep_ci,
                            "definition": "GRAIL depth-1 minus SyGMa two-step (the paper's 0.193)"},
            "depth_matched": {"point": gap_matched, "ci95": gap_matched_ci,
                              "definition": "GRAIL depth-1 minus SyGMa depth-1"},
        },
    }
    print(json.dumps(rep["reach"], indent=1), flush=True)
    print(json.dumps({"engine": rep["engine_contribution"], "gap": rep["gap"]}, indent=1), flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

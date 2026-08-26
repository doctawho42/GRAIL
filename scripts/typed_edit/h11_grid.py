"""The H11 check: does the emission rule's lead survive the grid, or live in a cell?

H11 fixes the rule -- emit what the trained rule budget produces, ordered by the H7 fusion, with
nothing truncated -- and predicts it beats MetaTox on macro F1 in every cell of five matching
criteria crossed with the budgets MetaTox is read at. A lead in one cell is what this series
exists to distrust, so the verdict is dominance or it is not a verdict.

The rule is GRAIL's own emission and is therefore not swept: it emits what it emits, about
sixteen candidates, and the question is whether that beats MetaTox whatever budget MetaTox is
read at. MetaTox's own emission is included as a column beside the declared budgets, since that
is the cell a service comparison actually lives in, and the counts are reported both ways rather
than the more favourable one being chosen.

Precision, recall and mean output are reported in the same table. Precision counts an unannotated
true metabolite as a false positive, which penalises volume, and this rule emits about half of
what it is compared against; the bias runs toward the rule and has to be visible where the F1 is.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import stamp  # noqa: E402

from _rrf import rrf_order  # noqa: E402
from grail_metabolism.metrics import _match_keys, f1 as f1_of, precision as p_of, recall as r_of  # noqa: E402
from scripts.run_match_sensitivity import _dedup_canon  # noqa: E402
from vs_metatox import population  # noqa: E402

CRITERIA = ["canonical", "inchikey", "inchi_no_stereo", "tanimoto1", "inchikey_tautomer"]
BUDGETS = [1, 3, 5, 8, 10, 15, 20, 32, 50, 74]
METATOX = ROOT / "results/metatox_smirks_preds.json"
# results/vs_metatox.json records MetaTox macro recall of 0.6708 at a budget of 50, not at its
# own emission: that artifact caps its predictions at 50 and this grid does not. The gate reads
# the column that number was computed in.
GATE_CELL = ("inchikey_tautomer", "50", 0.6708)


def key_list(smiles, match):
    return [next(iter(_match_keys([s], match))) for s in _dedup_canon(list(smiles))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "results/widepools_k30/all.json"))
    ap.add_argument("--out", default=str(ROOT / "results/h11_grid.json"))
    args = ap.parse_args()

    pools = {}
    for f in sorted(glob.glob(args.pools)) or [args.pools]:
        pools.update(json.loads(Path(f).read_text())["pools"])
    subs_all, truth, _ = population()
    subs = sorted(s for s in pools if truth.get(s))
    print(f"{len(subs)} substrates of {len(subs_all)}", file=sys.stderr, flush=True)

    rule = {s: [c["smiles"] for c in rrf_order(pools[s])] for s in subs}
    mtx = json.loads(METATOX.read_text())["predictions"]
    comp = {s: list(mtx.get(s, [])) for s in subs}

    mean_out = {"rule": round(sum(len(_dedup_canon(rule[s])) for s in subs) / len(subs), 2),
                "metatox": round(sum(len(_dedup_canon(comp[s])) for s in subs) / len(subs), 2)}

    grid, lost, lost_declared = {}, [], []
    for crit in CRITERIA:
        print(f"  keys for {crit} ...", file=sys.stderr, flush=True)
        real = {s: _match_keys(truth[s], crit) for s in subs}
        kr = {s: key_list(rule[s], crit) for s in subs}
        km = {s: key_list(comp[s], crit) for s in subs}

        def macro(fn, keys, k=None):
            return round(sum(fn(keys[s][:k] if k else keys[s], real[s]) for s in subs)
                         / len(subs), 4)

        row = {"rule": {"f1": macro(f1_of, kr), "precision": macro(p_of, kr),
                        "recall": macro(r_of, kr)},
               "metatox": {"uncapped": {"f1": macro(f1_of, km), "precision": macro(p_of, km),
                                        "recall": macro(r_of, km)}}}
        for k in BUDGETS:
            row["metatox"][str(k)] = {"f1": macro(f1_of, km, k), "precision": macro(p_of, km, k),
                                      "recall": macro(r_of, km, k)}
        grid[crit] = row

        for col in [str(k) for k in BUDGETS] + ["uncapped"]:
            if row["rule"]["f1"] <= row["metatox"][col]["f1"]:
                lost.append({"criterion": crit, "metatox_read_at": col,
                             "rule_f1": row["rule"]["f1"],
                             "metatox_f1": row["metatox"][col]["f1"]})
                if col != "uncapped":
                    lost_declared.append((crit, col))

    crit_g, col_g, want_g = GATE_CELL
    got = grid[crit_g]["metatox"][col_g]["recall"]
    gate_ok = abs(round(got, 4) - want_g) <= 1e-9
    mism = [] if gate_ok else [f"MetaTox macro recall under {crit_g} at k={col_g}: "
                               f"{got} vs committed {want_g}"]

    n_declared = len(CRITERIA) * len(BUDGETS)
    verdict = "supported" if not lost_declared else "failed"
    rep = {"provenance": stamp(__file__), "hypothesis": "H11",
           "population": {"n": len(subs), "source": "the 291 of results/four_method_291.json"},
           "aggregation": "macro, the mean of per-substrate F1",
           "rule": "the trained rule budget's pool, ordered by the H7 fusion, untruncated",
           "criteria": CRITERIA, "budgets": BUDGETS,
           "gate": {"reproduces_committed_metatox_recall": gate_ok, "mismatches": mism},
           "mean_output": mean_out,
           "cells_declared": n_declared, "cells_lost_declared": len(lost_declared),
           "cells_including_metatox_own_emission": n_declared + len(CRITERIA),
           "cells_lost_including_own_emission": len(lost),
           "lost": lost, "grid": grid,
           "bias_note": "precision counts an unannotated true metabolite as a false positive, "
                        "which penalises volume; the rule emits "
                        f"{mean_out['rule']} against MetaTox's {mean_out['metatox']}, so F1 is "
                        "biased toward the rule and its components are tabulated beside it",
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(rep, indent=1))

    print(f"\nmean output: rule {mean_out['rule']}   metatox {mean_out['metatox']}")
    print(f"\n{'criterion':<20}{'rule F1':>9}" + "".join(f"{k:>7}" for k in BUDGETS) + f"{'own':>8}")
    for crit in CRITERIA:
        r = grid[crit]
        print(f"{crit:<20}{r['rule']['f1']:>9.4f}"
              + "".join(f"{r['metatox'][str(k)]['f1']:>7.3f}" for k in BUDGETS)
              + f"{r['metatox']['uncapped']['f1']:>8.3f}")
    print(f"\ngate reproduces the committed MetaTox recall: {gate_ok}"
          + ("" if gate_ok else f"  {mism}"))
    print(f"cells lost, declared grid {len(CRITERIA)}x{len(BUDGETS)}: {len(lost_declared)}")
    print(f"cells lost, including MetaTox's own emission: {len(lost)}")
    for x in lost[:8]:
        print(f"    {x['criterion']} at {x['metatox_read_at']}: "
              f"rule {x['rule_f1']:.4f} vs metatox {x['metatox_f1']:.4f}")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

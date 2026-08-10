#!/usr/bin/env python3
"""How much of the engine term is the convention and how much is the missing call?

Section 4 reports that applying the same 152 rules through two engines differs by +0.188 of reach,
and attributes the difference to a hydrogen convention: one engine draws the hydrogens before
matching and the other does not. The loop that draws them never contracts the product. Expanding a
substrate and leaving the drawn hydrogen on the reacting atom is what makes the product unreadable,
and one call removes it.

This measures the three arms on the same substrates and the same rules:

    expanded, as the loop stands       AddHs -> RunReactants -> sanitize
    expanded, loop completed           AddHs -> RunReactants -> RemoveHs -> sanitize
    implicit                           RunReactants -> sanitize

If the completed arm meets the implicit one, the engine term is a missing call and not a convention,
and the paper's fourth section has to say so. If a gap survives, that gap is the convention, and its
size is what the section is entitled to claim.
"""
from __future__ import annotations

import argparse, json, multiprocessing, os, pathlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from rdkit import Chem, RDLogger

from engine_knobs import DEFAULT, apply_with, shared_rules
from _population import POPULATIONS, population_items, tagged_out
from grail_metabolism.metrics import _tautomer_inchikey  # noqa: F401  (parity with the reach runs)
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
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
    _CTX["rules"] = rules


def _worker(item):
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return sub, 0, 0, 0, 0
    r = _CTX["rules"]
    arms = {
        "expanded": apply_with(mol, r, True, DEFAULT["norm"], DEFAULT["drop_invalid"]),
        "completed": apply_with(mol, r, True, DEFAULT["norm"], DEFAULT["drop_invalid"],
                                remove_hs=True),
        "implicit": apply_with(mol, r, False, DEFAULT["norm"], DEFAULT["drop_invalid"]),
    }
    usable, hits = 0, {}
    for k, pool in arms.items():
        usable, hits[k], _ = _tautomer_recovered(trues, sorted(pool), audit=False)
    return sub, int(usable), int(hits["expanded"]), int(hits["completed"]), int(hits["implicit"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="clean_test", choices=POPULATIONS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default=str(ROOT / "results" / "completed_loop_reach.json"))
    args = ap.parse_args()
    args.out = tagged_out(args.out, args.population)

    rules = shared_rules()
    items = population_items(args.population)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} substrates, {len(rules)} shared rules, three arms", flush=True)

    rows = []
    with multiprocessing.get_context("spawn").Pool(args.workers, _init, (rules,)) as pool:
        for n, r in enumerate(pool.imap_unordered(_worker, items, 2), 1):
            rows.append(r)
            if n % 50 == 0 or n == len(items):
                print(f"  {n}/{len(items)}", flush=True)
    rows.sort(key=lambda r: r[0])

    U = np.array([r[1] for r in rows], dtype=float)
    E = np.array([r[2] for r in rows], dtype=float)
    C = np.array([r[3] for r in rows], dtype=float)
    I = np.array([r[4] for r in rows], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), (N_BOOT, len(rows)))
    den = np.maximum(U[idx].sum(axis=1), 1)

    def arm(X):
        bt = X[idx].sum(axis=1) / den
        return {"reach": round(float(X.sum() / max(U.sum(), 1)), 4),
                "ci95": [round(float(np.quantile(bt, .025)), 4),
                         round(float(np.quantile(bt, .975)), 4)]}

    def paired(X, Y):
        d = X - Y
        bt = d[idx].sum(axis=1) / den
        lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
        return {"delta": round(float(d.sum() / max(U.sum(), 1)), 4),
                "ci95": [round(lo, 4), round(hi, 4)], "excludes_zero": bool(lo * hi > 0)}

    rep = {"config": {**_code_version(), "population": args.population, "n_substrates": len(rows),
                      "n_rules": len(rules), "match": "inchikey_tautomer",
                      "n_boot": N_BOOT, "seed": SEED,
                      "arms": {"expanded": "AddHs -> RunReactants -> sanitize, as the loop stands",
                               "completed": "AddHs -> RunReactants -> RemoveHs -> sanitize",
                               "implicit": "RunReactants -> sanitize"}},
           "references": int(U.sum()),
           "reach": {"expanded": arm(E), "completed": arm(C), "implicit": arm(I)},
           "the_missing_call": paired(C, E),
           "what_survives_as_convention": paired(I, C),
           "the_whole_gap_as_published": paired(I, E)}
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\n  expanded {rep['reach']['expanded']['reach']}  completed "
          f"{rep['reach']['completed']['reach']}  implicit {rep['reach']['implicit']['reach']}")
    print(f"  the missing call is worth {rep['the_missing_call']['delta']:+.4f} "
          f"{rep['the_missing_call']['ci95']}")
    print(f"  what survives as convention {rep['what_survives_as_convention']['delta']:+.4f} "
          f"{rep['what_survives_as_convention']['ci95']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

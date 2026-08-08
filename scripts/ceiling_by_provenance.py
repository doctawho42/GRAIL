#!/usr/bin/env python3
"""How much of the coverage ceiling comes from curated rules and how much from mined ones?

The bank is a deduplicated union of four published rule sets and a much larger body of templates
mined from the training split, and the paper's headline is that the bank is not the bottleneck. A
reader's first question about that headline is which half of the bank earns the ceiling: if the
mined templates carry it, the result is about automated mining rather than about curated chemistry,
and if the curated ones do, the mined tail is doing less than its size suggests.

The split is exact, not estimated: resources/mined_only.txt lists the mined templates verbatim, so
membership in the bank partitions its 7,581 rules into 5,866 mined and 1,715 curated with nothing
left over. Each subset is then applied on its own, with the same uncapped depth-1 pass, the same
tautomer-aware matcher and the same substrates as the full-bank ceiling, so the three numbers are
comparable by construction and their overlap is measurable rather than inferred.

Reported per substrate as well as pooled, because the interesting quantity is not only how much each
subset reaches but how much it reaches that the other does not -- a mined template that only
duplicates a curated one adds nothing to the ceiling however often it fires.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import multiprocessing
import os

from rdkit import Chem, RDLogger

sys.path.insert(0, str(ROOT / "scripts"))
from _population import POPULATIONS, ceiling_target, population_items, tagged_out
from engine_knobs import apply_with
from grail_metabolism.utils.preparation import apply_rules_to_molecule
from run_benchmark import _tautomer_recovered   # the paper's own per-substrate recovery count

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
# The union of the two subsets is the whole bank, so it must reproduce the ceiling already stored
# per substrate in recall_factorization.json for exactly these 245. Anything else means this pass
# applies rules differently from the one the paper reports.
#
# The target is READ from that artifact rather than frozen here. A literal goes stale the moment the
# measurement moves, and then the gate passes self-consistently on a superseded number, which is
# worse than failing: it certifies the old value silently. This gate held 0.7284 through a ceiling
# correction that took the same quantity to 0.8171.
TOL = 0.002


def _rel(p) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


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


def _init(curated, mined):
    """Each worker holds both rule subsets and stays single-threaded, as factorize_recall does."""
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    global _CUR, _MIN
    _CUR, _MIN = curated, mined


def _worker(item):
    """One substrate: references recovered by curated rules, by mined rules, and by their union."""
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return (sub, 0, 0, 0, 0)
    # the ceiling's own primitives: hydrogens as the deployed generator fires rules, no validity
    # floor, so the union of the two subsets is the same object the ceiling measures
    pc = apply_with(mol, _CUR, False, "canonical", False)
    pm = apply_with(mol, _MIN, False, "canonical", False)
    u, c_cur, _ = _tautomer_recovered(trues, pc, audit=False)
    _, c_min, _ = _tautomer_recovered(trues, pm, audit=False)
    _, c_all, _ = _tautomer_recovered(trues, pc + pm, audit=False)
    return (sub, int(u), int(c_cur), int(c_min), int(c_all))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="grail_metabolism/resources/extended_smirks.txt")
    ap.add_argument("--mined", default="grail_metabolism/resources/mined_only.txt")
    ap.add_argument("--out", default=str(ROOT / "results" / "ceiling_by_provenance.json"))
    ap.add_argument("--workers", type=int, default=0,
                    help="0 leaves two cores free; set it lower to share the machine with other runs")
    ap.add_argument("--population", default="clean_test", choices=POPULATIONS,
                    help="subsample245 reproduces the committed artifact; clean_test is the split")
    args = ap.parse_args()
    args.out = tagged_out(args.out, args.population)

    bank = [l.strip() for l in open(ROOT / args.bank) if l.strip()]
    mined_all = {l.strip() for l in open(ROOT / args.mined) if l.strip()}
    mined = [r for r in bank if r in mined_all]
    curated = [r for r in bank if r not in mined_all]
    assert len(mined) + len(curated) == len(bank), "the split does not partition the bank"
    print(f"bank {len(bank)}: mined {len(mined)}, curated {len(curated)}", flush=True)

    items = population_items(args.population)
    print(f"substrates: {len(items)}", flush=True)

    ap_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    print(f"applying both subsets to every substrate on {ap_workers} workers", flush=True)
    ctx = multiprocessing.get_context("spawn")
    t = time.perf_counter()
    with ctx.Pool(ap_workers, initializer=_init, initargs=(curated, mined)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 4), 1):
            rows.append(r)
            if n % 25 == 0 or n == len(items):
                print(f"  {n}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)
    # imap_unordered returns in completion order, so the row order varies between runs.
    # A sum over rows does not care; a bootstrap that resamples row indices does, and the
    # published interval moved in the fourth decimal on re-run because of it.
    rows.sort(key=lambda r: r[0])
    U = [r[1] for r in rows]
    hits = {"curated": [r[2] for r in rows], "mined": [r[3] for r in rows],
            "full": [r[4] for r in rows]}
    subsets = {"curated": curated, "mined": mined, "full": bank}

    U = np.array(U)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(U), (N_BOOT, len(U)))
    rep = {"config": {**_code_version(), "bank": _rel(ROOT / args.bank),
                      "mined_list": _rel(ROOT / args.mined), "n_rules": len(bank),
                      "n_mined": len(mined), "n_curated": len(curated),
                      "population": args.population},
           "n": len(items), "match": "inchikey_tautomer", "subsets": {}}
    per = {}
    for name in subsets:
        h = np.array(hits[name])
        cov = h.sum() / U.sum()
        bt = np.array([h[j].sum() / U[j].sum() for j in idx])
        # Macro is the mean of per-substrate coverages: a different estimand, not a different
        # rounding of the same one, and the appendix compares an arm against it.
        per[name] = np.divide(h, U, out=np.zeros(len(h), dtype=float), where=U > 0)
        bt_macro = per[name][idx].mean(axis=1)
        rep["subsets"][name] = {
            "n_rules": len(subsets[name]), "coverage": round(float(cov), 4),
            "ci95": [round(float(np.quantile(bt, .025)), 4), round(float(np.quantile(bt, .975)), 4)],
            "coverage_macro": round(float(per[name].mean()), 4),
            "ci95_macro": [round(float(np.quantile(bt_macro, .025)), 4),
                           round(float(np.quantile(bt_macro, .975)), 4)]}
        print(f"  {name:8} {len(subsets[name]):>5} rules -> micro {cov:.4f} "
              f"{rep['subsets'][name]['ci95']}  macro {per[name].mean():.4f} "
              f"{rep['subsets'][name]['ci95_macro']}", flush=True)

    c, m, f = (rep["subsets"][k]["coverage"] for k in ("curated", "mined", "full"))
    target = ceiling_target([r[0] for r in rows])
    if abs(f - target) > TOL:
        raise SystemExit(f"union of the subsets covers {f:.4f} against the committed {target:.4f} on "
                         f"these substrates -- this pass is not the one the ceiling came from")
    rep["ceiling_gate"] = {"committed": round(target, 4), "reproduced": f,
                           "source": "results/recall_factorization.json, restricted to these substrates"}
    rep["exclusive"] = {"curated_only": round(f - m, 4), "mined_only": round(f - c, 4),
                        "shared": round(c + m - f, 4)}
    rep["per_substrate"] = [{"sub": r[0], "u": int(r[1]), "curated": int(r[2]),
                             "mined": int(r[3]), "full": int(r[4])} for r in rows]
    print(f"\n  reachable only by curated rules: {f-m:+.4f}")
    print(f"  reachable only by mined rules  : {f-c:+.4f}")
    print(f"  reachable by both              : {c+m-f:+.4f}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

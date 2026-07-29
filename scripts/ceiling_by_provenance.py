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
from grail_metabolism.utils.preparation import apply_rules_to_molecule
from run_benchmark import _tautomer_recovered   # the paper's own per-substrate recovery count

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
# The union of the two subsets is the whole bank, so it must reproduce the ceiling already
# stored per substrate in recall_factorization.json for exactly these 245. Anything else
# means this pass applies rules differently from the one the paper reports.
CEILING_SUBSET, TOL = 0.7284, 0.002
_CUR: list = []
_MIN: list = []


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
    pc = list(apply_rules_to_molecule(mol, _CUR, normalization_mode="canonical").keys())
    pm = list(apply_rules_to_molecule(mol, _MIN, normalization_mode="canonical").keys())
    u, c_cur, _ = _tautomer_recovered(trues, pc, audit=False)
    _, c_min, _ = _tautomer_recovered(trues, pm, audit=False)
    _, c_all, _ = _tautomer_recovered(trues, pc + pm, audit=False)
    return (sub, int(u), int(c_cur), int(c_min), int(c_all))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="grail_metabolism/resources/extended_smirks.txt")
    ap.add_argument("--mined", default="grail_metabolism/resources/mined_only.txt")
    ap.add_argument("--substrates", default="results/filter_vs_prior_ci.json",
                    help="artifact whose per_substrate block fixes the substrate set")
    ap.add_argument("--out", default=str(ROOT / "results" / "ceiling_by_provenance.json"))
    args = ap.parse_args()

    bank = [l.strip() for l in open(ROOT / args.bank) if l.strip()]
    mined_all = {l.strip() for l in open(ROOT / args.mined) if l.strip()}
    mined = [r for r in bank if r in mined_all]
    curated = [r for r in bank if r not in mined_all]
    assert len(mined) + len(curated) == len(bank), "the split does not partition the bank"
    print(f"bank {len(bank)}: mined {len(mined)}, curated {len(curated)}", flush=True)

    src = json.loads((ROOT / args.substrates).read_text())["per_substrate"]
    subs = [r["sub"] for r in src]
    refs_raw = json.loads((ROOT / "results" / "test_references.json").read_text())
    items = [(s, refs_raw[s]) for s in subs if refs_raw.get(s)]
    print(f"substrates: {len(items)}", flush=True)

    ap_workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"applying both subsets to every substrate on {ap_workers} workers", flush=True)
    ctx = multiprocessing.get_context("spawn")
    t = time.perf_counter()
    with ctx.Pool(ap_workers, initializer=_init, initargs=(curated, mined)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 4), 1):
            rows.append(r)
            if n % 25 == 0 or n == len(items):
                print(f"  {n}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)
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
                      "substrate_source": args.substrates},
           "n": len(items), "match": "inchikey_tautomer", "subsets": {}}
    for name in subsets:
        h = np.array(hits[name])
        cov = h.sum() / U.sum()
        bt = np.array([h[j].sum() / U[j].sum() for j in idx])
        rep["subsets"][name] = {
            "n_rules": len(subsets[name]), "coverage": round(float(cov), 4),
            "ci95": [round(float(np.quantile(bt, .025)), 4), round(float(np.quantile(bt, .975)), 4)]}
        print(f"  {name:8} {len(subsets[name]):>5} rules -> coverage {cov:.4f} "
              f"{rep['subsets'][name]['ci95']}", flush=True)

    c, m, f = (rep["subsets"][k]["coverage"] for k in ("curated", "mined", "full"))
    if abs(f - CEILING_SUBSET) > TOL:
        raise SystemExit(f"union of the subsets covers {f:.4f} against a stored {CEILING_SUBSET} on "
                         f"these substrates -- this pass is not the one the ceiling came from")
    rep["ceiling_gate"] = {"stored": CEILING_SUBSET, "reproduced": f}
    rep["exclusive"] = {"curated_only": round(f - m, 4), "mined_only": round(f - c, 4),
                        "shared": round(c + m - f, 4)}
    print(f"\n  reachable only by curated rules: {f-m:+.4f}")
    print(f"  reachable only by mined rules  : {f-c:+.4f}")
    print(f"  reachable by both              : {c+m-f:+.4f}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

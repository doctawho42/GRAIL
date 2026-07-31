#!/usr/bin/env python3
"""A third rule-grounded method for the decomposition, and the one with a real selection stage.

The decomposition has run on two methods, and a reviewer's objection is that one of them is
degenerate in the factor the paper is about: SyGMa's selection is 1 by construction, so the
comparison rests on a single contrast. BioTransformer answers it. Its templates are enumerable --
669 SMIRKS in its shipped metabolicReactions.json -- and unlike SyGMa it does select, applying
reactions conditioned on biosystem and enzyme rather than firing everything applicable.

What this measures and what it does not. Applying those templates uniformly with RDKit is not
BioTransformer's own Java implementation, which layers enzyme predicates, cofactor logic and
depth control on top of the same SMIRKS. So the number here is what the template set can express
under uniform application -- an upper bound on the bank's reach that its own pipeline may not
realise -- and it is reported as that rather than as BioTransformer's coverage factor. Realised
recall comes from the frozen predictions used everywhere else in this paper, so the conversion
ratio below is exactly what its shipped configuration converts of what its templates could reach.

Scored on the 150 shared substrates, the only population where BioTransformer's predictions exist,
with GRAIL and SyGMa recomputed on the same substrates so the three rows are comparable.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import Chem, RDLogger

from grail_metabolism.utils.preparation import apply_rules_to_molecule
from run_benchmark import _tautomer_recovered

RDLogger.DisableLog("rdApp.*")
N_BOOT, SEED = 10000, 0
# Templates live in more than one file: the human/mammalian set, the environmental-microbial set,
# and a standardisation set. Parsing only the first under-counts the bank, which shows up
# immediately as a reach below BioTransformer's own realised recall -- impossible if the parsed
# templates bounded its output.
BT_DB = [ROOT / "artifacts/tier2/biotransformer/database/metabolicReactions.json",
         ROOT / "artifacts/tier2/biotransformer/database/ENVMICRO/metabolicReactions.json",
         ROOT / "artifacts/tier2/biotransformer/database/standardizationReactions.json"]
_RULES: list = []


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


def load_bt_templates(paths) -> list[str]:
    """SMIRKS out of BioTransformer's reaction databases, which carry // comments before the JSON."""
    out, seen = [], set()
    for path in paths:
        if not path.exists():
            continue
        for t in re.findall(r'"smirks"\s*:\s*"([^"]{5,})"', path.read_text(errors="ignore")):
            t = t.replace("\\\\", "\\")
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _init(rules):
    RDLogger.DisableLog("rdApp.*")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    global _RULES
    _RULES = rules


def _worker(item):
    sub, trues = item
    mol = Chem.MolFromSmiles(sub)
    if mol is None or not trues:
        return (sub, 0, 0)
    prods = list(apply_rules_to_molecule(mol, _RULES, normalization_mode="canonical").keys())
    u, c, _ = _tautomer_recovered(trues, prods, audit=False)
    return (sub, int(u), int(c))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "decompose_biotransformer.json"))
    args = ap.parse_args()

    rules = load_bt_templates(BT_DB)
    print(f"BioTransformer templates parsed: {len(rules)}", flush=True)
    if len(rules) < 100:
        raise SystemExit("too few templates parsed -- the database format is not what was assumed")

    subs = json.loads((ROOT / "artifacts/tier2/substrates.json").read_text())
    truth = json.loads((ROOT / "results" / "test_references.json").read_text())
    items = [(s, truth[s]) for s in subs if truth.get(s)]
    if args.limit:
        items = items[: args.limit]
    print(f"substrates with references: {len(items)}", flush=True)

    workers = max(1, (os.cpu_count() or 4) - 2)
    ctx = multiprocessing.get_context("spawn")
    t = time.perf_counter()
    with ctx.Pool(workers, initializer=_init, initargs=(rules,)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_worker, items, 4), 1):
            rows.append(r)
            if n % 25 == 0 or n == len(items):
                print(f"  {n}/{len(items)} ({time.perf_counter()-t:.0f}s)", flush=True)

    U = np.array([r[1] for r in rows])
    C = np.array([r[2] for r in rows])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(U), (N_BOOT, len(U)))
    cov = C.sum() / U.sum()
    bt = np.array([C[j].sum() / U[j].sum() for j in idx])
    lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))

    # realised recall from the frozen predictions everything else in the paper uses
    cm = json.loads((ROOT / "results" / "cross_method_decomposition.json").read_text())
    realised = {m["method"]: m for m in cm["methods"]}
    rep = {"config": {**_code_version(), "templates": [_rel(x) for x in BT_DB], "n_templates": len(rules),
                      "substrates": "artifacts/tier2/substrates.json"},
           "n": len(items), "match": "inchikey_tautomer",
           "biotransformer": {"template_reach": round(float(cov), 4), "ci95": [round(lo, 4), round(hi, 4)],
                              "n_templates": len(rules)},
           "context": {"grail_bank_ceiling_on_shared": cm.get("grail_bank_ceiling_on_shared"),
                       "sygma_pool_reach": realised.get("SyGMa", {}).get("pool_coverage_recall_inf"),
                       "grail_realised": realised.get("GRAIL", {}).get("recall@15"),
                       "sygma_realised": realised.get("SyGMa", {}).get("recall@15")}}
    print(f"\n  BioTransformer templates reach {cov:.4f} [{lo:.4f},{hi:.4f}] of the references "
          f"on these {len(items)} substrates")
    print(f"  for context, GRAIL's bank reaches {cm.get('grail_bank_ceiling_on_shared')} here and "
          f"SyGMa's pool {realised.get('SyGMa', {}).get('pool_coverage_recall_inf')}")
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

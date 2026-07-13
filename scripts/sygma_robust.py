"""Crash-robust SyGMa recall@k reproduction (plain + tautomer InChIKey).

Reproduces `run_benchmark.sygma_baseline`'s recall@k numbers (0.558 plain / 0.572 tautomer @15,
n=1168 committed) while surviving the native segfault that kills the in-process run: SyGMa's
enumeration (a C-extension) segfaults on some substrate, and a segfault cannot be caught by
try/except, so the whole benchmark process dies mid-loop. This driver runs the enumeration in an
isolated worker subprocess that flushes each result; when the worker dies, it records the crasher
and relaunches past it, then computes recall with sygma_baseline's *exact* matching logic.

Every substrate that enumerates contributes a byte-identical recall term to sygma_baseline; only a
substrate that natively segfaults is dropped (recorded), so the reproduced mean matches to within
the weight of the dropped substrate(s).

Usage: python scripts/sygma_robust.py [--ks 5 10 12 15] [--out results/sygma_robust.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_metabolism.metrics import _inchikey, _tautomer_inchikey
from scripts.run_benchmark import ik_set, load_test_map

PYTHON = sys.executable
WORKER = ROOT / "scripts" / "sygma_enum_worker.py"


def enumerate_all(order_file: Path, out_jsonl: Path) -> List[int]:
    """Run the worker, resuming past each native crash. Returns the list of crasher indices."""
    n_subs = len(order_file.read_text().splitlines())
    if out_jsonl.exists():
        out_jsonl.unlink()
    crashers: List[int] = []
    start = 0
    while start < n_subs:
        rc = subprocess.run([PYTHON, str(WORKER), str(order_file), str(out_jsonl), str(start)]).returncode
        # how far did it get? last line's index + 1
        done = -1
        if out_jsonl.exists():
            for line in out_jsonl.read_text().splitlines():
                if line.strip():
                    done = json.loads(line)["i"]
        if rc == 0:
            break  # finished cleanly
        # worker died (segfault/other). The crasher is the substrate after the last completed one.
        crasher = done + 1
        crashers.append(crasher)
        print(f"  [robust] worker exited rc={rc} after index {done}; substrate {crasher} crashed "
              f"enumeration -> dropping, resuming at {crasher + 1}", flush=True)
        # record a null-ranked row for the crasher so the aggregation treats it as a drop
        with open(out_jsonl, "a") as fh:
            fh.write(json.dumps({"i": crasher, "sub": None, "ranked": None, "crashed": True}) + "\n")
            fh.flush()
        start = crasher + 1
    return crashers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 12, 15])
    ap.add_argument("--out", default=str(ROOT / "results" / "sygma_robust.json"))
    ap.add_argument("--scratch", default=None, help="dir for the order file + enumeration jsonl")
    args = ap.parse_args()
    ks = args.ks
    kmax = max(ks)

    scratch = Path(args.scratch) if args.scratch else ROOT / "results"
    scratch.mkdir(parents=True, exist_ok=True)
    order_file = scratch / "sygma_order.txt"
    out_jsonl = scratch / "sygma_enum.jsonl"

    # deterministic test order — identical to sygma_baseline's list(test_map.items())
    test_map: Dict[str, Set[str]] = load_test_map(None, 42)
    items = list(test_map.items())
    order_file.write_text("\n".join(sub for sub, _ in items))
    print(f"[robust] {len(items)} test substrates; isolating SyGMa enumeration in worker subprocesses", flush=True)

    t0 = time.time()
    crashers = enumerate_all(order_file, out_jsonl)
    print(f"[robust] enumeration done in {(time.time()-t0)/60:.1f} min; {len(crashers)} native crash(es): {crashers}", flush=True)

    # index -> ranked SMILES list (or None)
    ranked_by_i: Dict[int, object] = {}
    for line in out_jsonl.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            ranked_by_i[row["i"]] = row["ranked"]

    # --- recall aggregation: sygma_baseline's exact logic (run_benchmark.py:517-552) ---
    recall_at = {k: 0.0 for k in ks}
    recall_at_t = {k: 0.0 for k in ks}
    out_sizes: List[int] = []
    n = 0
    for i, (sub, true_prods) in enumerate(items):
        ranked_smiles = ranked_by_i.get(i)
        if not isinstance(ranked_smiles, list):  # mol None, sygma None, or native crash
            continue
        # plain-InChIKey preds, parent dropped by plain IK
        preds: List[str] = []
        for smi in ranked_smiles:
            ikey = _inchikey(smi)
            if ikey not in preds:
                preds.append(ikey)
        parent_ik = _inchikey(sub)
        preds = [p for p in preds if p != parent_ik]
        out_sizes.append(len(preds))
        true_ik = ik_set(true_prods)
        n += 1
        for k in ks:
            hit = len(set(preds[:k]) & true_ik)
            recall_at[k] += hit / len(true_ik) if true_ik else 0.0
        # tautomer-InChIKey preds, parent dropped by tautomer key, first kmax tautomer-distinct
        parent_tk = _tautomer_inchikey(sub)
        preds_t: List[str] = []
        for smi in ranked_smiles:
            tk = _tautomer_inchikey(smi)
            if tk == parent_tk or tk in preds_t:
                continue
            preds_t.append(tk)
            if len(preds_t) >= kmax:
                break
        true_tk = {_tautomer_inchikey(t) for t in true_prods}
        for k in ks:
            hit_t = len(set(preds_t[:k]) & true_tk)
            recall_at_t[k] += hit_t / len(true_tk) if true_tk else 0.0

    report = {
        "n_substrates_scored": n,
        "n_native_crashes": len(crashers),
        "crasher_indices": crashers,
        "crasher_smiles": [items[c][0] for c in crashers if c < len(items)],
        "mean_output_size": sum(out_sizes) / n if n else 0.0,
        "recall_at": {str(k): recall_at[k] / n for k in ks} if n else {},
        "recall_at_tautomer": {str(k): recall_at_t[k] / n for k in ks} if n else {},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n=== SyGMa robust reproduction ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"\n(committed sygma_baseline: recall@15 0.558 plain / 0.572 tautomer, n=1168)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

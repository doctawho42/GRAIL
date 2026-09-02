#!/usr/bin/env python3
"""Generate BioTransformer 3.0 predictions for a set of parent SMILES -> a predictions JSON
{parent_smiles: [metabolite_smiles ...]} that drops into the match-sensitivity engine.

Runs the JAR per parent (so every metabolite, across generations up to --steps, maps to its
original parent) from the JAR's own dir (it loads btkb/supportfiles relatively). Output is a
rule-based unranked set (BioTransformer gives no confidence score).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
import os
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
BT_DIR = Path(os.environ.get("BIOTRANSFORMER_DIR",
                             ROOT / "artifacts" / "tier2" / "biotransformer"))
JAR = Path(os.environ.get("BIOTRANSFORMER_JAR", BT_DIR / "biotransformer-3.0.0.jar"))
# The bundled JNI InChI artefact is cached for MAC-X86_64. On an arm64 machine the jar exits
# before predicting anything under the default JDK, and runs under an x86_64 Java 8 one. Naming
# the runtime here is the difference between "this tool cannot be run" and "this tool needs a
# runtime we have", and the two are not the same claim about a comparator.
JAVA = os.environ.get("BIOTRANSFORMER_JAVA", "java")


def load_parents(spec: str) -> List[str]:
    p = Path(spec)
    raw = p.read_text()
    if spec.endswith(".json"):
        data = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw))
        return [x["smiles"] for x in data if x.get("smiles")]
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def bt_one(smiles: str, steps: int) -> List[str]:
    out = Path(tempfile.mktemp(suffix=".csv"))
    try:
        subprocess.run(
            [JAVA, "-jar", str(JAR), "-k", "pred", "-b", "allHuman", "-cm", "3",
             "-s", str(steps), "-ismi", smiles, "-ocsv", str(out)],
            cwd=str(BT_DIR), capture_output=True, timeout=600,
        )
        mets = []
        if out.exists():
            with open(out) as f:
                for row in csv.DictReader(f):
                    smi = (row.get("SMILES") or "").strip()
                    if smi:
                        mets.append(smi)
        return mets
    except Exception:
        return []
    finally:
        out.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", required=True, help="GLORYx json, or a txt of one SMILES per line")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="parents in flight at once; the jar is a process per parent")
    ap.add_argument("--resume", action="store_true",
                    help="keep what --out already holds and predict only the rest, since a run "
                         "over the whole evaluated test set is hours and a lost run is hours")
    args = ap.parse_args()

    parents = load_parents(args.parents)
    if args.limit:
        parents = parents[: args.limit]
    preds: Dict[str, List[str]] = {}
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        try:
            preds = json.loads(out_path.read_text())
        except Exception:
            preds = {}
    todo = [p for p in parents if p not in preds]
    print(f"BioTransformer on {len(todo)} of {len(parents)} parents (steps={args.steps}, "
          f"{args.workers} at a time)", flush=True)
    t = time.perf_counter()
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (p, mets) in enumerate(
                    zip(todo, pool.map(lambda s: bt_one(s, args.steps), todo)), 1):
                preds[p] = mets
                if i == 1 or i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} ({time.perf_counter()-t:.0f}s) "
                          f"last={len(mets)} mets", flush=True)
                    out_path.write_text(json.dumps(preds, indent=2))
    else:
        for i, p in enumerate(todo, 1):
            preds[p] = bt_one(p, args.steps)
            if i == 1 or i % 5 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({time.perf_counter()-t:.0f}s) "
                      f"last={len(preds[p])} mets", flush=True)
    Path(args.out).write_text(json.dumps(preds, indent=2))
    print(f"wrote {args.out} ({sum(len(v) for v in preds.values())} total metabolites)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

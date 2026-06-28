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
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
BT_DIR = Path("/Users/nikitapolomosnov/PycharmProjects/GRAIL_baselines/biotransformer")
JAR = BT_DIR / "BioTransformer3.0_20230525.jar"


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
            ["java", "-jar", str(JAR), "-k", "pred", "-b", "allHuman", "-cm", "3",
             "-s", str(steps), "-ismi", smiles, "-ocsv", str(out)],
            cwd=str(BT_DIR), capture_output=True, timeout=180,
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
    args = ap.parse_args()

    parents = load_parents(args.parents)
    if args.limit:
        parents = parents[: args.limit]
    print(f"BioTransformer on {len(parents)} parents (steps={args.steps})", flush=True)
    preds: Dict[str, List[str]] = {}
    t = time.perf_counter()
    for i, p in enumerate(parents, 1):
        preds[p] = bt_one(p, args.steps)
        if i == 1 or i % 5 == 0 or i == len(parents):
            print(f"  {i}/{len(parents)} ({time.perf_counter()-t:.0f}s) last={len(preds[p])} mets", flush=True)
    Path(args.out).write_text(json.dumps(preds, indent=2))
    print(f"wrote {args.out} ({sum(len(v) for v in preds.values())} total metabolites)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

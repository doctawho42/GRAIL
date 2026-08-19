#!/usr/bin/env python3
"""Step 0 on the real bank, with a declared substrate sample.

The combinatorics of part C cost about 24 s per substrate against 7,580 templates, so the
full clean test split of 1,170 substrates is roughly eight hours. The sample records its
cap and its seed: without them the substrate set is not recoverable, since sampling runs
without replacement and a different cap yields a different set of the same size rather
than a subset.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from rdkit import Chem  # noqa: E402

from step0 import load_rules, run  # noqa: E402

BANK = ROOT / "grail_metabolism" / "resources" / "extended_smirks.txt"
SUBS = ROOT / "results" / "match_sensitivity_fulln.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=str(BANK))
    ap.add_argument("--cap", type=int, default=100, help="substrates to draw (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radii", default="0,1,2")
    ap.add_argument("--out", default=str(ROOT / "results" / "typed_edit_step0.json"))
    args = ap.parse_args()

    rules, load_stats = load_rules(args.rules)
    print(f"bank: {load_stats['lines']} lines, {load_stats['parsed']} rules, "
          f"{load_stats.get('unparseable', 0)} unparseable", file=sys.stderr, flush=True)

    pool = json.loads(SUBS.read_text())["substrates"]
    picked = (random.Random(args.seed).sample(pool, args.cap)
              if args.cap and args.cap < len(pool) else list(pool))
    subs = [(s, Chem.MolFromSmiles(s)) for s in picked]
    subs = [(n, m) for n, m in subs if m is not None]
    print(f"substrates: {len(subs)} of {len(pool)} (cap={args.cap}, seed={args.seed})",
          file=sys.stderr, flush=True)

    res = run(rules, subs, tuple(int(x) for x in args.radii.split(",")))
    res["rule_bank"] = {"path": str(Path(args.rules).relative_to(ROOT)), **load_stats}
    res["substrate_sample"] = {
        "source": str(SUBS.relative_to(ROOT)), "population": len(pool),
        "cap": args.cap, "seed": args.seed, "n": len(subs), "smiles": [n for n, _ in subs],
    }
    Path(args.out).write_text(json.dumps(res, indent=1))

    short = {k: v for k, v in res.items() if k != "rows"}
    short["substrate_sample"] = {k: v for k, v in short["substrate_sample"].items()
                                 if k != "smiles"}
    print(json.dumps(short, indent=1))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

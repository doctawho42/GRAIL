"""SyGMa enumeration worker (crash-isolated).

Enumerates SyGMa's ranked metabolite SMILES for each test substrate, starting at a given
index, appending one JSON line per substrate to an output file and FLUSHING after each, so a
native segfault (uncatchable by try/except, the reason the in-process run dies) loses at most
the single substrate being processed. The driver (`sygma_robust.py`) inspects the output to
find the last completed index, records the crasher, and relaunches this worker past it.

Only the enumeration (`_sygma_ranked`) can segfault; all matching/recall is done downstream by
the driver using run_benchmark's own functions, so the reproduced numbers are byte-identical to
`sygma_baseline` for every substrate that enumerates.

Usage: python scripts/sygma_enum_worker.py <order_file> <out_jsonl> <start_index>
  order_file: one SMILES per line, the deterministic test order (list(test_map.items()))
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem

from scripts.run_benchmark import _sygma_ranked, _sygma_scenario


def main() -> int:
    order_file, out_jsonl, start_index = sys.argv[1], sys.argv[2], int(sys.argv[3])
    subs = Path(order_file).read_text().splitlines()
    scenario = _sygma_scenario()
    mode = "a" if start_index > 0 else "w"
    with open(out_jsonl, mode) as fh:
        for i in range(start_index, len(subs)):
            sub = subs[i]
            ranked_smiles = None
            mol = Chem.MolFromSmiles(sub)
            if mol is not None:
                ranked = _sygma_ranked(mol, scenario)  # <-- the call that can segfault
                if ranked is not None:
                    ranked_smiles = [entry[0] for entry in ranked]
            fh.write(json.dumps({"i": i, "sub": sub, "ranked": ranked_smiles}) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the tier-2 external-tool input for the FULL 1170-substrate clean test.

The n=150 tier-2 run used artifacts/tier2/{substrates.json, sub_index_map.json, mp_input.csv}.
This produces the n≈1170 analogues under artifacts/tier2_1170/, in the SAME deterministic order as
results/recall_factorization.json (the 1170 evaluated test substrates), so the resulting tier-2
predictions re-score against the same reference set that GRAIL/SyGMa use.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "tier2_1170"


def main() -> int:
    rows = json.loads((ROOT / "results" / "recall_factorization.json").read_text())["per_substrate"]
    subs = [r["sub"] for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "substrates.json").write_text(json.dumps(subs, indent=1))
    index_map = {f"sub_{i}": s for i, s in enumerate(subs)}
    (OUT / "sub_index_map.json").write_text(json.dumps(index_map, indent=1))
    # MetaPredictor input: `sub_i,SMILES` per line, no header (matches artifacts/tier2/mp_input.csv)
    (OUT / "mp_input.csv").write_text("".join(f"sub_{i},{s}\n" for i, s in enumerate(subs)))

    print(f"wrote {len(subs)} substrates to {OUT}/ (substrates.json, sub_index_map.json, mp_input.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Where the third factor of the decomposition actually loses its references.

The ranking factor is |H|/|C_bud|, and a reader is entitled to ask how it can be below one when
the paper also says a top-15 re-truncation of GRAIL's emitted list removes nothing. Both are true
and they are different statements: GRAIL's deployed output policy is itself a cap at k=15, so the
output reaches the cap upstream of any re-truncation. What is measurable here is emission AT the
cap, not that the cap removed something -- a substrate whose budgeted pool is exactly fifteen emits
at the cap losing nothing, and no artifact stores the pool size. The field is named accordingly.
This counts how often the output reaches the cap and where the lost references sit.

Reads only results/recall_factorization.json (per-substrate U/Cfull/Cbud/H plus the deployed
top-15 list), so it adds no new measurement -- it reports a partition of one already committed.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "recall_factorization.json"
OUT = ROOT / "results" / "truncation_binding.json"


def main() -> int:
    d = json.loads(SRC.read_text())
    ps = d["per_substrate"]
    k = d["k"]
    n = len(ps)
    at_cap = [r for r in ps if len(r["deployed_top15"]) >= k]
    losing = [r for r in ps if r["H"] < r["Cbud"]]
    losing_at_cap = [r for r in losing if len(r["deployed_top15"]) >= k]
    sum_cbud = sum(r["Cbud"] for r in ps)
    sum_h = sum(r["H"] for r in ps)

    rep = {
        "source": str(SRC.relative_to(ROOT)),
        "k": k,
        "n_substrates": n,
        "n_emitting_at_cap": len(at_cap),
        "share_emitting_at_cap": len(at_cap) / n,
        "n_substrates_losing_between_Cbud_and_H": len(losing),
        "n_losing_substrates_that_are_at_cap": len(losing_at_cap),
        "references_lost_between_Cbud_and_H": sum_cbud - sum_h,
        "sum_Cbud": sum_cbud,
        "sum_H": sum_h,
        "ranking_factor": sum_h / sum_cbud,
        "gate": "ranking_factor must reproduce factors.ranking_conversion.point",
        "gate_committed": d["factors"]["ranking_conversion"]["point"],
        "note": "every substrate that loses a reference between the budgeted pool and the emitted "
                "output is one whose emission is at the cap, so the third factor's loss is "
                "truncation at the deployed budget rather than a score threshold discarding "
                "candidates below the cap",
    }
    assert abs(rep["ranking_factor"] - rep["gate_committed"]) < 1e-12, "gate failed"
    OUT.write_text(json.dumps(rep, indent=1))
    for key, val in rep.items():
        if key not in ("note", "gate", "source"):
            print(f"{key:42} {val}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

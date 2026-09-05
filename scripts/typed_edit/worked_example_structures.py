#!/usr/bin/env python3
"""The compounds the manuscript names, as structures rather than as hashes.

This journal's submission checklist asks for SMILES for the compounds represented in a manuscript,
and the only compounds this one represents individually are the worked example's: gemcitabine and
the four metabolites the corpus annotates for it. Everywhere else the paper speaks of populations,
and those are released as the descriptors the matching criteria are decided by, for the licence
reason the availability section gives.

Nothing here is a new measurement. The substrate is the one the case-study artifact ran on and the
four references are the candidates at the ranks that artifact records as reference hits, so the
structures come out of the run rather than being typed from a reference work. The gate is that
those four ranks must all be hits and must number what the artifact says the reference set holds.

    python scripts/typed_edit/worked_example_structures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _provenance import record_inputs, stamp  # noqa: E402

CASE = ROOT / "results" / "case_study_exhaustive.json"


def main() -> int:
    d = json.loads(CASE.read_text())
    cands, ranks = d["candidates"], d["reference_ranks"]
    if len(ranks) != d["n_references"]:
        print(f"REFUSING: the artifact records {d['n_references']} references and "
              f"{len(ranks)} reference ranks", file=sys.stderr)
        return 1

    hits = []
    for r in sorted(ranks):
        if r < 1 or r > len(cands):
            print(f"REFUSING: reference rank {r} is outside the candidate list of "
                  f"{len(cands)}", file=sys.stderr)
            return 1
        c = cands[r - 1]
        hits.append({"rank": r, "smiles": c["smiles"], "inchikey": c["key"],
                     "rule_id": c.get("rule_id"), "rule_source": c.get("rule_source")})

    report = {
        "provenance": stamp(__file__),
        "inputs": record_inputs([CASE]),
        "what_this_is": ("the compounds the manuscript represents individually, as SMILES, for the "
                         "submission checklist that asks for them"),
        "why_only_these": ("every other compound in this paper is a member of a population, and "
                           "those are released as matching descriptors rather than as structures "
                           "for the licence reason the availability section states"),
        "substrate": {"smiles": d["substrate"],
                      "as_the_corpus_stores_it": d["corpus_substrate"],
                      "name": "gemcitabine"},
        "annotated_metabolites": hits,
        "note": ("the metabolite structures are the candidates at the ranks the case-study "
                 "artifact records as reference hits, so they are the run's own output and not a "
                 "transcription"),
    }
    out = ROOT / "results" / "worked_example_structures.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"substrate and {len(hits)} annotated metabolites")
    for h in hits:
        print(f"  rank {h['rank']:>3}  {h['smiles']}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
